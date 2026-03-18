#!/usr/bin/env python3
"""
Evaluate production models on the held-out test set using consensus inference.

For each test subject:
1. Run each production model to get probability maps
2. Average probability maps across models
3. Select top-3 markers from averaged probabilities
4. Compare against ground truth segmentation

Reports per-subject and aggregate sensitivity/precision.
"""

import os
import sys
import json
import argparse
import glob

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import torchio as tio
import scipy.ndimage
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import UNet3D


def find_nearest_compatible_size(input_shape, min_factor=32):
    """Find nearest larger size divisible by min_factor."""
    return tuple(
        ((dim + min_factor - 1) // min_factor) * min_factor
        for dim in input_shape
    )


def pad_or_crop_numpy(vol, target):
    """Crop/pad volume to target shape."""
    slices = []
    for i in range(3):
        diff = vol.shape[i] - target[i]
        if diff > 0:
            start = diff // 2
            slices.append(slice(start, start + target[i]))
        else:
            slices.append(slice(0, vol.shape[i]))
    vol_c = vol[tuple(slices)]

    pad_width = []
    for i in range(3):
        diff = target[i] - vol_c.shape[i]
        if diff > 0:
            before = diff // 2
            after = diff - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))
    return np.pad(vol_c, pad_width, mode='constant', constant_values=0)


def select_top_n_markers(probability_map, n_markers=3, threshold=0.1):
    """Select top N markers from probability map based on mean probability."""
    structure = np.ones((3, 3, 3), dtype=bool)
    binary_mask = (probability_map > threshold).astype(np.int32)
    labeled_array, num_features = scipy.ndimage.label(binary_mask, structure=structure)

    if num_features == 0:
        return np.zeros_like(probability_map, dtype=np.uint8)

    if num_features <= n_markers:
        return binary_mask.astype(np.uint8)

    component_scores = []
    for label_id in range(1, num_features + 1):
        mask = (labeled_array == label_id)
        mean_prob = probability_map[mask].mean()
        component_scores.append((label_id, mean_prob))

    component_scores.sort(key=lambda x: x[1], reverse=True)
    top_labels = [lid for lid, _ in component_scores[:n_markers]]

    output_mask = np.zeros_like(probability_map, dtype=np.uint8)
    for lid in top_labels:
        output_mask[labeled_array == lid] = 1
    return output_mask


def compute_subject_metrics(pred_seg, targ_seg):
    """Compute marker detection metrics for a single subject.

    Uses binary dilation then per-component overlap matching to determine
    true positives, false negatives, and false positives.

    A predicted component is a TP if it overlaps any target component.
    A target component is detected (TP) if any predicted component overlaps it.
    """
    structure = np.ones((3, 3, 3), dtype=bool)

    pred_marker = scipy.ndimage.binary_dilation((pred_seg == 1).astype(np.int32)).astype(np.int32)
    targ_marker = scipy.ndimage.binary_dilation((targ_seg == 1).astype(np.int32)).astype(np.int32)

    pred_labeled, pred_n = scipy.ndimage.label(pred_marker, structure=structure)
    targ_labeled, targ_n = scipy.ndimage.label(targ_marker, structure=structure)

    # For each target component, check if any predicted component overlaps
    detected_targets = set()
    matched_preds = set()
    for t_id in range(1, targ_n + 1):
        t_mask = (targ_labeled == t_id)
        overlapping_preds = set(pred_labeled[t_mask]) - {0}
        if overlapping_preds:
            detected_targets.add(t_id)
            matched_preds.update(overlapping_preds)

    tp = len(detected_targets)
    fn = targ_n - tp
    fp = pred_n - len(matched_preds)

    return {
        'predicted_markers': pred_n,
        'actual_markers': targ_n,
        'true_positive': tp,
        'false_negative': fn,
        'false_positive': fp,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate production models on test set')
    parser.add_argument('--model-dir', type=str, default='models/production',
                        help='Directory containing production model checkpoints')
    parser.add_argument('--data-dir', type=str, default='data/test/prepared',
                        help='Directory containing prepared test subjects')
    parser.add_argument('--splits', type=str, default='data/splits.json',
                        help='Path to splits.json')
    parser.add_argument('--subset', type=str, default='test',
                        choices=['test', 'val', 'all'],
                        help='Which subset to evaluate (default: test)')
    parser.add_argument('--output', type=str, default='results/production_eval.csv',
                        help='Output CSV path')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Probability threshold for marker detection')
    parser.add_argument('--n-markers', type=int, default=3,
                        help='Number of markers to select per subject')
    args = parser.parse_args()

    print("=" * 80)
    print("PRODUCTION MODEL EVALUATION")
    print("=" * 80)

    # Find models
    model_paths = sorted(glob.glob(os.path.join(args.model_dir, '*-best.pth')))
    print(f"Found {len(model_paths)} models:")
    for p in model_paths:
        print(f"  {os.path.basename(p)}")
    print()

    if len(model_paths) == 0:
        print("ERROR: No models found!")
        return

    # Load splits
    with open(args.splits) as f:
        splits = json.load(f)

    if args.subset == 'all':
        subjects = sorted(splits['val'] + splits['test'])
    else:
        subjects = sorted(splits[args.subset])

    print(f"Evaluating on {len(subjects)} {args.subset} subjects")
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load all models
    print("Loading models...")
    models = []
    for model_path in model_paths:
        net = UNet3D(in_channels=1, out_channels=3).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)
        net.eval()
        models.append(net)
    print(f"Loaded {len(models)} models")
    print()

    # Evaluate each subject
    results = []
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_actual = 0

    for i, subject_id in enumerate(subjects):
        print(f"[{i+1}/{len(subjects)}] {subject_id}")

        # Find input and ground truth
        subject_dir = os.path.join(args.data_dir, subject_id)
        mri_path = os.path.join(subject_dir, f"{subject_id}_MRI_homogeneity-corrected.nii")
        seeds_path = os.path.join(subject_dir, "roi_niftis_mri_space", f"{subject_id}_seeds.nii.gz")

        if not os.path.exists(mri_path):
            print(f"  WARNING: MRI not found: {mri_path}")
            continue
        if not os.path.exists(seeds_path):
            print(f"  WARNING: Seeds not found: {seeds_path}")
            continue

        # Load and preprocess
        img = tio.ScalarImage(mri_path)
        orig_shape = img.data.numpy()[0].shape
        compatible_shape = find_nearest_compatible_size(orig_shape)
        sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
        input_tensor = sample.data.unsqueeze(0).to(device)

        # Load ground truth
        seeds_nii = nib.load(seeds_path)
        seeds_data = seeds_nii.get_fdata().astype(np.int32)

        # Run each model and collect seed probability maps
        seed_probs = []
        with torch.no_grad():
            for net in models:
                outputs = net(input_tensor)
                prob_maps = F.softmax(outputs, dim=1).cpu().numpy()[0]
                # Crop/pad back to original shape
                seed_prob = pad_or_crop_numpy(prob_maps[1], orig_shape)
                seed_probs.append(seed_prob)

        # Average and select top-N
        avg_prob = np.mean(seed_probs, axis=0)
        consensus_seg = select_top_n_markers(avg_prob, n_markers=args.n_markers, threshold=args.threshold)

        # Compute metrics
        metrics = compute_subject_metrics(consensus_seg, seeds_data)

        sensitivity = metrics['true_positive'] / metrics['actual_markers'] if metrics['actual_markers'] > 0 else 0
        precision = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive']) if (metrics['true_positive'] + metrics['false_positive']) > 0 else 0

        print(f"  Markers: {metrics['actual_markers']}, TP: {metrics['true_positive']}, "
              f"FN: {metrics['false_negative']}, FP: {metrics['false_positive']}, "
              f"Sens: {sensitivity:.2f}, Prec: {precision:.2f}")

        results.append({
            'subject_id': subject_id,
            'actual_markers': metrics['actual_markers'],
            'predicted_markers': metrics['predicted_markers'],
            'true_positive': metrics['true_positive'],
            'false_negative': metrics['false_negative'],
            'false_positive': metrics['false_positive'],
            'sensitivity': sensitivity,
            'precision': precision,
        })

        total_tp += metrics['true_positive']
        total_fn += metrics['false_negative']
        total_fp += metrics['false_positive']
        total_actual += metrics['actual_markers']

    # Aggregate metrics
    print()
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    agg_sensitivity = total_tp / total_actual if total_actual > 0 else 0
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    print(f"Subjects evaluated: {len(results)}")
    print(f"Models used: {len(models)} (consensus)")
    print(f"Total actual markers: {total_actual}")
    print(f"True positives: {total_tp}")
    print(f"False negatives: {total_fn}")
    print(f"False positives: {total_fp}")
    print(f"Sensitivity: {agg_sensitivity:.4f}")
    print(f"Precision: {agg_precision:.4f}")
    print()

    # Per-subject summary
    df = pd.DataFrame(results)
    print("Per-subject sensitivity:")
    print(f"  Mean: {df['sensitivity'].mean():.4f} ± {df['sensitivity'].std():.4f}")
    print(f"  Min:  {df['sensitivity'].min():.4f}")
    print(f"  Max:  {df['sensitivity'].max():.4f}")
    print()
    print("Per-subject precision:")
    print(f"  Mean: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"  Min:  {df['precision'].min():.4f}")
    print(f"  Max:  {df['precision'].max():.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
