#!/usr/bin/env python3
"""
Ablation study: how does the number of models in the consensus affect performance?

Tests 1, 2, 3, and 4 models. For subsets of models, tests all combinations
and reports mean/std to avoid bias from a particular model selection.

Precomputes per-model probability maps once, then sweeps combinations.
"""

import os
import sys
import json
import glob
from itertools import combinations

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
    return tuple(
        ((dim + min_factor - 1) // min_factor) * min_factor
        for dim in input_shape
    )


def pad_or_crop_numpy(vol, target):
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
    structure = np.ones((3, 3, 3), dtype=bool)

    pred_marker = scipy.ndimage.binary_dilation((pred_seg == 1).astype(np.int32)).astype(np.int32)
    targ_marker = scipy.ndimage.binary_dilation((targ_seg == 1).astype(np.int32)).astype(np.int32)

    pred_labeled, pred_n = scipy.ndimage.label(pred_marker, structure=structure)
    targ_labeled, targ_n = scipy.ndimage.label(targ_marker, structure=structure)

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


def evaluate_combination(subject_data, model_indices, threshold=0.1):
    """Evaluate a specific combination of models on all subjects."""
    total_tp = total_fn = total_fp = total_actual = 0
    n_perfect = 0

    for sd in subject_data:
        # Average probability maps for selected models
        selected_probs = [sd['per_model_probs'][i] for i in model_indices]
        avg_prob = np.mean(selected_probs, axis=0)

        consensus_seg = select_top_n_markers(avg_prob, n_markers=3, threshold=threshold)
        metrics = compute_subject_metrics(consensus_seg, sd['seeds_data'])

        total_tp += metrics['true_positive']
        total_fn += metrics['false_negative']
        total_fp += metrics['false_positive']
        total_actual += metrics['actual_markers']

        if metrics['false_negative'] == 0 and metrics['false_positive'] == 0:
            n_perfect += 1

    sens = total_tp / total_actual if total_actual > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    return {
        'sensitivity': sens,
        'precision': prec,
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
        'actual': total_actual,
        'n_perfect': n_perfect,
    }


def main():
    model_dir = 'models/production'
    data_dir = 'data/test/prepared'
    splits_path = 'data/splits.json'

    print("=" * 80)
    print("MODEL COUNT ABLATION STUDY")
    print("=" * 80)

    with open(splits_path) as f:
        splits = json.load(f)

    val_subjects = sorted(splits['val'])
    test_subjects = sorted(splits['test'])
    all_subjects = sorted(val_subjects + test_subjects)

    model_paths = sorted(glob.glob(os.path.join(model_dir, '*-best.pth')))
    n_models = len(model_paths)
    print(f"Models: {n_models}")
    for p in model_paths:
        print(f"  {os.path.basename(p)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    models = []
    for mp in model_paths:
        net = UNet3D(in_channels=1, out_channels=3).to(device)
        ckpt = torch.load(mp, map_location=device)
        if 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        models.append(net)
    print(f"Loaded {len(models)} models\n")

    # Precompute per-model probability maps for all subjects
    print("Precomputing per-model probability maps for all 34 subjects...")
    subject_data_map = {}
    for i, subject_id in enumerate(all_subjects):
        subject_dir = os.path.join(data_dir, subject_id)
        mri_path = os.path.join(subject_dir, f"{subject_id}_MRI_homogeneity-corrected.nii")
        seeds_path = os.path.join(subject_dir, "roi_niftis_mri_space", f"{subject_id}_seeds.nii.gz")

        if not os.path.exists(mri_path) or not os.path.exists(seeds_path):
            print(f"  SKIP {subject_id}")
            continue

        img = tio.ScalarImage(mri_path)
        orig_shape = img.data.numpy()[0].shape
        compatible_shape = find_nearest_compatible_size(orig_shape)
        sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
        input_tensor = sample.data.unsqueeze(0).to(device)

        seeds_nii = nib.load(seeds_path)
        seeds_data = seeds_nii.get_fdata().astype(np.int32)

        per_model_probs = []
        with torch.no_grad():
            for net in models:
                outputs = net(input_tensor)
                prob_maps = F.softmax(outputs, dim=1).cpu().numpy()[0]
                seed_prob = pad_or_crop_numpy(prob_maps[1], orig_shape)
                per_model_probs.append(seed_prob)

        subject_data_map[subject_id] = {
            'subject_id': subject_id,
            'per_model_probs': per_model_probs,
            'seeds_data': seeds_data,
        }
        print(f"  [{i+1}/{len(all_subjects)}] {subject_id} done")

    print(f"\nPrecomputed {len(subject_data_map)} subjects\n")

    val_data = [subject_data_map[s] for s in val_subjects if s in subject_data_map]
    test_data = [subject_data_map[s] for s in test_subjects if s in subject_data_map]
    all_data = val_data + test_data

    # Run ablation for each model count
    model_indices = list(range(n_models))
    seed_names = ['42', '123', '456', '789']

    summary_rows = []

    for n in range(1, n_models + 1):
        combos = list(combinations(model_indices, n))
        print("=" * 80)
        print(f"{n} MODEL(S) — {len(combos)} combination(s)")
        print("=" * 80)

        for subset_label, data in [('val', val_data), ('test', test_data), ('all', all_data)]:
            combo_results = []
            for combo in combos:
                res = evaluate_combination(data, combo, threshold=0.1)
                combo_results.append(res)

                combo_name = '+'.join(seed_names[i] for i in combo)
                summary_rows.append({
                    'n_models': n,
                    'combination': combo_name,
                    'subset': subset_label,
                    'sensitivity': res['sensitivity'],
                    'precision': res['precision'],
                    'tp': res['tp'], 'fn': res['fn'], 'fp': res['fp'],
                    'actual': res['actual'],
                    'n_perfect': res['n_perfect'],
                })

            sensitivities = [r['sensitivity'] for r in combo_results]
            precisions = [r['precision'] for r in combo_results]
            perfects = [r['n_perfect'] for r in combo_results]

            if len(combos) > 1:
                print(f"  {subset_label:5s}: Sens={np.mean(sensitivities):.4f}±{np.std(sensitivities):.4f}  "
                      f"Prec={np.mean(precisions):.4f}±{np.std(precisions):.4f}  "
                      f"Perfect={np.mean(perfects):.1f}±{np.std(perfects):.1f}/{len(data)}")
            else:
                print(f"  {subset_label:5s}: Sens={sensitivities[0]:.4f}  "
                      f"Prec={precisions[0]:.4f}  "
                      f"Perfect={perfects[0]}/{len(data)}")

            # Also print individual combos for small counts
            if len(combos) <= 6:
                for combo, res in zip(combos, combo_results):
                    combo_name = '+'.join(seed_names[i] for i in combo)
                    print(f"    {combo_name:20s}  Sens={res['sensitivity']:.4f}  Prec={res['precision']:.4f}  Perfect={res['n_perfect']}/{len(data)}")

        print()

    # Save
    df = pd.DataFrame(summary_rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/model_count_ablation.csv', index=False)
    print(f"Results saved to: results/model_count_ablation.csv")


if __name__ == '__main__':
    main()
