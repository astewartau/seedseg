#!/usr/bin/env python3
"""
Experiment with different consensus strategies on the validation set,
then report results on both val and test.

Strategies tested:
A) Baseline: top-3 selection, threshold=0.1 (current approach)
B) Threshold sweep: top-3 selection with different detection thresholds
C) Adaptive: threshold-only (no fixed marker count) — keep all components above confidence floor
D) Hybrid: top-3 then drop any component below a confidence floor

Precomputes probability maps once, then evaluates all strategies.
"""

import os
import sys
import json
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


def get_components(probability_map, threshold):
    """Get connected components and their mean probabilities."""
    structure = np.ones((3, 3, 3), dtype=bool)
    binary_mask = (probability_map > threshold).astype(np.int32)
    labeled_array, num_features = scipy.ndimage.label(binary_mask, structure=structure)

    components = []
    for label_id in range(1, num_features + 1):
        mask = (labeled_array == label_id)
        mean_prob = probability_map[mask].mean()
        size = mask.sum()
        components.append({
            'label_id': label_id,
            'mean_prob': mean_prob,
            'size': int(size),
            'mask': mask,
        })
    components.sort(key=lambda x: x['mean_prob'], reverse=True)
    return components, labeled_array


def strategy_top_n(components, n=3):
    """Select top N components by mean probability."""
    selected = components[:n]
    mask = np.zeros_like(components[0]['mask'], dtype=np.uint8) if components else np.array([])
    for c in selected:
        mask[c['mask']] = 1
    return mask


def strategy_adaptive(components, confidence_floor):
    """Keep all components above confidence floor."""
    selected = [c for c in components if c['mean_prob'] >= confidence_floor]
    if not components:
        return np.array([])
    mask = np.zeros_like(components[0]['mask'], dtype=np.uint8)
    for c in selected:
        mask[c['mask']] = 1
    return mask


def strategy_hybrid(components, n=3, confidence_floor=0.3):
    """Top N, then drop any below confidence floor."""
    selected = [c for c in components[:n] if c['mean_prob'] >= confidence_floor]
    if not components:
        return np.array([])
    mask = np.zeros_like(components[0]['mask'], dtype=np.uint8)
    for c in selected:
        mask[c['mask']] = 1
    return mask


def compute_subject_metrics(pred_seg, targ_seg):
    """Compute marker detection metrics using per-component overlap matching."""
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


def evaluate_strategy(subject_data, strategy_fn, strategy_kwargs, detection_threshold=0.1):
    """Evaluate a strategy across all subjects."""
    total_tp = total_fn = total_fp = total_actual = 0
    per_subject = []

    for sd in subject_data:
        components, _ = get_components(sd['avg_prob'], detection_threshold)
        if not components:
            pred_seg = np.zeros_like(sd['seeds_data'], dtype=np.uint8)
        else:
            pred_seg = strategy_fn(components, **strategy_kwargs)

        metrics = compute_subject_metrics(pred_seg, sd['seeds_data'])
        tp, fn, fp = metrics['true_positive'], metrics['false_negative'], metrics['false_positive']
        actual = metrics['actual_markers']

        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_actual += actual

        sens = tp / actual if actual > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        per_subject.append({
            'subject_id': sd['subject_id'],
            'actual': actual,
            'predicted': metrics['predicted_markers'],
            'tp': tp, 'fn': fn, 'fp': fp,
            'sens': sens, 'prec': prec,
        })

    agg_sens = total_tp / total_actual if total_actual > 0 else 0
    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    return {
        'sensitivity': agg_sens,
        'precision': agg_prec,
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
        'actual': total_actual,
        'per_subject': per_subject,
    }


def main():
    model_dir = 'models/production'
    data_dir = 'data/test/prepared'
    splits_path = 'data/splits.json'

    print("=" * 80)
    print("CONSENSUS STRATEGY EXPERIMENTS")
    print("=" * 80)

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    val_subjects = sorted(splits['val'])
    test_subjects = sorted(splits['test'])
    all_subjects = sorted(val_subjects + test_subjects)

    # Find and load models
    model_paths = sorted(glob.glob(os.path.join(model_dir, '*-best.pth')))
    print(f"Models: {len(model_paths)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

    # Precompute probability maps for all subjects
    print("Precomputing probability maps for all 34 subjects...")
    subject_data = {}
    for i, subject_id in enumerate(all_subjects):
        subject_dir = os.path.join(data_dir, subject_id)
        mri_path = os.path.join(subject_dir, f"{subject_id}_MRI_homogeneity-corrected.nii")
        seeds_path = os.path.join(subject_dir, "roi_niftis_mri_space", f"{subject_id}_seeds.nii.gz")

        if not os.path.exists(mri_path) or not os.path.exists(seeds_path):
            print(f"  SKIP {subject_id}: missing files")
            continue

        img = tio.ScalarImage(mri_path)
        orig_shape = img.data.numpy()[0].shape
        compatible_shape = find_nearest_compatible_size(orig_shape)
        sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
        input_tensor = sample.data.unsqueeze(0).to(device)

        seeds_nii = nib.load(seeds_path)
        seeds_data = seeds_nii.get_fdata().astype(np.int32)

        seed_probs = []
        with torch.no_grad():
            for net in models:
                outputs = net(input_tensor)
                prob_maps = F.softmax(outputs, dim=1).cpu().numpy()[0]
                seed_prob = pad_or_crop_numpy(prob_maps[1], orig_shape)
                seed_probs.append(seed_prob)

        avg_prob = np.mean(seed_probs, axis=0)

        subject_data[subject_id] = {
            'subject_id': subject_id,
            'avg_prob': avg_prob,
            'seeds_data': seeds_data,
        }
        print(f"  [{i+1}/{len(all_subjects)}] {subject_id} done")

    print(f"\nPrecomputed {len(subject_data)} subjects\n")

    val_data = [subject_data[s] for s in val_subjects if s in subject_data]
    test_data = [subject_data[s] for s in test_subjects if s in subject_data]
    all_data = val_data + test_data

    # ================================================================
    # Run experiments
    # ================================================================
    summary_rows = []

    def run_and_report(name, data_label, data, strategy_fn, strategy_kwargs, det_threshold=0.1):
        res = evaluate_strategy(data, strategy_fn, strategy_kwargs, detection_threshold=det_threshold)
        summary_rows.append({
            'strategy': name,
            'subset': data_label,
            'det_threshold': det_threshold,
            'subjects': len(data),
            'actual': res['actual'],
            'tp': res['tp'], 'fn': res['fn'], 'fp': res['fp'],
            'sensitivity': res['sensitivity'],
            'precision': res['precision'],
        })
        return res

    # --- A) Baseline: top-3, threshold=0.1 ---
    print("=" * 80)
    print("A) BASELINE: top-3, detection threshold=0.1")
    print("=" * 80)
    for label, data in [('val', val_data), ('test', test_data), ('all', all_data)]:
        res = run_and_report('A_baseline_top3', label, data, strategy_top_n, {'n': 3}, det_threshold=0.1)
        print(f"  {label:5s}: Sens={res['sensitivity']:.4f}  Prec={res['precision']:.4f}  TP={res['tp']} FN={res['fn']} FP={res['fp']}")
    print()

    # --- B) Detection threshold sweep (still top-3) ---
    print("=" * 80)
    print("B) THRESHOLD SWEEP: top-3 with different detection thresholds")
    print("=" * 80)
    for det_thresh in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        for label, data in [('val', val_data), ('test', test_data)]:
            res = run_and_report(f'B_top3_det{det_thresh}', label, data, strategy_top_n, {'n': 3}, det_threshold=det_thresh)
        val_res = [r for r in summary_rows if r['strategy'] == f'B_top3_det{det_thresh}' and r['subset'] == 'val'][-1]
        test_res = [r for r in summary_rows if r['strategy'] == f'B_top3_det{det_thresh}' and r['subset'] == 'test'][-1]
        print(f"  det={det_thresh:.2f}  val: Sens={val_res['sensitivity']:.4f} Prec={val_res['precision']:.4f}  |  test: Sens={test_res['sensitivity']:.4f} Prec={test_res['precision']:.4f}")
    print()

    # --- C) Adaptive: no fixed count, confidence floor only ---
    print("=" * 80)
    print("C) ADAPTIVE: keep all components above confidence floor")
    print("=" * 80)
    for conf_floor in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        for label, data in [('val', val_data), ('test', test_data)]:
            res = run_and_report(f'C_adaptive_cf{conf_floor}', label, data, strategy_adaptive, {'confidence_floor': conf_floor}, det_threshold=0.05)
        val_res = [r for r in summary_rows if r['strategy'] == f'C_adaptive_cf{conf_floor}' and r['subset'] == 'val'][-1]
        test_res = [r for r in summary_rows if r['strategy'] == f'C_adaptive_cf{conf_floor}' and r['subset'] == 'test'][-1]
        print(f"  floor={conf_floor:.2f}  val: Sens={val_res['sensitivity']:.4f} Prec={val_res['precision']:.4f}  |  test: Sens={test_res['sensitivity']:.4f} Prec={test_res['precision']:.4f}")
    print()

    # --- D) Hybrid: top-3 then drop below confidence floor ---
    print("=" * 80)
    print("D) HYBRID: top-3, drop below confidence floor")
    print("=" * 80)
    for conf_floor in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        for label, data in [('val', val_data), ('test', test_data)]:
            res = run_and_report(f'D_hybrid_cf{conf_floor}', label, data, strategy_hybrid, {'n': 3, 'confidence_floor': conf_floor}, det_threshold=0.05)
        val_res = [r for r in summary_rows if r['strategy'] == f'D_hybrid_cf{conf_floor}' and r['subset'] == 'val'][-1]
        test_res = [r for r in summary_rows if r['strategy'] == f'D_hybrid_cf{conf_floor}' and r['subset'] == 'test'][-1]
        print(f"  floor={conf_floor:.2f}  val: Sens={val_res['sensitivity']:.4f} Prec={val_res['precision']:.4f}  |  test: Sens={test_res['sensitivity']:.4f} Prec={test_res['precision']:.4f}")
    print()

    # Save all results
    df = pd.DataFrame(summary_rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/threshold_experiments.csv', index=False)
    print(f"Full results saved to: results/threshold_experiments.csv")

    # Print best strategies by F1 on val set
    print()
    print("=" * 80)
    print("BEST STRATEGIES BY F1 ON VALIDATION SET")
    print("=" * 80)
    val_df = df[df['subset'] == 'val'].copy()
    val_df['f1'] = 2 * val_df['sensitivity'] * val_df['precision'] / (val_df['sensitivity'] + val_df['precision']).replace(0, 1)
    val_df = val_df.sort_values('f1', ascending=False)

    for _, row in val_df.head(10).iterrows():
        # Find corresponding test result
        test_match = df[(df['strategy'] == row['strategy']) & (df['subset'] == 'test')]
        if len(test_match) > 0:
            tr = test_match.iloc[0]
            print(f"  {row['strategy']:30s}  val: Sens={row['sensitivity']:.4f} Prec={row['precision']:.4f} F1={row['f1']:.4f}  |  test: Sens={tr['sensitivity']:.4f} Prec={tr['precision']:.4f}")
        else:
            print(f"  {row['strategy']:30s}  val: Sens={row['sensitivity']:.4f} Prec={row['precision']:.4f} F1={row['f1']:.4f}")


if __name__ == '__main__':
    main()
