#!/usr/bin/env python3
"""
Model count ablation that also saves per-subject per-combination metrics so
paired tests (Wilcoxon signed-rank etc.) can be run on the test set.

Splits: val = data/val_subjects.txt, test = remaining prepared subjects.

Outputs:
  results/model_count_ablation_summary.csv     -- one row per (n_models, combination, subset)
  results/model_count_ablation_per_subject.csv -- one row per (n_models, combination, subset, subject)
"""

import os
import sys
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
    return tuple(((dim + min_factor - 1) // min_factor) * min_factor for dim in input_shape)


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
            pad_width.append((before, diff - before))
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
        component_scores.append((label_id, probability_map[mask].mean()))
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
    return {
        'predicted_markers': pred_n,
        'actual_markers': targ_n,
        'true_positive': tp,
        'false_negative': targ_n - tp,
        'false_positive': pred_n - len(matched_preds),
    }


def main():
    model_dir = 'models/production'
    data_dir = 'data/test/prepared'
    val_subjects_path = 'data/val_subjects.txt'

    print("=" * 80)
    print("MODEL COUNT ABLATION (with per-subject output)")
    print("=" * 80)

    with open(val_subjects_path) as f:
        val_subjects = sorted({l.strip() for l in f if l.strip()})
    all_prepared = sorted(d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d)))
    test_subjects = sorted(set(all_prepared) - set(val_subjects))
    all_subjects = sorted(val_subjects + test_subjects)
    print(f"Val: {len(val_subjects)}   Test: {len(test_subjects)}   Total: {len(all_subjects)}")

    model_paths = sorted(glob.glob(os.path.join(model_dir, '*-best.pth')))
    n_models = len(model_paths)
    print(f"Models: {n_models}")
    for p in model_paths:
        print(f"  {os.path.basename(p)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    models = []
    for mp in model_paths:
        net = UNet3D(in_channels=1, out_channels=3).to(device)
        ckpt = torch.load(mp, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        models.append(net)
    print(f"Loaded {len(models)} models\n")

    print(f"Precomputing per-model probability maps for {len(all_subjects)} subjects...")
    subject_data_map = {}
    for i, subject_id in enumerate(all_subjects):
        subject_dir = os.path.join(data_dir, subject_id)
        mri_path = os.path.join(subject_dir, f"{subject_id}_MRI_homogeneity-corrected.nii")
        seeds_path = os.path.join(subject_dir, "roi_niftis_mri_space", f"{subject_id}_seeds.nii.gz")
        if not os.path.exists(mri_path) or not os.path.exists(seeds_path):
            print(f"  SKIP {subject_id} (missing files)")
            continue
        img = tio.ScalarImage(mri_path)
        orig_shape = img.data.numpy()[0].shape
        compatible_shape = find_nearest_compatible_size(orig_shape)
        sample = tio.ZNormalization()(tio.CropOrPad(compatible_shape)(img))
        input_tensor = sample.data.unsqueeze(0).to(device)
        seeds_data = nib.load(seeds_path).get_fdata().astype(np.int32)
        per_model_probs = []
        with torch.no_grad():
            for net in models:
                outputs = net(input_tensor)
                prob_maps = F.softmax(outputs, dim=1).cpu().numpy()[0]
                per_model_probs.append(pad_or_crop_numpy(prob_maps[1], orig_shape))
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

    model_indices = list(range(n_models))
    seed_names = ['42', '123', '456', '789']

    summary_rows = []
    per_subject_rows = []

    for n in range(1, n_models + 1):
        combos = list(combinations(model_indices, n))
        print("=" * 80)
        print(f"{n} MODEL(S) -- {len(combos)} combination(s)")
        print("=" * 80)

        for subset_label, data in [('val', val_data), ('test', test_data), ('all', all_data)]:
            for combo in combos:
                combo_name = '+'.join(seed_names[i] for i in combo)
                total_tp = total_fn = total_fp = total_actual = 0
                n_perfect = 0
                for sd in data:
                    selected_probs = [sd['per_model_probs'][i] for i in combo]
                    avg_prob = np.mean(selected_probs, axis=0)
                    seg = select_top_n_markers(avg_prob, n_markers=3, threshold=0.1)
                    m = compute_subject_metrics(seg, sd['seeds_data'])
                    per_subject_rows.append({
                        'n_models': n,
                        'combination': combo_name,
                        'subset': subset_label,
                        'subject_id': sd['subject_id'],
                        'actual_markers': m['actual_markers'],
                        'predicted_markers': m['predicted_markers'],
                        'true_positive': m['true_positive'],
                        'false_negative': m['false_negative'],
                        'false_positive': m['false_positive'],
                        'sensitivity': m['true_positive'] / m['actual_markers'] if m['actual_markers'] else float('nan'),
                        'precision': m['true_positive'] / (m['true_positive'] + m['false_positive']) if (m['true_positive'] + m['false_positive']) else float('nan'),
                        'perfect': int(m['false_negative'] == 0 and m['false_positive'] == 0),
                    })
                    total_tp += m['true_positive']
                    total_fn += m['false_negative']
                    total_fp += m['false_positive']
                    total_actual += m['actual_markers']
                    if m['false_negative'] == 0 and m['false_positive'] == 0:
                        n_perfect += 1
                sens = total_tp / total_actual if total_actual else 0
                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
                summary_rows.append({
                    'n_models': n,
                    'combination': combo_name,
                    'subset': subset_label,
                    'sensitivity': sens,
                    'precision': prec,
                    'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
                    'actual': total_actual,
                    'n_perfect': n_perfect,
                    'n_subjects': len(data),
                })
            sub_rows = [r for r in summary_rows if r['n_models']==n and r['subset']==subset_label]
            sens_arr = [r['sensitivity'] for r in sub_rows]
            prec_arr = [r['precision'] for r in sub_rows]
            if len(sens_arr) > 1:
                print(f"  {subset_label:5s}: Sens={np.mean(sens_arr):.4f}+/-{np.std(sens_arr):.4f}  Prec={np.mean(prec_arr):.4f}+/-{np.std(prec_arr):.4f}")
            else:
                print(f"  {subset_label:5s}: Sens={sens_arr[0]:.4f}  Prec={prec_arr[0]:.4f}")
        print()

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(summary_rows).to_csv('results/model_count_ablation_summary.csv', index=False)
    pd.DataFrame(per_subject_rows).to_csv('results/model_count_ablation_per_subject.csv', index=False)
    print("Saved:")
    print("  results/model_count_ablation_summary.csv     ({} rows)".format(len(summary_rows)))
    print("  results/model_count_ablation_per_subject.csv ({} rows)".format(len(per_subject_rows)))


if __name__ == '__main__':
    main()
