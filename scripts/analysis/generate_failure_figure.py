#!/usr/bin/env python3
"""
Generate a failure case figure in the same style as the scanner overlay figure.

For each failure subject, shows a single axial slice centered on the failure
point with filled semi-transparent overlays (colorblind-friendly palette):
  - Blue: true positive (correctly detected marker)
  - Orange: false negative (missed marker)
  - Reddish purple: false positive (spurious detection)
"""

import os
import sys
import glob

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def select_top_n_markers(probability_map, n_markers=3, threshold=0.05,
                         confidence_floor=0.2):
    structure = np.ones((3, 3, 3), dtype=bool)
    binary_mask = (probability_map > threshold).astype(np.int32)
    labeled_array, num_features = scipy.ndimage.label(binary_mask,
                                                      structure=structure)
    if num_features == 0:
        return np.zeros_like(probability_map, dtype=np.uint8)

    component_scores = []
    for label_id in range(1, num_features + 1):
        mask = (labeled_array == label_id)
        mean_prob = probability_map[mask].mean()
        component_scores.append((label_id, mean_prob))
    component_scores.sort(key=lambda x: x[1], reverse=True)

    output_mask = np.zeros_like(probability_map, dtype=np.uint8)
    for lid, score in component_scores[:n_markers]:
        if score >= confidence_floor:
            output_mask[labeled_array == lid] = 1
    return output_mask


def run_consensus_inference(mri_path, model_paths, device='cpu'):
    """Run 4-model consensus inference and return seed segmentation."""
    img = nib.load(mri_path)
    volume = img.get_fdata().astype(np.float32)
    original_shape = volume.shape

    mean_val = volume.mean()
    std_val = volume.std()
    if std_val > 0:
        volume = (volume - mean_val) / std_val

    padded_shape = find_nearest_compatible_size(original_shape)
    padded = pad_or_crop_numpy(volume, padded_shape)
    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(device)

    prob_sum = None
    for mp in model_paths:
        model = UNet3D(in_channels=1, out_channels=3)
        ckpt = torch.load(mp, map_location=device)
        model.load_state_dict(ckpt['model_state_dict']
                              if 'model_state_dict' in ckpt else ckpt)
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            marker_prob = probs[0, 1].cpu().numpy()
        marker_prob = pad_or_crop_numpy(marker_prob, original_shape)
        if prob_sum is None:
            prob_sum = marker_prob
        else:
            prob_sum += marker_prob
        del model

    consensus_prob = prob_sum / len(model_paths)
    seed_mask = select_top_n_markers(consensus_prob)
    return seed_mask


def find_failures(pred_seg, targ_seg):
    """Identify TP, FN, FP components and their centroids."""
    structure = np.ones((3, 3, 3), dtype=bool)

    pred_dilated = scipy.ndimage.binary_dilation(
        (pred_seg > 0).astype(np.int32)
    ).astype(np.int32)
    targ_dilated = scipy.ndimage.binary_dilation(
        (targ_seg == 1).astype(np.int32)
    ).astype(np.int32)

    pred_labeled, pred_n = scipy.ndimage.label(pred_dilated, structure=structure)
    targ_labeled, targ_n = scipy.ndimage.label(targ_dilated, structure=structure)

    detected_targets = set()
    matched_preds = set()
    for t_id in range(1, targ_n + 1):
        t_mask = (targ_labeled == t_id)
        overlapping_preds = set(pred_labeled[t_mask]) - {0}
        if overlapping_preds:
            detected_targets.add(t_id)
            matched_preds.update(overlapping_preds)

    failures = []
    for t_id in range(1, targ_n + 1):
        centroid = tuple(int(round(c)) for c in
                         scipy.ndimage.center_of_mass(targ_labeled == t_id))
        if t_id in detected_targets:
            failures.append({'type': 'TP', 'centroid': centroid})
        else:
            failures.append({'type': 'FN', 'centroid': centroid})

    for p_id in range(1, pred_n + 1):
        if p_id not in matched_preds:
            centroid = tuple(int(round(c)) for c in
                             scipy.ndimage.center_of_mass(pred_labeled == p_id))
            failures.append({'type': 'FP', 'centroid': centroid})

    return failures, pred_labeled, targ_labeled, detected_targets, matched_preds


def build_classified_mask(pred_seg, targ_seg):
    """Build a per-voxel classification mask where each marker component gets
    exactly one label: 1=TP (blue), 2=FN (orange), 3=FP (reddish purple).

    - TP: ground truth components that overlap a prediction -> show GT region
    - FN: ground truth components with no overlapping prediction -> show GT region
    - FP: predicted components with no overlapping ground truth -> show pred region
    """
    structure = np.ones((3, 3, 3), dtype=bool)

    pred_dilated = scipy.ndimage.binary_dilation(
        (pred_seg > 0).astype(np.int32)
    ).astype(np.int32)
    targ_dilated = scipy.ndimage.binary_dilation(
        (targ_seg == 1).astype(np.int32)
    ).astype(np.int32)

    pred_labeled, pred_n = scipy.ndimage.label(pred_dilated, structure=structure)
    targ_labeled, targ_n = scipy.ndimage.label(targ_dilated, structure=structure)

    # Match targets to predictions
    detected_targets = set()
    matched_preds = set()
    for t_id in range(1, targ_n + 1):
        t_mask = (targ_labeled == t_id)
        overlapping_preds = set(pred_labeled[t_mask]) - {0}
        if overlapping_preds:
            detected_targets.add(t_id)
            matched_preds.update(overlapping_preds)

    # Build output: 0=background, 1=TP, 2=FN, 3=FP
    classified = np.zeros_like(pred_seg, dtype=np.uint8)

    # TP: detected ground truth components (show GT region)
    for t_id in detected_targets:
        classified[targ_labeled == t_id] = 1

    # FN: undetected ground truth components (show GT region)
    for t_id in range(1, targ_n + 1):
        if t_id not in detected_targets:
            classified[targ_labeled == t_id] = 2

    # FP: unmatched predicted components (show pred region)
    for p_id in range(1, pred_n + 1):
        if p_id not in matched_preds:
            classified[pred_labeled == p_id] = 3

    return classified


def make_overlay_slice(mri_slice, classified_slice):
    """Create an RGB image with filled colored overlays on grayscale MRI.
    classified_slice values: 0=bg, 1=TP(blue), 2=FN(orange), 3=FP(purple)."""
    vmin = np.percentile(mri_slice[mri_slice > 0], 1) if np.any(mri_slice > 0) else 0
    vmax = np.percentile(mri_slice[mri_slice > 0], 99) if np.any(mri_slice > 0) else 1
    if vmax <= vmin:
        vmax = vmin + 1
    mri_norm = np.clip((mri_slice - vmin) / (vmax - vmin), 0, 1)

    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    alpha = 0.45

    # Colorblind-friendly palette (Wong, Nature Methods 2011)
    colors = {
        1: np.array([0.00, 0.45, 0.70]),  # TP = blue
        2: np.array([0.90, 0.62, 0.00]),  # FN = orange
        3: np.array([0.80, 0.47, 0.65]),  # FP = reddish purple
    }

    for label, color in colors.items():
        mask = (classified_slice == label)
        if np.any(mask):
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

    return np.clip(rgb, 0, 1)


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    os.chdir(project_dir)

    output_dir = 'paper_draft/images'
    os.makedirs(output_dir, exist_ok=True)

    model_paths = sorted(glob.glob('models/production/*-best.pth'))
    print(f"Found {len(model_paths)} production models")

    # All 5 failure subjects
    failure_subjects = ['S044', 'S078', 'S110']

    def get_paths(sid):
        for base in ['data/test/prepared', 'data/train']:
            for ext in ['', '.gz']:
                p = f'{base}/{sid}/{sid}_MRI_homogeneity-corrected.nii{ext}'
                if os.path.exists(p):
                    seeds_dir = f'{base}/{sid}/roi_niftis_mri_space'
                    seeds_file = f'{seeds_dir}/{sid}_seeds.nii.gz'
                    if os.path.exists(seeds_file):
                        return p, seeds_file
        return None, None

    subject_data = []
    for sid in failure_subjects:
        mri_path, seeds_path = get_paths(sid)
        if mri_path is None:
            print(f"WARNING: Could not find data for {sid}, skipping")
            continue

        print(f"Processing {sid}")
        mri = nib.load(mri_path).get_fdata().astype(np.float32)
        targ_seg = nib.load(seeds_path).get_fdata().astype(np.int32)
        pred_seg = run_consensus_inference(mri_path, model_paths)

        results = find_failures(pred_seg, targ_seg)
        failures, pred_labeled, targ_labeled, detected_targets, matched_preds = results

        # Build per-voxel classified mask (1=TP, 2=FN, 3=FP)
        classified = build_classified_mask(pred_seg, targ_seg)

        # Find the most interesting failure slice
        fn_fps = [f for f in failures if f['type'] in ('FN', 'FP')]
        if not fn_fps:
            print(f"  No failures found, skipping")
            continue

        fn_entries = [f for f in fn_fps if f['type'] == 'FN']
        fp_entries = [f for f in fn_fps if f['type'] == 'FP']

        # Use the FN centroid if available (shows the missed marker)
        if fn_entries:
            focus = fn_entries[0]
        else:
            focus = fp_entries[0]

        focus_slice = focus['centroid'][2]
        print(f"  Failures: {len(fn_entries)} FN, {len(fp_entries)} FP, "
              f"showing slice {focus_slice} ({focus['type']})")

        # Crop around all marker centroids (both GT and pred)
        all_centroids = [f['centroid'] for f in failures]
        marker_rows = [c[0] for c in all_centroids]
        marker_cols = [c[1] for c in all_centroids]
        crop_margin = 50
        r_center = (min(marker_rows) + max(marker_rows)) // 2
        c_center = (min(marker_cols) + max(marker_cols)) // 2
        r_min = max(0, r_center - crop_margin)
        r_max = min(mri.shape[0], r_center + crop_margin)
        c_min = max(0, c_center - crop_margin)
        c_max = min(mri.shape[1], c_center + crop_margin)

        mri_crop = mri[r_min:r_max, c_min:c_max, focus_slice]
        classified_crop = classified[r_min:r_max, c_min:c_max, focus_slice]

        overlay = make_overlay_slice(mri_crop, classified_crop)

        subject_data.append({
            'id': sid,
            'overlay': overlay,
            'failures': failures,
            'fn_count': len(fn_entries),
            'fp_count': len(fp_entries),
            'focus_type': focus['type'],
        })

    if not subject_data:
        print("No subjects processed!")
        return

    # --- Generate figure: 1 row ---
    n = len(subject_data)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8))
    if n == 1:
        axes = [axes]

    for col, sd in enumerate(subject_data):
        axes[col].imshow(sd['overlay'].transpose(1, 0, 2), origin='lower',
                         aspect='equal')
        axes[col].set_xticks([])
        axes[col].set_yticks([])

        # Title with failure summary
        fn_str = f"{sd['fn_count']} FN" if sd['fn_count'] > 0 else ""
        fp_str = f"{sd['fp_count']} FP" if sd['fp_count'] > 0 else ""
        summary = ", ".join(filter(None, [fn_str, fp_str]))
        axes[col].set_title(f'Subject {col + 1}\n({summary})',
                            fontsize=11, fontweight='bold', pad=8)

    # Legend
    legend_elements = [
        Patch(facecolor=(0.00, 0.45, 0.70, 0.7), edgecolor='black',
              linewidth=0.5, label='True positive'),
        Patch(facecolor=(0.90, 0.62, 0.00, 0.7), edgecolor='black',
              linewidth=0.5, label='False negative'),
        Patch(facecolor=(0.80, 0.47, 0.65, 0.7), edgecolor='black',
              linewidth=0.5, label='False positive'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.subplots_adjust(wspace=0.08)

    out_path = os.path.join(output_dir, 'failure_cases.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
