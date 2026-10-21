#!/usr/bin/env python3
"""
Generate a figure showing scanner-style marker segmentation overlays.

Replicates the OpenRecon on-scanner visualization approach:
  1. Dilate seed mask (3x3x3)
  2. Fill holes
  3. Find outer boundaries
  4. Set boundary voxels to max intensity (bright white outlines)

Layout: 2 rows x 3 columns
  Top row:    plain T1-weighted MRI for 3 subjects
  Bottom row: same slices with marker boundary overlay
"""

import os
import sys
import glob

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import scipy.ndimage
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation as sk_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def openrecon_overlay(data, seed_mask):
    """Apply OpenRecon-style boundary overlay: dilate, fill, find boundaries,
    set to max intensity."""
    overlay = data.copy()
    mask = (seed_mask > 0)
    mask = sk_dilation(mask, footprint=np.ones((3, 3, 3)))
    mask = binary_fill_holes(mask)
    boundaries = find_boundaries(mask, mode='outer')
    overlay[boundaries] = overlay.max()
    return overlay, boundaries


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


def find_marker_centroids(seed_mask):
    """Find centroids of each detected marker."""
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n = scipy.ndimage.label(seed_mask, structure=structure)
    centroids = []
    for i in range(1, n + 1):
        com = scipy.ndimage.center_of_mass(labeled == i)
        centroids.append(tuple(int(round(c)) for c in com))
    centroids.sort(key=lambda c: c[2])
    return centroids


def best_slice(centroids):
    """Pick the axial slice with the most markers visible.
    If tied, pick the one closest to the median slice."""
    from collections import Counter
    slice_counts = Counter(c[2] for c in centroids)
    max_count = max(slice_counts.values())
    best_candidates = [s for s, count in slice_counts.items()
                       if count == max_count]
    # If all on different slices, pick the middle one
    if max_count == 1:
        all_slices = sorted(c[2] for c in centroids)
        return all_slices[len(all_slices) // 2]
    median_sl = np.median([c[2] for c in centroids])
    return min(best_candidates, key=lambda s: abs(s - median_sl))


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    os.chdir(project_dir)

    output_dir = 'paper_draft/images'
    os.makedirs(output_dir, exist_ok=True)

    model_paths = sorted(glob.glob('models/production/*-best.pth'))
    print(f"Found {len(model_paths)} production models")

    subjects = ['S019', 'S121', 'S048']

    def get_mri_path(sid):
        for base in ['data/test/prepared', 'data/train']:
            for ext in ['', '.gz']:
                p = f'{base}/{sid}/{sid}_MRI_homogeneity-corrected.nii{ext}'
                if os.path.exists(p):
                    return p
        return None

    subject_data = []
    for sid in subjects:
        mri_path = get_mri_path(sid)
        if mri_path is None:
            print(f"WARNING: Could not find MRI for {sid}, skipping")
            continue
        print(f"Processing {sid}: {mri_path}")
        mri = nib.load(mri_path).get_fdata().astype(np.float32)
        seed_mask = run_consensus_inference(mri_path, model_paths)
        centroids = find_marker_centroids(seed_mask)
        print(f"  {len(centroids)} markers, centroids: {centroids}")

        if len(centroids) == 0:
            print(f"  WARNING: No markers detected, skipping")
            continue

        overlay, boundaries = openrecon_overlay(mri, seed_mask)
        sl = best_slice(centroids)
        print(f"  Best slice: {sl}")

        subject_data.append({
            'id': sid,
            'mri': mri,
            'overlay': overlay,
            'boundaries': boundaries,
            'seed_mask': seed_mask,
            'centroids': centroids,
            'slice': sl,
        })

    if not subject_data:
        print("No subjects processed!")
        return

    # --- Generate figure: 1 row x N cols (overlay only) ---
    n = len(subject_data)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8))
    if n == 1:
        axes = [axes]

    for col, sd in enumerate(subject_data):
        mri = sd['mri']
        sl = sd['slice']
        centroids = sd['centroids']

        # Crop around markers for visibility
        marker_rows = [c[0] for c in centroids]
        marker_cols = [c[1] for c in centroids]
        crop_margin = 50
        r_center = (min(marker_rows) + max(marker_rows)) // 2
        c_center = (min(marker_cols) + max(marker_cols)) // 2
        r_min = max(0, r_center - crop_margin)
        r_max = min(mri.shape[0], r_center + crop_margin)
        c_min = max(0, c_center - crop_margin)
        c_max = min(mri.shape[1], c_center + crop_margin)

        mri_crop = mri[r_min:r_max, c_min:c_max, sl]
        boundary_crop = sd['boundaries'][r_min:r_max, c_min:c_max, sl]

        vmin = np.percentile(mri[mri > 0], 1) if np.any(mri > 0) else 0
        vmax = np.percentile(mri[mri > 0], 99) if np.any(mri > 0) else 1

        # Grayscale MRI + colored boundary overlay
        axes[col].imshow(mri_crop.T, cmap='gray', origin='lower',
                         vmin=vmin, vmax=vmax, aspect='equal')
        if boundary_crop.any():
            boundary_rgba = np.zeros((*boundary_crop.T.shape, 4))
            boundary_rgba[boundary_crop.T > 0] = [1, 0.35, 0, 1]  # orange-red
            axes[col].imshow(boundary_rgba, origin='lower', aspect='equal')
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        axes[col].set_title(f'Subject {col + 1}', fontsize=12,
                            fontweight='bold', pad=8)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08)

    out_path = os.path.join(output_dir, 'scanner_overlay.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
