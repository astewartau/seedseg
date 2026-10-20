#!/usr/bin/env python3
"""Figure 1 (pipeline schematic) for the v2 manuscript, with thumbnails from one patient.
usage: figure1_v2.py --out <dir> [--subject z0163505]
"""
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import nibabel as nib
import numpy as np
import pandas as pd

P = '/scratch/user/uqaste15/prostate-v2'
BOX, EDGE, TXT, ARR = '#dfe6ee', '#5b7083', '#1d2a36', '#3d5163'
ORANGE, BLUE = '#ff9f1c', '#2ec4f1'


def box(ax, x, y, w, h, text, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                                fc=BOX, ec=EDGE, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=TXT)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, lw=1.2, color=ARR))


def thumb(fig, rect, img, z, overlay=None, color=None, prob=None, title='', c=(160, 160), h=62):
    ax = fig.add_axes(rect)
    sl = img[:, :, z]; lo, hi = np.percentile(sl, [1, 99.5])
    ax.imshow(sl.T, cmap='gray', vmin=lo, vmax=hi, origin='lower')
    if prob is not None:
        ax.imshow(np.ma.masked_less(prob[:, :, z].T, 0.05), cmap='viridis', vmin=0, vmax=1, origin='lower', alpha=0.9)
    if overlay is not None:
        for (x, y) in overlay:
            ax.add_patch(plt.Circle((x, y), 5, fill=False, lw=1.4, color=color))
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(EDGE)
    ax.set_title(title, fontsize=8.5, color=TXT, pad=3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subject', default='z0163505')
    a = ap.parse_args(); s = a.subject
    img = np.asarray(nib.load(f'{P}/images/{s}.nii').dataobj, np.float32)
    prob = np.asarray(nib.load(f'{P}/eval/probs_511/{s}.nii.gz').dataobj, np.float32)
    refs = np.array(json.load(open(f'{P}/labelfix/refined/{s}.json'))['markers_final_vox'])
    z = int(round(np.median(refs[:, 2])))
    pts = [(r[0], r[1]) for r in refs]
    cc = (refs[:, 0].mean(), refs[:, 1].mean() + 15)  # all markers, projected onto the shown section
    c = pd.read_csv(f'{P}/eval/results/nnunet511_all127_candidates.csv')
    det = [(r.x, r.y) for r in c[(c.subject == s) & (c['rank'] <= 3)].itertuples()]
    pmax = prob.max(axis=2, keepdims=True).repeat(img.shape[2], axis=2)

    fig = plt.figure(figsize=(10, 5.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    for y0, h, lab in [(0.53, 0.45, 'Training'), (0.13, 0.37, 'Inference')]:
        ax.add_patch(FancyBboxPatch((0.01, y0), 0.98, h, boxstyle='round,pad=0,rounding_size=0.01', fc='#f4f6f8', ec='#c9d2db', lw=1))
        ax.text(0.025, y0 + h / 2, lab, rotation=90, ha='center', va='center', fontsize=11, color=TXT)

    # training row
    box(ax, 0.05, 0.84, 0.15, 0.1, 'Planning structure set\n(marker + CTV contours)')
    box(ax, 0.05, 0.60, 0.15, 0.1, 'T1-weighted MRI\n(N4 corrected)')
    box(ax, 0.25, 0.84, 0.15, 0.1, 'Map to MRI with\nclinical registration')
    box(ax, 0.45, 0.84, 0.15, 0.1, 'Refine to MR\nsignal voids')
    box(ax, 0.45, 0.60, 0.15, 0.1, 'QC: inter-marker\ndistances vs CT')
    box(ax, 0.66, 0.72, 0.13, 0.1, 'nnU-Net\n5-fold CV\n(5 networks)')
    arrow(ax, 0.20, 0.89, 0.25, 0.89); arrow(ax, 0.40, 0.89, 0.45, 0.89); arrow(ax, 0.525, 0.84, 0.525, 0.70)
    arrow(ax, 0.20, 0.65, 0.66, 0.75); arrow(ax, 0.60, 0.89, 0.66, 0.80)
    thumb(fig, [0.83, 0.585, 0.14, 0.335], img, z, overlay=pts, color=BLUE, title='Training labels', c=cc)
    arrow(ax, 0.83, 0.77, 0.79, 0.77)

    # inference row
    box(ax, 0.05, 0.26, 0.12, 0.1, 'T1-weighted MRI\n(N4 corrected)')
    box(ax, 0.21, 0.26, 0.12, 0.1, '5 networks\n(sliding window)')
    box(ax, 0.37, 0.26, 0.12, 0.1, 'Average\nprobabilities')
    box(ax, 0.53, 0.26, 0.12, 0.1, 'Keep 3 highest-\nscoring markers')
    box(ax, 0.53, 0.155, 0.12, 0.075, 'Fewer than implanted?\nFlag for review', fs=7.5)
    for x0, x1 in [(0.17, 0.21), (0.33, 0.37), (0.49, 0.53)]: arrow(ax, x0, 0.31, x1, 0.31)
    arrow(ax, 0.59, 0.26, 0.59, 0.23)
    thumb(fig, [0.68, 0.16, 0.14, 0.31], img, z, prob=pmax, title='Marker probability', c=cc)
    thumb(fig, [0.84, 0.16, 0.14, 0.31], img, z, overlay=det, color=ORANGE, title='Detected markers', c=cc)
    arrow(ax, 0.65, 0.31, 0.68, 0.31)

    # deployment strip
    ax.add_patch(FancyBboxPatch((0.01, 0.015), 0.98, 0.095, boxstyle='round,pad=0,rounding_size=0.01', fc='#e9edf1', ec='#c9d2db', lw=1))
    for x, t in [(0.2, 'GPU / CPU (Python)'), (0.5, 'MRI scanner (OpenRecon)'), (0.8, 'Web browser (ONNX / WebAssembly)')]:
        ax.text(x, 0.062, t, ha='center', va='center', fontsize=9.5, color=TXT)
    fig.savefig(f'{a.out}/Figure1.png', dpi=300); fig.savefig(f'{a.out}/Figure1.tiff', dpi=300)


if __name__ == '__main__':
    main()
