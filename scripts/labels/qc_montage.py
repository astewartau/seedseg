#!/usr/bin/env python3
"""Step 5: visual QC pages. One row per corrected marker: slices z-1, z, z+1
(z = slice of the corrected marker centre), 28 mm field of view. Orange solid =
corrected marker sphere, magenta dashed = original label, blue = corrected CTV.
Row titles carry the flags from refine_summary.csv."""
import argparse, json, os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--per-page', type=int, default=8)
a = ap.parse_args()
R = a.root
os.makedirs(a.out, exist_ok=True)
S = pd.read_csv(os.path.join(R, 'labelfix', 'refine_summary.csv')).set_index('subject')
subs = list(S.index)
H = 20
for page in range(0, len(subs), a.per_page):
    rows = []
    for sid in subs[page:page + a.per_page]:
        rec = json.load(open(os.path.join(R, 'labelfix', 'refined', f'{sid}.json')))
        vol = np.asarray(nib.load(os.path.join(R, 'images', f'{sid}.nii')).dataobj, np.float32)
        new = np.asarray(nib.load(os.path.join(R, 'labels', 'corrected', f'{sid}.nii.gz')).dataobj)
        old = np.asarray(nib.load(os.path.join(R, 'labels', 'original', f'{sid}.nii.gz')).dataobj)
        for k, (q, v) in enumerate(zip(rec['markers_final_vox'], rec['void_strength'])):
            flag = []
            if v < 0.45: flag.append('LOW-VOID')
            if rec['accepted']: flag.append('REFINED')
            if rec['status'] != 'ok': flag.append(rec['status'])
            rows.append((sid, k + 1, np.round(q).astype(int), v, ' '.join(flag), vol, new, old, rec['move_mm'][k]))
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.3, 2.1 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (sid, k, c, v, flag, vol, new, old, mv) in enumerate(rows):
        x0, x1 = max(c[0] - H, 0), min(c[0] + H, vol.shape[0]); y0, y1 = max(c[1] - H, 0), min(c[1] + H, vol.shape[1])
        crop = vol[x0:x1, y0:y1]
        lo, hi = np.percentile(crop, [1, 99])
        for j, dz in enumerate((-1, 0, 1)):
            ax = axes[r, j]; ax.set_xticks([]); ax.set_yticks([])
            zz = int(np.clip(c[2] + dz, 0, vol.shape[2] - 1))
            ax.imshow(crop[:, :, zz].T, cmap='gray', vmin=lo, vmax=hi, origin='lower')
            for m, col, ls in [((old[x0:x1, y0:y1, zz] == 1), '#CC79A7', '--'), ((new[x0:x1, y0:y1, zz] == 2), '#56B4E9', '-'),
                               ((new[x0:x1, y0:y1, zz] == 1), '#E69F00', '-')]:
                if m.any() and not m.all():
                    ax.contour(m.T.astype(float), levels=[0.5], colors=[col], linewidths=0.9, linestyles=ls, origin='lower')
            if j == 0:
                ax.set_ylabel(f'{sid} M{k}\nV={v:.2f} d={mv:.1f}mm\n{flag}', fontsize=6.5)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(a.out, f'qc_{page // a.per_page:02d}.png'), dpi=90)
    plt.close()
    print('page', page // a.per_page, flush=True)
