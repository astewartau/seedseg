#!/usr/bin/env python3
"""QC page of flagged markers only, 5 slices, larger field of view (42 mm)."""
import json, os, sys
import numpy as np, nibabel as nib, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R, out = sys.argv[1], sys.argv[2]
extra = set(sys.argv[3:])  # subjects to include in full
S = pd.read_csv(os.path.join(R, 'labelfix', 'refine_summary.csv')).set_index('subject')
rows = []
for sid in S.index:
    rec = json.load(open(os.path.join(R, 'labelfix', 'refined', f'{sid}.json')))
    for k, (q, v) in enumerate(zip(rec['markers_final_vox'], rec['void_strength'])):
        why = []
        if v < 0.45: why.append('LOW-VOID')
        if rec['accepted']: why.append('REFINED')
        if rec['status'] != 'ok': why.append(rec['status'])
        if sid in extra: why.append('listed')
        if why: rows.append((sid, k + 1, np.round(q).astype(int), v, ' '.join(why), rec['move_mm'][k]))
H = 30
per = 16
for p in range(0, len(rows), per):
    chunk = rows[p:p + per]
    fig, axes = plt.subplots(len(chunk), 5, figsize=(10, 2.1 * len(chunk)))
    axes = np.atleast_2d(axes)
    cache = {}
    for r, (sid, k, c, v, why, mv) in enumerate(chunk):
        if sid not in cache:
            cache = {sid: [np.asarray(nib.load(f).dataobj) for f in (os.path.join(R, 'images', f'{sid}.nii'),
                     os.path.join(R, 'labels', 'corrected', f'{sid}.nii.gz'), os.path.join(R, 'labels', 'original', f'{sid}.nii.gz'))]}
        vol, new, old = cache[sid]
        x0, x1 = max(c[0] - H, 0), min(c[0] + H, vol.shape[0]); y0, y1 = max(c[1] - H, 0), min(c[1] + H, vol.shape[1])
        lo, hi = np.percentile(vol[x0:x1, y0:y1], [1, 99])
        for j, dz in enumerate(range(-2, 3)):
            ax = axes[r, j]; ax.set_xticks([]); ax.set_yticks([])
            zz = int(np.clip(c[2] + dz, 0, vol.shape[2] - 1))
            ax.imshow(vol[x0:x1, y0:y1, zz].T, cmap='gray', vmin=lo, vmax=hi, origin='lower')
            for m, col, ls in [((old[x0:x1, y0:y1, zz] == 1), '#CC79A7', '--'), ((new[x0:x1, y0:y1, zz] == 1), '#E69F00', '-')]:
                if m.any() and not m.all():
                    ax.contour(m.T.astype(float), levels=[0.5], colors=[col], linewidths=1.0, linestyles=ls, origin='lower')
            if j == 0: ax.set_ylabel(f'{sid} M{k}\nV={v:.2f} d={mv:.1f}\n{why}', fontsize=7)
            if r == 0: ax.set_title(f'z{dz:+d}', fontsize=8)
    plt.tight_layout(pad=0.3); plt.savefig(os.path.join(out, f'flagged_{p // per:02d}.png'), dpi=80); plt.close()
print(len(rows), 'flagged rows')
