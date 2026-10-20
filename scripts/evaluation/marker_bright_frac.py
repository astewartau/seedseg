#!/usr/bin/env python3
"""Per reference marker: fraction of T1-bright voxels (> 1.6 x local tissue level, the
re-ranker's blood-product feature) within 6 mm. Writes eval/analysis/marker_bright_frac.csv
with (subject, marker) keys matching compare_arms.py outcomes."""
import json, os, sys
import numpy as np, nibabel as nib, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rerank'))
from rerank import cached_maps  # noqa: E402
P = '/scratch/user/uqaste15/prostate-v2'
rows = []
for s in open(P + '/subjects.txt').read().split():
    rec = json.load(open(f'{P}/labelfix/refined/{s}.json')); ex = set(rec.get('excluded', []))
    vol, img, V, I_s, B = cached_maps(s, P); z = np.array(img.header.get_zooms()[:3], float)
    bright = vol > 1.6 * np.maximum(B, 1e-6)
    refs = [v for i, v in enumerate(rec['markers_final_vox']) if i not in ex]
    for j, q in enumerate(np.array(refs, float).reshape(-1, 3)):
        r = np.ceil(6 / z).astype(int); qi = np.round(q).astype(int)
        lo = np.maximum(qi - r, 0); hi = np.minimum(qi + r + 1, vol.shape)
        g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing='ij'), -1)
        d = np.sqrt((((g - q) * z) ** 2).sum(-1)); sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        rows.append(dict(subject=s, marker=j, bright_frac_6mm=float(bright[sl][d <= 6].mean())))
pd.DataFrame(rows).to_csv(P + '/eval/analysis/marker_bright_frac.csv', index=False)
print(pd.DataFrame(rows).bright_frac_6mm.describe(percentiles=[.5, .75, .9, .95]))
