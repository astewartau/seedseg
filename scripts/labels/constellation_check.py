#!/usr/bin/env python3
"""Step 6: rigid-invariant consistency check. For each subject, compare the
pairwise inter-marker distances of the corrected MR positions with those of the
CT (RTSTRUCT) constellation. Max absolute difference > 3 mm is flagged."""
import json, os, sys, itertools
import numpy as np, nibabel as nib, pandas as pd
R = sys.argv[1]
rows = []
for sid in open(os.path.join(R, 'subjects.txt')).read().split():
    rf = json.load(open(os.path.join(R, 'labelfix', 'refined', f'{sid}.json')))
    mp = os.path.join(R, 'labelfix', 'markers', f'{sid}.json')
    if not os.path.exists(mp):
        rows.append(dict(subject=sid, check='no_ct_constellation')); continue
    mk = json.load(open(mp))
    z = np.array(nib.load(os.path.join(R, 'images', f'{sid}.nii')).header.get_zooms()[:3])
    mr = np.array(rf['markers_final_vox']) * z
    ct = np.array([m['ct_lps'] for m in mk['markers']])
    pairs = list(itertools.combinations(range(len(ct)), 2))
    diff = [abs(np.linalg.norm(mr[i] - mr[j]) - np.linalg.norm(ct[i] - ct[j])) for i, j in pairs]
    init = np.array(rf['markers_init_vox']) * z
    diff0 = [abs(np.linalg.norm(init[i] - init[j]) - np.linalg.norm(ct[i] - ct[j])) for i, j in pairs]
    rows.append(dict(subject=sid, check='ok' if max(diff) <= 3 else 'FLAG', max_pair_diff_mm=round(max(diff), 2),
                     max_pair_diff_before_mm=round(max(diff0), 2), status=mk.get('status')))
d = pd.DataFrame(rows); d.to_csv(os.path.join(R, 'labelfix', 'constellation_check.csv'), index=False)
print(d.check.value_counts().to_dict()); print('max pair diff (after):', d.max_pair_diff_mm.describe().round(2).to_dict())
print(d[d.check != 'ok'].to_string())
