#!/usr/bin/env python3
"""Step 7: apply human QC. Markers rejected at visual QC (listed in
qc_exclusions.csv: subject, marker index 1-based, reason) get the ignore label
(3) over their 3-mm sphere in labels/corrected and in any synthetic labels
derived from that subject; they are flagged `excluded` in the refined record so
evaluation drops them from the reference."""
import csv, glob, json, os, sys
import numpy as np, nibabel as nib
R = sys.argv[1]
SPHERE_MM, IGNORE = 3.0, 3
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'records', 'qc_exclusions.csv'))))
for r in rows:
    sid, k = r['subject'], int(r['marker']) - 1
    recp = os.path.join(R, 'labelfix', 'refined', f'{sid}.json'); rec = json.load(open(recp))
    q = np.array(rec['markers_final_vox'][k])
    rec.setdefault('excluded', [])
    if k not in rec['excluded']:
        rec['excluded'].append(k)
    rec['exclusion_reason'] = r['reason']
    json.dump(rec, open(recp, 'w'), indent=1)
    for f in [os.path.join(R, 'labels', 'corrected', f'{sid}.nii.gz')] + glob.glob(os.path.join(R, 'synthetic', 'corrected', 'labels', f'{sid}_syn*.nii.gz')):
        img = nib.load(f); lab = np.asarray(img.dataobj).copy(); z = np.array(img.header.get_zooms()[:3])
        g = np.indices(lab.shape).reshape(3, -1).T
        m = ((((g - q) * z) ** 2).sum(1) <= SPHERE_MM ** 2).reshape(lab.shape) & (lab == 1)
        lab[m] = IGNORE
        nib.save(nib.Nifti1Image(lab.astype(img.get_data_dtype()), img.affine, img.header), f)
        print(sid, k + 1, os.path.basename(f), int(m.sum()), 'voxels -> ignore')
