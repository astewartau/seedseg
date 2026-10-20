#!/usr/bin/env python
"""Mean of per-fold marker probability maps: <dir>/fold{k}/{sid}.nii.gz -> <dir>/ens/{sid}.nii.gz
(only subjects with all requested folds). Usage: ensemble_maps.py <dir> [--folds 0 1 2 3 4]"""
import argparse, glob, os

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('dir'); ap.add_argument('--folds', nargs='+', default=['0', '1', '2', '3', '4'])
ap.add_argument('--name', default='ens')
a = ap.parse_args()
os.makedirs(os.path.join(a.dir, a.name), exist_ok=True)
n = 0
for f0 in sorted(glob.glob(os.path.join(a.dir, f'fold{a.folds[0]}', '*.nii.gz'))):
    sid = os.path.basename(f0)[:-7]
    fs = [os.path.join(a.dir, f'fold{f}', f'{sid}.nii.gz') for f in a.folds]
    if not all(map(os.path.exists, fs)):
        continue
    ims = [nib.load(f) for f in fs]
    p = np.mean([np.asarray(i.dataobj, np.float32) for i in ims], 0)
    nib.save(nib.Nifti1Image(p, ims[0].affine, ims[0].header), os.path.join(a.dir, a.name, f'{sid}.nii.gz'))
    n += 1
print(f'{n} ensemble maps in {os.path.join(a.dir, a.name)}')
