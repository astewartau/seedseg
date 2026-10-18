#!/usr/bin/env python3
"""Control label variant: original labels with each marker blob replaced by a
3-mm-radius sphere at the blob centroid (same format as 'corrected', original
positions and original CTV). Separates the effect of label shape from position."""
import os, sys
import numpy as np, nibabel as nib, scipy.ndimage as ndi
R = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, 'labels', 'original_sphere'); os.makedirs(out, exist_ok=True)
for sid in open(os.path.join(R, 'subjects.txt')).read().split():
    img = nib.load(os.path.join(R, 'labels', 'original', f'{sid}.nii.gz'))
    old = np.asarray(img.dataobj).astype(np.int16); z = np.array(img.header.get_zooms()[:3])
    lab = np.where(old >= 1, 2, 0).astype(np.uint8)
    l, n = ndi.label(old == 1, np.ones((3, 3, 3)))
    g = np.indices(old.shape).reshape(3, -1).T
    for i in range(1, n + 1):
        c = np.array(ndi.center_of_mass(l == i))
        # index with a mask of lab's shape: nibabel arrays are Fortran-ordered, so
        # lab.reshape(-1) is a copy and assigning into it would be silently lost
        lab[((((g - c) * z) ** 2).sum(1) <= 9.0).reshape(old.shape)] = 1
    assert (lab == 1).sum() > 0 or n == 0, f'{sid}: no marker voxels written'
    nib.save(nib.Nifti1Image(lab, img.affine), os.path.join(out, f'{sid}.nii.gz'))
print('done')
