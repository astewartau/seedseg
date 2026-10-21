#!/usr/bin/env python3
"""Average the out-of-fold 3-class probability maps of repeated runs (same folds).

Writes {sid}_probs.nii.gz (all classes, for constellation selection) and
{sid}_marker_prob.nii.gz to --out, copies oof_folds.json, and symlinks the
marker maps as {sid}.nii.gz into --mprob for evaluate_detections.py.
usage: average_runs.py --runs <oof dir> <oof dir> ... --out <dir> --mprob <dir>
"""
import argparse, filecmp, json, os, shutil
import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--mprob', required=True)
a = ap.parse_args()
for r in a.runs[1:]:  # averaging is only out-of-fold if every run used the same folds
    assert filecmp.cmp(f'{a.runs[0]}/oof_folds.json', f'{r}/oof_folds.json', shallow=False), r
os.makedirs(a.out, exist_ok=True); os.makedirs(a.mprob, exist_ok=True)
shutil.copy(f'{a.runs[0]}/oof_folds.json', a.out)
for sid in sorted(json.load(open(f'{a.runs[0]}/oof_folds.json'))):
    ims = [nib.load(f'{r}/{sid}_probs.nii.gz') for r in a.runs]
    for im in ims[1:]:
        assert np.allclose(im.affine, ims[0].affine) and im.shape == ims[0].shape, sid
    p = np.mean([np.asarray(im.dataobj, np.float32) for im in ims], 0)
    nib.save(nib.Nifti1Image(p, ims[0].affine), f'{a.out}/{sid}_probs.nii.gz')
    nib.save(nib.Nifti1Image(p[..., 1], ims[0].affine), f'{a.out}/{sid}_marker_prob.nii.gz')
    link = f'{a.mprob}/{sid}.nii.gz'
    if os.path.lexists(link): os.remove(link)
    os.symlink(f'{os.path.abspath(a.out)}/{sid}_marker_prob.nii.gz', link)
print(f'averaged {len(a.runs)} runs -> {a.out}')
