#!/usr/bin/env python
"""Compare ONNX-pipeline marker probability maps with nnUNetv2_predict references.

  compare_parity.py --ours <run_parity out dir> --ref-fold '<pattern with {f} and {sid}>'
      [--ref-ens '<pattern with {sid}>'] --subjects <file> --out <csv> [--k 3]

The reference ensemble is the given map or, without --ref-ens, the mean of the
reference fold maps (= mean of per-fold softmax, as evaluated). For each subject and
map (ens, fold k) it reports max |dp|, whether the top-K detections are the same
components in the same order (centroid distance <= 1 voxel), the largest centroid
shift (voxels) and mean-probability difference of the top-K, and whether the number
of candidates / count flag agree."""
import argparse, json, os, sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedseg_onnx import candidates  # noqa: E402


def load(p):
    return np.asarray(nib.load(p).dataobj, np.float32)


def compare(po, pr, k, thr=0.1):
    """Top-K of ours vs reference: the same set of components (each matched to a reference
    top-K centroid within 1 voxel), the same order, worst centroid shift and mean-p difference."""
    co, cr = candidates(po, threshold=thr), candidates(pr, threshold=thr)
    to, tr = co[:k], cr[:k]
    same_set = len(to) == len(tr)
    dmax, pmax, order = 0.0, 0.0, True
    for i, a in enumerate(to):
        d = [float(np.linalg.norm(np.subtract(a['centroid_vox'], b['centroid_vox']))) for b in tr]
        j = int(np.argmin(d)) if d else -1
        if j < 0 or d[j] > 1.0:
            same_set = False
            continue
        order &= j == i
        dmax, pmax = max(dmax, d[j]), max(pmax, abs(a['mean_prob'] - tr[j]['mean_prob']))
    return dict(max_abs_dp=float(np.abs(po - pr).max()), same_topk=bool(same_set), same_order=bool(same_set and order),
                topk_centroid_shift_vox=dmax, topk_mean_prob_diff=pmax, n_cand_ours=len(co), n_cand_ref=len(cr),
                count_flag_same=(len(co) < k) == (len(cr) < k),
                ref_topk=json.dumps([[round(c['mean_prob'], 5)] + [round(v, 2) for v in c['centroid_vox']] for c in tr]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--ref-fold', required=True)
    ap.add_argument('--ref-ens')
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--folds', nargs='+', default=['0', '1', '2', '3', '4'])
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []
    for sid in open(a.subjects).read().split():
        refs = {f: load(a.ref_fold.format(f=f, sid=sid)) for f in a.folds}
        ens_ref = load(a.ref_ens.format(sid=sid)) if a.ref_ens else np.mean([refs[f] for f in a.folds], 0)
        ours = {f: os.path.join(a.ours, f'fold{f}', f'{sid}.nii.gz') for f in a.folds}
        ours = {f: load(p) for f, p in ours.items() if os.path.exists(p)}
        fn = os.path.join(a.ours, 'ens', f'{sid}.nii.gz')
        if os.path.exists(fn):
            ens = load(fn)
        elif len(ours) == len(a.folds):
            ens = np.mean([ours[f] for f in a.folds], 0)
        else:
            ens = None
        if ens is not None:
            rows.append(dict(subject=sid, map='ens', **compare(ens, ens_ref, a.k)))
        for f, po in ours.items():
            rows.append(dict(subject=sid, map=f'fold{f}', **compare(po, refs[f], a.k)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    for name, g in df.groupby('map', sort=False):
        print(f'{name}: {len(g)} subj | same top-{a.k} {g.same_topk.sum()}/{len(g)} (same order {g.same_order.sum()}) | max centroid shift '
              f'{g.topk_centroid_shift_vox.max():.3f} vox | max |d mean p| {g.topk_mean_prob_diff.max():.2e} | '
              f'max |dp| {g.max_abs_dp.max():.2e} (median {g.max_abs_dp.median():.2e}) | count flag same '
              f'{g.count_flag_same.sum()}/{len(g)} | n_cand same {(g.n_cand_ours == g.n_cand_ref).sum()}/{len(g)}')


if __name__ == '__main__':
    main()
