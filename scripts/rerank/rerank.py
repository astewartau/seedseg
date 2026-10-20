#!/usr/bin/env python3
"""Second stage: learned re-ranking of marker candidates.

Each candidate (a connected component of the segmentation network's marker
probability, from evaluate_detections.py) is described by
  * network outputs: mean / max probability, mass, log volume, rank by mean,
    prostate probability at the centroid (optional),
  * local image appearance around the probability-weighted centroid, from the
    void map V = clip((B - I_s)/B, 0, 1) (see scripts/labels):
    V at the centre; mean V in spherical shells 0-1, 1-2, 2-3, 3-5 mm; void
    extent in-plane and through-plane (V > 0.45 voxels connected to the centre);
    minimum smoothed intensity relative to local tissue; fraction of T1-bright
    voxels (> 1.6 x local tissue level) within 6 mm (blood products),
  * context: distance to the nearest other candidate, number of candidates.
A gradient-boosted tree classifier (scikit-learn HistGradientBoosting) predicts
P(candidate is a marker). Training uses out-of-fold candidates only, and the
re-ranker for CV fold k is trained on subjects outside fold k (nested by the
same subject folds as the segmentation network). Selection then keeps the top 3
by P (evaluate_detections.py --rescore).
"""
import argparse, json, os, sys
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'labels'))
from refine_and_rasterise import void_map  # noqa: E402

SHELLS = [(0, 1), (1, 2), (2, 3), (3, 5)]


def cached_maps(sid, root):
    """Void map V, smoothed image I_s and local tissue level B, cached per subject."""
    cp = os.path.join(root, 'cache', 'voidmap', f'{sid}.npz')
    img = nib.load(os.path.join(root, 'images', f'{sid}.nii'))
    vol = np.asarray(img.dataobj, np.float32)
    if os.path.exists(cp):
        d = np.load(cp)
        return vol, img, d['V'], d['I_s'], d['B']
    I_s = ndi.gaussian_filter(vol, (0.5, 0.5, 0.25)); B = ndi.median_filter(vol, size=(9, 9, 3))
    V = np.clip((B - I_s) / np.maximum(B, 1e-6), 0, 1)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.savez(cp, V=V.astype(np.float32), I_s=I_s.astype(np.float32), B=B.astype(np.float32))
    return vol, img, V, I_s, B


def image_features(sid, C, root, prostate_dir):
    vol, img, V, I_s, B = cached_maps(sid, root)
    z = np.array(img.header.get_zooms()[:3], float)
    bright = vol > 1.6 * np.maximum(B, 1e-6)
    pr = None
    if prostate_dir and os.path.exists(os.path.join(prostate_dir, f'{sid}.nii.gz')):
        pr = np.asarray(nib.load(os.path.join(prostate_dir, f'{sid}.nii.gz')).dataobj, np.float32)
    feats = []
    pos = C[['x', 'y', 'z']].values
    for k, (_, c) in enumerate(C.iterrows()):
        q = np.array([c.x, c.y, c.z])
        r = np.ceil(6 / z).astype(int); qi = np.round(q).astype(int)
        lo = np.maximum(qi - r, 0); hi = np.minimum(qi + r + 1, vol.shape)
        g = np.stack(np.meshgrid(*[np.arange(a_, b_) for a_, b_ in zip(lo, hi)], indexing='ij'), -1)
        d = np.sqrt((((g - q) * z) ** 2).sum(-1))
        sl = tuple(slice(a_, b_) for a_, b_ in zip(lo, hi))
        v = V[sl]; f = {}
        f['v_centre'] = float(V[tuple(np.clip(qi, 0, np.array(vol.shape) - 1))])
        for a_, b_ in SHELLS:
            m = (d >= a_) & (d < b_)
            f[f'v_shell_{a_}_{b_}'] = float(v[m].mean()) if m.any() else 0.0
        core = (v > 0.45) & (d <= 4)
        lab, _ = ndi.label(core, np.ones((3, 3, 3)))
        cid = lab[tuple(np.clip(qi - lo, 0, np.array(v.shape) - 1))]
        blob = lab == cid if cid > 0 else np.zeros_like(core)
        idx = np.argwhere(blob)
        f['void_extent_inplane_mm'] = float((np.ptp(idx[:, :2], 0).max() + 1) * z[0]) if len(idx) else 0.0
        f['void_extent_slices'] = float(np.ptp(idx[:, 2]) + 1) if len(idx) else 0.0
        f['min_rel_intensity'] = float((I_s[sl][d <= 3] / np.maximum(B[sl][d <= 3], 1e-6)).min())
        f['bright_frac_6mm'] = float(bright[sl][d <= 6].mean())
        others = np.delete(pos, k, axis=0)
        f['dist_nearest_cand_mm'] = float(np.linalg.norm((others - q) * z, axis=1).min()) if len(others) else 100.0
        if pr is not None:
            f['prostate_prob'] = float(pr[tuple(np.clip(qi, 0, np.array(vol.shape) - 1))])
        feats.append(f)
    return pd.DataFrame(feats, index=C.index)


def build(cands_csv, root, prostate_dir):
    C = pd.read_csv(cands_csv)
    from multiprocessing import Pool
    groups = list(C.groupby('subject'))
    with Pool(int(os.environ.get('RERANK_PROCS', 8))) as pool:
        parts = pool.starmap(image_features, [(sid, g, root, prostate_dir) for sid, g in groups])
    F = pd.concat(parts).loc[C.index]
    C = pd.concat([C, F], axis=1)
    C['log_size'] = np.log(C.size_mm3); C['n_cand'] = C.groupby('subject').cand.transform('count')
    return C


NET = ['mean', 'max', 'mass', 'log_size', 'rank', 'n_cand']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True, help='candidate CSV (out-of-fold) from evaluate_detections.py')
    ap.add_argument('--root', required=True)
    ap.add_argument('--folds', required=True, help='JSON: list of 5 {"train": [...], "val": [...]} (nnU-Net splits_final.json)')
    ap.add_argument('--prostate-probs', help='dir with {sid}.nii.gz prostate probability maps')
    ap.add_argument('--out', required=True, help='output CSV: subject,cand,rerank_p')
    a = ap.parse_args()
    C = build(a.candidates, a.root, a.prostate_probs)
    feats = [c for c in C.columns if c in NET or c.startswith(('v_', 'void_', 'min_rel', 'bright', 'dist_nearest', 'prostate_prob'))]
    folds = json.load(open(a.folds))
    C['rerank_p'] = np.nan
    for k, fd in enumerate(folds):
        val = set(fd['val'])
        tr = C[~C.subject.isin(val)]; te = C.subject.isin(val)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                             class_weight='balanced', random_state=0)
        clf.fit(tr[feats], tr.is_marker.astype(int))
        C.loc[te, 'rerank_p'] = clf.predict_proba(C.loc[te, feats])[:, 1]
        print(f'fold {k}: trained on {len(tr)} candidates ({int(tr.is_marker.sum())} markers), scored {int(te.sum())}', flush=True)
    C.to_csv(a.out.replace('.csv', '_features.csv'), index=False)
    C[['subject', 'cand', 'rerank_p']].to_csv(a.out, index=False)
    json.dump(dict(features=feats), open(a.out.replace('.csv', '_meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
