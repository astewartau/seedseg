#!/usr/bin/env python3
"""Marker-level evaluation by centroid matching.

Input: per-subject marker-class probability maps as NIfTI (`{sid}.nii.gz`, on the
image grid) and reference marker centres from the label correction records
(`labelfix/refined/{sid}.json`, key markers_final_vox).

Detection: connected components (26-connectivity) of p > --threshold; each
component's position is its probability-weighted centroid; components are ranked
by --score (mean or max probability, or a column from --rescore CSV) and the top
--top-k are kept.

Matching: one-to-one (Hungarian) between kept detections and reference markers,
with a distance tolerance --tol-mm (default 5 mm). TP = matched pairs; FN =
unmatched reference markers; FP = unmatched detections.

Outputs: per-subject CSV, candidate table (all components with features and a
`is_marker` flag = within tol of a reference marker; used to train the re-ranker),
and a summary with Clopper-Pearson and subject-bootstrap 95% CIs.
"""
import argparse, glob, json, os
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

S26 = np.ones((3, 3, 3), bool)


def clopper_pearson(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def candidates(p, z, thr):
    lab, n = ndi.label(p > thr, structure=S26)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    mean = ndi.mean(p, lab, idx); mx = ndi.maximum(p, lab, idx)
    size = ndi.sum(np.ones_like(p), lab, idx); mass = ndi.sum(p, lab, idx)
    cen = ndi.center_of_mass(p, lab, idx)
    out = []
    for i in range(n):
        out.append(dict(cand=int(idx[i]), mean=float(mean[i]), max=float(mx[i]), size_mm3=float(size[i] * z.prod()),
                        mass=float(mass[i]), x=cen[i][0], y=cen[i][1], z=cen[i][2]))
    return out


def match(det_mm, ref_mm, tol):
    if len(det_mm) == 0 or len(ref_mm) == 0:
        return [], np.zeros((len(det_mm), len(ref_mm)))
    D = np.linalg.norm(det_mm[:, None, :] - ref_mm[None, :, :], axis=2)
    r, c = linear_sum_assignment(np.where(D <= tol, D, 1e6))
    return [(i, j, D[i, j]) for i, j in zip(r, c) if D[i, j] <= tol], D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True, help='dir with {sid}.nii.gz marker probability maps')
    ap.add_argument('--root', required=True, help='prostate-v2 root (images/, labelfix/refined/)')
    ap.add_argument('--subjects', help='file with subject IDs (default: all prob maps)')
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--threshold', type=float, default=0.1)
    ap.add_argument('--score', default='mean')
    ap.add_argument('--rescore', help='CSV with subject,cand,<score column> to rank by')
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--tol-mm', type=float, default=5.0)
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args()
    subs = [l.strip() for l in open(a.subjects)] if a.subjects else sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(a.probs, '*.nii.gz')))
    resc = pd.read_csv(a.rescore).set_index(['subject', 'cand'])[a.score] if a.rescore else None
    per, cands_all = [], []
    for sid in subs:
        img = nib.load(os.path.join(a.probs, f'{sid}.nii.gz'))
        p = np.asarray(img.dataobj, np.float32); z = np.array(img.header.get_zooms()[:3], float)
        rec = json.load(open(os.path.join(a.root, 'labelfix', 'refined', f'{sid}.json')))
        allref = np.array(rec['markers_final_vox']) * z
        excl = set(rec.get('excluded', []))  # markers rejected at human QC
        ref = np.array([r for i, r in enumerate(allref) if i not in excl]).reshape(-1, 3)
        exref = np.array([r for i, r in enumerate(allref) if i in excl]).reshape(-1, 3)
        cs = candidates(p, z, a.threshold)
        cpos = np.array([[c['x'], c['y'], c['z']] for c in cs]) * z if cs else np.zeros((0, 3))
        # candidate-level truth (for the re-ranker): nearest reference within tol
        for c, q in zip(cs, cpos):
            d = np.linalg.norm(ref - q, axis=1).min() if len(ref) else np.inf
            c.update(subject=sid, dist_ref_mm=float(d), is_marker=bool(d <= a.tol_mm))
            if resc is not None:
                c['rescore'] = float(resc.get((sid, c['cand']), -np.inf))
        key = 'rescore' if resc is not None else a.score
        order = sorted(range(len(cs)), key=lambda i: cs[i][key], reverse=True)[:a.top_k]
        for r_, i in enumerate(sorted(range(len(cs)), key=lambda i: cs[i][key], reverse=True)):
            cs[i]['rank'] = r_ + 1
        kept = cpos[order] if order else np.zeros((0, 3))
        if len(exref):  # detections on an excluded marker count neither way
            kept = kept[[np.linalg.norm(exref - q, axis=1).min() > a.tol_mm for q in kept]] if len(kept) else kept
        m, _ = match(kept, ref, a.tol_mm)
        tp = len(m); fn = len(ref) - tp; fp = len(kept) - tp
        per.append(dict(subject=sid, n_excluded=len(exref), n_ref=len(ref), n_det=len(kept), n_cand=len(cs), tp=tp, fn=fn, fp=fp,
                        perfect=(fn == 0 and fp == 0), loc_err_mm=float(np.mean([d for _, _, d in m])) if m else np.nan))
        cands_all += cs
    P = pd.DataFrame(per); C = pd.DataFrame(cands_all)
    P.to_csv(a.out + '_subjects.csv', index=False); C.to_csv(a.out + '_candidates.csv', index=False)
    TP, FN, FP = P.tp.sum(), P.fn.sum(), P.fp.sum()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(a.n_boot):
        s = P.iloc[rng.integers(0, len(P), len(P))]
        bs.append((s.tp.sum() / max(s.tp.sum() + s.fn.sum(), 1), s.tp.sum() / max(s.tp.sum() + s.fp.sum(), 1)))
    bs = np.array(bs)
    summ = dict(n_subjects=len(P), markers=int(TP + FN), tp=int(TP), fn=int(FN), fp=int(FP),
                sensitivity=TP / (TP + FN), sens_cp95=clopper_pearson(TP, TP + FN), sens_boot95=tuple(np.percentile(bs[:, 0], [2.5, 97.5])),
                precision=TP / max(TP + FP, 1), prec_cp95=clopper_pearson(TP, TP + FP), prec_boot95=tuple(np.percentile(bs[:, 1], [2.5, 97.5])),
                perfect=int(P.perfect.sum()), loc_err_mm_median=float(P.loc_err_mm.median()),
                cand_recall=float(C.groupby('subject').is_marker.sum().sum() / (TP + FN)) if len(C) else 0.0,
                settings=dict(threshold=a.threshold, score=a.score if resc is None else f'rescore:{a.score}', top_k=a.top_k, tol_mm=a.tol_mm))
    json.dump(summ, open(a.out + '_summary.json', 'w'), indent=1, default=float)
    print(f"{len(P)} subj | sens {TP}/{TP+FN} = {summ['sensitivity']:.1%} | prec {TP}/{TP+FP} = {summ['precision']:.1%} | perfect {summ['perfect']}/{len(P)} | loc err {summ['loc_err_mm_median']:.2f} mm")


if __name__ == '__main__':
    main()
