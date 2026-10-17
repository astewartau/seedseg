#!/usr/bin/env python3
"""Per-marker outcomes from evaluate_detections.py candidate tables, paired arm comparison
and miss categories. Re-derives matching (top-k by rank, 5 mm Hungarian) and checks totals."""
import json, sys, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
import nibabel as nib
P = '/scratch/user/uqaste15/prostate-v2'; R = P + '/eval/results'
TOL, K = 5.0, 3
subs = open(P + '/subjects.txt').read().split(); test = set(open(P + '/eval/test25.txt').read().split())
refs = {}
for s in subs:
    rec = json.load(open(f'{P}/labelfix/refined/{s}.json')); z = np.array(nib.load(f'{P}/images/{s}.nii').header.get_zooms()[:3])
    ex = set(rec.get('excluded', [])); a = np.array(rec['markers_final_vox']) * z
    refs[s] = (np.array([r for i, r in enumerate(a) if i not in ex]).reshape(-1, 3), np.array([r for i, r in enumerate(a) if i in ex]).reshape(-1, 3), z)

def outcomes(cands, rankcol='rank'):
    C = pd.read_csv(cands); rows = []; fp = {}
    for s in subs:
        ref, exr, z = refs[s]; c = C[C.subject == s]
        pos = c[['x', 'y', 'z']].to_numpy() * z if len(c) else np.zeros((0, 3))
        rk = c[rankcol].to_numpy() if len(c) else np.zeros(0)
        keep = rk <= K
        if len(exr) and len(pos):
            keep &= np.array([np.linalg.norm(exr - q, axis=1).min() > TOL for q in pos])
        kp = pos[keep]; matched = {}
        if len(kp) and len(ref):
            D = np.linalg.norm(kp[:, None] - ref[None], axis=2); r, cc = linear_sum_assignment(np.where(D <= TOL, D, 1e6))
            matched = {j: i for i, j in zip(r, cc) if D[i, j] <= TOL}
        fp[s] = int(keep.sum()) - len(matched)
        for j, q in enumerate(ref):
            d = np.linalg.norm(pos - q, axis=1) if len(pos) else np.array([np.inf])
            near = d <= TOL
            best = int(rk[near].min()) if near.any() else None
            if j in matched: cat = 'hit'
            elif best is None: cat = 'no_candidate'
            elif len(ref) > K and best <= K: cat = 'topk_cap'   # candidate in top-k but used by another marker / cap
            else: cat = 'ranked_out'
            rows.append(dict(subject=s, marker=j, n_ref=len(ref), cat=cat, best_rank=best, test=s in test))
    return pd.DataFrame(rows), pd.Series(fp)

def boot_diff(A, B, fA, fB, sel, n=5000):
    sA = A[A.subject.isin(sel)]; sB = B[B.subject.isin(sel)]
    tA = sA.groupby('subject').apply(lambda d: (d.cat == 'hit').sum()); tB = sB.groupby('subject').apply(lambda d: (d.cat == 'hit').sum())
    nm = sA.groupby('subject').size(); ss = list(tA.index)
    rng = np.random.default_rng(0); ds, dp = [], []
    for _ in range(n):
        i = rng.choice(ss, len(ss))
        ds.append((tA[i].sum() - tB[i].sum()) / nm[i].sum())
        pa = tA[i].sum() / (tA[i].sum() + fA[i].sum()); pb = tB[i].sum() / (tB[i].sum() + fB[i].sum()); dp.append(pa - pb)
    return np.percentile(ds, [2.5, 97.5]), np.percentile(dp, [2.5, 97.5])

if __name__ == '__main__':
    arms = dict(a.split('=') for a in sys.argv[1:])  # name=candidates.csv[:rankcol]
    res = {}
    for k, v in arms.items():
        f, _, col = v.partition(':'); res[k] = outcomes(f, col or 'rank')
        O, F = res[k]
        for nm, sel in [('all127', subs), ('test25', sorted(test))]:
            o = O[O.subject.isin(sel)]
            print(f"{k:12s} {nm}: TP {(o.cat=='hit').sum()}/{len(o)} FP {F[sel].sum()} | misses: " + ', '.join(f"{c} {n}" for c, n in o[o.cat!='hit'].cat.value_counts().items()))
    names = list(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]; A, fA = res[a]; B, fB = res[b]
            for nm, sel in [('all127', subs), ('test25', sorted(test))]:
                m = A.merge(B, on=['subject', 'marker']); m = m[m.subject.isin(sel)]
                ha = m.cat_x == 'hit'; hb = m.cat_y == 'hit'
                (ls, hs), (lp, hp) = boot_diff(A, B, fA, fB, sel)
                print(f"{a} vs {b} {nm}: markers only-{a} {int((ha&~hb).sum())}, only-{b} {int((~ha&hb).sum())}; "
                      f"dSens 95% boot [{ls:+.3f},{hs:+.3f}], dPrec [{lp:+.3f},{hp:+.3f}]; FP {fA[sel].sum()} vs {fB[sel].sum()}")
    for k, (O, F) in res.items():
        O.to_csv(f'{P}/eval/analysis/markers_{k}.csv', index=False)
