#!/usr/bin/env python3
"""Constellation-aware marker selection on out-of-fold candidates.

Instead of keeping the top 3 candidates by network score, choose the set of K
candidates that maximises

    sum_i log(score_i) + lam * [sum_i log p_pos(x_i) + sum_i log p_surf(s_i) + sum_{i<j} log p_pair(d_ij)]

where the geometric prior is learned from reference marker positions of the
training folds only:
  p_pos   3-D Gaussian KDE of marker position in prostate-normalised coordinates
          (predicted prostate: centroid + principal axes scaled by their SD),
  p_surf  1-D KDE of signed distance to the predicted prostate surface (mm, <0 inside),
  p_pair  1-D KDE of pairwise inter-marker distances (mm).
The prostate comes from the same out-of-fold nnU-Net prediction (class 2), so no
reference information from the evaluated subject is used.

K = number of implanted markers (from the CT-derived record, as a clinical implant
count would be), or a fixed --k. Optionally adds low-threshold candidates
(components of p_marker > --extra-thr that contain no voxel > --thr) so that a
marker the network scored below the main threshold can still be chosen.

lam is chosen per outer fold by inner cross-validation over the remaining 4 folds
(objective: TP - FP), so held-out folds are never used for tuning.

Output: candidates CSV in evaluate_detections.py format, with `rank` 1..|S| for
the chosen set and >= 4 for the rest, plus `chosen`, `cscore` columns; and a
summary JSON. Matching/metrics reuse scripts/evaluation/compare_arms.py.
"""
import argparse, itertools, json, os, sys
import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))
from evaluate_detections import candidates as find_candidates, clopper_pearson  # noqa: E402

P = '/scratch/user/uqaste15/prostate-v2'
OOF = P + '/nnunet/predictions/Dataset511_SeedSegCvAll_corrected/nnUNetTrainer__nnUNetResEncUNetMPlans/oof'
LAMS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def subject_context(sid, extra_thr, thr, cache_dir):
    """Prostate frame, surface distance at candidates, and extra low-threshold candidates."""
    cf = os.path.join(cache_dir, f'{sid}_{extra_thr:g}.json')
    if os.path.exists(cf):
        return json.load(open(cf))
    img = nib.load(f'{OOF}/{sid}_probs.nii.gz')
    pr = np.asarray(img.dataobj, np.float32); z = np.array(img.header.get_zooms()[:3], float)
    pm, pp = pr[..., 1], pr[..., 2]
    mask = pp > 0.5
    lab, n = ndi.label(mask)
    if n > 1:  # keep largest component
        mask = lab == (np.argmax(ndi.sum(mask, lab, range(1, n + 1))) + 1)
    pts = np.argwhere(mask) * z
    cen = pts.mean(0); ev, evec = np.linalg.eigh(np.cov((pts - cen).T))
    # signed distance to prostate surface in mm (negative inside)
    din = ndi.distance_transform_edt(mask, sampling=z); dout = ndi.distance_transform_edt(~mask, sampling=z)
    sdist = np.where(mask, -din, dout)
    extras = []
    if extra_thr is not None and extra_thr < thr:
        for c in find_candidates(pm, z, extra_thr):
            if c['max'] > thr:
                continue  # already represented by a main candidate
            extras.append(c)
    ctx = dict(cen=cen.tolist(), evec=evec.tolist(), sd=np.sqrt(ev).tolist(), zooms=z.tolist(), extras=extras,
               sdist_shape=list(sdist.shape))
    np.save(os.path.join(cache_dir, f'{sid}_sdist.npy'), sdist.astype(np.float32))
    json.dump(ctx, open(cf, 'w'))
    return ctx


def geo(ctx, vox, sdist):
    z = np.array(ctx['zooms']); mm = np.atleast_2d(vox) * z
    q = (mm - np.array(ctx['cen'])) @ np.array(ctx['evec']) / np.array(ctx['sd'])
    q = q * np.sign(np.array(ctx['evec'])[2:3, :] + 1e-12)  # orient axes consistently (+z component)
    idx = np.clip(np.round(np.atleast_2d(vox)).astype(int), 0, np.array(sdist.shape) - 1)
    s = sdist[idx[:, 0], idx[:, 1], idx[:, 2]]
    return q, s, mm


def fit_prior(subs, refs):
    Q, S, D = [], [], []
    for sid in subs:
        q, s, mm = refs[sid]
        Q.append(q); S.append(s)
        D += [np.linalg.norm(a - b) for a, b in itertools.combinations(mm, 2)]
    return dict(pos=gaussian_kde(np.vstack(Q).T), surf=gaussian_kde(np.concatenate(S)), pair=gaussian_kde(np.array(D)))


def select(cs, K, prior, lam):
    """cs: DataFrame of candidates of one subject with q0..q2, sdist, xmm.. columns."""
    n = len(cs)
    if n == 0:
        return []
    k = min(K, n)
    ls = np.log(np.clip(cs['mean'].to_numpy(), 1e-30, None))
    if lam == 0:
        return list(np.argsort(-ls, kind='stable')[:k])
    q = cs[['q0', 'q1', 'q2']].to_numpy().T; mm = cs[['xmm', 'ymm', 'zmm']].to_numpy()
    lu = ls + lam * (np.log(prior['pos'](q) + 1e-300) + np.log(prior['surf'](cs['sdist'].to_numpy()) + 1e-300))
    D = np.linalg.norm(mm[:, None] - mm[None], axis=2)
    lpair = np.log(prior['pair'](D.ravel()) + 1e-300).reshape(n, n)
    # prune to the 12 best by unary score to bound the search
    pool = list(np.argsort(-lu)[:12])
    best, bs = None, -np.inf
    for S in itertools.combinations(pool, k):
        v = lu[list(S)].sum() + lam * sum(lpair[i, j] for i, j in itertools.combinations(S, 2))
        if v > bs:
            best, bs = list(S), v
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidates', default=P + '/eval/results/nnunet511_all127_candidates.csv')
    ap.add_argument('--folds', default='/scratch/user/uqaste15/src/seedseg-nnunet/scripts/nnunet/splits/splits_final_Dataset511_SeedSegCvAll_corrected.json')
    ap.add_argument('--thr', type=float, default=0.1)
    ap.add_argument('--extra-thr', type=float, default=None, help='add low-threshold candidates (e.g. 1e-3)')
    ap.add_argument('--k', type=int, default=None, help='fixed set size (default: implant count from record)')
    ap.add_argument('--lam', type=float, default=None, help='fixed lambda (default: inner-CV per outer fold)')
    ap.add_argument('--cache', default=P + '/eval/cache_constellation')
    ap.add_argument('--out', required=True, help='output prefix')
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    folds = json.load(open(a.folds)); subs = [s for f in folds for s in f['val']]
    fold_of = {s: k for k, f in enumerate(folds) for s in f['val']}
    C = pd.read_csv(a.candidates)
    rows, refs, Kof = [], {}, {}
    for sid in subs:
        ctx = subject_context(sid, a.extra_thr if a.extra_thr is not None else 1.0, a.thr, a.cache)
        sdist = np.load(os.path.join(a.cache, f'{sid}_sdist.npy'))
        rec = json.load(open(f'{P}/labelfix/refined/{sid}.json')); ex = set(rec.get('excluded', []))
        allv = np.array(rec['markers_final_vox'])
        Kof[sid] = a.k if a.k else len(allv)
        q, s, mm = geo(ctx, np.array([v for i, v in enumerate(allv) if i not in ex]), sdist)
        refs[sid] = (q, s, mm)
        c = C[C.subject == sid].copy(); c['extra'] = False
        if a.extra_thr is not None and ctx['extras']:
            e = pd.DataFrame(ctx['extras']); e['cand'] = 1000 + np.arange(len(e)); e['subject'] = sid; e['extra'] = True
            c = pd.concat([c, e], ignore_index=True)
        if len(c):
            q, s, mm = geo(ctx, c[['x', 'y', 'z']].to_numpy(), sdist)
            c[['q0', 'q1', 'q2']] = q; c['sdist'] = s; c[['xmm', 'ymm', 'zmm']] = mm
        rows.append(c)
    C = pd.concat(rows, ignore_index=True)
    from compare_arms import outcomes  # noqa: E402  (same 5 mm Hungarian matching)

    def run(eval_subs, prior_subs_fn, lam):
        out = []
        for sid in eval_subs:
            c = C[C.subject == sid].reset_index(drop=True)
            S = select(c, Kof[sid], prior_subs_fn(sid), lam)
            c['chosen'] = False; c.loc[S, 'chosen'] = True
            order = list(S) + [i for i in np.argsort(-c['mean'].to_numpy(), kind='stable') if i not in S]
            c['rank'] = 0
            for r, i in enumerate(S):
                c.loc[i, 'rank'] = r + 1
            for r, i in enumerate([i for i in order if i not in S]):
                c.loc[i, 'rank'] = 4 + r
            out.append(c)
        return pd.concat(out, ignore_index=True)

    # nested lambda choice
    lam_of, priors = {}, {}
    for k in range(5):
        train = [j for j in range(5) if j != k]
        if a.lam is not None:
            lam_of[k] = a.lam
        else:
            best = None
            for lam in LAMS:
                tot = 0
                for j in train:
                    pr = fit_prior([s for i in train if i != j for s in folds[i]['val']], refs)
                    df = run(folds[j]['val'], lambda s, pr=pr: pr, lam)
                    tot += score_subset(df, folds[j]['val'], outcomes, a.out)
                if best is None or tot > best[1]:
                    best = (lam, tot)
            lam_of[k] = best[0]
        priors[k] = fit_prior([s for i in train for s in folds[i]['val']], refs)
        print(f'outer fold {k}: lambda {lam_of[k]}', flush=True)
    res = pd.concat([run(folds[k]['val'], lambda s, k=k: priors[k], lam_of[k]) for k in range(5)], ignore_index=True)
    res.to_csv(a.out + '_candidates.csv', index=False)
    O, F = outcomes(a.out + '_candidates.csv')
    test = set(open(P + '/eval/test25.txt').read().split())
    summ = {}
    for nm, sel in [('all127', subs), ('test25', sorted(test))]:
        o = O[O.subject.isin(sel)]; tp = int((o.cat == 'hit').sum()); fp = int(F[sel].sum()); n = len(o)
        per = o.groupby('subject').apply(lambda d: (d.cat != 'hit').sum()) + F[sel]
        ndet = res[res.subject.isin(sel) & res.chosen].groupby('subject').size().reindex(sel).fillna(0)
        flag = int(sum(ndet[s] < Kof[s] for s in sel))
        summ[nm] = dict(tp=tp, markers=n, fp=fp, sens=tp / n, sens_cp95=clopper_pearson(tp, n), prec=tp / max(tp + fp, 1),
                        prec_cp95=clopper_pearson(tp, tp + fp), perfect=int((per == 0).sum()), n_subjects=len(sel),
                        count_flagged=flag, misses=o[o.cat != 'hit'].cat.value_counts().to_dict())
        print(f"{nm}: sens {tp}/{n} prec {tp}/{tp+fp} FP {fp} perfect {summ[nm]['perfect']}/{len(sel)} count-flagged {flag} misses {summ[nm]['misses']}")
    summ['lambda_per_fold'] = lam_of
    summ['settings'] = dict(thr=a.thr, extra_thr=a.extra_thr, k=a.k or 'implant count', lam=a.lam if a.lam is not None else 'inner-CV', grid=LAMS)
    json.dump(summ, open(a.out + '_summary.json', 'w'), indent=1, default=float)


def score_subset(df, sel, outcomes, prefix):
    tmp = prefix + f'_tmp{os.getpid()}.csv'; df.to_csv(tmp, index=False)
    O, F = outcomes(tmp); os.remove(tmp)
    O = O[O.subject.isin(sel)]
    return int((O.cat == 'hit').sum()) - int(F[sel].sum())


if __name__ == '__main__':
    main()
