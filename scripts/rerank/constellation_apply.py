#!/usr/bin/env python3
"""Apply constellation selection to an external set (no CV): the geometric prior is fitted on
the reference markers of all 127 study subjects (prostate frame from the given arm's OOF maps),
lambda is fixed (default 0.1, the value inner CV chose in most folds), K = reference marker count.
Writes a rescore CSV (subject, cand, cscore) for evaluate_detections.py --rescore --score cscore:
chosen candidates get 2 + mean probability, the rest their mean probability."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constellation import subject_context, geo, fit_prior, select, P  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--candidates', required=True, help='evaluate_detections.py candidates CSV of the external set')
ap.add_argument('--root', required=True, help='external root (images/, labelfix/refined/)')
ap.add_argument('--probs3', required=True, help='dir with {sid}_probs.nii.gz (3-class) for the external set')
ap.add_argument('--study-oof', required=True, help='OOF dir of the same arm for the 127 (prior frames)')
ap.add_argument('--study-cache', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--lam', type=float, default=0.1)
ap.add_argument('--thr', type=float, default=0.1)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.cache, exist_ok=True); os.makedirs(a.study_cache, exist_ok=True)


def refs_of(sid, root, ctx, sdist):
    rec = json.load(open(f'{root}/labelfix/refined/{sid}.json')); ex = set(rec.get('excluded', []))
    allv = np.array(rec['markers_final_vox'])
    return geo(ctx, np.array([v for i, v in enumerate(allv) if i not in ex]), sdist), len(allv)


study = open(P + '/subjects.txt').read().split(); refs = {}
for sid in study:
    ctx = subject_context(sid, 1.0, a.thr, a.study_cache, a.study_oof)
    sd = np.load(os.path.join(a.study_cache, f'{sid}_sdist.npy'))
    refs[sid] = refs_of(sid, P, ctx, sd)[0]
prior = fit_prior(study, refs)
C = pd.read_csv(a.candidates); out = []
for sid in C.subject.unique():
    ctx = subject_context(sid, 1.0, a.thr, a.cache, a.probs3)
    sd = np.load(os.path.join(a.cache, f'{sid}_sdist.npy'))
    K = refs_of(sid, a.root, ctx, sd)[1]
    c = C[C.subject == sid].reset_index(drop=True)
    q, s, mm = geo(ctx, c[['x', 'y', 'z']].to_numpy(), sd)
    c[['q0', 'q1', 'q2']] = q; c['sdist'] = s; c[['xmm', 'ymm', 'zmm']] = mm
    S = select(c, K, prior, a.lam)
    c['cscore'] = c['mean'] + np.where(np.isin(np.arange(len(c)), S), 2.0, 0.0)
    out.append(c[['subject', 'cand', 'cscore']])
pd.concat(out).to_csv(a.out, index=False); print('wrote', a.out)
