#!/usr/bin/env python3
"""Manuscript figures for the v2 model (511): detections (cross-validation and independent
test) and failure cases. Each panel is one axial section; detections are the kept (top-3)
components of p_marker > 0.1, drawn as solid outlines and labelled TP/FP; missed reference
markers are dashed circles labelled FN.
usage: figures_v2.py --out <dir>
"""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

P = '/scratch/user/uqaste15/prostate-v2'
OOF = P + '/eval/probs_511'
CV_CANDS = P + '/eval/results/nnunet511_all127_candidates.csv'
MARKERS = P + '/eval/analysis/markers_511.csv'
F = P + '/fresh'
FR_PROBS = F + '/pred/511_ens/mprob'
FR_CANDS = F + '/pred/eval_511_ens_candidates.csv'
HALF_MM = 30.0
COL = {'TP': '#ff9f1c', 'FP': '#e71d73', 'FN': '#2ec4f1'}


def load(root, probs, sid):
    img = np.asarray(nib.load(f'{root}/images/{sid}.nii').dataobj, np.float32)
    im = nib.load(f'{probs}/{sid}.nii.gz')
    prob = np.asarray(im.dataobj, np.float32)
    rec = json.load(open(f'{root}/labelfix/refined/{sid}.json'))
    ex = set(rec.get('excluded', []))
    refs = np.array([m for i, m in enumerate(rec['markers_final_vox']) if i not in ex])
    zoom = np.array(im.header.get_zooms()[:3])
    return img, prob, refs, zoom


def panel(ax, root, probs, cands, sid, z=None, fn_refs=(), title=''):
    img, prob, refs, zoom = load(root, probs, sid)
    kept = cands[(cands.subject == sid) & (cands['rank'] <= 3)]
    pts = [(r.x, r.y, r.z, 'TP' if r.is_marker else 'FP') for r in kept.itertuples()]
    pts += [(*refs[j], 'FN') for j in fn_refs]
    if z is None:  # section with most events (ties: first)
        zs = [int(round(p[2])) for p in pts]
        z = max(set(zs), key=zs.count)
    on = [p for p in pts if abs(p[2] - z) <= 1.0] or pts
    cx, cy = np.mean([p[0] for p in on]), np.mean([p[1] for p in on])
    h = HALF_MM / zoom[0]
    if len(on) > 1:  # widen the crop so every event on this section fits
        h = max(h, (np.max(np.abs(np.array([[p[0] - cx, p[1] - cy] for p in on]))) + 12 / zoom[0]))
    x0, x1 = int(max(cx - h, 0)), int(min(cx + h, img.shape[0]))
    y0, y1 = int(max(cy - h, 0)), int(min(cy + h, img.shape[1]))
    sl = img[x0:x1, y0:y1, z]
    lo, hi = np.percentile(img[..., z], [1, 99.5])
    # array axes are (x, y); show x left-right and y top-bottom flipped so anterior is up
    ax.imshow(np.flipud(sl.T), cmap='gray', vmin=lo, vmax=hi, extent=(x0, x1, y0, y1), interpolation='lanczos')
    lab, _ = ndimage.label(prob > 0.1, structure=np.ones((3, 3, 3)))
    for x, y, zz, kind in pts:
        if abs(zz - z) > 1.0:
            continue
        if kind == 'FN':
            ax.add_patch(plt.Circle((x, y), 3.0 / zoom[0] * 1.6, fill=False, ls='--', lw=1.6, color=COL['FN']))
        else:
            k = lab[int(round(x)), int(round(y)), int(round(zz))]
            if k:
                m = (lab[x0:x1, y0:y1, z] == k).astype(float)
                ax.contour(np.arange(x0, x1), np.arange(y0, y1), m.T, levels=[0.5], colors=COL[kind], linewidths=1.6)
        ax.text(x + 6, y + 6, kind, color=COL[kind], fontsize=8, fontweight='bold')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    sb = 10.0 / zoom[0]  # 10 mm scale bar
    ax.plot([x1 - 4 - sb, x1 - 4], [y0 + 5, y0 + 5], color='white', lw=2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cv = pd.read_csv(CV_CANDS); fr = pd.read_csv(FR_CANDS)
    mk = pd.read_csv(MARKERS)
    subj = pd.read_csv(P + '/eval/results/nnunet511_all127_subjects.csv')
    subj['K'] = subj.n_ref + subj.n_excluded; subj['flag'] = subj.n_cand < subj.K

    # Figure 2: perfect detections, 3 cross-validation patients (top) and 3 independent (bottom)
    perfect = subj[subj.perfect & (subj.K == 3)].subject.tolist()
    cv_pick = [perfect[i] for i in (5, len(perfect) // 2, len(perfect) - 6)]
    fr_pick = ['E05', 'E03', 'E02']
    fig, axs = plt.subplots(2, 3, figsize=(9, 6.3))
    for ax, s in zip(axs[0], cv_pick): panel(ax, P, OOF, cv, s, title='Cross-validation')
    for ax, s in zip(axs[1], fr_pick): panel(ax, F, FR_PROBS, fr, s, title='Independent test')
    for ax, l in zip(axs.flat, 'ABCDEF'): ax.text(0.03, 0.95, l, transform=ax.transAxes, color='white', fontsize=11, fontweight='bold', va='top')
    fig.tight_layout(); fig.savefig(f'{a.out}/Figure2.png', dpi=300); fig.savefig(f'{a.out}/Figure2.tiff', dpi=300)

    # Figure 3: failures. Top: flagged (no candidate). Bottom: unflagged substitutions.
    flagged = subj[subj.flag].subject.tolist()
    subst = subj[~subj.flag & (subj.fn == 1) & (subj.fp == 1)].subject.tolist()
    fig, axs = plt.subplots(2, 3, figsize=(9, 6.3))
    for ax, s in zip(axs[0], flagged[:3]):
        j = mk[(mk.subject == s) & (mk.cat != 'hit')].marker.tolist()
        _, _, refs, _ = load(P, OOF, s)
        panel(ax, P, OOF, cv, s, z=int(round(refs[j[0]][2])), fn_refs=j, title='Flagged for review')
    # unflagged substitutions: the one patient whose FN and FP share a section, then one patient
    # shown at its FN section and at its FP section
    rows = [('S107', 'fn', 'Not flagged'), ('S091', 'fn', 'Not flagged (missed marker)'),
            ('S091', 'fp', 'Not flagged (selected instead)')]
    for ax, (s, which, ttl) in zip(axs[1], rows):
        j = mk[(mk.subject == s) & (mk.cat != 'hit')].marker.tolist()
        _, _, refs, _ = load(P, OOF, s)
        fpz = cv[(cv.subject == s) & (cv['rank'] <= 3) & (~cv.is_marker)].z.iloc[0]
        z = int(round(refs[j[0]][2])) if which == 'fn' else int(round(fpz)) if which == 'fp' else None
        panel(ax, P, OOF, cv, s, z=z, fn_refs=j, title=ttl)
    subst = [r[0] for r in rows]
    for ax, l in zip(axs.flat, 'ABCDEF'): ax.text(0.03, 0.95, l, transform=ax.transAxes, color='white', fontsize=11, fontweight='bold', va='top')
    fig.tight_layout(); fig.savefig(f'{a.out}/Figure3.png', dpi=300); fig.savefig(f'{a.out}/Figure3.tiff', dpi=300)
    print('Figure 2:', cv_pick, fr_pick); print('Figure 3:', flagged[:3], subst[:3])


if __name__ == '__main__':
    main()
