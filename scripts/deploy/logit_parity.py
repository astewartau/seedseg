#!/usr/bin/env python
"""Max |logit difference| between the original fp32 fold graph and other exports of it
(e.g. webgpu, webgpu16, int8), on the central patch of each case's normalised image.
  logit_parity.py --models <dir> --inputs <dir> --subjects <file> --fold 0 \
      --variant webgpu=<dir> --variant webgpu16=<dir> --variant int8=<dir> --out <json>"""
import argparse, json, os, sys
from pathlib import Path

import nibabel as nib
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedseg_onnx import nonzero_bbox, zscore  # noqa: E402


def centre_patch(path, patch):
    vol = np.asarray(nib.load(path).dataobj, np.float32).T
    box = nonzero_bbox(vol)
    d = zscore(vol[tuple(slice(a, b) for a, b in box)])
    d = np.pad(d, [(0, max(0, p - s)) for p, s in zip(patch, d.shape)])
    c = [(s - p) // 2 for s, p in zip(d.shape, patch)]
    return np.ascontiguousarray(d[tuple(slice(o, o + p) for o, p in zip(c, patch))][None, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', required=True, type=Path)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--fold', required=True)
    ap.add_argument('--variant', action='append', default=[], help='key=dir (key in that dir\'s seedseg-v2.json)')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = len(os.sched_getaffinity(0))
    meta = json.loads((a.models / 'seedseg-v2.json').read_text())
    patch = meta['patch_size']
    sess = lambda d, k: ort.InferenceSession(str(Path(d) / json.loads((Path(d) / 'seedseg-v2.json').read_text())['folds'][a.fold][k]['file']), so)
    ref = sess(a.models, 'fp32')
    var = {k: sess(d, k) for k, d in (v.split('=', 1) for v in a.variant)}
    res = {k: {} for k in var}
    for sid in open(a.subjects).read().split():
        x = centre_patch(os.path.join(a.inputs, f'{sid}_0000.nii.gz'), patch)
        r = ref.run(None, {'input': x})[0]
        for k, s in var.items():
            res[k][sid] = float(np.abs(s.run(None, {'input': x})[0] - r).max())
        print(sid, {k: f'{v[sid]:.3g}' for k, v in res.items()}, flush=True)
    summary = {k: dict(max_abs_dlogit=max(v.values()), median=float(np.median(list(v.values()))), per_case=v) for k, v in res.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(fold=a.fold, **summary), indent=1))
    print({k: v['max_abs_dlogit'] for k, v in summary.items()})


if __name__ == '__main__':
    main()
