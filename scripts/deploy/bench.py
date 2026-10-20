#!/usr/bin/env python
"""Time the ONNX pipeline on one raw (uncorrected) T1: N4, session load, and one-fold /
five-fold inference with and without mirroring. Prints one JSON line.
  bench.py --models <dir> --input <raw T1 .nii> --device cpu|cuda [--threads N] [--five-mirror]
Five-fold mirrored inference on CPU takes long; without --five-mirror it is reported as
5 x the one-fold time (the folds run sequentially, so this is exact up to noise)."""
import argparse, json, os, sys, time

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedseg_onnx import SeedSegONNX, n4_correct_xyz  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--models', required=True); ap.add_argument('--input', required=True)
ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int)
ap.add_argument('--precision', default='fp32'); ap.add_argument('--five-mirror', action='store_true')
a = ap.parse_args()
img = nib.load(a.input); xyz = np.asarray(img.dataobj, np.float64)
r = dict(device=a.device, threads=a.threads or len(os.sched_getaffinity(0)), precision=a.precision, shape=list(xyz.shape))
t = time.time(); c = n4_correct_xyz(xyz); r['n4_s'] = time.time() - t
vol = c.T.astype(np.float32); sp = np.array(img.header.get_zooms()[:3])[::-1]
for folds, key in [(['0'], '1fold'), (['0', '1', '2', '3', '4'], '5fold')]:
    t = time.time(); m = SeedSegONNX(a.models, folds, mirror=True, device=a.device, threads=a.threads, precision=a.precision, log=None)
    r[f'load_{key}_s'] = time.time() - t
    r['providers'] = m.providers
    for mir in (False, True):
        if key == '5fold' and mir and not a.five_mirror:
            r['5fold_mirror_s'] = 5 * r['1fold_mirror_s']; r['5fold_mirror_extrapolated'] = True
            continue
        m.mirror_axes = tuple(m.meta['mirror_axes']) if mir else ()
        if a.device == 'cuda':
            m.batch = 8 if mir else 1
            m.predict(vol, sp)  # warm-up (cuDNN autotune, allocations)
        t = time.time(); m.predict(vol, sp); r[f'{key}_{"mirror" if mir else "nomirror"}_s'] = time.time() - t
    del m
print(json.dumps(r), flush=True)
