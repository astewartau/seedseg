#!/usr/bin/env python
"""Run the ONNX pipeline (seedseg_onnx.py) on nnU-Net-style inputs ({sid}_0000.nii.gz,
already N4-corrected) and save marker probability maps for each fold and the ensemble:
  {out}/fold{k}/{sid}.nii.gz, {out}/ens/{sid}.nii.gz, {out}/timing.json
Used for the parity check against nnUNetv2_predict (compare_parity.py)."""
import argparse, json, os, sys, time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedseg_onnx import SeedSegONNX  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--subjects', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--folds', nargs='+')
    ap.add_argument('--no-mirror', action='store_true')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int)
    ap.add_argument('--precision', default='fp32')
    ap.add_argument('--batch', type=int)
    a = ap.parse_args()
    m = SeedSegONNX(a.models, a.folds, mirror=not a.no_mirror, device=a.device, threads=a.threads, precision=a.precision,
                    batch_size=a.batch, log=None)
    print('providers', m.providers, 'folds', m.folds, 'mirror', m.mirror_axes, 'batch', m.batch, flush=True)
    subs = open(a.subjects).read().split()
    timing = {}
    write_ens = len(m.folds) > 1
    for d in (['ens'] if write_ens else []) + [f'fold{f}' for f in m.folds]:
        (a.out / d).mkdir(parents=True, exist_ok=True)
    for sid in subs:
        img = nib.load(os.path.join(a.inputs, f'{sid}_0000.nii.gz'))
        xyz = np.asarray(img.dataobj, np.float32)
        t = time.time()
        ens, folds = m.predict(xyz.T, np.array(img.header.get_zooms()[:3])[::-1], return_folds=True)
        dt = time.time() - t
        timing[sid] = dt
        c = m.meta['marker_class']
        hdr = img.header.copy(); hdr.set_data_dtype(np.float32); hdr['scl_slope'], hdr['scl_inter'] = 1, 0
        if write_ens:
            nib.save(nib.Nifti1Image(ens[c].T, img.affine, hdr), a.out / 'ens' / f'{sid}.nii.gz')
        for f, p in zip(m.folds, folds):
            nib.save(nib.Nifti1Image(p[c].T, img.affine, hdr), a.out / f'fold{f}' / f'{sid}.nii.gz')
        print(f'{sid}: {dt:.1f} s', flush=True)
    (a.out / f'timing_{"_".join(m.folds)}.json').write_text(json.dumps(dict(device=a.device, providers=m.providers, folds=m.folds,
                                                      mirror=bool(m.mirror_axes), per_case_s=timing), indent=1))


if __name__ == '__main__':
    main()
