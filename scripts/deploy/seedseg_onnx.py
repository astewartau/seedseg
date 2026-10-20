#!/usr/bin/env python
"""SeedSeg v2 inference with ONNX Runtime (no PyTorch, no nnU-Net install).

Re-implements nnU-Net v2 ``nnUNetv2_predict`` for the exported fold networks
(``export_onnx.py``), so the same code runs in the OpenRecon container (CPU or
CUDA) and as the reference for the browser worker (``web/js/inference-worker.js``):

1. N4 bias-field correction exactly as for the training images
   (``homogeneity_correction.py``: SimpleITK Otsu mask + N4 with default settings,
   applied to the NIfTI-order array).
2. nnU-Net preprocessing: crop to the bounding box of nonzero voxels, per-image
   z-score (no mask), resampling to the plans spacing (a no-op for 2 x 0.6875 x
   0.6875 mm images; otherwise cubic spline as in nnU-Net, via scipy).
3. Gaussian-weighted sliding window (step 0.5, sigma 1/8 of the patch, scaled to
   max 10), optional test-time mirroring over all axis combinations (logits
   averaged), softmax per fold, ensemble = mean of the fold softmax maps.
4. Candidates: 26-connected components of marker probability > threshold, ranked
   by mean probability; position = probability-weighted centroid. The top K
   (implant count, default 3) are the detections; ``count_flag`` is set when fewer
   than K components exist.

Command line (NIfTI in, JSON + probability NIfTI out)::

  seedseg_onnx.py --models <dir with seedseg-v2.json> --input t1.nii[.gz] --out <prefix>
      [--folds 0 1 2 3 4] [--no-mirror] [--no-n4] [--k 3] [--threshold 0.1]
      [--device cpu|cuda] [--threads N] [--precision fp32|w16|fp16]

Without ``--no-n4`` the input must be the uncorrected T1 (as from dcm2niix); the
nnU-Net training inputs were N4-corrected, so pass ``--no-n4`` for those.
"""
import argparse, json, os, time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

S26 = np.ones((3, 3, 3), bool)


# ----------------------------------------------------------------------------- preprocessing

def n4_correct_xyz(arr_xyz):
    """N4 as used for the training data: arr in NIfTI (x, y, z) order, float64 in/out."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.asarray(arr_xyz, np.float64))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    return sitk.GetArrayFromImage(sitk.N4BiasFieldCorrection(img, mask))


def nonzero_bbox(vol):
    """Bounding box (list of [lo, hi)) of nonzero voxels; nnU-Net crop_to_nonzero
    (hole filling does not change the bounding box)."""
    nz = vol != 0
    if not nz.any():
        return [[0, s] for s in vol.shape]
    box = []
    for ax in range(vol.ndim):
        a = np.any(nz, axis=tuple(i for i in range(vol.ndim) if i != ax))
        idx = np.where(a)[0]
        box.append([int(idx[0]), int(idx[-1]) + 1])
    return box


def zscore(vol):
    v = vol.astype(np.float32, copy=True)
    m, s = v.mean(), v.std()
    v -= m
    v /= max(s, 1e-8)
    return v


def new_shape_for(shape, spacing, target):
    return np.array([int(round(sp / t * n)) for sp, t, n in zip(spacing, target, shape)])


def resample(vol, new_shape, order=3):
    """nnU-Net resample_data_or_seg (no separate-z for anisotropy < 3) via skimage-equivalent
    spline zoom with edge mode. Only used when the image spacing differs from the plans."""
    shape = np.array(vol.shape)
    if np.all(shape == new_shape):
        return vol
    from skimage.transform import resize
    return resize(vol.astype(float), tuple(int(x) for x in new_shape), order, mode='edge',
                  anti_aliasing=False).astype(np.float32)


# ----------------------------------------------------------------------------- sliding window

def gaussian_map(patch, sigma_scale=1 / 8, value_scaling_factor=10):
    tmp = np.zeros(patch)
    tmp[tuple(p // 2 for p in patch)] = 1
    g = ndi.gaussian_filter(tmp, [p * sigma_scale for p in patch], 0, mode='constant', cval=0)
    g /= g.max() / value_scaling_factor
    g = g.astype(np.float32)
    g[g == 0] = g[g > 0].min()
    return g


def window_steps(size, patch, step=0.5):
    out = []
    for i, k in zip(size, patch):
        n = int(np.ceil((i - k) / (k * step))) + 1
        s = (i - k) / (n - 1) if n > 1 else 0
        out.append([int(np.round(s * j)) for j in range(n)])
    return out


def mirror_sets(axes):
    """Axis combinations of nnU-Net test-time mirroring (spatial axes, 0-based), identity first."""
    return [()] + [c for i in range(len(axes)) for c in combinations(axes, i + 1)]


def softmax(x, axis=0):
    x = x - x.max(axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis, keepdims=True)
    return x


class SeedSegONNX:
    """Ensemble of exported fold networks. ``predict`` returns class probabilities on the
    input grid (C, z, y, x) for a (z, y, x) volume in SimpleITK array order."""

    def __init__(self, model_dir, folds=None, mirror=True, device='cpu', threads=None, precision='fp32',
                 batch_size=None, log=print):
        import onnxruntime as ort
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / 'seedseg-v2.json').read_text())
        self.patch = tuple(self.meta['patch_size'])
        self.spacing = np.array(self.meta['spacing_zyx'], float)
        self.mirror_axes = tuple(self.meta['mirror_axes']) if mirror else ()
        self.folds = [str(f) for f in (folds if folds is not None else self.meta['folds'].keys())]
        self.log = log or (lambda *a: None)
        so = ort.SessionOptions()
        # explicit count: ORT's default (all host cores, pinned) breaks under Slurm cgroups
        so.intra_op_num_threads = int(threads or len(os.sched_getaffinity(0)))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if device == 'cuda':
            try:
                ort.preload_dlls()
            except Exception:
                pass
            prov = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC', 'use_tf32': 0}), 'CPUExecutionProvider']
        else:
            prov = ['CPUExecutionProvider']
        self.batch = batch_size or (len(mirror_sets(self.mirror_axes)) if device == 'cuda' else 1)
        self.sessions = []
        for f in self.folds:
            fn = self.dir / self.meta['folds'][f][precision]['file']
            self.sessions.append(ort.InferenceSession(str(fn), so, providers=prov))
        self.providers = self.sessions[0].get_providers()
        self.gauss = gaussian_map(self.patch, self.meta['gaussian']['sigma_scale'],
                                  self.meta['gaussian']['value_scaling_factor'])

    def _run(self, sess, x):
        outs = []
        for i in range(0, len(x), self.batch):
            outs.append(sess.run(None, {'input': np.ascontiguousarray(x[i:i + self.batch])})[0])
        return np.concatenate(outs, 0)

    def _logits(self, sess, data):
        """Sliding-window logits (C, z, y, x) of one network on a normalised (z, y, x) volume."""
        shape = np.array(data.shape)
        pshape = np.maximum(shape, self.patch)
        pad_lo = (pshape - shape) // 2
        if np.any(pshape != shape):
            data = np.pad(data, [(lo, p - s - lo) for lo, p, s in zip(pad_lo, pshape, shape)])
        flips = mirror_sets(self.mirror_axes)
        acc = None
        norm = np.zeros(data.shape, np.float32)
        steps = window_steps(data.shape, self.patch, self.meta['tile_step_size'])
        for sz in steps[0]:
            for sy in steps[1]:
                for sx in steps[2]:
                    sl = (slice(sz, sz + self.patch[0]), slice(sy, sy + self.patch[1]), slice(sx, sx + self.patch[2]))
                    tile = data[sl]
                    x = np.stack([np.flip(tile, a) if a else tile for a in flips])[:, None].astype(np.float32)
                    y = self._run(sess, x)  # (F, C, z, y, x)
                    pred = sum(np.flip(y[i], tuple(a_ + 1 for a_ in a)) if a else y[i] for i, a in enumerate(flips))
                    pred = pred / len(flips)
                    if acc is None:
                        acc = np.zeros((pred.shape[0], *data.shape), np.float32)
                    acc[(slice(None), *sl)] += pred * self.gauss
                    norm[sl] += self.gauss
        acc /= norm
        crop = tuple(slice(lo, lo + s) for lo, s in zip(pad_lo, shape))
        return acc[(slice(None), *crop)]

    def predict(self, vol_zyx, spacing_zyx=None, return_folds=False):
        """vol_zyx: (z, y, x) image (already N4-corrected). Returns probs (C, z, y, x) float32,
        the fold-mean of softmax maps, and optionally per-fold probs."""
        vol = np.asarray(vol_zyx, np.float32)
        box = nonzero_bbox(vol)
        sl = tuple(slice(lo, hi) for lo, hi in box)
        data = zscore(vol[sl])
        sp = np.array(spacing_zyx if spacing_zyx is not None else self.spacing, float)
        ns = new_shape_for(data.shape, sp, self.spacing)
        resampled = np.any(ns != np.array(data.shape))
        if resampled:
            self.log(f'resampling {data.shape} -> {tuple(ns)} (spacing {sp} -> {self.spacing})')
            data = resample(data, ns)
        mean, per_fold = None, []
        for f, sess in zip(self.folds, self.sessions):
            t = time.time()
            p = softmax(self._logits(sess, data), 0)
            if resampled:  # nnU-Net resamples logits (order 1) before softmax; close enough off-protocol
                p = np.stack([resample(c, np.array(vol[sl].shape), order=1) for c in p])
            self.log(f'fold {f}: {time.time() - t:.1f} s')
            mean = p.copy() if mean is None else mean + p
            if return_folds:
                per_fold.append(p)
        mean /= len(self.sessions)
        out = np.zeros((mean.shape[0], *vol.shape), np.float32)
        out[0] = 1.0  # outside the crop: background
        out[(slice(None), *sl)] = mean
        if return_folds:
            folds = []
            for p in per_fold:
                o = np.zeros_like(out); o[0] = 1.0; o[(slice(None), *sl)] = p; folds.append(o)
            return out, folds
        return out


# ----------------------------------------------------------------------------- candidates

def candidates(p, zooms=(1, 1, 1), threshold=0.1):
    """Connected components (26-conn) of p > threshold, sorted by mean probability (desc).
    Positions are probability-weighted centroids in voxel coordinates of p's axes."""
    lab, n = ndi.label(p > threshold, structure=S26)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    mean = ndi.mean(p, lab, idx); mx = ndi.maximum(p, lab, idx)
    size = ndi.sum(np.ones_like(p), lab, idx)
    cen = ndi.center_of_mass(p, lab, idx)
    vox = float(np.prod(zooms))
    out = [dict(label=int(i), mean_prob=float(m), max_prob=float(x), n_voxels=int(s), volume_mm3=float(s * vox),
                centroid_vox=[float(c) for c in cc]) for i, m, x, s, cc in zip(idx, mean, mx, size, cen)]
    out.sort(key=lambda c: c['mean_prob'], reverse=True)
    for r, c in enumerate(out):
        c['rank'] = r + 1
    return out


def detect(p, zooms=(1, 1, 1), k=3, threshold=0.1):
    cs = candidates(p, zooms, threshold)
    return dict(k=int(k), threshold=threshold, n_candidates=len(cs), markers=cs[:k], count_flag=len(cs) < k,
                others=cs[k:])


def marker_mask(p, dets, threshold=0.1):
    """uint8 mask with value r (1..K) on the voxels of the r-th detected component."""
    lab, _ = ndi.label(p > threshold, structure=S26)
    m = np.zeros(p.shape, np.uint8)
    for r, c in enumerate(dets['markers']):
        m[lab == c['label']] = r + 1
    return m


# ----------------------------------------------------------------------------- NIfTI front end

def run_nifti(model, path, n4=True, k=3, threshold=0.1, log=print):
    """Full pipeline on a NIfTI T1. Returns (detections dict, marker prob (x,y,z), nibabel image).
    Detection centroids are given in NIfTI voxel (x, y, z) and world (RAS mm) coordinates."""
    import nibabel as nib
    img = nib.load(str(path))
    xyz = np.asarray(img.dataobj, np.float64)
    if xyz.ndim == 4:
        xyz = xyz[..., 0]
    t = time.time()
    if n4:
        xyz = n4_correct_xyz(xyz)
        log(f'N4: {time.time() - t:.1f} s')
    zooms = np.array(img.header.get_zooms()[:3], float)
    probs = model.predict(xyz.T, spacing_zyx=zooms[::-1])  # (C, z, y, x)
    pm = probs[model.meta['marker_class']].T  # (x, y, z)
    d = detect(pm, zooms, k, threshold)
    for c in d['markers'] + d['others']:
        c['centroid_world_mm'] = [float(v) for v in nib.affines.apply_affine(img.affine, c['centroid_vox'])]
    return d, pm, img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--folds', nargs='+')
    ap.add_argument('--no-mirror', action='store_true')
    ap.add_argument('--no-n4', action='store_true')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--threshold', type=float, default=0.1)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--threads', type=int)
    ap.add_argument('--precision', default='fp32', choices=['fp32', 'w16', 'fp16'],
                    help='w16: fp16-stored weights, fp32 compute; fp16: fp16 compute (GPU)')
    a = ap.parse_args()
    import nibabel as nib
    t0 = time.time()
    m = SeedSegONNX(a.models, a.folds, mirror=not a.no_mirror, device=a.device, threads=a.threads,
                    precision=a.precision)
    t1 = time.time()
    d, pm, img = run_nifti(m, a.input, n4=not a.no_n4, k=a.k, threshold=a.threshold)
    t2 = time.time()
    d.update(folds=m.folds, mirror=bool(m.mirror_axes), providers=m.providers, load_s=t1 - t0, infer_s=t2 - t1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    hdr = img.header.copy(); hdr.set_data_dtype(np.float32); hdr['scl_slope'], hdr['scl_inter'] = 1, 0
    nib.save(nib.Nifti1Image(pm.astype(np.float32), img.affine, hdr), a.out + '_marker_prob.nii.gz')
    Path(a.out + '_detections.json').write_text(json.dumps(d, indent=1) + '\n')
    print(json.dumps({k: d[k] for k in ('n_candidates', 'count_flag', 'infer_s')}),
          [(round(c['mean_prob'], 4), [round(v, 1) for v in c['centroid_vox']]) for c in d['markers']])


if __name__ == '__main__':
    main()
