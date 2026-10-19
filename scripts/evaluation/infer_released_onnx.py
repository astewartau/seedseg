"""Run the 4 released SeedSeg ONNX models on prepared subjects, replicating
evaluate_production_models.py preprocessing (CropOrPad to multiple of 32,
z-normalise over the padded volume, softmax, centre-crop back).

Saves per subject: per-model marker prob, mean prostate prob (float16 npz).
The released-ensemble marker map used by evaluate_detections.py is the mean of the
4 per-model marker probabilities, written as eval/released_probs/{sid}.nii.gz on the
affine of images/{sid}.nii. (Copied from the 2026-10-16 diagnostic session.)
"""
import argparse, glob, os, sys, time
import numpy as np
import nibabel as nib
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', required=True)
ap.add_argument('--models', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--subjects', nargs='*')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

so = ort.SessionOptions()
so.intra_op_num_threads = 16
so.inter_op_num_threads = 1
so.log_severity_level = 3
sessions = [ort.InferenceSession(m, so, providers=['CPUExecutionProvider']) for m in sorted(args.models)]
print('models:', [os.path.basename(m) for m in sorted(args.models)], flush=True)


def tio_pad(vol, target):
    # torchio CropOrPad: ini = ceil(diff/2), fin = floor(diff/2)
    pw = []
    for s, t in zip(vol.shape, target):
        d = t - s
        pw.append((int(np.ceil(d / 2)), int(np.floor(d / 2))))
    return np.pad(vol, pw, mode='constant', constant_values=0), pw


def crop_back(vol, orig):
    # evaluate_production_models.pad_or_crop_numpy: start = diff // 2
    sl = []
    for s, t in zip(vol.shape, orig):
        d = s - t
        st = d // 2 if d > 0 else 0
        sl.append(slice(st, st + t))
    return vol[tuple(sl)]


subs = args.subjects or sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data_dir, 'z*')))
for sid in subs:
    outp = os.path.join(args.out, f'{sid}.npz')
    if os.path.exists(outp):
        continue
    mri = os.path.join(args.data_dir, sid, f'{sid}_MRI_homogeneity-corrected.nii')
    if not os.path.exists(mri):
        print('skip (no MRI)', sid, flush=True)
        continue
    t0 = time.time()
    img = nib.load(mri)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    orig = vol.shape
    target = tuple(((d + 31) // 32) * 32 for d in orig)
    padded, pw = tio_pad(vol, target)
    x = (padded - padded.mean()) / padded.std()
    x = np.ascontiguousarray(x[None, None].astype(np.float32))
    marker = []
    prostate = []
    for s in sessions:
        logits = s.run(None, {'input': x})[0][0]
        logits = logits - logits.max(axis=0, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=0, keepdims=True)
        marker.append(crop_back(p[1], orig))
        prostate.append(crop_back(p[2], orig))
    np.savez_compressed(outp,
                        marker=np.stack(marker).astype(np.float16),
                        prostate=np.mean(prostate, axis=0).astype(np.float16),
                        pad=np.array(pw))
    print(f'{sid} {orig} {time.time() - t0:.1f}s', flush=True)
