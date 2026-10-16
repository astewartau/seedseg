#!/usr/bin/env python
"""Convert nnU-Net probability .npz files to NIfTI in the original image space.

nnU-Net (SimpleITKIO) stores ``probabilities`` as (C, z, y, x); we transpose
to (x, y, z) and use the affine of the matching raw image, so outputs overlay
directly on /scratch/user/uqaste15/prostate-v2/images/{sid}.nii.

Per case writes:
  {sid}_marker_prob.nii.gz   float32 (x,y,z), class 1 (gold marker) probability
  {sid}_probs.nii.gz         float32 (x,y,z,3), all class probabilities
  {sid}_seg.nii.gz           nnU-Net argmax segmentation (copied)

Modes:
  oof  --model <trainer__plans__3d_fullres dir> --raw <raw dataset> --folds 0 1 2 3 4
       collect fold_X/validation/*.npz (each case appears in exactly one fold)
  dir  --pred <nnUNetv2_predict output with --save_probabilities> --images <imagesTs>
"""
import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def export_case(npz, seg, ref_img, out_dir, sid):
    probs = np.load(npz)["probabilities"].astype(np.float32)  # (C, z, y, x)
    probs = np.transpose(probs, (3, 2, 1, 0))  # (x, y, z, C)
    ref = nib.load(str(ref_img))
    if probs.shape[:3] != ref.shape[:3]:
        raise ValueError(f"{sid}: probs {probs.shape} vs image {ref.shape}")
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr["scl_slope"], hdr["scl_inter"] = 1, 0
    nib.save(nib.Nifti1Image(probs[..., 1], ref.affine, hdr), str(out_dir / f"{sid}_marker_prob.nii.gz"))
    nib.save(nib.Nifti1Image(probs, ref.affine, hdr), str(out_dir / f"{sid}_probs.nii.gz"))
    if seg.exists():
        shutil.copy(seg, out_dir / f"{sid}_seg.nii.gz")
        s = np.asarray(nib.load(str(seg)).dataobj)
        agree = (s == probs.argmax(-1)).mean()
        if agree < 0.999:
            print(f"warning: {sid}: argmax(probs) agrees with seg on only {agree:.4f} of voxels")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    o = sub.add_parser("oof")
    o.add_argument("--model", type=Path, required=True)
    o.add_argument("--raw", type=Path, required=True)
    o.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    o.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("dir")
    d.add_argument("--pred", type=Path, required=True)
    d.add_argument("--images", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if a.mode == "oof":
        splits = json.loads((a.raw / "splits_final.json").read_text())
        index = {}
        for f in a.folds:
            vdir = a.model / f"fold_{f}" / "validation"
            for sid in splits[f]["val"]:
                npz = vdir / f"{sid}.npz"
                if not npz.exists():
                    print(f"missing {npz}")
                    continue
                export_case(npz, vdir / f"{sid}.nii.gz", a.raw / "imagesTr" / f"{sid}_0000.nii.gz", a.out, sid)
                index[sid] = f
        (a.out / "oof_folds.json").write_text(json.dumps(index, indent=1) + "\n")
        print(f"exported {len(index)} out-of-fold cases to {a.out}")
    else:
        n = 0
        for npz in sorted(a.pred.glob("*.npz")):
            sid = npz.stem
            export_case(npz, a.pred / f"{sid}.nii.gz", a.images / f"{sid}_0000.nii.gz", a.out, sid)
            n += 1
        print(f"exported {n} cases to {a.out}")


if __name__ == "__main__":
    main()
