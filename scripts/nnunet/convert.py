#!/usr/bin/env python
"""Build nnU-Net v2 raw datasets for SeedSeg from a label variant.

Designs
-------
cvall     ("Design A") all subjects that have a label in the variant, 5-fold CV
          with a fixed fold assignment (splits/folds_cvall_seed0.json).
temporal  ("Design B") the train+val subjects of splits_original.json in 5-fold
          CV (splits/folds_temporal_seed0.json); the 25 test subjects go to
          imagesTs/labelsTs and are never used for training.

Fold assignments of real subjects are generated ONCE (``make-folds``) from the
full ID lists with seed 0 and stored in the repo, so every label variant uses
the same folds. Subjects missing from a variant are simply dropped from their
fold.

Synthetic cases (``--synthetic``) named ``{sid}_syn{k}`` are added to imagesTr,
but in splits_final.json they only ever appear in the TRAINING list of folds
where ``sid`` itself is a training case. They are never validation cases, and
synthetic cases of temporal-test subjects are skipped altogether.

Dataset IDs: 500 + 10 * VARIANT_INDEX[variant] + DESIGN_CODE[design(+syn)]
  e.g. original -> 501 cvall, 502 temporal, 503 cvall+syn, 504 temporal+syn
       corrected -> 511, 512, 513, 514; original_sphere -> 521, 522, 523, 524
New variants must be appended to VARIANT_INDEX (never reorder it).

Usage
-----
  python scripts/nnunet/convert.py make-folds            # once; already done
  python scripts/nnunet/convert.py build --variant original --design cvall
  python scripts/nnunet/convert.py build --variant original --design temporal
  python scripts/nnunet/convert.py id --variant original --design cvall   # print ID
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("/scratch/user/uqaste15/prostate-v2")
REPO_SPLITS = Path(__file__).resolve().parent / "splits"
NNUNET_ROOT = ROOT / "nnunet"
N_FOLDS = 5
SEED = 0

VARIANT_INDEX = {"original": 0, "corrected": 1, "original_sphere": 2}  # append only
DESIGN_CODE = {("cvall", False): 1, ("temporal", False): 2,
               ("cvall", True): 3, ("temporal", True): 4}
DESIGN_NAME = {"cvall": "SeedSegCvAll", "temporal": "SeedSegTemporal"}
LABELS = {"background": 0, "marker": 1, "prostate": 2}
# Markers rejected at human QC carry the nnU-Net ignore label (must be the highest value)
IGNORE = 3
SEEN_IGNORE = []


def dataset_id(variant, design, synthetic=False):
    if variant not in VARIANT_INDEX:
        sys.exit(f"Unknown variant '{variant}': add it to VARIANT_INDEX in {__file__}")
    return 500 + 10 * VARIANT_INDEX[variant] + DESIGN_CODE[(design, synthetic)]


def dataset_name(variant, design, synthetic=False):
    did = dataset_id(variant, design, synthetic)
    return f"Dataset{did:03d}_{DESIGN_NAME[design]}{'Syn' if synthetic else ''}_{variant}"


def all_subject_ids():
    return sorted(p.name[:-4] for p in (ROOT / "images").glob("*.nii"))


def assign_folds(ids, seed=SEED, n_folds=N_FOLDS):
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {sid: i % n_folds for i, sid in enumerate(ids)}


def make_folds(args):
    REPO_SPLITS.mkdir(parents=True, exist_ok=True)
    orig = json.loads((ROOT / "splits_original.json").read_text())
    ids = all_subject_ids()
    have = set(ids)
    temporal_pool = [s for s in orig["train"] + orig["val"] if s in have]
    test = [s for s in orig["test"] if s in have]
    for name, pool, extra in [("cvall", ids, {}),
                              ("temporal", temporal_pool, {"test": sorted(test)})]:
        out = REPO_SPLITS / f"folds_{name}_seed{SEED}.json"
        if out.exists() and not args.force:
            print(f"{out} exists (use --force to regenerate); keeping it")
            continue
        folds = assign_folds(pool)
        doc = {"seed": SEED, "n_folds": N_FOLDS, "n_subjects": len(folds),
               "folds": dict(sorted(folds.items())), **extra}
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out} ({len(folds)} subjects)")


def load_folds(design):
    return json.loads((REPO_SPLITS / f"folds_{design}_seed{SEED}.json").read_text())


def write_image(src, dst):
    img = nib.load(str(src))
    out = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
    nib.save(out, str(dst))


def write_label(src, dst, ref_img):
    lab = nib.load(str(src))
    data = np.rint(np.asarray(lab.dataobj, dtype=np.float32)).astype(np.uint8)
    vals = set(np.unique(data))
    if IGNORE in vals:
        SEEN_IGNORE.append(str(src))
    bad = vals - set(LABELS.values()) - {IGNORE}
    if bad:
        raise ValueError(f"{src}: unexpected label values {bad}")
    ref = nib.load(str(ref_img))
    if data.shape != ref.shape or not np.allclose(lab.affine, ref.affine, atol=1e-4):
        raise ValueError(f"{src}: shape/affine mismatch with {ref_img}")
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"], hdr["scl_inter"] = 1, 0
    nib.save(nib.Nifti1Image(data, ref.affine, hdr), str(dst))


def build(args):
    variant, design, syn = args.variant, args.design, args.synthetic
    did = dataset_id(variant, design, syn)
    name = dataset_name(variant, design, syn)
    raw = NNUNET_ROOT / "nnUNet_raw" / name
    lab_dir = ROOT / "labels" / variant
    if not lab_dir.is_dir():
        sys.exit(f"no label variant at {lab_dir}")
    labelled = {p.name[:-7] for p in lab_dir.glob("*.nii.gz")}

    fdoc = load_folds(design)
    folds = {s: f for s, f in fdoc["folds"].items() if s in labelled}
    dropped = sorted(set(fdoc["folds"]) - set(folds))
    test = [s for s in fdoc.get("test", []) if s in labelled]

    if raw.exists():
        if not args.overwrite:
            sys.exit(f"{raw} exists; pass --overwrite to rebuild")
        shutil.rmtree(raw)
    for d in ["imagesTr", "labelsTr"] + (["imagesTs", "labelsTs"] if test else []):
        (raw / d).mkdir(parents=True)

    def add(sid, img, lab, split):
        write_image(img, raw / f"images{split}" / f"{sid}_0000.nii.gz")
        write_label(lab, raw / f"labels{split}" / f"{sid}.nii.gz", img)

    for sid in sorted(folds):
        add(sid, ROOT / "images" / f"{sid}.nii", lab_dir / f"{sid}.nii.gz", "Tr")
    for sid in test:
        add(sid, ROOT / "images" / f"{sid}.nii", lab_dir / f"{sid}.nii.gz", "Ts")

    # synthetic cases: parent -> [case ids]
    syn_cases = {}
    if syn:
        sdir = ROOT / "synthetic" / variant
        pat = re.compile(r"^(?P<sid>.+)_syn(?P<k>\d+)\.nii(\.gz)?$")
        for p in sorted((sdir / "images").glob("*_syn*.nii*")):
            m = pat.match(p.name)
            if not m:
                continue
            sid = m["sid"]
            if sid not in folds:  # test subject, excluded subject, or unknown
                continue
            cid = p.name.split(".nii")[0]
            lab = sdir / "labels" / f"{cid}.nii.gz"
            if not lab.exists():
                print(f"warning: no label for synthetic {cid}; skipped")
                continue
            add(cid, p, lab, "Tr")
            syn_cases.setdefault(sid, []).append(cid)

    splits = []
    for f in range(N_FOLDS):
        val = sorted(s for s, ff in folds.items() if ff == f)
        tr_real = sorted(s for s, ff in folds.items() if ff != f)
        tr = tr_real + sorted(c for s in tr_real for c in syn_cases.get(s, []))
        splits.append({"train": tr, "val": val})
    # sanity: no synthetic in val, no parent leakage
    for sp in splits:
        assert not any("_syn" in v for v in sp["val"])
        vset = set(sp["val"])
        assert not any(c.split("_syn")[0] in vset for c in sp["train"])

    n_tr = len(folds) + sum(len(v) for v in syn_cases.values())
    dataset_json = {
        "channel_names": {"0": "T1"},
        "labels": {**LABELS, **({"ignore": IGNORE} if SEEN_IGNORE else {})},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "name": name,
        "description": (f"SeedSeg gold marker + prostate, design={design}, "
                        f"label variant={variant}, synthetic={syn}"),
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset_json, indent=1) + "\n")
    meta = {"dataset_id": did, "variant": variant, "design": design,
            "synthetic": syn, "n_real_train": len(folds),
            "n_synthetic": n_tr - len(folds), "n_test": len(test),
            "dropped_from_folds": dropped, "test": test}
    (raw / "seedseg_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (raw / "splits_final.json").write_text(json.dumps(splits, indent=1) + "\n")

    # repo copy + preprocessed copy (nnU-Net reads it from nnUNet_preprocessed)
    (REPO_SPLITS / f"splits_final_{name}.json").write_text(json.dumps(splits, indent=1) + "\n")
    install_splits(name)
    print(f"built {raw}: {len(folds)} real train, {n_tr - len(folds)} synthetic, "
          f"{len(test)} test; dropped {dropped}")


def install_splits(name):
    """Copy splits_final.json from the raw dataset into nnUNet_preprocessed."""
    pre = NNUNET_ROOT / "nnUNet_preprocessed" / name
    pre.mkdir(parents=True, exist_ok=True)
    shutil.copy(NNUNET_ROOT / "nnUNet_raw" / name / "splits_final.json", pre / "splits_final.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-folds")
    m.add_argument("--force", action="store_true")
    for c in ("build", "id", "name", "install-splits"):
        b = sub.add_parser(c)
        b.add_argument("--variant", required=True)
        b.add_argument("--design", required=True, choices=["cvall", "temporal"])
        b.add_argument("--synthetic", action="store_true")
        if c == "build":
            b.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.cmd == "make-folds":
        make_folds(a)
    elif a.cmd == "build":
        build(a)
    elif a.cmd == "id":
        print(dataset_id(a.variant, a.design, a.synthetic))
    elif a.cmd == "name":
        print(dataset_name(a.variant, a.design, a.synthetic))
    elif a.cmd == "install-splits":
        install_splits(dataset_name(a.variant, a.design, a.synthetic))


if __name__ == "__main__":
    main()
