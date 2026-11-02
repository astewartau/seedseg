#!/usr/bin/env python3
"""Step 6: package selected synthetic cases as the public demo dataset.

Selection: cases that pass the safe gate and, on both input paths (shaded raw image with
the default N4 pipeline, and the N4-like image), have all 3 markers found with no false
positive and no extra candidate. From those, --n cases spread over the number of
distractors are taken in case order and renumbered demo-001, demo-002, ...

Per case (all derived from the renderer output; nothing patient-specific):
  demo-XXX_T1w.nii.gz       shaded T1w magnitude as from the scanner (the app applies N4)
  demo-XXX_T1w.json         acquisition sidecar (BIDS-style) marked as synthetic
  demo-XXX_labels.nii.gz    1 = gold marker (3 mm sphere at the centre), 2 = prostate
  demo-XXX_truth.json       marker centres (voxel and RAS mm), axes, susceptibility; the
                            simulated distractors (calcification / blood-product-like)
  dicom/demo-XXX/*.dcm      the same image as a DICOM MR series (one file per slice)
"""
import argparse, datetime, json, os, shutil
import numpy as np
import nibabel as nib
import pandas as pd

ACQ = dict(MagneticFieldStrength=3.0, RepetitionTime=0.6, EchoTime=0.00666, FlipAngle=80,
           SliceThickness=2.0, MRAcquisitionType="2D", ScanningSequence="GR", SequenceVariant="SP",
           PulseSequenceType="2D spoiled gradient echo (FLASH)", ProtocolName="t1_fl2d_tra",
           SequenceName="*fl2d1r", PixelBandwidth=400, ParallelReductionFactorInPlane=2,
           InPlanePhaseEncodingDirection="ROW")


def select(src, n):
    w = pd.read_csv(f"{src}/gate_works.csv")
    s = pd.read_csv(f"{src}/gate_safe.csv")
    ok = w.assign(good=(w.found == 3) & (w.fp == 0) & (w.extra_candidates == 0)).groupby("case").good.all()
    keep = s.safe & (s.plausible if "plausible" in s else True)
    ok = ok[ok].index.intersection(s[keep].case)
    ncalc = {c: len(json.load(open(f"{src}/params/{c}.json"))["calcs"]) for c in ok}
    # round-robin over distractor counts so the release shows 1, 2 and 3 distractors
    groups = {k: sorted(c for c in ok if ncalc[c] == k) for k in sorted(set(ncalc.values()))}
    out = []
    while len(out) < n and any(groups.values()):
        for k in list(groups):
            if groups[k] and len(out) < n:
                out.append(groups[k].pop(0))
    return sorted(out), len(ok)


def write_dicom(vol, aff, folder, case_id, series_uid_root):
    """One MR Image Storage file per axial slice. Voxel (i, j, k) of the NIfTI is pixel
    (row j, column i) of slice k; orientation and positions follow from the affine."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    os.makedirs(folder, exist_ok=True)
    lps = np.diag([-1, -1, 1, 1]) @ aff
    row_dir = lps[:3, 0] / np.linalg.norm(lps[:3, 0])     # along i (columns of the pixel array)
    col_dir = lps[:3, 1] / np.linalg.norm(lps[:3, 1])     # along j (rows)
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    data = np.clip(np.round(vol), 0, 32767).astype(np.uint16)
    study, series, frame = (generate_uid(prefix=None, entropy_srcs=[series_uid_root, x]) for x in ("st", "se", "for"))
    date = "20260101"
    for k in range(vol.shape[2]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid(entropy_srcs=[series_uid_root, "sop", str(k)])
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series; ds.FrameOfReferenceUID = frame
        ds.Modality = "MR"; ds.PatientName = f"Synthetic^{case_id}"; ds.PatientID = f"SEEDSEG-{case_id.upper()}"
        ds.PatientSex = "M"; ds.PatientPosition = "HFS"; ds.BodyPartExamined = "PELVIS"
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = date; ds.StudyTime = ds.SeriesTime = "120000"
        ds.StudyDescription = "SeedSeg synthetic demo"; ds.SeriesDescription = "t1_fl2d_tra_synthetic"
        ds.Manufacturer = "SeedSeg synthetic"; ds.InstitutionName = "Synthetic"
        ds.ImageType = ["DERIVED", "SECONDARY", "M"]
        ds.SeriesNumber = 1; ds.InstanceNumber = k + 1; ds.AcquisitionNumber = 1
        ds.MagneticFieldStrength = ACQ["MagneticFieldStrength"]; ds.RepetitionTime = ACQ["RepetitionTime"] * 1000
        ds.EchoTime = ACQ["EchoTime"] * 1000; ds.FlipAngle = ACQ["FlipAngle"]; ds.SliceThickness = ACQ["SliceThickness"]
        ds.SpacingBetweenSlices = float(sp[2]); ds.MRAcquisitionType = "2D"
        ds.ScanningSequence = "GR"; ds.SequenceVariant = "SP"; ds.ScanOptions = ""
        # protocol tags the SeedSeg dicompare schema checks (Manufacturer is left synthetic)
        ds.ProtocolName = ACQ["ProtocolName"]; ds.SequenceName = ACQ["SequenceName"]
        ds.PixelBandwidth = ACQ["PixelBandwidth"]; ds.NumberOfAverages = 1
        ds.ParallelReductionFactorInPlane = float(ACQ["ParallelReductionFactorInPlane"])
        ds.InPlanePhaseEncodingDirection = ACQ["InPlanePhaseEncodingDirection"]
        ds.ImageOrientationPatient = [float(x) for x in np.r_[row_dir, col_dir]]
        ds.ImagePositionPatient = [float(x) for x in (lps @ np.array([0, 0, k, 1]))[:3]]
        ds.SliceLocation = float(ds.ImagePositionPatient[2])
        ds.PixelSpacing = [float(sp[1]), float(sp[0])]   # row spacing (along j), column spacing (along i)
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[0]
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        ds.PixelData = np.ascontiguousarray(data[:, :, k].T).tobytes()
        fn = os.path.join(folder, f"{case_id}_{k + 1:03d}.dcm")
        try:
            ds.save_as(fn, enforce_file_format=True)
        except TypeError:                       # pydicom < 3
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(fn, write_like_original=False)


def truth_record(rec, aff, new_id):
    w = lambda v: [float(x) for x in nib.affines.apply_affine(aff, v)]
    return dict(
        case=new_id, synthetic=True, n_markers=len(rec["markers"]),
        markers=[dict(centre_vox=m["center_vox"], centre_ras_mm=w(m["center_vox"]), axis_vox=m["axis"],
                      angle_to_b0_deg=m["theta_deg"], delta_chi_ppm=m["dchi_ppm"],
                      length_mm=m["length_mm"], diameter_mm=2 * m["radius_mm"]) for m in rec["markers"]],
        distractors=[dict(kind="calcification-like" if c["dchi_ppm"] < 0 else "blood-product-like",
                          centre_vox=c["center_vox"], centre_ras_mm=w(c["center_vox"]), delta_chi_ppm=c["dchi_ppm"],
                          equivalent_diameter_mm=c.get("equiv_diam_mm")) for c in rec["calcs"]],
        labels={"1": "gold marker (3 mm sphere at the marker centre)", "2": "prostate"},
        acquisition=dict(ACQ, effective_coils=rec["noise"]["n_coils"]),
        generator=dict(seed=rec["seed"], render=dict(shape_modes=rec["render"]["shape_z"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="render.py output with gate_works.csv and gate_safe.csv")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--no-dicom", action="store_true")
    a = ap.parse_args()
    cases, n_pass = select(a.src, a.n)
    if os.path.exists(a.dst):
        shutil.rmtree(a.dst)
    os.makedirs(a.dst)
    index = []
    for i, c in enumerate(cases, 1):
        nid = f"demo-{i:03d}"
        img = nib.load(f"{a.src}/raw/{c}.nii.gz")
        vol = np.asarray(img.dataobj, np.float32)
        aff = img.affine
        hdr = nib.Nifti1Header(); hdr.set_data_dtype(np.int16); hdr.set_xyzt_units("mm", "sec")
        out = nib.Nifti1Image(np.clip(np.round(vol), 0, 32767).astype(np.int16), aff, hdr)
        out.set_qform(aff, 1); out.set_sform(aff, 1)
        nib.save(out, f"{a.dst}/{nid}_T1w.nii.gz")
        lab = np.asarray(nib.load(f"{a.src}/labels/{c}.nii.gz").dataobj).astype(np.uint8)
        lo = nib.Nifti1Image(lab, aff); lo.set_qform(aff, 1); lo.set_sform(aff, 1)
        nib.save(lo, f"{a.dst}/{nid}_labels.nii.gz")
        rec = json.load(open(f"{a.src}/params/{c}.json"))
        sidecar = dict(ACQ, Synthetic=True, ImageType=["DERIVED", "SECONDARY", "M"],
                       Description="Synthetic T1w FLASH (SeedSeg demo); not patient data")
        json.dump(sidecar, open(f"{a.dst}/{nid}_T1w.json", "w"), indent=1)
        json.dump(truth_record(rec, aff, nid), open(f"{a.dst}/{nid}_truth.json", "w"), indent=1)
        if not a.no_dicom:
            write_dicom(vol, aff, f"{a.dst}/dicom/{nid}", nid, f"seedseg-demo-{nid}")
        index.append(dict(case=nid, source=c, n_distractors=len(rec["calcs"])))
        print(nid, "<-", c, flush=True)
    json.dump(dict(created=datetime.date.today().isoformat(), cases=index, candidates_passing=int(n_pass)),
              open(f"{a.dst}/_index_private.json", "w"), indent=1)


if __name__ == "__main__":
    main()
