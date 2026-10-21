#!/usr/bin/env python3
"""Step 1 of label correction: derive marker positions from the clinical DICOM.

For each subject archive (MR/, RS/, RE/ folders):
  * collect all RTSTRUCT ROIs whose name denotes a gold marker (GS1, GS1_01,
    Marker1, "Marker GS1", ...; helper/override structures are ignored),
  * take each ROI's centroid in the CT frame of reference (POINT ROIs: the
    point; contour ROIs: area-weighted centroid of the planar contours),
  * merge ROI centroids closer than 4 mm (duplicate copies of one marker),
  * map the unique marker positions to MR patient coordinates with the
    clinical spatial registration object (RE),
  * also record the CTV ROI used for the prostate label.

Output: a JSON per subject and a summary CSV.
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
import pydicom

MARKER_RE = re.compile(r'^(z?gs\s*\d|marker|gold\s*marker|seed\s*\d)', re.I)
EXCLUDE_RE = re.compile(r'artifact|ovrd|ovrde|override|opti|avoid|prv|ring', re.I)
CTV_PRIORITY = ['CTV_High', 'CTV_Low', 'CTVp']


def is_marker(name):
    n = name.strip()
    return bool(MARKER_RE.search(n) or re.search(r'marker\s*gs\d|gs\d\s*marker', n, re.I)) and not EXCLUDE_RE.search(n)


def roi_centroid(contour_seq):
    pts_all, areas, cents = [], [], []
    for c in contour_seq:
        p = np.array(c.ContourData, float).reshape(-1, 3)
        geom = getattr(c, 'ContourGeometricType', 'CLOSED_PLANAR')
        if geom == 'POINT' or len(p) < 3:
            pts_all.append(p)
            continue
        x, y = p[:, 0], p[:, 1]
        a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if abs(a) < 1e-6:
            pts_all.append(p)
            continue
        cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * a)
        areas.append(abs(a)); cents.append([cx, cy, p[:, 2].mean()])
    if areas:
        return np.average(np.array(cents), axis=0, weights=np.array(areas)), 'contour'
    if pts_all:
        return np.concatenate(pts_all).mean(0), 'point'
    return None, None


def load_reg(re_ds, mr_for):
    """Return 4x4 matrix M with x_ref = M @ x_mr (DICOM SRO convention), where
    ref is the SRO's registered frame (the CT/RTSTRUCT frame)."""
    for item in re_ds.RegistrationSequence:
        if item.FrameOfReferenceUID == mr_for:
            m = item.MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
            return np.array(m, float).reshape(4, 4)
    return None


def match_mr_series(d, nifti_path):
    """Pick the MR series whose volume centre matches the training NIfTI; return its FoR."""
    import nibabel as nib
    img = nib.load(nifti_path)
    ctr_ras = img.affine @ np.append((np.array(img.shape[:3]) - 1) / 2, 1)
    ctr_lps = ctr_ras[:3] * np.array([-1, -1, 1])
    series = {}
    for f in glob.glob(f'{d}/MR/*'):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        series.setdefault(ds.SeriesInstanceUID, []).append(ds)
    best = []
    for uid, dss in series.items():
        ds0 = dss[0]
        iop = np.array(ds0.ImageOrientationPatient, float); ps = np.array(ds0.PixelSpacing, float)
        cents = [np.array(x.ImagePositionPatient, float) + iop[:3] * ps[1] * (x.Columns - 1) / 2 + iop[3:] * ps[0] * (x.Rows - 1) / 2 for x in dss]
        best.append((np.linalg.norm(np.mean(cents, 0) - ctr_lps), len(dss), ds0.FrameOfReferenceUID))
    best.sort()
    note = f'{len(series)} series; centre offset {best[0][0]:.1f} mm'
    if len(best) > 1 and best[1][0] - best[0][0] < 1.0:
        note += '; AMBIGUOUS'
    return best[0][2], note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dicom-root', required=True)
    ap.add_argument('--subjects', required=True, help='text file with subject IDs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--images', required=True, help='dir with {sid}.nii training images')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for sid in [l.strip() for l in open(a.subjects) if l.strip()]:
        dd = sorted(glob.glob(os.path.join(a.dicom_root, f'*_{sid}')))
        row = dict(subject=sid, status='ok')
        if not dd:
            row['status'] = 'no_archive'; rows.append(row); continue
        d = dd[-1]
        mr_for, mr_note = match_mr_series(d, os.path.join(a.images, f'{sid}.nii'))
        row['mr_series_match'] = mr_note
        # choose RTSTRUCT with marker ROIs (latest structure set if several)
        cands = []
        for f in glob.glob(f'{d}/RS/*'):
            rs = pydicom.dcmread(f)
            names = {r.ROINumber: r.ROIName for r in rs.StructureSetROISequence}
            if any(is_marker(n) for n in names.values()):
                cands.append((str(getattr(rs, 'StructureSetDate', '')) + str(getattr(rs, 'StructureSetTime', '')), f, rs, names))
        if not cands:
            row['status'] = 'no_marker_rois'; rows.append(row); continue
        cands.sort()
        _, rs_path, rs, names = cands[-1]
        rs_for = rs.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
        reg, re_path = None, None
        for f in glob.glob(f'{d}/RE/*'):
            r = pydicom.dcmread(f)
            if any(it.FrameOfReferenceUID == rs_for for it in r.RegistrationSequence):
                reg = load_reg(r, mr_for)
                if reg is not None:
                    re_path = f; break
        if rs_for == mr_for:
            reg = np.eye(4); re_path = 'same_frame'
        if reg is None:
            # registration references another MR frame from the same session (or the
            # RTSTRUCT is on a synthetic CT): apply its non-identity matrix as is and flag
            for f in glob.glob(f'{d}/RE/*'):
                r = pydicom.dcmread(f)
                if any(it.FrameOfReferenceUID == rs_for for it in r.RegistrationSequence):
                    for it in r.RegistrationSequence:
                        if it.FrameOfReferenceUID != rs_for:
                            reg = load_reg(r, it.FrameOfReferenceUID); re_path = f
                            row['status'] = 'mr_frame_mismatch'
        if reg is None:
            row['status'] = 'no_registration'; rows.append(row); continue
        contours = {c.ReferencedROINumber: c for c in getattr(rs, 'ROIContourSequence', [])}
        rois = []
        for num, name in names.items():
            if is_marker(name) and num in contours and hasattr(contours[num], 'ContourSequence'):
                c, kind = roi_centroid(contours[num].ContourSequence)
                if c is not None:
                    rois.append(dict(name=name, kind=kind, ct_lps=c.tolist()))
        # merge duplicates (< 4 mm)
        markers = []
        for r in rois:
            p = np.array(r['ct_lps'])
            for m in markers:
                if np.linalg.norm(np.mean(m['pts'], 0) - p) < 4.0:
                    m['pts'].append(p); m['names'].append(r['name']); break
            else:
                markers.append(dict(pts=[p], names=[r['name']]))
        inv = np.linalg.inv(reg)
        out_m = []
        for m in markers:
            ct = np.mean(m['pts'], 0)
            mr = (inv @ np.append(ct, 1))[:3]
            out_m.append(dict(names=m['names'], ct_lps=ct.tolist(), mr_lps=mr.tolist(),
                              dup_spread_mm=float(max(np.linalg.norm(np.array(m['pts']) - ct, axis=1))) if len(m['pts']) > 1 else 0.0))
        ctv = next((n for want in CTV_PRIORITY for n in names.values() if n.strip().lower() == want.lower()), None)
        rec = dict(subject=sid, archive=os.path.basename(d), rtstruct=os.path.basename(rs_path),
                   registration=os.path.basename(re_path) if re_path else None, mr_for=mr_for, rs_for=rs_for,
                   reg_matrix=reg.tolist(), markers=out_m, ctv_roi=ctv,
                   marker_roi_names=sorted({n for n in names.values() if is_marker(n)}))
        json.dump(rec, open(os.path.join(a.out, f'{sid}.json'), 'w'), indent=1)
        rec['status'] = row['status']
        json.dump(rec, open(os.path.join(a.out, f'{sid}.json'), 'w'), indent=1)
        row.update(n_markers=len(out_m), n_marker_rois=len(rois), ctv_roi=ctv,
                   max_dup_spread_mm=max([m['dup_spread_mm'] for m in out_m] or [0]))
        rows.append(row)
        print(sid, row['n_markers'], rec['marker_roi_names'], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
