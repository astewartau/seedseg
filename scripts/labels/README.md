# Reference labels

These scripts derive the reference marker positions and training labels from the clinical
DICOM data (planning structure set, spatial registration object and T1-weighted series).
Run them in order; each step writes its outputs under `<root>/labelfix/` and `<root>/labels/`.

| Step | Script | Output |
|---|---|---|
| 1 | `derive_markers.py` | per-patient marker positions (CT and MR frames), the registration used, flags |
| 2 | `refine_and_rasterise.py` | `labels/corrected/{sid}.nii.gz`, `labelfix/refined/{sid}.json`, `labelfix/refine_summary.csv` |
| 3 | `qc_montage.py`, `qc_flagged.py` | QC pages: all markers, then the flagged markers enlarged |
| 4 | `constellation_check.py` | `labelfix/constellation_check.csv` (inter-marker distances, MR vs CT) |
| 5 | `apply_qc_exclusions.py` | sets markers rejected at QC (`records/qc_exclusions.csv`) to the ignore label |
| – | `auto_ct_labels.py` | reference positions for patients without a usable registration (below) |

## Method

**Marker positions.** Every structure-set ROI whose name denotes a gold marker (e.g. GS1,
GS1_01, Marker1) gives one centroid in CT coordinates: the point itself for point ROIs, or the
area-weighted centroid of the planar contours. Centroids closer than 4 mm are treated as copies
of one marker and averaged. Positions are mapped to MR patient coordinates with the clinical
DICOM spatial registration object.

**Refinement to the MR signal voids.** Clinical CT–MR registrations are not marker-based, so
positions are refined against the T1-weighted image:

- *Void map*: V = clip((B − I_s)/B, 0, 1), where I_s is the image smoothed with a Gaussian
  (σ = 0.5, 0.5, 0.25 voxels) and B a 9 × 9 × 3-voxel median filter (the local tissue level).
- *Rigid constellation refinement*: the marker constellation is rotated about its centroid
  (±6°, 3° steps) and translated (±6 mm) to maximise the summed void strength at the marker
  positions (after a 1.5 mm maximum filter). It is accepted only if the sum increases by more
  than 0.4.
- *Per-marker snap*: each marker moves to the void-weighted centroid of the strongest void
  within 2 mm, if its strength is at least 0.45; otherwise it keeps its position and is flagged.

**Labels.** Markers are rasterised as 3-mm-radius spheres (class 1) and the CTV, mapped with the
same registration and refinement, as the prostate (class 2). Markers rejected at QC get label 3,
which nnU-Net is configured to ignore.

**Quality control.** Rigid motion preserves inter-marker distances, so the MR distances are
compared with the CT distances (`constellation_check.py`). Flagged markers (weak void, refined
constellation, atypical registration or marker count) are reviewed visually on the QC pages.

## Patients without a usable registration (`auto_ct_labels.py`)

Reference positions are derived without a trained network and without manual annotation:

1. **Source markers.** From the planning CT, gold is detected by intensity: components above
   2000 HU are split at maxima at least 5 mm apart; parts of 4–80 mm³ with a peak of at least
   2500 HU lying at least 30 mm inside the body are clustered (60 mm single linkage), and the
   cluster with the most saturated members is the marker set. Without a CT, the structure-set
   marker points are used.
2. **MR candidates.** Components of the void map with V ≥ 0.45, 2–60 mm³, within 55 mm in-plane
   of the image centre.
3. **Rigid matching.** Source triples are matched to candidate triples with the same
   inter-marker distances (within 2.5 mm) and a rigid transform is fitted (Kabsch); if no triple
   fits, pairs are matched with a rotation search about the pair axis. Implausible hypotheses
   (rotation > 13°, large shifts from the image centre) are dropped, the rest are scored by the
   summed void strength at all transformed markers, and the winner is refined as above.
4. **Acceptance.** Rotation ≤ 13°, residual ≤ 1.0 mm root mean square, and at least two markers
   on a clear void. Accepted cases are checked visually on their QC montage.

`prepare` converts the T1-weighted DICOM series to the voxel layout of the study images and
applies the same N4 correction.
