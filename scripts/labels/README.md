# Reference label correction

These scripts rebuild every reference label from the clinical DICOM, using a single
pipeline. They replace the earlier training labels, which had errors in 13 of 93
training subjects. Run them in order. Each step writes its outputs under
`<root>/labelfix/` and `<root>/labels/`.

| Step | Script | Output |
|---|---|---|
| 1 | `derive_markers.py` | per-subject marker positions (CT and MR frames), the registration used, flags |
| 2–4 | `refine_and_rasterise.py` | `labels/corrected/{sid}.nii.gz`, `labelfix/refined/{sid}.json`, `labelfix/refine_summary.csv` |
| 5 | `qc_montage.py`, `qc_flagged.py` | QC pages: all markers, then the flagged markers enlarged |
| 6 | `constellation_check.py` | `labelfix/constellation_check.csv` |
| – | `make_original_sphere.py` | control variant `labels/original_sphere/` (old positions, new sphere format) |

The QC outcome for each subject is in `labelfix/qc_decisions.csv`.

## Methods (draft)

**Marker positions from the clinical data.** In each patient's RTSTRUCT, every ROI
whose name denotes a gold marker was collected (e.g. GS1, GS1_01, Marker1, "Marker
GS1"). Helper and override structures were excluded. Each ROI gave one centroid in
the CT frame of reference: the point itself for point ROIs, or the area-weighted
centroid of the planar contours for contour ROIs. Centroids closer than 4 mm were
treated as copies of the same marker and averaged; copies agreed to within 2.1 mm
(median 0.1 mm). The RTSTRUCTs give three markers for 122 patients and two for 3.

The positions were mapped into MR patient coordinates with the clinical DICOM
spatial registration object. Where an archive held several MR series, the series
was matched to the training volume by its volume centre. In 7 patients the
registration referred to a different MR frame of reference from the same session;
its matrix was applied unchanged and the case flagged. In 2 patients, whose
contours were drawn on a synthetic CT, no registration linked the contours to the
MR, and the positions were initialised from the previous labels.

This re-derivation reproduced the previous labels within 1.5 mm in 111 of 125
patients. The other differences were errors in the previous labels:
- 12–55 mm displacements of whole marker sets;
- 9 markers missing across 7 patients, including one patient with no marker labels.

The previous prostate labels were wrong in the same patients.

**Refinement to the MR signal voids.** The clinical CT–MR registrations were not
marker-based, so marker positions were refined against the T1-weighted images.

- *Void map.* A signal-void map was computed as
  V = clip((B − I_s)/B, 0, 1), where:
  - I_s is the image smoothed with a Gaussian (σ = 0.5, 0.5, 0.25 voxels);
  - B is a 9 × 9 × 3-voxel (≈6 mm) median filter of the image, i.e. the local tissue level.
- *Rigid constellation refinement.* The three-marker constellation was rotated about
  its centroid (±6°, 3° steps per axis) and translated (±6 mm) to maximise the summed
  void strength at the marker positions, after a 1.5 mm max-filter tolerance. The
  refinement was accepted only if it raised the summed void strength by more than 0.4;
  the gains split into two groups, ≤0.24 or ≥0.43. It was accepted for 4 patients,
  with translations of 1.6–5.8 mm.
- *Per-marker snap.* Each marker was then moved to the void-weighted centroid of the
  strongest void within 2 mm, provided its strength was at least 0.45. That is about
  the 99th percentile at random prostate locations ≥8 mm from any marker (median there
  0.16), whereas real markers have a median of 0.58. This applied to 361 of 378 markers.
  The other 17 kept the constellation position and were flagged.
- *Displacement from the clinical positions.* Median 0.6 mm (IQR 0.4–0.8); 9 markers
  moved more than 2 mm.

**Quality control.**
- *Constellation shape.* Rigid motion preserves inter-marker distances, so the
  corrected MR distances were compared with the CT distances: all 125 patients with CT
  positions agreed to within 1.8 mm (median 0.7 mm).
- *Visual check.* Every flagged marker (low void strength, refined registration,
  mismatched or missing registration, ambiguous MR series, two-marker patients: 66
  markers) was reviewed visually on five contiguous slices, together with a sample of
  unflagged markers. No marker was rejected.

**Label format.** Markers were rasterised as 3-mm-radius spheres at the corrected
positions (class 1). Gold markers measure 1 × 3 mm, and the previous labels were
CT-contour blobs of about 7 mm, with a median of 232 voxels. The CTV contours were
mapped with the same clinical registration plus the accepted refinement and
rasterised on the MR grid (class 2). Where the previous CTV was correct, the
re-rasterised CTV matches it with a median Dice of 0.95.

## Caveats
- The visual QC so far was done by Claude. A human reader should confirm the 66
  flagged rows (`labelfix/qc/flagged_*.png`) before publication.
- Rotations hit the ±6° search bound in the 4 refined patients. Their constellation
  shape still matches the CT, and they were checked visually.
