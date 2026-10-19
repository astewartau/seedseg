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

## Automatic labels for new patients without a registration (`auto_ct_labels.py`)

For patients outside the 127 whose archives have no usable CT→MR registration, reference marker
positions on the T1 are derived without any network and without manual annotation:

1. **Source markers.**
   - From the CT, where there is one: gold saturates the CT (about 3070 HU). Components of HU > 2000 are
     split at HU maxima ≥ 5 mm apart. Parts of 4–80 mm³ with a peak ≥ 2500 HU and lying ≥ 30 mm inside the
     body (this excludes skin, couch and index-bar markers) are clustered with 60 mm single linkage. The
     cluster with the most saturated members is the marker set.
   - Otherwise, the RTSTRUCT marker ROIs in whatever frame they are in.
2. **MR void blobs.** Components of the void map V ≥ 0.45, 2–60 mm³, within 55 mm in-plane of the image
   centre.
3. **Rigid matching.** Source triples are matched to blob triples whose inter-marker distances agree
   within 2.5 mm (Kabsch). If no triple fits, source pairs are matched to blob pairs with a spin search.
   - Hypotheses are dropped if they rotate more than 13°, shift the centroid left–right more than
     20 mm, or put the centroid more than 30 mm in-plane or 25 mm through-plane from the image centre.
   - The rest are scored by the summed 1.5-mm-max-filtered void strength at all transformed markers,
     then polished by snapping and refitting.
   - The winning transform places every source marker. The standard refinement (`refine()`: small rigid
     search, then per-marker void snap) gives the final position.
4. **Acceptance.** Rotation ≤ 13°, rms residual (source vs final) ≤ 1.0 mm, and ≥ 2 markers on a clear
   void (≥ 0.45). Every accepted case is then checked visually on its QC montage.

T1 preparation (`prepare`) converts the DICOM series in the voxel layout of the study images and applies
the same N4 correction. On a study subject this reproduces `images/{sid}.nii` to 1e-5 relative.

**Validation on the 60 study subjects with CT**, run blind to their labels and compared with the
corrected references:
- The rotation gate and the field-of-view centre gate were set on the first half (subjects.txt order).
- CT detection found 175 of 176 RTSTRUCT markers with no false detections. The miss peaks at 1250 HU.
- Held-out half (31 subjects): 29 accepted, 2 rejected, both of which were wrong matches.
  - No accepted subject had any marker more than 5 mm from the reference.
  - Of the 86 reference markers in accepted subjects, 83.7% were within 2 mm, 98.8% within 3 mm and
    100% within 5 mm.
  - The 2–3 mm differences are mostly where the void snap lands along the same elongated void.
- Caveats:
  - The CT detection limits and the 20 mm left–right gate were chosen looking at all study subjects'
    CTs and clinical registrations, not only the tuning half.
  - For the one held-out subject with a QC-excluded marker (no MR void at the CT position), the
    pipeline still places that marker.

Outputs for new patients go in `prostate-v2/fresh/`:
- `images/{sid}.nii`;
- `labels/{sid}.json` (`markers_final_vox` on that grid, plus fit statistics) and `{sid}_qc.png`;
- `labels/summary.csv`, `subjects_fresh.txt`.

There are no prostate (CTV) labels for these patients.
