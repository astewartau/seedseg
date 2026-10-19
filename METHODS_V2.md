# Methods (SeedSeg v2, draft)

Draft methods for the revision of the preprint (bioRxiv 10.64898/2026.06.18.733061).
Items marked **[pending]** were still running on 2026-10-19.

## Data and cohort

The study used the cohort of the preprint: 127 patients undergoing prostate
radiotherapy with implanted gold fiducial markers, each with a T1-weighted MRI and
a clinical radiotherapy structure set. Images were acquired on a 3T scanner
(Siemens Skyra) with an axial 2D FLASH sequence (TE 6.66 ms, flip angle 80°,
0.6875 × 0.6875 × 2.0 mm voxels, matrix 320 × 320 × 40; one examination had 42
slices). The 61 examinations from 2020–2021 were acquired with software version
E11 and TR 689 ms, and the 66 from 2025–2026, including all 25 test patients,
with software version XA60 and TR 600 ms; geometry and the other parameters were
unchanged. Because the exported headers were de-identified, the protocol
parameters were taken from series exported directly from the scanner in each
period. Multi-channel magnitude images were used. Each image was corrected for
intensity inhomogeneity with N4 (SimpleITK `N4BiasFieldCorrection`, default
parameters, mask from SimpleITK `OtsuThreshold` with 200 histogram bins), as in
the preprint, and used without further preprocessing. All 127 images were the
preprint's `*_MRI_homogeneity-corrected.nii` files, produced with this same
correction function (preprint scripts `homogeneity_correction.py` and
`prepare_test_data.py`, and a one-off script applying the same function to two
training patients). One test image was an online reconstruction by a
vendor-platform application rather than the standard scanner reconstruction, with
an intensity scale about 2.6 times that of the other test images. It was kept
as acquired; the network input was z-score normalised per image.

The 127 patients comprise the preprint's 93 training, 9 validation and 25 test
patients. The 25 test patients were acquired after the other 102 (temporal
split). Of the patients in the original training collection, two lacked the
T1-weighted marker sequence and one had a registration linking the CT to a
different MR frame of reference than the T1-weighted series, so its labels could
not be derived; all three were excluded, as in the preprint. Two further patients
listed in the preprint's training split had no marker contours in their
structure sets (one had only a CTV, the other only a body contour) and were
excluded, which leaves the preprint's 93 training patients. Other patients with
a T1-weighted examination who were scanned in 2025 were not included because they
had no usable reference labels: no structure set or registration on the
T1-weighted frame of reference, or no marker contours.

The primary analysis was 5-fold cross-validation over all 127 patients. Fold
membership was assigned once, by shuffling the sorted patient list with a fixed
seed (0) and assigning patients to folds in turn (folds of 26, 26, 25, 25 and 25
patients). The same assignment was used for every label variant and every arm. A
second design reproduced the preprint's temporal split: 5-fold cross-validation
over the 102 earlier patients (folds of 20–21) and a test set of the 25 later
patients that was never used for training or model selection.

## Reference labels

All reference labels were rebuilt from the clinical DICOM data with one pipeline
(`scripts/labels/`), replacing the labels used in the preprint.

**Marker positions from the structure sets.** In each patient's RTSTRUCT, every
ROI whose name denoted a gold marker (for example GS1, GS1_01, Marker1, "Marker
GS1") was collected; helper structures (names containing artifact, override,
opti, avoid, PRV or ring) were excluded. Where several structure sets held marker
ROIs, the latest was used. Each ROI gave one centroid in the CT frame of
reference: the point itself for point ROIs, and the area-weighted centroid of
the planar contours for contour ROIs. Centroids closer than 4 mm were treated as
copies of the same marker and averaged; copies agreed to within 2.1 mm (median
0.1 mm). The structure sets gave three markers for 122 patients and two for 3.

**CT to MR mapping.** Positions were mapped into MR patient coordinates with the
clinical DICOM spatial registration object (x_MR = M⁻¹ x_CT). Where an archive
held several MR series, the series whose volume centre matched the training
image was used. In 7 patients the registration referred to a different MR frame
of reference from the same session; its matrix was applied unchanged and the
case was flagged. In 2 patients, whose contours had been drawn on a synthetic CT,
no registration linked the structure set to the MR; their marker positions were
initialised from the centroids of the previous label components, and their
prostate labels were taken from the previous labels.

The re-derivation reproduced the previous labels within 1.5 mm in 111 of 125
patients. The remaining differences were errors in the previous labels:
displacements of whole marker sets by 12–55 mm, and 9 markers missing across 7
patients (one of whom had no marker label at all). The previous prostate labels
were wrong in the same patients.

**Refinement to the MR signal voids.** The clinical CT–MR registrations were not
marker-based, so marker positions were refined against the T1-weighted image.

- *Void map.* V = clip((B − I_s)/B, 0, 1), where I_s is the image smoothed with a
  Gaussian of σ = (0.5, 0.5, 0.25) voxels and B is a 9 × 9 × 3-voxel median
  filter of the image (about 6 mm, the local tissue level). For the
  constellation search, V was max-filtered over a 1.5 mm sphere (V_d) to tolerate
  small deformations between CT and MR.
- *Rigid constellation refinement.* The patient's marker constellation was
  rotated about its centroid (−6° to +6° in 3° steps about each axis) and
  translated (±6 mm; steps of one in-plane voxel in-plane and half a slice
  through-plane), maximising the summed V_d at the marker positions. The refined
  transform was accepted only if it raised the sum by more than 0.4; otherwise
  the clinical registration was kept. The observed gains fell in two groups
  (≤ 0.24 and ≥ 0.43). The refinement was accepted in 4 patients, with
  translations of 1.6–5.8 mm.
- *Per-marker snap.* Each marker was then moved to the V-weighted centroid of the
  strongest void within 2 mm (voxels above half the peak value, connected to the
  peak), provided that the peak V was at least 0.45. This threshold is about the
  99th percentile of V at random prostate locations at least 8 mm from any marker
  (median 0.16 there), whereas real markers have a median of 0.58. The snap
  applied to 361 of 378 markers; the other 17 kept their constellation position
  and were flagged as low-void.
- *Displacement from the clinical positions.* Median 0.6 mm (IQR 0.4–0.8 mm);
  9 markers moved by more than 2 mm.

**Rasterisation.** Markers were rasterised on the MR grid as spheres of 3 mm
radius (in mm, hence ellipsoidal in voxels) at the final positions (class 1).
Gold markers measure 1 × 3 mm; the previous labels were CT-contour blobs of about
7 mm (median 232 voxels). The prostate label (class 2) was the CTV (ROI priority
CTV_High, then CTV_Low, then CTVp; used in 122, 1 and 2 patients), mapped with
the clinical registration followed by any accepted rigid refinement. Contours
were filled on a 0.3 mm in-plane grid, spread over the contour slice thickness in
0.5 mm steps, mapped to MR voxels, closed with a 3 × 3 × 1 structuring element and
hole-filled slice by slice. Marker voxels overwrite prostate voxels. Where the
previous CTV was correct, the new CTV agreed with it with a median Dice of 0.95.

**Quality control.**

- *Constellation shape.* Rigid motion preserves inter-marker distances, so for
  each patient the pairwise distances of the final MR positions were compared
  with those of the CT positions (flag if any difference exceeded 3 mm). All 125
  patients with CT positions agreed to within 1.8 mm (median of the per-patient
  maximum 0.7 mm). This check validates the snapped markers only; markers that
  kept their constellation position preserve the CT distances by construction
  and were covered by the visual check.
- *Visual check.* All markers were displayed on three contiguous slices (28 mm
  field of view) with the new and previous labels overlaid. Every flagged marker
  (low void strength, accepted refinement, mismatched or missing registration,
  ambiguous MR series, two-marker patients; 66 markers) was reviewed on five
  contiguous slices (42 mm field of view) by one author (A.S.). One marker was
  rejected: there was no MR void at its CT-confirmed position (V = 0.44).
  Unflagged markers were not reviewed visually.

**Ignore label.** The rejected marker was given label 3 over its 3 mm sphere, in
the corrected label and in the synthetic cases derived from that patient
(`apply_qc_exclusions.py`). nnU-Net was configured with 3 as its ignore label,
so these voxels did not contribute to the loss. In evaluation the marker was
removed from the reference, and any detection within 5 mm of it was discarded
before matching, so it counted neither as a true nor as a false positive. The
reference therefore contained 377 markers (74 in the 25 test patients).

## Independent external test set with automatic reference labels

To test the final models on patients that played no part in any design decision, we
assembled an external set from patients outside the 127 who had the same T1 marker
sequence (standard reconstruction; vendor-application reconstructions were never
used) and either a planning CT or a marker structure set, but no usable CT→MR
registration. Their reference labels were derived automatically, without any
trained network and without manual annotation.

**Source markers.** Where a CT was available, gold markers were detected by
intensity: gold saturates the CT (about 3070 HU). Components above 2000 HU were
split at intensity maxima at least 5 mm apart. Parts of 4–80 mm³ with a peak of at
least 2500 HU, lying at least 30 mm inside the body contour (this excludes skin,
couch and index-bar markers), were clustered with 60 mm single linkage, and the
cluster with the most saturated members was taken as the marker set. Without a CT,
the marker points of the planning structure set were used.

**Registration to the MR image.** Candidate voids were the components of the
void map (the same hand-crafted map used to refine the study labels; V ≥ 0.45,
2–60 mm³, within 55 mm in-plane of the image centre). Because inter-marker distances
are invariant to rigid motion, triples of source markers were matched to triples of
MR voids whose pairwise distances agreed within 2.5 mm, and a rigid transform was
fitted to each match (Kabsch). If no triple fitted, pairs were matched with a search
over rotation about the pair axis. Hypotheses were discarded if they rotated by more
than 13°, shifted the marker centroid by more than 20 mm left–right, or placed it more
than 30 mm in-plane or 25 mm through-plane from the image centre. The remaining
hypotheses were scored by the summed void strength (after a 1.5 mm maximum filter)
at all transformed markers. The best transform placed every source marker, so a
marker with a weak void was still placed. The same refinement as for the study
labels (a small rigid search, then a per-marker snap to the void) gave the final
positions. A case was accepted if the rotation was at most 13°, the rms residual
between source and final positions was at most 1.0 mm, and at least two markers lay
on a clear void (V ≥ 0.45). Every accepted case was then checked visually on a QC
montage.

**Validation of the automatic labels.** The pipeline was run blind on the 60 study
patients that had a CT, and compared with their corrected reference labels. The
rotation limit and the image-centre limits were set on the first half (29 patients).
Results are reported on the second half (31 patients).
- CT detection found 175 of 176 markers, with no false detections. The missed marker
  peaked at 1250 HU.
- In the held-out half, 29 patients were accepted and 2 were rejected; both rejected
  cases were wrong matches.
- In the accepted patients, 83.7% of the 86 reference markers were within 2 mm of
  the automatic position, 98.8% within 3 mm and 100% within 5 mm, the matching
  tolerance used for evaluation.
- The CT detection limits and the 20 mm left–right limit were chosen while looking at
  all 60 patients, not only the tuning half.

The T1 images of the external patients were converted and N4-corrected in the same
way as the study images. On a study patient, this reproduced the study image to a
relative difference of 1e-5.

**External set.** Nine patients were accepted, each with 3 markers: 4 acquired in
2021–2023 (scanner software E11, TR 689 ms) and 5 in 2025 (XA60, TR 600 ms). For two
of them, clinical marker positions defined directly on the T1 were available. The
automatic labels agreed with them within 1.1 mm and 2.0 mm. One patient had a strong
void-like structure that every model detected. Its automatic label was confirmed
because the CT inter-marker distances matched the labelled triple within 0.8 mm, and
every alternative triple including the structure within no better than 3.9 mm. No
prostate contours were derived for this set, and none were needed for marker
evaluation.

**Inference on the external set.** Each fold model of an arm was applied to the
external images separately (nnU-Net, CPU, the same settings and test-time mirroring
as for the out-of-fold predictions). The 5-fold ensemble was the voxel-wise mean of
the five fold models' probabilities. Single fold models are reported individually,
because the deployed model is a single network. The released ensemble was applied
as for the study patients. For constellation selection on this set, the geometric
prior was fitted on the reference markers of all 127 study patients, λ was fixed at
0.1 (the value chosen by inner cross-validation in 9 of the 10 outer folds of 511
and 513), and K was the number of reference markers. The external set was analysed
after the study-set analyses of 501, 511, 512 and 513 were complete. It was then used,
together with the study-set results, to choose the deployed configuration. Because
the external set informed that choice, its results for the chosen configuration are
not a fully blind estimate. [pending: the 250-epoch and 521 models are applied to it
when their training finishes.]

## Label variants

Three label variants were built on the same images, each with classes 0
(background), 1 (marker) and 2 (prostate):

| Variant | Marker label | Prostate label | Controls |
|---|---|---|---|
| `original` | Previous labels: CT-contour blobs at the previous positions (including the errors above) | Previous CTV | Reproduces the training labels of the preprint |
| `corrected` | 3 mm spheres at the refined MR positions; one QC-rejected marker set to ignore | Re-rasterised CTV | The new reference standard; also used for all evaluation |
| `original_sphere` | 3 mm spheres at the centroids of the previous marker components (26-connectivity) | Previous CTV plus the previous marker voxels | Same positions as `original`, same format as `corrected`: separates the effect of label shape from that of label position |

All arms were evaluated against the `corrected` reference positions, whatever
labels they were trained on. In `original_sphere`, the one patient without
previous marker labels has no marker voxels.

## Physics-based synthetic markers and distractors

Synthetic training cases were made by inserting simulated gold markers (labelled)
and calcification- or blood-product-like susceptibility sources (unlabelled hard
negatives) into the real N4-corrected images (`scripts/synth/`).

**Sources.** A gold marker was modelled as a solid cylinder of 1.0 mm diameter
and 3.0 mm length. Distractors were unions of 1–3 randomly rotated ellipsoids
with an equivalent diameter of 0.5–3 mm (semi-axes scaled by U(0.6, 1.4) per
axis; secondary ellipsoids scaled by U(0.4, 0.8) and offset by up to half the
diameter). Each source was rasterised as a fractional occupancy map o(r) on a
fine grid that oversampled each MR voxel 7 × 7 in-plane and 14 times
through-plane (0.098 × 0.098 × 0.143 mm), with 3³ supersampling per fine cell for
partial volume. The susceptibility difference to tissue was Δχ(r) = Δχ·o(r).

**Field.** The relative field perturbation was computed with the dipole kernel in
k-space on a grid zero-padded by a factor of 1.5 in each direction,
ΔB/B₀ = F⁻¹{[1/3 − (k·b̂)²/|k|²]·F{Δχ}}, with the B₀ direction b̂ taken from the
image affine (the slice axis). The off-resonance frequency was Δf = γ̄B₀·ΔB/B₀
(γ̄ = 42.577 MHz/T, B₀ = 3 T).

**Signal.** Each fine cell carried transverse magnetisation
m(r) = ρ(r)·exp(−i2πΔf(r)·TE), with ρ = 1 − o + o·ρ_src; gold has ρ_src = 0. The
complex signal of slice s at in-plane position x was
S_s(x) = Σ_z p(z + δz(r) − z_s)·m(r), where p is a 2.0 mm boxcar slice profile and
δz = Δf/G_ss is the off-resonance displacement of the excited slice
(G_ss = RF bandwidth/slice thickness; RF bandwidth 1300 Hz, positive polarity).
The sum over fine z captures through-slice signal cancellation, which dominates
for 2 mm slices. Readout displacement was not modelled. The in-plane point-spread
function was the boxcar average over the MR pixel. The complex voxel attenuation
was A = S/S₀, with S₀ the same computation without the source. Each source was
simulated in its own 25 × 25 × 9-voxel patch (17 × 17 × 18 mm), and the
attenuations of different sources were multiplied, which is exact when their
perturbations do not overlap; the placement rules below ensure this.

**Noise model and insertion.** Signal-free regions such as rectal gas sit at
3.6–5 times the tissue noise SD, rather than at the 1.25 times of a Rician floor,
consistent with a sum-of-squares multi-channel reconstruction (non-central χ
with 2N degrees of freedom). In the 21 purest gas regions the effective channel
count N_eff was about 6.7–15 (median about 12). N was therefore drawn per case
as a uniform integer in [6, 14]. The synthetic magnitude was formed in the power
domain,

M′² = |A|²·M² + (1 − |A|²)₊·σ²·Σ_{c=1..N} |n_c|²,  n_c ~ CN(0, 1),

with M the original magnitude and σ the per-channel noise SD, estimated per
patient from the median absolute deviation of an in-plane Laplacian residual
within the dilated CTV (3 dilation iterations). The noise was spatially
correlated with lag-1 coefficients of 0.38 and 0.22 along the two in-plane axes,
as measured in the purest gas regions. The image is unchanged where A = 1, and the
void floor follows the scanner's noise-floor distribution. Output images kept the
source's int16 storage, so quantisation gave no cue to synthetic regions.

**Placement and parameters.** Each case received 1–3 synthetic markers (uniform
integer). Marker centres (continuous, sub-voxel) were at least 1.5 mm inside the
prostate (classes 1 ∪ 2), at least 10 mm from any real-marker label voxel and at
least 8 mm from each other. Each marker was assigned a placement context with
probabilities 0.5/0.25/0.25: *random* (anywhere admissible), *boundary* (1.5–4 mm
from the gland edge) or *bright* (within 4 mm of a voxel brighter than 1.6 times
the CTV median, within 6 voxels of the gland, i.e. near T1-bright blood
products); if a context had no admissible location, placement fell back to
random. The polar angle of the marker axis to B₀ was drawn from the fitted
orientation prior (below, on a 10° grid) plus U(−5°, 5°), with uniform azimuth;
Δχ was drawn from U(−35, −15) ppm around the nominal −25 ppm (χ_Au − χ_tissue =
−34 − (−9) ppm). Each case received 0–4 distractors inside the prostate, at least
6 mm from synthetic marker centres and at least 3 mm from real-marker label
voxels. Of these, 75% were diamagnetic (Δχ log-uniform in [−5, −1] ppm,
ρ_src ∈ U(0, 0.3), calcification-like) and 25% paramagnetic (Δχ log-uniform in
[+1, +10] ppm, ρ_src ∈ U(0.3, 1), blood-product-like). Distractors kept the label
of the voxels they replaced.

**Labels.** The patient's `corrected` label was kept unchanged, and each
synthetic marker was added as a 3 mm-radius sphere (class 1) at its centre.

**Generated set.** Two cases were generated per patient (seed 0; deterministic
in seed, patient and case index), giving 254 cases with 1–3 synthetic markers
(75, 92 and 87 cases with 1, 2 and 3) and 0–4 distractors. Every case's
parameters were recorded in a JSON file.

**Validation.** Real and simulated voids were located (smoothed minimum within a
3.5 mm sphere, then deficit-weighted centroid) and described with the same code:
per-slice radial profiles, per-slice minimum intensity and void area (below
0.6 times the CTV median), void volume, in-plane elongation and number of slices
with a void. The real set was the 361 markers that had been snapped to a void
(V ≥ 0.45). To fit the orientation and Δχ priors, a library of 360 simulations
(polar angle 0–90° in 10° steps × Δχ ∈ {−15, −25, −35} ppm × 12 random marker-free
prostate locations) was matched to each real void by nearest neighbours in its
rotation-invariant descriptor, giving a per-void posterior over (θ, Δχ); the
pooled posterior defined the generation prior. A fresh set of 361 simulations
drawn from the prior was then compared with the real set by two-sample
Kolmogorov–Smirnov tests. The orientation prior used for generation came from
the fit with a fixed coil count N = 10 and was used unchanged when N was drawn
from [6, 14] for generation; the comparison below uses N ∈ [6, 14]. Medians [IQR], real vs simulated:
minimum intensity relative to the CTV median 0.34 [0.29–0.39] vs 0.33
[0.29–0.38] (KS p = 0.08); void volume 41.6 [33–55] vs 43.5 [29–57] mm³
(p = 0.02); in-plane elongation 1.19 [1.11–1.30] vs 1.18 [1.12–1.29] (p = 0.64);
slices with void 4 [3–4] vs 3 [3–4] (p = 0.06). The RMS difference of the median
radial profiles was 0.033 (in CTV-median units). Δχ was not identifiable over
−15 to −35 ppm (pooled posterior 0.30/0.37/0.34), which motivated the uniform Δχ
prior. Off-resonance slice displacement reproduced the cranio-caudal asymmetry
of the real voids; the RF bandwidth is not stored in the anonymised DICOM, so it
was chosen as plausible for the sequence rather than fitted. Known mismatches
are a per-patient scalar σ, voids about one slice shorter than real ones in z
(attributed to the idealised boxcar slice profile), and homogeneous simulated
backgrounds. Numerically, the dipole field of a sphere matched the analytic
solution to within 0.012 ppm (1.5 Hz) outside 1.5 radii, and the default grid
gave attenuation errors with a 99th percentile of at most 0.06 inside the void
relative to 9 × 9 × 20 oversampling.

## Network and training

All arms used nnU-Net v2 (nnunetv2 2.8.1, PyTorch 2.6.0 with CUDA 12.4,
Python 3.11) with the residual-encoder planner (`nnUNetPlannerResEncM`, ResEnc-M
preset) and the `3d_fullres` configuration. The plans were identical for every
dataset: target spacing 2.0 × 0.6875 × 0.6875 mm (z, y, x; the native spacing),
median image size 40 × 317 × 316 voxels after cropping, patch size 32 × 256 × 224
voxels, batch size 2, per-image z-score normalisation (no 3d_lowres
configuration was created). The network was a ResidualEncoderUNet with 6 stages
(32, 64, 128, 256, 320 and 320 features; 1, 3, 4, 6, 6 and 6 residual blocks;
first-stage kernel 1 × 3 × 3, all others 3 × 3 × 3; strides 1 × 1 × 1, 1 × 2 × 2,
2 × 2 × 2 three times, then 1 × 2 × 2), instance normalisation and leaky ReLU,
with one convolution per decoder stage. The output had three classes (background,
marker, prostate).

Training used the default `nnUNetTrainer`: 1000 epochs of 250 iterations, SGD with
Nesterov momentum 0.99, weight decay 3 × 10⁻⁵, initial learning rate 0.01 with
polynomial decay, Dice plus cross-entropy loss with deep supervision, 33%
foreground oversampling and the default nnU-Net data augmentation. The model
after the last epoch (`checkpoint_final`) was used for all predictions. No
checkpoint was selected on validation data, because in the cross-validation
design the validation fold is also the evaluation fold.

For the repeat-run analysis, 511 was additionally trained three times with
`nnUNetTrainer_250epochs` (identical except for 250 epochs, so the learning rate
decays over 250 epochs), in separate results directories (r1–r3), on the same
dataset and folds. nnU-Net 2.8.1 sets no random seed (and runs cuDNN in
non-deterministic mode), and none was set by us, so the repeats differ in
initialisation, data sampling and augmentation. **[pending:
running as of 2026-10-19.]**

Each fold was trained on one NVIDIA H100 GPU (12 CPU cores, 96 GB RAM, 10 data
augmentation workers) on the UQ Bunya cluster. A 1000-epoch fold took about
27–29 s per epoch, about 8 h in total (7 h 49 min to 8 h 27 min).

## Experimental arms

Dataset IDs follow 500 + 10 × variant (original 0, corrected 1, original_sphere
2) + design (cross-validation over all 127 = 1, temporal = 2, cross-validation
with synthetic cases = 3).

| Arm | Labels | Training data and design | Question |
|---|---|---|---|
| Released | Previous labels | The preprint's 4-model ensemble (custom 3D U-Net, 93 training patients); probabilities averaged over the 4 models | Baseline: the published method under the new evaluation |
| 501 | `original` | 5-fold CV, all 127 | nnU-Net with the old labels (effect of the training framework) |
| 511 | `corrected` | 5-fold CV, all 127 | Main model; 511 vs 501 = effect of the label correction |
| 512 | `corrected` | 5-fold CV over the 102 earlier patients; 25 later patients predicted by the 5-fold ensemble | Comparison with the preprint on its temporal test set; ensemble vs single model (512 ensemble vs 511 OOF on the same 25) |
| 513 | `corrected` + 254 synthetic cases | 5-fold CV, all 127; synthetic cases only in the training lists of folds where their source patient is a training patient | Effect of the synthetic markers and distractors (513 vs 511) |
| 521 | `original_sphere` | 5-fold CV, all 127 | Shape vs position: 501 → 521 changes only the label shape; 521 → 511 changes position and CTV |
| 511 r1–r3 | `corrected` | As 511, 250 epochs, three independent runs | Effect of a shorter schedule, run-to-run variability, and whether averaging repeat runs of one model helps |
| 511 r-avg | `corrected` | Voxel-wise mean of the r1–r3 probability maps (marker and all three classes) | As above; the average is still out-of-fold because the runs share folds |

In 513, each training fold contained 101–102 real and 202–204 synthetic cases;
synthetic cases were never in a validation fold, so no patient contributed to
both the training and the evaluation of a fold model.

**521 [pending].** The first 521 run was discarded: a label-generation bug left
its labels without any marker voxels. The labels were regenerated with the
corrected script (126 of 127 patients have marker voxels) and the arm was
retrained with the same design; this run was in progress as of 2026-10-19.

**250-epoch repeats [pending]**, as above.

## Inference and post-processing

**Out-of-fold probabilities.** For the cross-validation arms, every patient was
predicted only by the fold model that had not seen it in training: the
probabilities from nnU-Net's final validation of each fold (`nnUNetv2_train
--npz`; sliding-window inference with tile step 0.5, Gaussian weighting and
mirroring), exported as NIfTI on the original image grid. For 512 the 25 test
patients were predicted with `nnUNetv2_predict` using all 5 fold models
(averaged softmax, default settings). Affine and orientation of the exported
maps were spot-checked against the images.

**Candidate extraction (default selection).** Candidates were the connected
components (26-connectivity) of the marker-class probability above 0.1. Each
candidate's position was its probability-weighted centroid and its score its
mean probability. The 3 highest-scoring candidates were kept per patient (as in
the preprint), regardless of the patient's marker count.

**Constellation selection.** As an alternative to top-3 selection, the set S of
K candidates was chosen that maximised

Σ_{i∈S} log s_i + λ [Σ_{i∈S} log p_pos(x_i) + Σ_{i∈S} log p_surf(d_i) + Σ_{i<j∈S} log p_pair(‖x_i − x_j‖)],

where s_i is the mean probability, and the geometric prior comprises Gaussian
kernel density estimates (Scott's bandwidth) of (i) marker position in
prostate-normalised coordinates (centroid and principal axes of the predicted
prostate, scaled by their SDs, axes oriented consistently), (ii) signed distance
to the predicted prostate surface, and (iii) pairwise inter-marker distance. The
prostate was the largest connected component of the same out-of-fold prediction
(class 2 probability > 0.5), so no reference information from the evaluated
patient was used. The KDEs were fitted on the reference markers of the training
folds only. The search was exhaustive over the 12 candidates with the best
unary score. K was the number of implanted markers recorded in the clinical
structure set, standing in for the implant record; for the two patients without
a structure-set record, K was the number of markers in their previous labels
(three in both).
λ was chosen per outer fold from {0, 0.1, 0.25, 0.5, 1, 2, 4} by inner
cross-validation over the other four folds (objective: true positives minus
false positives, with the prior refitted for each inner split); the outer
fold's prior was then fitted on all four training folds.

**Count flag.** When fewer than K candidates were available, so that fewer than
K markers were output, the patient was flagged for review.

**Extra low-threshold candidates (exploratory).** In a variant of constellation
selection, components of p > 10⁻³ or p > 10⁻⁴ that contained no voxel above 0.1
were added as candidates.

**Learned re-ranker (exploratory).** A gradient-boosted tree classifier
(scikit-learn `HistGradientBoostingClassifier`; 300 iterations, learning rate
0.05, 15 leaf nodes, L2 regularisation 1.0, balanced class weights) was trained
to predict whether a candidate lies within 5 mm of a reference marker, from
network outputs (mean and maximum probability, mass, log volume, rank, number of
candidates), void-map features around the centroid (V at the centre and in
shells of 0–1, 1–2, 2–3 and 3–5 mm; void extent in-plane and through-plane;
minimum smoothed intensity relative to the local tissue level within 3 mm; the
bright fraction, below) and the distance to the nearest other candidate. For
each outer fold, it was trained on the out-of-fold candidates of the other four
folds, and the top 3 candidates by its output were kept. It was evaluated on 511
only.

## Evaluation

**Matching.** Kept detections were matched one-to-one to the reference marker
positions (`corrected`, refined MR positions) by the Hungarian algorithm on
centroid distance, with a 5 mm tolerance. Matched pairs were true positives,
unmatched reference markers false negatives and unmatched detections false
positives. Sensitivity was TP/(TP + FN) and precision TP/(TP + FP), pooled over
markers. A patient was *perfect* if it had no false negative and no false
positive. The localisation error was the mean distance of matched pairs per
patient. The same evaluator was applied to every arm, including the released
ensemble. Its marker probability map was the mean of the four released ONNX
models' marker softmax outputs (ONNX Runtime 1.23.2, CPU), computed on the same
N4-corrected images with the preprint's inference preprocessing: zero-padding to
a multiple of 32 voxels per axis, z-score normalisation over the padded volume,
and cropping back (`scripts/evaluation/infer_released_onnx.py`). This reproduced
the preprint's test results exactly.

**Populations.** The primary population was all 127 patients with out-of-fold
predictions. Results were also reported for the 25 temporal test patients and
per fold. For the cross-validation arms, the test-25 results are a subset of the
out-of-fold predictions, not a temporal test (their fold models also trained on
later patients); 512 provides the temporal test. For the released ensemble, the
all-127 results are partly in-sample (its models were trained on 93 of these
patients), which biases comparisons in its favour.

**Confidence intervals.** 95% CIs for sensitivity and precision were exact
Clopper–Pearson intervals. Paired comparisons of two arms on the same patients
used per-marker outcomes: 95% CIs for the differences in sensitivity and
precision came from a patient-level bootstrap (5000 resamples of patients with
replacement, fixed seed), and the per-marker discordance was tested with an
exact McNemar test (two-sided binomial test on the markers found by only one of
the two arms; SciPy `binomtest`, `scripts/evaluation/compare_arms.py`).

**Miss categories.** Each missed marker was classified as *no candidate* (no
candidate centroid within 5 mm at the 0.1 threshold) or *ranked out* (a
candidate within 5 mm existed but was not among the kept detections or was not
matched to it). Candidate recall, the upper bound for any selection rule at
this threshold, was the fraction of reference markers with a candidate within
5 mm, each marker counted once.

**Bright fraction.** To examine markers near T1-bright blood products, each
reference marker was assigned a bright fraction: the proportion of voxels within
6 mm of its position whose intensity exceeded 1.6 times the local tissue level
B (the 9 × 9 × 3-voxel median filter of the image, as in the void map). Markers in the top 10%
(bright fraction > 0.0085, n = 37) and top 5% (n = 19) were compared across arms.

## Statistics and pre-specification

Pre-specified before any nnU-Net results were seen: the arms 501, 511, 512, 513
and 521 and their design (fold assignment, plans, 1000 epochs,
`checkpoint_final`); the evaluation (5 mm one-to-one centroid matching, threshold
0.1, top 3 by mean probability, Clopper–Pearson CIs); the primary population
(all 127, out-of-fold); and the main contrasts 511 vs 501 (label correction),
511 vs 521 (label position vs shape), 513 vs 511 (synthetic data) and single
model vs 5-fold ensemble (512).

Post hoc, after the 501 and 511 results were seen:

- Constellation selection. It was developed on the 511 out-of-fold output. Its
  settings (λ grid, inner-CV choice of λ, K = implant count, threshold 0.1) were
  fixed before it was applied to 513 and the other arms, and these settings are
  the ones referred to as pre-specified in the results. The λ grid and the
  inner-CV objective were written before the method was first run on 511 and were
  not changed afterwards. The λ = 0 and fixed K = 3 runs on 511 were sensitivity
  analyses.
- The extra low-threshold candidates (10⁻³, 10⁻⁴). These thresholds were chosen
  after looking at the misses, not blind, and results with them are optimistic.
- The 250-epoch schedule and the repeat-run average, motivated by the training
  curves (validation marker Dice peaking at about epochs 120–250).
- The learned re-ranker was planned as a second stage, but its feature set was
  chosen with knowledge of the remaining errors of the earlier models.
- The bright-fraction analysis and its 10%/5% cut points; the 1.6 threshold was
  taken from the re-ranker's feature.
- Comparisons on the 25 test patients for the cross-validation arms, and
  comparisons with averaged probability maps of different arms.
- Miss categories are descriptive.

No correction for multiple comparisons was applied. The pre-specified primary
contrasts were on all 127 patients: 511 vs 501 (label correction), 513 vs 511
(synthetic data) and 521 vs 501 and 511 (label shape vs position); 512 vs the
released ensemble on the 25 temporal test patients was the pre-specified
comparison with the preprint. All other paired tests, including contrasts on the
25 test patients for the cross-validation arms, are exploratory.

## Software and code

Label derivation, evaluation and post-processing used Python with pydicom,
nibabel, SciPy, pandas and scikit-learn (Python 3.10.4, NumPy 1.24.3, SciPy
1.10.1, nibabel 5.2.1, pandas 2.0.3, scikit-learn 1.7.2, pydicom 2.4.4, SimpleITK
2.5.6). The synthetic-marker generator used Python 3.9.12 with NumPy 2.0.2, SciPy
1.13.1, nibabel 5.3.3 and pydicom 2.4.4. nnU-Net 2.8.1 ran with PyTorch 2.6.0
(CUDA 12.4) on Python 3.11.9. Code: `scripts/labels/`,
`scripts/synth/`, `scripts/nnunet/`, `scripts/evaluation/` and `scripts/rerank/`
in the SeedSeg repository.
