# SeedSeg v2: gold fiducial marker detection in prostate T1 MRI

This folder holds everything from the v2 revision of SeedSeg (preprint bioRxiv 10.64898/2026.06.18.733061):
data, reference labels, trained models, predictions, evaluation results, code and documentation.
It is kept in sync from the working copy on Bunya scratch (`/scratch/user/uqaste15/prostate-v2/`) by
`scripts/backup_v2_to_rdm.sh` (in the code bundle). That sync is incremental and never deletes anything here.

**Start with `docs/V2_PLAN.md`** (design, decisions, all results, status) and `docs/METHODS_V2.md` (methods draft).

Contains patient study IDs (`z…`): do not share outside the project.

## Layout

| Folder | Contents |
|---|---|
| `docs/` | `V2_PLAN.md` (plan, results, decisions), `METHODS_V2.md` (paper methods draft), READMEs for the label pipeline, the synthetic-marker simulator and the nnU-Net setup |
| `code/` | `seedseg-v2.bundle`: git bundle of all v2 branches (`v2-labels`: labels, evaluation, post-processing, docs; `v2-nnunet`: nnU-Net scripts; `v2-synth`: simulator). Restore with `git clone seedseg-v2.bundle seedseg` and then `git checkout <branch>`. `branches.txt` lists the branch heads. |
| `data/study/` | The 127-patient study set: `images/{sid}.nii` (N4-corrected 3T T1, the network input), `labels/{original,corrected,original_sphere}/{sid}.nii.gz` (0 background, 1 marker, 2 prostate/CTV, 3 ignore), `labelfix/` (label rebuild from DICOM: derived marker positions, refinement records `refined/{sid}.json` with the reference marker positions used for evaluation, QC pages, CSVs), `subjects.txt`, `splits_original.json` (preprint split), `test25.txt` (the 25 later-acquired test patients), `val9.txt`, `train102_temporal.txt` |
| `data/synthetic/` | 254 physics-simulated cases (markers and distractors inserted into study images; `images/`, `labels/`, `params/`) and the simulator validation (`validation/`) |
| `data/external/` | External test set: 9 patients outside the 127, labelled **automatically** (gold from CT intensity or the planning structure set, rigidly matched to MR voids, no network, no manual annotation; method and validation in `docs/labels_README.md` and the methods). `images/{sid}.nii` (N4 T1), `labels/{sid}.json` (`markers_final_vox` on the image grid, fit statistics), `labels/{sid}_qc.png`, `labels/summary.csv`, `labels/validation_study60.csv` (validation against the study references) |
| `models/nnUNet_results/` | nnU-Net v2 ResEnc-M models: `Dataset{501,511,512,513,521}_…/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_{0..4}/checkpoint_final.pth`, plus plans, dataset.json and training logs. 501 = original labels, 511 = corrected labels (main), 512 = corrected labels with a temporal split, 513 = corrected labels + synthetic cases, 521 = original positions drawn as spheres (shape control). Use with `nnUNetv2_predict -d <id> -c 3d_fullres -p nnUNetResEncUNetMPlans -f <fold>` after copying into an `nnUNet_results` dir. |
| `models/nnUNet_results_r{1,2,3}/` | Three independent 250-epoch runs of 511 (`nnUNetTrainer_250epochs`) |
| `models/splits/` | Fixed fold assignments (`splits_final_Dataset*.json`, shared by all arms) |
| `models/released_onnx/` | The 4 preprint models (ONNX, also on OSF f7hjv), the baseline |
| `predictions/study_predictions*/` | Per-arm probability maps on the image grid: `oof/{sid}_marker_prob.nii.gz`, `{sid}_probs.nii.gz` (3 classes), `{sid}_seg.nii.gz`. Out-of-fold (each patient predicted by the fold model that did not train on it); for 512 also `test_ensemble/` (the 25 test patients, 5-fold ensemble). `_r1`–`_r3` = 250-epoch runs. |
| `predictions/study_released/` | Released-ensemble marker probability maps for the 127 |
| `predictions/external*/` | Maps for the external set: `{arm}_fold{k}/probs/` (each fold model), `{arm}_ens/` (5-fold mean), `released_probs/`, `eval_*` (evaluation tables) |
| `evaluation/results/` | Evaluation tables per arm and set (`*_summary.json`, `*_subjects.csv`, `*_candidates.csv`), constellation-selection and re-ranker outputs |
| `evaluation/analysis/` | Per-marker outcomes (`markers_*.csv`), miss analyses, figures, DICOM series inventory, acquisition comparability report (`acq_comparability_2025.md`, `acq_params.csv`), bright-fraction table |
| `evaluation/external_labelling_work/` | Working outputs of the automatic labelling (validation runs, CT component table) |
| `evaluation/slurm_logs/` | Slurm logs of all training, export and check jobs |
| `archive/invalid_521_20261019/` | Record of the discarded first 521 run, whose labels had no marker class because of a label-generation bug: the faulty labels, its all-zero evaluation and a README. Its models are not kept. |
| `archive/snapshots-2026-10/` | Earlier dated full-tar snapshots (2026-10-16 to 2026-10-19), superseded by this tree |

## Elsewhere on this RDM (not duplicated here)

- `../2026-02-19-prostate-project.tar.gz`: the original project data (train/test NIfTI, original labels, DICOM-derived files). The study images here are its `*_MRI_homogeneity-corrected.nii` files.
- `../dicoms-organized/`: per-patient DICOM archives (MR, CT, RTSTRUCT, REG), the source of the v2 label rebuild and of the external set.
- `../preprint-2023/`, `../raw-kspace/`: earlier material.

## Not stored (regenerable)

| Item | How to regenerate |
|---|---|
| `nnUNet_raw` datasets | `python scripts/nnunet/convert.py build --variant <v> --design <d> [--synthetic]` (v2-nnunet) from `data/study` and `data/synthetic` |
| `nnUNet_preprocessed` | `sbatch --export=ALL,DATASET=<id> scripts/nnunet/preprocess.slurm` |
| `checkpoint_best/latest`, validation `.npz` | Not needed: the exported OOF maps are in `predictions/` |
| Extracted DICOMs, void-map cache, constellation caches | Re-extract from `../dicoms-organized/`; caches are rebuilt on demand |
