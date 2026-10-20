#!/bin/bash
# Evaluate the fresh (automatically labelled, external) set: each fold model alone, the
# 5-fold ensemble (mean of fold maps), and ensemble + constellation (prior from the 127).
# usage: eval_fresh.sh <arm id> <study OOF dir> <study constellation cache>
set -e
id=$1; soof=$2; scache=$3
F=/scratch/user/uqaste15/prostate-v2/fresh; S=/scratch/user/uqaste15/src/seedseg/scripts; R=$F/pred
ev() { python3 $S/evaluation/evaluate_detections.py --probs $1 --root $F --subjects $F/subjects_fresh.txt --out $2 "${@:3}"; }
for f in 0 1 2 3 4; do
  d=$R/${id}_fold$f/probs; m=$R/${id}_fold$f/mprob; mkdir -p $m
  for x in $d/*_marker_prob.nii.gz; do ln -sf $x $m/$(basename $x _marker_prob.nii.gz).nii.gz; done
  echo -n "$id fold$f: "; ev $m $R/eval_${id}_fold$f
  python3 $S/rerank/constellation_apply.py --candidates $R/eval_${id}_fold${f}_candidates.csv --root $F \
    --probs3 $d --study-oof $soof --study-cache $scache --cache $R/${id}_fold$f/cache --out $R/${id}_fold${f}_constellation.csv > /dev/null
  echo -n "$id fold$f+constellation: "; ev $m $R/eval_${id}_fold${f}_const --rescore $R/${id}_fold${f}_constellation.csv --score cscore
done
python3 - "$R" "$id" <<'PY'
import sys, os, numpy as np, nibabel as nib
R, i = sys.argv[1:]; F = os.path.dirname(R)
os.makedirs(f'{R}/{i}_ens/mprob', exist_ok=True); os.makedirs(f'{R}/{i}_ens/probs3', exist_ok=True)
for s in open(F + '/subjects_fresh.txt').read().split():
    ims = [nib.load(f'{R}/{i}_fold{f}/probs/{s}_probs.nii.gz') for f in range(5)]
    p = np.mean([np.asarray(im.dataobj, np.float32) for im in ims], 0)
    nib.save(nib.Nifti1Image(p, ims[0].affine), f'{R}/{i}_ens/probs3/{s}_probs.nii.gz')
    nib.save(nib.Nifti1Image(p[..., 1], ims[0].affine), f'{R}/{i}_ens/mprob/{s}.nii.gz')
PY
echo -n "$id ensemble: "; ev $R/${id}_ens/mprob $R/eval_${id}_ens
python3 $S/rerank/constellation_apply.py --candidates $R/eval_${id}_ens_candidates.csv --root $F \
  --probs3 $R/${id}_ens/probs3 --study-oof $soof --study-cache $scache --cache $R/${id}_ens/cache --out $R/${id}_ens_constellation.csv
echo -n "$id ensemble+constellation: "; ev $R/${id}_ens/mprob $R/eval_${id}_ens_const --rescore $R/${id}_ens_constellation.csv --score cscore
