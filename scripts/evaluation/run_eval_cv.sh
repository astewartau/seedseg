#!/bin/bash
# usage: run_eval.sh <id> <oof dir>   (evaluates all127, test25, and each fold's held-out subjects)
set -e
id=$1; oof=$2
P=/scratch/user/uqaste15/prostate-v2; E=$P/eval; R=$E/results; S=/scratch/user/uqaste15/src/seedseg/scripts
mkdir -p $E/folds_$id
python3 -c "
import json,sys; d=json.load(open('$oof/oof_folds.json'))
for f in range(5): open('$E/folds_$id/fold%d.txt'%f,'w').write('\n'.join(s for s,k in d.items() if k==f)+'\n')"
ev() { python3 $S/evaluation/evaluate_detections.py --probs $E/probs_$id --root $P --subjects $1 --out $R/nnunet${id}_$2; }
echo -n "all127: "; ev $P/subjects.txt all127
echo -n "test25: "; ev $E/test25.txt test25
for f in 0 1 2 3 4; do echo -n "fold$f: "; ev $E/folds_$id/fold$f.txt fold$f; done
