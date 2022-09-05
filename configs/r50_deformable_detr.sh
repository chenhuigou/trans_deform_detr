#!/usr/bin/env bash
#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} #> r50_deformable_detr_342_202208_.log 2>&1 &
#GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/r50_deformable_detr.sh #--resume exps/r50_deformable_detr_transform_invariant/checkpoint0004.pth \