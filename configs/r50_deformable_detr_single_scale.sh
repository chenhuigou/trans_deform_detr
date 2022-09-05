#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale_transform_invariant_new
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
