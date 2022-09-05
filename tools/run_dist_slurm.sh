#!/usr/bin/env bash
# --------------------------------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/open-mmlab/mmdetection/blob/3b53fe15d87860c6941f3dda63c0f27422da6266/tools/slurm_train.sh
# --------------------------------------------------------------------------------------------------------------------------

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
RUN_COMMAND=${@:4}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    -A puzzle\
    --mem=100G \
    --job-name=${JOB_NAME} \
    --gres=gpu:a100:4 \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=4 \
    -o logs/2_deform4.out \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${RUN_COMMAND}
#srun -p tier3 \
#    --job-name=2_deform_4\
#    -A puzzle \
#    -o logs/2_deform4.out \
#    
#    --gres=gpu:a100:4\
#    --ntasks=8 \
#    --ntasks-per-node=4 \
#    --cpus-per-task=4 \
#    bash configs/r50_deformable_detr.sh
    