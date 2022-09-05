function train {
    MASTER_ADDR="129.21.171.186" \ 
    MASTER_PORT="39599" \
    NODE_RANK=1 \
    GPUS_PER_NODE=4 \
    ./tools/run_dist_launch.sh 8 \
    ./configs/r50_deformable_detr.sh
}
export -f train
nohup bash -c train > logs/r50_deformable_detr_24_batch_gpu1.log &
#
#MASTER_ADDR="129.21.171.186" MASTER_PORT="39500" 