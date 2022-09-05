function train {
    GPUS_PER_NODE=2 
    tools/run_dist_launch.sh 2 \
    ./configs/r50_deformable_detr.sh
}
export -f train
nohup bash -c train > logs/train.log &

## multi-gpu training
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#tools/dist_train.sh \
#    local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py \
#    4 2>&1
