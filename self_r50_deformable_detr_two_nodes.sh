function train1 {
    #MASTER_ADDR="127.0.0.1"
    #MASTER_ADDR="127.0.0.2"
    #NODE_RANK=1
    #NODE_RANK=1
    #GPUS_PER_NODE=4
    #tools/run_dist_launch.sh 8 \
    #./configs/r50_deformable_detr.sh
    GPUS_PER_NODE=4 ./tools/run_dist_slurm.sh tier3 2_deformable_detr_each_4 8 configs/r50_deformable_detr.sh
}
#export CUDA_VISIBLE_DEVICES=4,5,6,7
export -f train1
#nohup bash -c train1
nohup bash -c train1 > logs/2_deformable_detr_each_4.log &

## multi-gpu training
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#tools/dist_train.sh \
#    local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py \
#    4 2>&1
