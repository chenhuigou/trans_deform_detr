function test {
    GPUS_PER_NODE=2 
    tools/run_dist_launch.sh 2 \
    ./configs/r50_deformable_detr.sh \
    --resume \
    configs/r50_deformable_detr.sh --resume  /home/dxleec/chenhui/Deformable-DETR/exps/r50_deformable_detr/checkpoint.pth \
    --eval

}
export -f test
nohup bash -c test > logs/test.log &

## multi-gpu training
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#tools/dist_train.sh \
#    local_configs/ddrformer/23slim/ddrformer.23slim.1024x1024.city.160k.py \
#    4 2>&1
