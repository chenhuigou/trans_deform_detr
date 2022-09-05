#python main.py --
#srun puzzle -p tier3 --nodes=1 --ntasks-per-node=4 --cpus-per-task=1 --gres=gpu:a100:1 --mem=100g  --pty bash

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh \ configs/r50_deformable_detr.sh --resume  /home/dxleec/chenhui/Deformable-DETR/exps/r50_deformable_detr/checkpoint0004.pth --eval