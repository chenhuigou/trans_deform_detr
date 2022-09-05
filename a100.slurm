#!/bin/bash
#SBATCH -J each_4_deformable_detr
#SBATCH -t 120:00:00
#SBATCH -p tier3
#SBATCH -A puzzle
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:4             
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH -c 4
#SBATCH -o each_4_deformable_detr.out
#spack load gcc@9.3.0/hufzekv
#spack load cuda@11.0.2/lrd2rcw
#conda activate openmmlab

# conda env list
#nvidia-smi
#GPUS_PER_NODE=4  ./tools/run_dist_slurm.sh tier3 deformable_detr 8 configs/r50_deformable_detr.sh
bash configs/r50_deformable_detr.sh


#srun -p tier3 \
#>     --job-name=2_deform_4\
#>     -A puzzle \
#>     -o logs/2_deform4.log \
#>     --mem=100G \
#>     --gres=gpu:a100:4\
#>     --ntasks=8 \
#>     --ntasks-per-node=4 \
#>     --cpus-per-task=4 \
#>     bash configs/r50_deformable_detr.sh
#srun -A puzzle -t 24:0:0 -p tier3  --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:a100:2 --mem=100g  --pty bash