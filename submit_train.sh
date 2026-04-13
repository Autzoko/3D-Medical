#!/bin/bash
#SBATCH --job-name=med3d
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=90:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

source activate med3d

cd /scratch/ll5582/Medical3D/3D-Medical

python train.py \
    --data_dir ../Data/acdc ../Data/amos ../Data/kits/kits23/dataset ../Data/brats \
    --arch vit3d_small \
    --batch_size 8 \
    --epochs 100 \
    --num_workers 8 \
    --output_dir ../checkpoints \
    --save_freq 10 \
    --log_freq 10
