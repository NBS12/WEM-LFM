#!/bin/bash
#SBATCH -J exp_l
#SBATCH -o WEM+LFM_param4.0.out
#SBATCH -p A100_40G
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

export CUDA_VISIBLE_DEVICES=2
python /home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/main.py \
    -t \
    --base /home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/configs/train.yaml  \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 1 \
    --logdir "/home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/logs" \
    --finetune_from /home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/sd-image-conditioned-v2.ckpt
    # --gpus 0 \
    
