#!/bin/bash
#SBATCH -J inf
#SBATCH -o gcdm_inference_lesion_aware_frequency_mask_pram4.0.out
#SBATCH -p A100_40G
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

python /home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/inference.py \
 