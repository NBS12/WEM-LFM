#!/bin/bash
#SBATCH -J eval
#SBATCH -o eval_middle_aware_lesion_frequency_mask4_0.out
#SBATCH -p A100_40G
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

python /home/zjs/sxx24/Gated-Conditional-Diffusion-Model-v2/evaluation_new.py \
 