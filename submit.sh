#!/bin/bash
#SBATCH --job-name=ml_pg
#SBATCH --output=ml_pg.out
#SBATCH --account=guest
#SBATCH -p guest-gpu
#SBATCH --gres=gpu:V100:1 # list_free_gpu gives a list of gpus
#SBATCH --qos=low-gpu

#Load modules required for your job

module load cuda/11.7
#module load share_modules/ANACONDA/5.3_py3

source activate pt

#Path to your executable

python explainer_ml.py