#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=vist-rcnn
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

module load 2020
module load Anaconda3/2020.02
module load CUDA

echo "Extract RCNN features"

source activate rcnnenv


cd ~/py-bottom-up-attention/demo/

python extract_rcnn_features_vist.py

conda deactivate
