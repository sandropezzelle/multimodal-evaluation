#!/bin/bash

#SBATCH -N1
#SBATCH --job-name=lmdb
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

# PATH=/home/bugliarello.e/data/mscoco

module load 2020
module load Anaconda3/2020.02 

source activate voltaen

pip install lmdb

python convert_cocovist_lmdb.py --split coco-vist 
# trainval --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta
# python convert_coco_lmdb.py --split test --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
