#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=jsonl
#SBATCH --partition=gpu_shared
#SBATCH --time=02:00:00
#SBACTH --gres=gpu:i1

module load 2020
module load Anaconda3/2020.02

source activate voltaenv

python convert_sentences_to_jsonl.py
