#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=vilbertLOOP
#SBATCH --time=60:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

# module load 2019
module load 2020
# module load eb
module load Anaconda3/2020.02

echo "Test model"

TASK=0
MODEL=ctrl_vilbert
MODEL_CONFIG=ctrl_vilbert_base
TASKS_CONFIG=ctrl_test_tasks
PRETRAINED=./models/${MODEL}_pretrained_cc.bin
OUTPUT_DIR=./results/vilbert

source activate voltaenv

cd ~/volta/

cp -r ${PRETRAINED} ${TMPDIR}/data

for mylayer in 14 16 18 20 22 24 26 28 30 32 34 36
do
	python get_representations_all.py --zero_shot --modality lang --targ_layer ${mylayer} \
		--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${TMPDIR}/data \
		--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test --batch_size 1 \
		--output_dir ${OUTPUT_DIR}

done

conda deactivate
