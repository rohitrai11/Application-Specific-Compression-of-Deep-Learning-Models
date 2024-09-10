#!/bin/bash

# Set environment variables
SEED="42"
GPU_ID="0"
RUN_NAME="pruned_bert_seed${SEED}"
PJ_NAME="pruned_bert"
SQuAD_DIR="squad_dataset"
RE_EXQA_OUT_DIR="output"
current_directory=$(pwd)
export analysis="${current_directory}/output/analysis"

# Run the Python script
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u run_squad.py --project $PJ_NAME \
--model_type pruned-bert \
--model_name_or_path saved_model_path_without_quotes --do_lower_case \
--do_eval --output_dir $RE_EXQA_OUT_DIR/$RUN_NAME --warmup_ratio 0.1 --overwrite_output_dir --threads 4 \
--predict_file $SQuAD_DIR/dev-v2.0.json --version_2_with_negative \
--seed $SEED > log/$RUN_NAME.txt &

echo "Script is running in the background. Check log/$RUN_NAME for output."


