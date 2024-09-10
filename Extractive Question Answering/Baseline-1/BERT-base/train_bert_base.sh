#!/bin/bash

# Set environment variables
SEED="42"
GPU_ID="0"
RUN_NAME="bert_base_seed${SEED}"
PJ_NAME="bert_base"
SQuAD_DIR="squad_dataset"
RE_EXQA_OUT_DIR="output"

# Run the Python script
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u run_squad.py --project $PJ_NAME \
--model_type bert-base \
--model_name_or_path google-bert/bert-base-uncased --do_lower_case \
--do_train --do_eval --output_dir $RE_EXQA_OUT_DIR/$RUN_NAME --warmup_ratio 0.1 --num_train_epochs 3 --logging_train_steps 1000 --log_before_train --evaluate_during_training --overwrite_output_dir --threads 4 --version_2_with_negative \
--train_file $SQuAD_DIR/train-v2.0.json \
--predict_file $SQuAD_DIR/dev-v2.0.json \
--seed $SEED > log/$RUN_NAME.txt &

echo "Script is running in the background. Check log/$RUN_NAME for output."
