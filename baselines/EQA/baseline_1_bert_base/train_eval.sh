#!/usr/bin/env bash
set -euo pipefail
mkdir -p checkpoints/eqa_bert_base logs
python3 -u common/qa_runner.py --model_name_or_path google-bert/bert-base-uncased --output_dir checkpoints/eqa_bert_base --train_file squad_dataset/train-v2.0.json --predict_file squad_dataset/dev-v2.0.json --do_train --do_eval --answerable_only --do_lower_case --num_train_epochs 3 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 3e-5 --warmup_ratio 0.1 --max_seq_length 384 --doc_stride 128 --max_query_length 64 --seed 42 2>&1 | tee logs/eqa_bert_base.log
