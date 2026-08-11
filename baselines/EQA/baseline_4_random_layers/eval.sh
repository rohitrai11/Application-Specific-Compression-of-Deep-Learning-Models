#!/usr/bin/env bash
set -euo pipefail
python3 -u common/qa_runner.py --model_name_or_path checkpoints/eqa_random_6_layers --output_dir checkpoints/eqa_random_6_layers --predict_file squad_dataset/dev-v2.0.json --do_eval --answerable_only --do_lower_case --per_gpu_eval_batch_size 32 --max_seq_length 384 --doc_stride 128 --max_query_length 64 --seed 42
