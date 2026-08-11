#!/usr/bin/env bash
set -euo pipefail
python3 baseline_4_random_layers/train.py --input_model_path checkpoints/baseline1_bert_base --output_dir checkpoints/baseline4_random_6_layers --dataset_dir snli_1.0 --remove_layers 6 --seed 42
python3 baseline_4_random_layers/test.py --input_model_path checkpoints/baseline4_random_6_layers --dataset_dir snli_1.0
