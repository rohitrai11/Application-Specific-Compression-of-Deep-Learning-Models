#!/usr/bin/env bash
set -euo pipefail
python3 -u baseline_2_quantization/quantize_and_eval.py --input_model_path checkpoints/eqa_bert_base --output_dir results/eqa_quantized_bert --predict_file squad_dataset/dev-v2.0.json --answerable_only --batch_size 32 --seed 42
