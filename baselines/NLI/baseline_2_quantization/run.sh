#!/usr/bin/env bash
set -euo pipefail
python3 baseline_2_quantization/quantize.py --input_model_path checkpoints/baseline1_bert_base --output_dir checkpoints/baseline2_quantized_bert
python3 baseline_2_quantization/test.py --input_model_path checkpoints/baseline2_quantized_bert --dataset_dir snli_1.0
