#!/usr/bin/env bash
set -euo pipefail
for MODEL in distilbert bert-medium bert-mini bert-tiny; do
  python3 baseline_3_off_the_shelf/train.py --model_key "$MODEL" --dataset_dir snli_1.0 --output_root checkpoints/baseline3
  python3 baseline_3_off_the_shelf/test.py --input_model_path "checkpoints/baseline3/$MODEL" --dataset_dir snli_1.0 --output_dir "predictions/baseline3/$MODEL" --prediction_prefix "snli_$MODEL"
done
