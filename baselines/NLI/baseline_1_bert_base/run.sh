#!/usr/bin/env bash
set -euo pipefail
python3 baseline_1_bert_base/train.py --dataset_dir snli_1.0 --output_dir checkpoints/baseline1_bert_base
python3 baseline_1_bert_base/test.py --input_model_path checkpoints/baseline1_bert_base --dataset_dir snli_1.0 --output_dir predictions/baseline1_bert_base
