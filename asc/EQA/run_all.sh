#!/usr/bin/env bash
set -euo pipefail

python3 -u EQA/train_base.py
python3 -u EQA/compute_similarity.py

for TH in 0.90 0.85 0.80; do
  TAG=$(python3 -c "print(int(round(float('${TH}')*100)))")
  python3 -u EQA/build_and_finetune.py --threshold "${TH}"
  python3 -u EQA/test.py \
    --input_model_path "EQA/checkpoints/asc_th_${TAG}" \
    --output_dir "EQA/predictions/asc_th_${TAG}"
done
