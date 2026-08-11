#!/usr/bin/env bash
set -euo pipefail
python3 -u PI/train_base.py
python3 -u PI/compute_similarity.py
for TH in 0.90 0.85 0.80; do TAG=$(python3 -c "print(int(round(float('${TH}')*100)))"); python3 -u PI/build_and_finetune.py --threshold "${TH}"; python3 -u PI/test.py --input_model_path "PI/checkpoints/asc_th_${TAG}" --output_dir "PI/predictions/asc_th_${TAG}"; done
