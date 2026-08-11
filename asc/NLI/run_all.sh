#!/usr/bin/env bash
set -euo pipefail
python3 -u NLI/train_base.py
python3 -u NLI/compute_similarity.py
for TH in 0.90 0.85 0.80; do TAG=$(python3 -c "print(int(round(float('${TH}')*100)))"); python3 -u NLI/build_and_finetune.py --threshold "${TH}"; python3 -u NLI/test.py --input_model_path "NLI/checkpoints/asc_th_${TAG}" --output_dir "NLI/predictions/asc_th_${TAG}"; done
