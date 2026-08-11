#!/usr/bin/env bash
set -euo pipefail
python3 -u NLI/compute_similarity.py
for TH in 0.90 0.85 0.80; do python3 -u NLI/build_and_finetune.py --threshold "${TH}"; done
python3 -u PI/compute_similarity.py
for TH in 0.90 0.85 0.80; do python3 -u PI/build_and_finetune.py --threshold "${TH}"; done
python3 -u EQA/compute_similarity.py
for TH in 0.90 0.85 0.80; do python3 -u EQA/build_and_finetune.py --threshold "${TH}"; done
