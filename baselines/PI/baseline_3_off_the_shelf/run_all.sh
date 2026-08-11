#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for model in distilbert bert-medium bert-mini bert-tiny; do python3 baseline_3_off_the_shelf/train.py --model_key "$model"; python3 baseline_3_off_the_shelf/test.py --model_key "$model"; done
