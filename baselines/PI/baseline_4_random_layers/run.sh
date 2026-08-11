#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 baseline_4_random_layers/train.py
python3 baseline_4_random_layers/test.py
