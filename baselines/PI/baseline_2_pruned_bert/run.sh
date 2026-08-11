#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 baseline_2_pruned_bert/train.py
python3 baseline_2_pruned_bert/test.py
