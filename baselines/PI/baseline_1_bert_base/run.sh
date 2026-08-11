#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 baseline_1_bert_base/train.py
python3 baseline_1_bert_base/test.py
