#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 baseline_2_quantization/quantize.py
python3 baseline_2_quantization/test.py
