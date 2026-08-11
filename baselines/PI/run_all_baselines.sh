#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash baseline_1_bert_base/run.sh
bash baseline_2_pruned_bert/run.sh
bash baseline_2_quantization/run.sh
bash baseline_3_off_the_shelf/run_all.sh
bash baseline_4_random_layers/run.sh
