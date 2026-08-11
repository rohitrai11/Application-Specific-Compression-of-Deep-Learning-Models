#!/usr/bin/env bash
set -euo pipefail
bash baseline_1_bert_base/train_eval.sh
bash baseline_2_pruned_bert/train_eval.sh
bash baseline_2_quantization/run.sh
bash baseline_3_off_the_shelf/run_all.sh
bash baseline_4_random_layers/train_eval.sh
