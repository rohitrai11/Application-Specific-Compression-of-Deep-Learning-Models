#!/usr/bin/env bash
set -euo pipefail
bash NLI/run_all.sh
bash PI/run_all.sh
bash EQA/run_all.sh
