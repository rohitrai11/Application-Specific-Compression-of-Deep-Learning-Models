#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/baselines/NLI"; bash run_all_baselines.sh
cd "$ROOT/baselines/PI"; bash run_all_baselines.sh
cd "$ROOT/baselines/EQA"; bash run_all_baselines.sh
