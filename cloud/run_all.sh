#!/usr/bin/env bash
# Full run on the cloud GPU: smoke test -> benchmark -> NPD night check -> stats.
# Resumable: re-running skips finished (image, level, method) rows.
#
# Run it detached so a closed VS Code window doesn't kill it:
#   POTHOLEMIX_ROOT=data/raw/pothole-mix NPD_ROOT=data/raw/NPD \
#     nohup bash cloud/run_all.sh > results/run.log 2>&1 &
#   tail -f results/run.log
# STOP THE INSTANCE when it finishes: billing is per hour.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${POTHOLEMIX_ROOT:?set POTHOLEMIX_ROOT to the pothole-mix folder that contains testing/}"
: "${NPD_ROOT:?set NPD_ROOT to the NPD folder that contains annotations/}"
export POTHOLEMIX_ROOT
mkdir -p results
nvidia-smi --query-gpu=name --format=csv,noheader | tee results/gpu.txt

echo "== smoke test (2 images)"
python scripts/benchmark.py --limit 2 --out results/smoke.csv
rm -f results/smoke.csv

echo "== benchmark"
python scripts/benchmark.py --overlays 10

echo "== NPD night check"
python scripts/night_check.py --npd "$NPD_ROOT" --split test

echo "== stats"
python scripts/analyze.py --night results/night_check.csv
echo "ALL DONE $(date)"
