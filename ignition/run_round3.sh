#!/bin/bash
# Round 3 of the ignition study: the big case set (cases_big.parquet, 22k cases, rings inside), one sampler
# at a time, each reading 8 units at once. Resumable: rerun as is, finished parts are skipped.
set -u
cd "$(dirname "$0")/.."
D=data/ignition
step() { echo "== $(date '+%H:%M') $1"; }
step "weather big (4271 groups, 8 processes)";   uv run ignition/sample_weather.py --tag big    > $D/weather_big.log 2>&1 || exit 1
step "nlcd big (223 parts, 8 threads)";           uv run ignition/sample_nlcd.py --tag big       > $D/nlcd_big.log 2>&1 || exit 1
step "embeddings big (60018 chunks, 8 threads)";  uv run ignition/sample_embeddings.py --tag big > $D/emb_big.log 2>&1 || exit 1
step "done"
