#!/bin/bash
# Round 2 of the ignition study, sequential so one remote reader runs at a time. Resumable: rerun as is.
set -u
cd "$(dirname "$0")/.."
D=data/ignition
step() { echo "== $(date '+%H:%M') $1"; }
step "weather wide (956 groups, ~70 min)";     uv run ignition/sample_weather.py --tag wide    > $D/weather_wide.log 2>&1 || exit 1
step "embeddings wide (7389 chunks, 8 at a time)"; uv run ignition/sample_embeddings.py --tag wide > $D/emb_wide.log 2>&1 || exit 1
step "nlcd wide (~15 min)";                     uv run ignition/sample_nlcd.py --tag wide       > $D/nlcd_wide.log 2>&1 || exit 1
step "embeddings 2 years before (8 at a time)";     uv run ignition/sample_embeddings.py --back 1   > $D/emb_b1.log 2>&1 || exit 1
step "embeddings 3 years before (8 at a time)";     uv run ignition/sample_embeddings.py --back 2   > $D/emb_b2.log 2>&1 || exit 1
step "done"
