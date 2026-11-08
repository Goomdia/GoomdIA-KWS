#!/usr/bin/env bash
# Evaluation of the open-set experiments (after run_open_set.py). LeTR streaming runs on CPU (no GPU lock);
# wav2vec 2.0 jobs run on the shared GPU, one job per `flock` call.
# usage: experiments/v3/run_eval_v3.sh PY CACHE EXTRA GSC LOCK
set -u
PY=$1; CACHE=$2; EXTRA=$3; GSC=$4; LOCK=$5
cd "$(dirname "$0")/../.."
V3=experiments/v3/runs
# closed-set baselines of the article (models saved in experiments/runs)
for R in letr128t-gsc-speaker-lr0.001-s10 w2v2-ssl-speaker-lr3e-05-s0; do
  O=$V3/closed30-$R; mkdir -p "$O"
  if [[ $R == w2v2* ]]; then G="flock $LOCK"; D=cuda; else G=""; D=cpu; fi
  [ -f "$O/summary.json" ] || $G "$PY" experiments/v3/eval_closed_isolated.py --run experiments/runs/$R --cache "$CACHE" \
      --extra "$EXTRA" --out "$O" --device $D > "$O/eval_isolated.log" 2>&1
  [ -f "$O/streaming_v3.json" ] || $G "$PY" experiments/v3/streaming_v3.py --run experiments/runs/$R --cache "$CACHE" \
      --gsc "$GSC" --extra "$EXTRA" --out "$O" --device $D > "$O/streaming_v3.log" 2>&1
done
# 32-class models
for O in $V3/open32-*/; do
  O=${O%/}; [ -f "$O/model.pt" ] && [ ! -f "$O/streaming_v3.json" ] || continue
  if [[ $O == *w2v2* ]]; then G="flock $LOCK"; D=cuda; else G=""; D=cpu; fi
  $G "$PY" experiments/v3/streaming_v3.py --run "$O" --cache "$CACHE" --gsc "$GSC" --extra "$EXTRA" --device $D \
      > "$O/streaming_v3.log" 2>&1
done
"$PY" analysis/08_open_set_results.py
