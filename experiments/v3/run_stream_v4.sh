#!/usr/bin/env bash
# E4 streaming tuning for the new 32-class models (centre scoring; dev-only selection; one test evaluation). Restartable.
HERE=$(cd "$(dirname "$0")" && pwd); PY=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/venv/bin/python; LOCK=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/gpu.lock
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
G="BC-ResNet-8 32 cl.,w2v2 32 cl. (2 seeds),MMS-300M 32 cl.,mHuBERT-147 32 cl."
for st in posteriors search final; do
  flock "$LOCK" nice -n 10 "$PY" "$HERE/streaming_tune.py" $st --mode centre --groups "$G" --device cuda --threads 4 --procs 2
  rc=$?; [ $rc -ne 0 ] && { echo "FAILED $st rc=$rc"; exit 1; }; sleep 20
done
echo STREAM DONE
