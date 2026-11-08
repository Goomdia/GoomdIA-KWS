#!/usr/bin/env bash
# Tail of the v4 pipeline (replaces finish_v4.sh; BC-ResNet GSC init dropped), restartable. usage: finish_v4b.sh MAIN_QUEUE_PID
# (1) after the p1 re-run (GSC jobs): extend the BC-ResNet LR grid to 1e-2 (p1 + mswc sweeps; existing runs skipped);
# (2) re-select the BC-ResNet-8 32-class init/LR (e4; no-op if unchanged); (3) after the main queue: subsets re-check;
# (4) 32-class mHuBERT-147; (5) streaming tuning; (6) summary tables.
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$HERE/../..; PY=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
cd "$HERE"; MAIN=$1

nice -n 10 $PY run_v4c.py > run_v4_lr1e-2.log 2>&1;         echo "LR 1e-2 sweeps finished $(date '+%F %T')"
nice -n 10 $PY run_v4.py e4 > run_v4_e4_redo.log 2>&1;             echo "e4 re-check finished $(date '+%F %T'): $(cat e4_bc8_choice.json)"
while kill -0 "$MAIN" 2>/dev/null; do sleep 60; done;              echo "main queue finished $(date '+%F %T')"
nice -n 10 $PY run_v4.py subsets > run_v4_subsets_redo.log 2>&1;   echo "subsets re-check finished $(date '+%F %T')"
nice -n 10 $PY run_v4b.py > run_v4b.log 2>&1;                      echo "mHuBERT 32 cl. finished $(date '+%F %T')"
./run_stream_v4.sh > run_stream_v4.log 2>&1;                        echo "streaming finished $(date '+%F %T')"
nice -n 10 $PY "$ROOT/analysis/11_results_v4.py" > run_analysis_v4.log 2>&1; echo "analysis finished $(date '+%F %T')"
echo ALL DONE
