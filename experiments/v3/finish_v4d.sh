#!/usr/bin/env bash
# Remaining v4 pipeline after the 2026-10-31 20:01 power cut, strictly sequential and restartable (finished runs are skipped).
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$HERE/../..; PY=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
cd "$HERE"
step() { echo ">> $1 $(date '+%F %T')"; shift; nice -n 10 "$@"; echo "   done $(date '+%F %T')"; }
step "LR 1e-2 sweeps"        $PY run_v4c.py           >> run_v4_lr1e-2.log 2>&1
step "e4 re-check"           $PY run_v4.py e4         >> run_v4_e4_redo.log 2>&1
step "subsets"               $PY run_v4.py subsets    >> run_v4_subsets.log 2>&1
step "e5 MMS curve"          $PY run_v4.py e5         >> run_v4_e5.log 2>&1
step "mHuBERT 32 cl."        $PY run_v4b.py           >> run_v4b.log 2>&1
step "streaming"             ./run_stream_v4.sh       >> run_stream_v4.log 2>&1
step "analysis"              $PY "$ROOT/analysis/11_results_v4.py" > run_analysis_v4.log 2>&1
echo ALL DONE
