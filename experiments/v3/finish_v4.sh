#!/usr/bin/env bash
# Tail of the v4 pipeline, restartable: (1) after the phase-p1 re-run (GSC jobs) repeat phase e4 so that the BC-ResNet-8 32-class init is
# re-selected now that GSC results exist (no-op if the choice is unchanged); (2) after the main queue ends repeat phase subsets (no-op if
# its init is unchanged); (3) streaming tuning of the 32-class models; (4) summary tables.   usage: finish_v4.sh MAIN_QUEUE_PID
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$HERE/../..; PY=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4
cd "$HERE"; MAIN=$1
while pgrep -f "[r]un_v4.py p1" >/dev/null; do sleep 60; done;  echo "p1 re-run finished $(date '+%F %T')"
nice -n 10 $PY run_v4.py e4 > run_v4_e4_redo.log 2>&1;           echo "e4 re-check finished $(date '+%F %T'): $(cat e4_bc8_choice.json)"
while kill -0 "$MAIN" 2>/dev/null; do sleep 60; done;            echo "main queue finished $(date '+%F %T')"
nice -n 10 $PY run_v4.py subsets > run_v4_subsets_redo.log 2>&1; echo "subsets re-check finished $(date '+%F %T')"
./run_stream_v4.sh > run_stream_v4.log 2>&1;                      echo "streaming finished $(date '+%F %T')"
nice -n 10 $PY "$ROOT/analysis/11_results_v4.py" > run_analysis_v4.log 2>&1; echo "analysis finished $(date '+%F %T')"
echo ALL DONE
