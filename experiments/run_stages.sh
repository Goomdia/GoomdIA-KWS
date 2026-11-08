#!/usr/bin/env bash
# usage: run_stages.sh PY CACHE OUT OOD "stage1:nworkers stage2:nworkers ..."
PY=$1; CACHE=$2; OUT=$3; OOD=$4; shift 4
for spec in $@; do
  st=${spec%%:*}; n=${spec##*:}
  for i in $(seq 1 $n); do "$PY" "$(dirname "$0")/run_queue.py" "$PY" "$CACHE" "$OUT" "$st" "$OOD" & sleep 20; done
  wait
  echo "STAGE $st DONE"
done
