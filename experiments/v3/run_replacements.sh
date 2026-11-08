#!/usr/bin/env bash
# Replacement seeds for runs that failed to train (best validation accuracy < 0.1, i.e. near chance = 1/30):
# adds seeds 3, 4 until each configuration has 3 successful seeds. Waits for run_big_models.sh (PID $4) first.
# usage: run_replacements.sh PY CACHE OOD [PID]
PY=$1; CACHE=$2; OOD=$3; PID=${4:-}
HERE=$(cd "$(dirname "$0")/.." && pwd); RUNS=$HERE/runs; LOCK=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/gpu.lock
REC=(--bs 8 --epochs 20 --patience 5 --wd 1e-2 --optim adamw8bit)
[ -n "$PID" ] && while kill -0 "$PID" 2>/dev/null; do sleep 60; done

ok() { "$PY" -c "import json,glob,sys;print(sum(json.load(open(f))['best_val_acc']>=0.1 for f in glob.glob(sys.argv[1]+'-s[0-4]/summary.json')))" "$RUNS/$1"; }
for cfg in w2v2:speaker w2v2:random mms300m:speaker; do
  m=${cfg%%:*}; sp=${cfg##*:}; base="$m-ssl-$sp-lr3e-05"
  for seed in 3 4; do
    [ "$(ok "$base")" -ge 3 ] && break
    name="$base-s$seed"; [ -f "$RUNS/$name/summary.json" ] && continue
    echo ">> $name $(date -Is)"
    flock "$LOCK" "$PY" "$HERE/train_kws.py" --cache "$CACHE" --out "$RUNS" --name "$name" --model $m --init ssl --split $sp \
      --lr 3e-5 --seed $seed "${REC[@]}" --ood "$OOD" > "$RUNS/$name.log" 2>&1
    echo "<< $name $(date -Is)"
  done
done
echo "REPLACEMENTS DONE $(date -Is)"
