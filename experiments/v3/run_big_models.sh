#!/usr/bin/env bash
# Large-encoder fine-tuning (wav2vec 2.0 Base extra seeds, MMS-300M), one GPU run per flock call.
# usage: run_big_models.sh PY CACHE OOD "phase ..."   phases: lrcheck w2v2 mms
PY=$1; CACHE=$2; OOD=$3; shift 3
HERE=$(cd "$(dirname "$0")/.." && pwd); RUNS=$HERE/runs; LRC=$HERE/v3/mms_lr_check
LOCK=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/gpu.lock
REC=(--bs 8 --epochs 20 --patience 5 --wd 1e-2 --optim adamw8bit)   # w2v2 recipe (run_queue.py, stage w2v2)

train() {  # out name args...
  local out=$1 name=$2; shift 2
  [ -f "$out/$name/summary.json" ] && return
  echo ">> $name $(date -Is)"
  flock "$LOCK" "$PY" "$HERE/train_kws.py" --cache "$CACHE" --out "$out" --name "$name" "$@" > "$out/$name.log" 2>&1
  echo "<< $name $(date -Is)"
}

for phase in "$@"; do
  if [ "$phase" = lrcheck ]; then          # MMS-300M learning rate on validation: seed 0, first 5 of 20 epochs
    mkdir -p "$LRC"
    for lr in 1e-05 3e-05 0.0001; do
      train "$LRC" "mms300m-ssl-speaker-lr$lr-s0-ep5" --model mms300m --init ssl --split speaker --lr $lr --seed 0 "${REC[@]}" --stop_after 5
    done
  fi
  if [ "$phase" = w2v2 ]; then
    for sp in speaker:2 random:1 random:2; do
      train "$RUNS" "w2v2-ssl-${sp%%:*}-lr3e-05-s${sp##*:}" --model w2v2 --init ssl --split ${sp%%:*} --lr 3e-5 --seed ${sp##*:} "${REC[@]}" --ood "$OOD"
    done
  fi
  if [ "$phase" = mms ]; then              # learning rate = best validation accuracy of the check
    lr=$("$PY" -c "import json,glob,sys;b=max((json.load(open(f)) for f in glob.glob(sys.argv[1]+'/*/summary.json')),key=lambda s:s['best_val_acc']);print('%g'%b['args']['lr'])" "$LRC")
    for seed in 0 1 2; do
      train "$RUNS" "mms300m-ssl-speaker-lr$lr-s$seed" --model mms300m --init ssl --split speaker --lr $lr --seed $seed "${REC[@]}" --ood "$OOD" $([ $seed = 0 ] && echo --save_model)
      t=$("$PY" -c "import json,sys;print(int(json.load(open(sys.argv[1]))['train_time_s']))" "$RUNS/mms300m-ssl-speaker-lr$lr-s$seed/summary.json")
      [ "$t" -gt 21600 ] && [ $seed -ge 1 ] && { echo "run > 6 h: stopping at 2 seeds"; break; }
    done
  fi
  echo "PHASE $phase DONE $(date -Is)"
done
