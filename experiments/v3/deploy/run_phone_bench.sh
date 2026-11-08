#!/bin/bash
# Phone benchmark with Google's TFLite benchmark_model (android_aarch64, XNNPACK CPU) over adb.
# Usage: run_phone_bench.sh [-o data/deploy_phone.csv] [-t "1 4"] file1.tflite [file2.tflite ...]
#   Each <name>.<fp32|int8>.tflite needs its <name>.<prec>.json written by export_tflite.py (size, max diff).
#   Export first, e.g.: deploy_venv/bin/python export_tflite.py --ckpt <model.pt>   (new checkpoints: same call).
# Needs: platform-tools (adb) in ${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/platform-tools, and the benchmark binary at
#   ${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/tflite_bm/android_aarch64_benchmark_model (nightly build from
#   storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/).
# Only /data/local/tmp is used; files are removed at the end. Reported latency = median of the tool's runs.
ADB=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/platform-tools/adb
BM=${GOOMDIA_CACHE:-$HOME/goomdia_kws_cache}/tflite_bm/android_aarch64_benchmark_model
OUT="$(dirname "$0")/../../../data/deploy_phone.csv"; THR="1 4"; RUNS=${RUNS:-50}; TMO=${TMO:-400}
while getopts "o:t:" o; do case $o in o) OUT=$OPTARG;; t) THR=$OPTARG;; esac; done; shift $((OPTIND-1))
$ADB devices | grep -q -w device || { echo "phone not authorised/connected"; exit 1; }
DEV="$($ADB shell getprop ro.product.model | tr -d '\r')"
$ADB push $BM /data/local/tmp/benchmark_model >/dev/null; $ADB shell chmod +x /data/local/tmp/benchmark_model
[ -f "$OUT" ] || echo "model,tflite,precision,frontend_in_graph,device,threads,size_mb,status,median_ms,avg_ms,p95_ms,peak_mem_mb,init_ms,max_abs_logit_diff_vs_torch,argmax_agreement" > "$OUT"
for f in "$@"; do
  j="${f%.tflite}.json"; read -r name prec fe diff agree < <(python3 -c "
import json;m=json.load(open('$j'));print(m['name'],m['precision'],m['frontend_in_graph'],m['max_abs_logit_diff_vs_torch'],m['argmax_agreement'])")
  size=$(python3 -c "import os;print(round(os.path.getsize('$f')/2**20,2))")
  $ADB push "$f" /data/local/tmp/m.tflite >/dev/null
  for t in $THR; do
    log=$(timeout $TMO $ADB shell "/data/local/tmp/benchmark_model --graph=/data/local/tmp/m.tflite --num_threads=$t --num_runs=$RUNS --warmup_runs=5 --max_secs=120 --use_xnnpack=true --report_peak_memory_footprint=true 2>&1" | tr -d '\r')
    st=ok; med=; avg=; p95=; mem=; init=
    L=$(echo "$log" | grep "count=" | tail -1)
    if [ -z "$L" ]; then st="fail: $(echo "$log" | grep -i -E "error|fail|abort|kill|Cannot" | head -1 | tr ',' ';' | cut -c1-120)"; [ "$st" = "fail: " ] && st="fail: no output (killed/timeout)"
    else
      med=$(echo "$L" | sed -E 's/.*median=([0-9.]+).*/\1/'); avg=$(echo "$L" | sed -E 's/.*avg=([0-9.]+).*/\1/'); p95=$(echo "$L" | sed -E 's/.*p95=([0-9.]+).*/\1/')
      med=$(python3 -c "print(round($med/1000,2))"); avg=$(python3 -c "print(round($avg/1000,2))"); p95=$(python3 -c "print(round($p95/1000,2))")
      mem=$(echo "$log" | grep "peak memory footprint (MB) via" | sed -E 's/.*: ([0-9.]+)/\1/'); init=$(echo "$log" | sed -nE 's/.*Init: ([0-9]+),.*/\1/p'); init=$(python3 -c "print(round(${init:-0}/1000,1))")
    fi
    echo "$name,$(basename $f),$prec,$fe,$DEV,$t,$size,$st,$med,$avg,$p95,$mem,$init,$diff,$agree" | tee -a "$OUT"
    sleep 20   # let the phone cool between runs
  done
done
$ADB shell rm -f /data/local/tmp/m.tflite /data/local/tmp/benchmark_model
