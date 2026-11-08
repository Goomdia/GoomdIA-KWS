"""Desktop (high-end tier) deployment cost: median latency of one 1.5 s window (feature extraction included, as
footprint_big.py), fp32 and PyTorch dynamic int8 (nn.Linear -> qint8), at 1, 4 and N threads, peak RSS, size on disk.
Each model x precision runs in a fresh subprocess so that peak RSS (ru_maxrss) is clean.
Timing holds an exclusive flock on gpu.lock (same lock as the GPU training jobs), so nothing else heavy runs.
  nice -n 10 python desktop_bench.py --ckpt <model.pt> [--ckpt ...] [--threads 1 4 16] [--acc-csv data/deploy_desktop_accuracy.csv]
Output data/deploy_desktop.csv (appended): one row per ckpt x precision x threads. N threads capped at 16.
"""
import argparse, fcntl, json, os, subprocess, sys, tempfile, time, resource
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', action='append', required=True, help='model.pt or random:ARCH[:K] (architecture only)')
ap.add_argument('--arch', default=None); ap.add_argument('--factory', default=None)
ap.add_argument('--threads', type=int, nargs='+', default=[1, 4, 16])
ap.add_argument('--precision', nargs='+', default=['fp32', 'int8'])
ap.add_argument('--runs', type=int, default=30); ap.add_argument('--warmup', type=int, default=5)
ap.add_argument('--out', default=os.path.join(C.ROOT, 'data', 'deploy_desktop.csv'))
ap.add_argument('--acc-csv', default=os.path.join(C.ROOT, 'data', 'deploy_desktop_accuracy.csv'))
ap.add_argument('--no-lock', action='store_true')
ap.add_argument('--_child', default=None)
a = ap.parse_args()


def child(spec):
    import torch
    torch.set_num_threads(1)
    ck, arch, factory, prec, ths, runs, warm = spec
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    model, arch, k = C.load_model(ck, arch, factory)
    if prec == 'int8':
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    sd = tempfile.mktemp(suffix='.pt'); torch.save(model.state_dict(), sd)     # size on disk of the state dict
    size = os.path.getsize(sd) / 2 ** 20; os.remove(sd)
    x = C.reference_input(model, 1.5, 0); lens = torch.tensor([x.shape[1]])
    res = {}
    for th in ths:
        torch.set_num_threads(th)
        with torch.inference_mode():
            for _ in range(warm): model(x, lens)
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter(); model(x, lens); ts.append(time.perf_counter() - t0)
        ts.sort(); res[th] = [1000 * ts[len(ts) // 2], 1000 * ts[int(.1 * len(ts))], 1000 * ts[int(.9 * len(ts))]]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print('RESULT' + json.dumps(dict(res=res, size_mb=size, peak_rss_mb=peak, base_rss_mb=base, arch=arch,
                                     params=sum(p.numel() for p in model.parameters()))))


if a._child:
    child(json.loads(a._child)); sys.exit()

import pandas as pd
acc = pd.read_csv(a.acc_csv) if os.path.exists(a.acc_csv) else None
rows = []
lk = open(C.LOCK, 'a')
if not a.no_lock:
    print('waiting for', C.LOCK, flush=True); fcntl.flock(lk, fcntl.LOCK_EX)
cpu = open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': \t')
try:
    for ck in a.ckpt:
        ck = ck if C.is_random(ck) else os.path.abspath(ck); rel = ck if C.is_random(ck) else os.path.relpath(ck, C.ROOT)
        for prec in a.precision:
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1')
            spec = [ck, a.arch, a.factory, prec, a.threads, a.runs, a.warmup]
            out = subprocess.run(['nice', '-n', '10', sys.executable, os.path.abspath(__file__), '--ckpt', ck, '--_child', json.dumps(spec)],
                                 capture_output=True, text=True, env=env, cwd=C.HERE).stdout
            line = [l for l in out.splitlines() if l.startswith('RESULT')]
            if not line: print('FAILED', rel, prec, out[-300:]); continue
            r = json.loads(line[0][6:])
            ac = acc[(acc.ckpt == rel) & (acc.precision == prec)] if acc is not None else []
            for th, (med, p10, p90) in r['res'].items():
                rows.append(dict(ckpt=rel, arch=r['arch'], precision=prec, threads=int(th), params=r['params'],
                                 latency_ms_median=round(med, 2), latency_ms_p10=round(p10, 2), latency_ms_p90=round(p90, 2),
                                 size_mb_state_dict=round(r['size_mb'], 2), peak_rss_mb=round(r['peak_rss_mb'], 0),
                                 baseline_rss_mb=round(r['base_rss_mb'], 0), window_s=1.5, frontend_included=True, cpu=cpu,
                                 test_acc=float(ac.test_acc.iloc[0]) if len(ac) else 'NA',
                                 delta_acc_vs_fp32_pp=float(ac.delta_acc_vs_fp32_pp.iloc[0]) if len(ac) else 'NA'))
                print(rows[-1], flush=True)
finally:
    if rows:
        df = pd.DataFrame(rows)
        if os.path.exists(a.out): df = pd.concat([pd.read_csv(a.out), df])
        df.to_csv(a.out, index=False)
