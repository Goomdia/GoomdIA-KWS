"""footprint.py method (params, CPU latency for one 1.5 s window, median of 30 after 5 warm-up calls) applied to
the fine-tuned large encoders; w2v2 is re-measured as a control against experiments/footprint.json."""
import json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train_kws import W2V2KWS, SSL

torch.manual_seed(0)
res = {}
x = torch.randn(1, 24000)
lens = torch.tensor([24000])
for name in ('w2v2', 'mms300m'):
    m = W2V2KWS(30, SSL[name]).eval()
    p = sum(t.numel() for t in m.parameters())
    r = dict(params=p, size_mb_fp32=round(p * 4 / 2 ** 20, 2))
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.inference_mode():
            for _ in range(5): m(x, lens)
            ts = []
            for _ in range(30):
                t0 = time.perf_counter(); m(x, lens); ts.append(time.perf_counter() - t0)
        ts.sort()
        r[f'latency_ms_{th}thr_median'] = round(1000 * ts[len(ts) // 2], 2)
    res[name] = r
    print(name, r, flush=True)
res['cpu'] = open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': \t')
json.dump(res, open(os.path.join(HERE, 'footprint_big.json'), 'w'), indent=1)
