"""Parameter count, model size and CPU latency for one 1.5 s analysis window (feature extraction
included), as used by the streaming detector (window 1.5-2 s, hop 1 s)."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_kws import LeTRKWS, W2V2KWS

torch.manual_seed(0)
res = {}
x = torch.randn(1, 24000)
lens = torch.tensor([24000])
for name in ('letr96', 'letr128t', 'letr256', 'w2v2'):
    m = (W2V2KWS(30) if name == 'w2v2' else LeTRKWS(name, 30)).eval()
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
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'footprint.json'), 'w'), indent=1)
