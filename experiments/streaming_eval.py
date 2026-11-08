"""Simulated streaming evaluation of the sliding-window detector used in the demonstrator.

Synthetic streams are built from speaker-disjoint TEST keywords separated by gaps of background
noise (GSC _background_noise_, random SNR), half of the gaps also containing an out-of-vocabulary
spoken word (English GSC test words). The detector classifies a window every `hop` seconds and emits
the arg-max keyword when its posterior exceeds `thr` (refractory period 0.5 s), as in the app.
A keyword occurrence is *detected* if the correct keyword is emitted by a window overlapping it;
any other emission is a false alarm (wrong keyword on a keyword = substitution, or emission over a
gap = false alarm on noise / OOV speech). Reports detection rate and false alarms per hour.
"""
import argparse, glob, json, os, random, sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_kws import LeTRKWS, W2V2KWS, Pool, SR, SSL

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='run directory containing model.pt and summary.json')
ap.add_argument('--cache', required=True)
ap.add_argument('--gsc', required=True, help='speech_commands_v0.02 folder')
ap.add_argument('--minutes', type=float, default=60)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed); np.random.seed(a.seed)
HERE = os.path.dirname(os.path.abspath(__file__))

s = json.load(open(os.path.join(a.run, 'summary.json')))
pool = Pool(a.cache)
k = len(pool.dictionary)
model = W2V2KWS(k, SSL[s['args']['model']]) if s['args']['model'] in SSL else LeTRKWS(s['args']['model'], k)
model.load_state_dict(torch.load(os.path.join(a.run, 'model.pt'), map_location='cpu'))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev).eval()
torch.set_num_threads(12)
BS = 32 if s['args']['model'] in SSL else 256

sp = pd.read_csv(os.path.join(HERE, '..', 'data', 'splits', 'splits_usable.csv'))
test_idx = [pool.index[o] for o in sp.oid[sp[s['args']['split']] == 'test']]
noises = [sf.read(f, dtype='float32')[0] for f in glob.glob(os.path.join(a.gsc, '_background_noise_', '*.wav'))]
oov = [os.path.join(a.gsc, l.strip()) for l in open(os.path.join(a.gsc, 'testing_list.txt'))]
oov = [p for p in oov if p.split('/')[-2] not in ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')]

def noise(n, level):
    z = rng.choice(noises); st = rng.randint(0, len(z) - n - 1)
    seg = z[st:st + n]
    return seg / (np.sqrt((seg ** 2).mean()) + 1e-8) * level

# ---- build the stream ---------------------------------------------------------------
total = int(a.minutes * 60 * SR)
audio, events = [], []          # events: (start, end, label or -1 for OOV)
pos = 0
while pos < total:
    gap = int(rng.uniform(1.5, 4.0) * SR)
    lvl = 10 ** (rng.uniform(-40, -25) / 20)                 # background RMS (re full scale)
    g = noise(gap, lvl)
    if rng.random() < 0.5:                                     # out-of-vocabulary speech in the gap
        w, _ = sf.read(rng.choice(oov), dtype='float32')
        w = w / (np.abs(w).max() + 1e-8) * 0.5
        o = rng.randint(0, max(0, gap - len(w)))
        g[o:o + len(w)] += w[:gap - o]
        events.append((pos + o, pos + o + min(len(w), gap - o), -1))
    audio.append(g); pos += gap
    i = rng.choice(test_idx)
    w = pool.wave(i).copy()
    w = w / (np.abs(w).max() + 1e-8) * rng.uniform(0.3, 0.9)
    w = w + noise(len(w), lvl)
    events.append((pos, pos + len(w), int(pool.labels[i])))
    audio.append(w.astype(np.float32)); pos += len(w)
x = np.concatenate(audio).astype(np.float32)
kw = [(st, en, l) for st, en, l in events if l >= 0]
hours = len(x) / SR / 3600

# ---- run detectors -------------------------------------------------------------------
def posteriors(win, hop):
    W, H = int(win * SR), int(hop * SR)
    starts = np.arange(0, len(x) - W, H)
    out = []
    with torch.no_grad():
        for b in range(0, len(starts), BS):
            seg = np.stack([x[st:st + W] for st in starts[b:b + BS]])
            lo, hi = seg.min(1, keepdims=True), seg.max(1, keepdims=True)
            seg = 2 * (seg - lo) / (hi - lo + 1e-8) - 1
            t = torch.from_numpy(seg).to(dev)
            lens = torch.full((len(seg),), W)
            out.append(model(t, lens).float().softmax(-1).cpu().numpy())
    return starts, np.concatenate(out), W

res = []
for win, hop in ((2.0, 1.0), (1.5, 0.5)):
    starts, P, W = posteriors(win, hop)
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
        last, emits = -1e9, []
        for st, p in zip(starts, P):
            c, v = int(p.argmax()), float(p.max())
            if v > thr and (st - last) / SR > 0.5:
                emits.append((st, st + W, c)); last = st
        hit = np.zeros(len(kw), bool); fa_noise = fa_oov = subs = 0
        for es, ee, c in emits:
            ov = [j for j, (ks, ke, l) in enumerate(kw) if es < ke and ee > ks]
            if any(kw[j][2] == c for j in ov):
                for j in ov:
                    if kw[j][2] == c: hit[j] = True
            elif ov:
                subs += 1
            elif any(es < oe and ee > os_ for os_, oe, l in events if l < 0):
                fa_oov += 1
            else:
                fa_noise += 1
        res.append(dict(window_s=win, hop_s=hop, threshold=thr, detection_rate=float(hit.mean()),
                        substitutions_per_h=subs / hours, fa_oov_per_h=fa_oov / hours, fa_noise_per_h=fa_noise / hours,
                        fa_total_per_h=(subs + fa_oov + fa_noise) / hours))
out = dict(run=os.path.basename(a.run.rstrip('/')), hours=hours, keywords=len(kw),
           oov_words=sum(1 for e in events if e[2] < 0), results=res)
json.dump(out, open(os.path.join(a.run, 'streaming_eval.json'), 'w'), indent=1)
print(pd.DataFrame(res).round(3).to_string())
