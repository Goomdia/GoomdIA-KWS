"""Streaming evaluation v3: closed-set (30-class) and open-set (32-class) detectors on two synthetic streams.

Streams (1 h each by default)
  english : the stream of experiments/streaming_eval.py, rebuilt with identical code and random draws
            (speaker-disjoint TEST keywords, gaps of GSC _background_noise_ at -40..-25 dBFS, English GSC test
            words, digits excluded, in half of the gaps). Reproduces Online Resource 1, Sect. S5.
  moore   : same construction, but the out-of-vocabulary speech comes from Moore broadcast excerpts (1-2 s)
            of the held-out TEST documents (open_set_clips.npz, subsource moore_broadcast, split test), and gap
            noise drawn with equal probability from the two GSC noise files held out from the open-set training
            (running_tap, pink_noise) and from a track of held-out field noise (test recording groups).
  moore_dev: as moore, but VALIDATION keywords (speakers), validation Moore documents and validation field noise
            (GSC running_tap / pink_noise again); used only to choose the operating thresholds.
Detector configurations
  w2.0-h1.0          window 2 s, hop 1 s, raw posteriors, refractory 0.5 s      (article)
  w1.5-h0.5          window 1.5 s, hop 0.5 s, raw posteriors, refractory 0.5 s  (article, Online Resource 1)
  w1.5-h0.5-avg3     hop 0.5 s, causal moving average of 3 windows, refractory 1 s
  w1.5-h0.25-avg4    hop 0.25 s, causal moving average of 4 windows (1 s), refractory 1 s
A window emits the arg-max class when it is a keyword and its (smoothed) posterior exceeds the threshold.
For 32-class models, _background_ / _unknown_ are never emitted (an arg-max on them means no emission).
Scoring is that of streaming_eval.py: an emission whose span overlaps a keyword occurrence of the same label
detects it; otherwise it is a substitution (overlaps a keyword), an FA on OOV speech (overlaps an OOV word)
or an FA on noise. For smoothed posteriors the span is the union of the averaged windows.
Output: <run or --out>/streaming_v3.json (one record per stream x configuration x threshold).
"""
import argparse, glob, json, os, random, sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from train_kws import LeTRKWS, W2V2KWS, Pool, SR

CONFIGS = [  # name, window (s), hop (s), smoothing (windows), refractory (s)
    ('w2.0-h1.0', 2.0, 1.0, 1, 0.5), ('w1.5-h0.5', 1.5, 0.5, 1, 0.5),
    ('w1.5-h0.5-avg3', 1.5, 0.5, 3, 1.0), ('w1.5-h0.25-avg4', 1.5, 0.25, 4, 1.0)]
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995,
              0.9999, 0.99999]
TEST_NOISE = ('running_tap', 'pink_noise')


def build_stream(pool, test_idx, noises, oov_waves, minutes, seed):
    """Verbatim stream construction of streaming_eval.py (same rng calls in the same order).
    oov_waves: callable(rng) -> waveform of one OOV word."""
    rng = random.Random(seed); np.random.seed(seed)

    def noise(n, level):
        z = rng.choice(noises); st = rng.randint(0, len(z) - n - 1)
        seg = z[st:st + n]
        return seg / (np.sqrt((seg ** 2).mean()) + 1e-8) * level
    total = int(minutes * 60 * SR)
    audio, events, pos = [], [], 0
    while pos < total:
        gap = int(rng.uniform(1.5, 4.0) * SR)
        lvl = 10 ** (rng.uniform(-40, -25) / 20)
        g = noise(gap, lvl)
        if rng.random() < 0.5:
            w = oov_waves(rng)
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
    return np.concatenate(audio).astype(np.float32), events


def english_stream(a, pool, test_idx):
    noises = [sf.read(f, dtype='float32')[0] for f in glob.glob(os.path.join(a.gsc, '_background_noise_', '*.wav'))]
    oov = [os.path.join(a.gsc, l.strip()) for l in open(os.path.join(a.gsc, 'testing_list.txt'))]
    oov = [p for p in oov if p.split('/')[-2] not in ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')]
    return build_stream(pool, test_idx, noises, lambda rng: sf.read(rng.choice(oov), dtype='float32')[0], a.minutes, a.seed)


def field_noise_track(z, seed, split='test'):
    """Held-out field noise (recording groups of `split`) as one track: clips normalised to unit RMS, shuffled and
    joined with 20 ms linear cross-fades."""
    sel = np.where((z['subsource'] == 'field_noise') & (z['split'] == split))[0]
    random.Random(seed + 1).shuffle(sel := list(sel))
    au, of, le = z['audio'], z['offsets'], z['lengths']
    fade = int(0.02 * SR); r = np.linspace(0, 1, fade, dtype=np.float32)
    out = np.zeros(0, np.float32)
    for j in sel:
        c = au[of[j]:of[j] + le[j]].astype(np.float32)
        c = c / (np.sqrt((c ** 2).mean()) + 1e-8)
        if len(out):
            out[-fade:] = out[-fade:] * (1 - r) + c[:fade] * r; c = c[fade:]
        out = np.concatenate([out, c])
    return out


def moore_stream(a, pool, test_idx, split='test'):
    noises = [sf.read(os.path.join(a.gsc, '_background_noise_', f'{n}.wav'), dtype='float32')[0] for n in TEST_NOISE]
    z = np.load(a.extra, allow_pickle=True)
    noises.append(field_noise_track(z, a.seed, split))
    sel = np.where((z['subsource'] == 'moore_broadcast') & (z['split'] == split))[0]
    au, of, le = z['audio'], z['offsets'], z['lengths']
    waves = [au[of[j]:of[j] + le[j]].copy() for j in sel]
    return build_stream(pool, test_idx, noises, lambda rng: rng.choice(waves), a.minutes, a.seed)


def posteriors(model, x, win, hop, dev, bs):
    W, H = int(win * SR), int(hop * SR)
    starts = np.arange(0, len(x) - W, H)
    out = []
    with torch.no_grad():
        for b in range(0, len(starts), bs):
            seg = np.stack([x[st:st + W] for st in starts[b:b + bs]])
            lo, hi = seg.min(1, keepdims=True), seg.max(1, keepdims=True)
            seg = 2 * (seg - lo) / (hi - lo + 1e-8) - 1
            t = torch.from_numpy(seg).to(dev)
            lens = torch.full((len(seg),), W)
            out.append(model(t, lens).float().softmax(-1).cpu().numpy())
    return starts, np.concatenate(out), W


def score(starts, P, W, events, hours, n_kw_classes, smooth, refractory, thr):
    kw = np.array([(s, e, l) for s, e, l in events if l >= 0])
    oov = np.array([(s, e) for s, e, l in events if l < 0]).reshape(-1, 2)
    if smooth > 1:
        c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), 0)
        Ps = np.vstack([c[i + 1] - c[max(0, i + 1 - smooth)] for i in range(len(P))])
        Ps /= np.minimum(np.arange(1, len(P) + 1), smooth)[:, None]
        span0 = starts[np.maximum(0, np.arange(len(P)) - smooth + 1)]
    else:
        Ps, span0 = P, starts
    cls, val = Ps.argmax(1), Ps.max(1)
    last, emits = -1e9, []
    for t in np.where((cls < n_kw_classes) & (val > thr))[0]:
        st = starts[t]
        if (st - last) / SR > refractory:
            emits.append((span0[t], st + W, cls[t])); last = st
    hit = np.zeros(len(kw), bool); fa_noise = fa_oov = subs = 0
    for es, ee, c in emits:
        lo, hi = np.searchsorted(kw[:, 1], es, 'right'), np.searchsorted(kw[:, 0], ee, 'left')
        ov = [j for j in range(lo, hi) if es < kw[j, 1] and ee > kw[j, 0]]
        good = [j for j in ov if kw[j, 2] == c]
        if good:
            hit[good] = True
        elif ov:
            subs += 1
        elif len(oov) and np.any((es < oov[:, 1]) & (ee > oov[:, 0])):
            fa_oov += 1
        else:
            fa_noise += 1
    n_oov = len(oov)
    return dict(threshold=thr, detection_rate=float(hit.mean()), wrong_kw_per_kw=subs / len(kw),
                fa_oov_per_oov_word=fa_oov / max(1, n_oov), substitutions_per_h=subs / hours,
                fa_oov_per_h=fa_oov / hours, fa_noise_per_h=fa_noise / hours,
                fa_nonkw_per_h=(fa_oov + fa_noise) / hours, fa_total_per_h=(subs + fa_oov + fa_noise) / hours,
                n_emissions=len(emits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run directory containing model.pt and summary.json')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--gsc', required=True)
    ap.add_argument('--extra', required=True, help='open_set_clips.npz (Moore test documents)')
    ap.add_argument('--streams', default='english,moore,moore_dev')
    ap.add_argument('--minutes', type=float, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save_stream', default='', help='optional folder for the stream audio/events (npz)')
    ap.add_argument('--out', default='', help='output folder (default: the run directory)')
    a = ap.parse_args()

    s = json.load(open(os.path.join(a.run, 'summary.json')))
    pool = Pool(a.cache)
    k = len(pool.dictionary)
    K = len(s.get('classes', pool.dictionary))
    model = W2V2KWS(K) if s['args']['model'] == 'w2v2' else LeTRKWS(s['args']['model'], K)
    model.load_state_dict(torch.load(os.path.join(a.run, 'model.pt'), map_location='cpu'))
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu'); model.to(dev).eval()
    torch.set_num_threads(12)
    bs = 32 if s['args']['model'] == 'w2v2' else 256
    sp = pd.read_csv(os.path.join(HERE, '..', '..', 'data', 'splits', 'splits_usable.csv'))
    test_idx = [pool.index[o] for o in sp.oid[sp[s['args']['split']] == 'test']]
    valid_idx = [pool.index[o] for o in sp.oid[sp[s['args']['split']] == 'valid']]

    res, info = [], {}
    for stream in a.streams.split(','):
        if stream == 'english': x, events = english_stream(a, pool, test_idx)
        elif stream == 'moore': x, events = moore_stream(a, pool, test_idx, 'test')
        else: x, events = moore_stream(a, pool, valid_idx, 'valid')   # moore_dev: threshold selection
        hours = len(x) / SR / 3600
        info[stream] = dict(hours=hours, keywords=sum(1 for e in events if e[2] >= 0),
                            oov_words=sum(1 for e in events if e[2] < 0),
                            oov_hours=sum(e[1] - e[0] for e in events if e[2] < 0) / SR / 3600)
        if a.save_stream:
            os.makedirs(a.save_stream, exist_ok=True)
            np.savez(os.path.join(a.save_stream, f'stream_{stream}_seed{a.seed}.npz'), audio=x, events=np.array(events))
        cache = {}
        for name, win, hop, smooth, refr in CONFIGS:
            if (win, hop) not in cache: cache[(win, hop)] = posteriors(model, x, win, hop, dev, bs)
            starts, P, W = cache[(win, hop)]
            for thr in THRESHOLDS:
                r = score(starts, P, W, events, hours, k, smooth, refr, thr)
                res.append(dict(stream=stream, config=name, window_s=win, hop_s=hop, smooth=smooth, refractory_s=refr, **r))
        print(stream, info[stream], flush=True)
    out = dict(run=os.path.basename(a.run.rstrip('/')), n_classes=K, streams=info, seed=a.seed, device=str(dev),
               results=res)
    os.makedirs(a.out or a.run, exist_ok=True)
    json.dump(out, open(os.path.join(a.out or a.run, 'streaming_v3.json'), 'w'), indent=1)
    R = pd.DataFrame(res)
    print(R[R.threshold.isin([0.5, 0.7, 0.9, 0.99])][['stream', 'config', 'threshold', 'detection_rate', 'wrong_kw_per_kw',
                                                      'fa_oov_per_oov_word', 'fa_noise_per_h', 'fa_nonkw_per_h']].round(3).to_string())


if __name__ == '__main__':
    main()
