"""Detector tuning for continuous keyword detection (models are NOT retrained or modified).

Stages (usage: streaming_tune.py STAGE [options])
  prep        events/hours of the three saved streams (cache/v3/streams/*.npz) + gate features (frame energy, Silero VAD)
  posteriors  window posteriors per (run, stream, window, normalisation) at hop 0.25 s (hop 0.5 s = every 2nd window)
              -> tune_cache/post/*.npy          (w2v2 on GPU: wrap in `flock .../gpu.lock`)
  search      factorial search on the dev stream (moore_dev) only; writes data/streaming_tuned_dev_ops.csv
  final       choose configs on dev, evaluate once on the test streams, demo defaults, ablations
              -> data/streaming_tuned_test.csv, data/streaming_tuned_selected.csv, data/streaming_tuned_ablation.csv
Scoring modes (--mode): 'window' = streaming_v3.py (emission span overlaps a keyword); 'centre' = 1 s span around the centre of
the evidence (strict; HEADLINE: the window mode rewards long windows, outputs suffixed _centre).
Budgets on moore_dev (BUDGETS): A <=5 FA/h outside keywords; B <=1 FA/h; C = A and <=0.05 wrong keywords per keyword.
Scoring is that of streaming_v3.py (an emission overlapping a keyword occurrence of the same label detects it; else
substitution / FA on OOV speech / FA on noise). `check` reproduces streaming_v3 numbers for the old detector.
"""
import argparse, itertools, json, multiprocessing as mp, os, sys, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TC = os.path.join(HERE, 'tune_cache')
STREAMS = os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')) + '/v3/streams'
POOL = os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')) + '/pool.npz'
# Silero VAD TorchScript model (snakers4/silero-vad, file src/silero_vad/data/silero_vad.jit); set SILERO_VAD_JIT.
VAD_JIT = os.environ.get('SILERO_VAD_JIT', os.path.join(os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')), 'silero_vad.jit'))
SR = 16000
DATA = os.path.join(ROOT, 'data')
SNAMES = ['moore_dev', 'moore', 'english']

LETR_CLOSED = 'closed30-letr128t-gsc-speaker-lr0.001-s10'
W2V_CLOSED = 'closed30-w2v2-ssl-speaker-lr3e-05-s0'
LETR_OPEN = [f'open32-letr128t-gsc-speaker-lr0.001-s{i}' for i in range(3)]
W2V_OPEN = ['open32-w2v2-ssl-speaker-lr3e-05-s0']
GROUPS = {'LeTR closed': [LETR_CLOSED], 'LeTR 32 cl. GSC': LETR_OPEN, 'w2v2 closed': [W2V_CLOSED], 'w2v2 32 cl.': W2V_OPEN}
RUNS = [r for g in GROUPS.values() for r in g]
# v4 groups (32-class models per hardware tier); selected with --groups, results are merged into the existing CSVs
_ch = os.path.join(HERE, 'e4_bc8_choice.json')
_bc = json.load(open(_ch)) if os.path.exists(_ch) else dict(bcresnet8_init='gsc', lr=1e-3)
GROUPS_NEW = {
    'BC-ResNet-8 32 cl.': [f"open32-bcresnet8-{_bc['bcresnet8_init']}-speaker-lr{_bc['lr']:g}-s{i}" for i in range(3)],
    'w2v2 32 cl. (2 seeds)': ['open32-w2v2-ssl-speaker-lr3e-05-s0', 'open32-w2v2-ssl-speaker-lr3e-05-s2'],
    'MMS-300M 32 cl.': [f'open32-mms300m-ssl-speaker-lr3e-05-s{i}' for i in range(2)],
    'mHuBERT-147 32 cl.': [f'open32-mhubert-ssl-speaker-lr3e-05-s{i}' for i in range(2)]}
WINDOWS = [1.0, 1.5, 2.0]
HOPS = [0.25, 0.5]
NORMS = ['minmax', 'rms', 'none']
RMS_DB = [-45, -40, -35, -30, -27, -24, -21, -18, -15]
GATES = ['none'] + [f'rms{d}' for d in RMS_DB] + ['vad0.3', 'vad0.5']
THRS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 0.99999]
BUDGETS = [(5.0, 99.0), (1.0, 99.0), (5.0, 0.05)]   # (max FA/h outside keywords, max wrong keywords per keyword)
MODE = 'window'      # scoring span: 'window' (as in streaming_v3.py) or 'centre' (1 s around the evidence centre)
METRICS = ['det', 'wrong_kw', 'fa_oov_word', 'fa_noise_h', 'fa_nonkw_h', 'fa_total_h', 'n_emis', 'subs_h']


def SUF(): return '' if MODE == 'window' else '_centre'


def is_w2v(run): return 'w2v2' in run or 'mms300m' in run or 'mhubert' in run
def norms_for(run): return ['minmax'] if is_w2v(run) else NORMS      # w2v2 standardises each window itself (norm is a no-op)
def post_norm(run, norm): return 'minmax' if is_w2v(run) else norm


def run_dir(name):
    if name.startswith('closed30-'): return os.path.join(ROOT, 'experiments', 'runs', name.split('-', 1)[1])
    return os.path.join(HERE, 'runs', name)


# ===================================================================================================== prep
def prep(a):
    import torch
    os.makedirs(os.path.join(TC, 'gates'), exist_ok=True)
    vad = torch.jit.load(VAD_JIT); vad.eval()
    for s in SNAMES:
        z = np.load(os.path.join(STREAMS, f'stream_{s}_seed0.npz'))
        x = z['audio'].astype(np.float32); ev = z['events']
        np.save(os.path.join(TC, f'events_{s}.npy'), ev)
        json.dump(dict(hours=len(x) / SR / 3600, n=len(x)), open(os.path.join(TC, f'meta_{s}.json'), 'w'))
        # 100 ms frames, 50 ms hop: RMS in dB
        n = len(x) // 800
        e0 = (x[:n * 800].astype(np.float64) ** 2).reshape(n, 800).sum(1)
        fr = 10 * np.log10((e0[:-1] + e0[1:]) / 1600 + 1e-12)
        # Silero VAD: probability per 512-sample chunk
        nc = len(x) // 512
        probs = np.zeros(nc, np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(x[:nc * 512]).view(nc, 512)
            vad.reset_states()
            for i in range(nc): probs[i] = vad(xt[i:i + 1], SR).item()
        out = dict(frame_db=fr.astype(np.float32), vad_prob=probs)
        for thr in (0.3, 0.5):
            out[f'vad{thr}'] = vad_mask(probs, thr)
        np.savez(os.path.join(TC, 'gates', f'{s}.npz'), **out)
        print(s, 'hours', len(x) / SR / 3600, 'speech frac', {t: float(out[f"vad{t}"].mean()) for t in (0.3, 0.5)}, flush=True)


def vad_mask(p, thr, neg_gap=0.15, min_speech=8, min_silence=3, pad=1):
    """Chunk-level speech mask from Silero probabilities (hysteresis thr / thr-0.15, bridge silences < 100 ms,
    drop speech < 250 ms, pad 30 ms)  -- the logic of silero_vad.get_speech_timestamps."""
    neg = max(thr - neg_gap, 0.01)
    m = np.zeros(len(p), bool); on = False
    for i, v in enumerate(p):
        if not on and v >= thr: on = True
        elif on and v < neg: on = False
        m[i] = on
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    segs = []
    for s_, e_ in zip(st, en):
        if segs and s_ - segs[-1][1] < min_silence: segs[-1][1] = e_
        else: segs.append([s_, e_])
    out = np.zeros(len(p), bool)
    for s_, e_ in segs:
        if e_ - s_ >= min_speech: out[max(0, s_ - pad):e_ + pad] = True
    return out


class Gates:
    def __init__(self, stream):
        g = np.load(os.path.join(TC, 'gates', f'{stream}.npz'))
        self.fr = g['frame_db']; self.vad = {0.3: g['vad0.3'], 0.5: g['vad0.5']}
        self.cache = {}

    def peak_db(self, W, n):
        key = ('p', W, n)
        if key not in self.cache:
            nw = int(W * SR) // 800 - 1
            mx = np.lib.stride_tricks.sliding_window_view(self.fr, nw).max(1)
            j0 = np.arange(n) * 5          # hop-0.25 s windows start every 5 frames of 50 ms
            self.cache[key] = mx[j0]
        return self.cache[key]

    def vad_win(self, thr, W, n):
        key = ('v', thr, W, n)
        if key not in self.cache:
            c = (np.arange(n) * 4000 + int(W * SR) // 2) // 512
            self.cache[key] = self.vad[thr][np.minimum(c, len(self.vad[thr]) - 1)]
        return self.cache[key]

    def get(self, name, W, n):
        if name == 'none': return np.ones(n, bool)
        if name.startswith('rms'): return self.peak_db(W, n) > float(name[3:])
        return self.vad_win(float(name[3:]), W, n)


# =============================================================================================== posteriors
def posteriors_stage(a):
    import torch
    sys.path.insert(0, os.path.join(HERE, '..'))
    from train_kws import LeTRKWS, W2V2KWS, BCResNetKWS, SSL, BC
    os.makedirs(os.path.join(TC, 'post'), exist_ok=True)
    dic = list(np.load(POOL, allow_pickle=True)['dictionary'])
    dev = torch.device('cuda' if a.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(a.threads)
    for run in a.runs.split(','):
        d = run_dir(run); s = json.load(open(os.path.join(d, 'summary.json')))
        K = len(s.get('classes', dic))
        mn = s['args']['model']
        model = W2V2KWS(K, SSL[mn]) if mn in SSL else BCResNetKWS(mn, K) if mn in BC else LeTRKWS(mn, K)
        model.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location='cpu'))
        model.to(dev).eval()
        bs = (16 if ('mms300m' in run or 'mhubert' in run) else 32) if is_w2v(run) else 256
        for st in SNAMES:
            x = np.load(os.path.join(STREAMS, f'stream_{st}_seed0.npz'))['audio'].astype(np.float32)
            for W in WINDOWS:
                for norm in norms_for(run):
                    f = os.path.join(TC, 'post', f'{run}__{st}__w{W}__{norm}.npy')
                    if os.path.exists(f): continue
                    Wn = int(W * SR); starts = np.arange(0, len(x) - Wn, 4000)
                    out = []
                    with torch.no_grad():
                        for b in range(0, len(starts), bs):
                            seg = np.stack([x[t:t + Wn] for t in starts[b:b + bs]])
                            if norm == 'minmax':
                                lo, hi = seg.min(1, keepdims=True), seg.max(1, keepdims=True)
                                seg = 2 * (seg - lo) / (hi - lo + 1e-8) - 1
                            elif norm == 'rms':
                                r = np.sqrt((seg.astype(np.float64) ** 2).mean(1, keepdims=True)).astype(np.float32)
                                seg = seg * (0.11 / np.maximum(r, 1e-8))
                            t = torch.from_numpy(np.ascontiguousarray(seg)).to(dev)
                            lens = torch.full((len(seg),), Wn)
                            out.append(model(t, lens).float().softmax(-1).cpu().numpy())
                    np.save(f, np.concatenate(out).astype(np.float32))
                    print(run, st, W, norm, flush=True)
        del model
        if dev.type == 'cuda': torch.cuda.empty_cache()


# ===================================================================================================== rules
class Ctx:
    """Everything a decision rule needs for one (run, stream, window, hop, norm, gate)."""
    def __init__(self, P, W, hop, k, gate):
        step = int(round(hop / 0.25))
        self.P = P[::step]; self.starts = (np.arange(len(P)) * 4000)[::step]; self.W = int(W * SR); self.hop = hop
        self.k = k; self.gate = gate; self._sm = {}; self._below = None

    def smoothed(self, m):
        if m not in self._sm:
            P = self.P
            if m > 1:
                c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), 0)
                i = np.arange(len(P))
                Ps = (c[i + 1] - c[np.maximum(0, i + 1 - m)]) / np.minimum(i + 1, m)[:, None]
                span0 = self.starts[np.maximum(0, i - m + 1)]
            else:
                Ps, span0 = P, self.starts
            self._sm[m] = (Ps.argmax(1), Ps.max(1), span0)
        return self._sm[m]

    def below(self):
        if self._below is None:
            b = (self.P[:, :self.k] < 0.15).astype(np.int32)
            self._below = np.vstack([np.zeros((1, b.shape[1]), np.int32), np.cumsum(b, 0)])
        return self._below


def _refractory(idx, st, r):
    out, last, lim = [], -10 ** 12, r * SR
    for t in idx:
        s = st[t]
        if s - last > lim: out.append(t); last = s
    return np.array(out, dtype=np.int64)


def rule_emit(rule, ctx, thr):
    """-> (es, ee, cls) arrays of emissions. rule = (family, params...)."""
    fam = rule[0]; st, W, k = ctx.starts, ctx.W, ctx.k
    if fam in ('thr', 'avg'):                           # threshold + refractory (on smoothed posteriors for avg)
        m = 1 if fam == 'thr' else rule[1]; r = rule[-1]
        cls, val, span0 = ctx.smoothed(m)
        idx = np.where((cls < k) & (val > thr) & ctx.gate)[0]
        idx = _refractory(idx, st, r)
        return span0[idx], st[idx] + W, cls[idx]
    if fam == 'kofn':                                   # k-of-n same-word consensus
        kk, n, r = rule[1], rule[2], rule[3]
        cls, val, _ = ctx.smoothed(1)
        idx = np.where((cls < k) & (val > thr) & ctx.gate)[0]
        if len(idx) == 0: return np.zeros(0), np.zeros(0), np.zeros(0, int)
        M = np.zeros((len(st), k), np.int16); M[idx, cls[idx]] = 1
        cs = np.vstack([np.zeros((1, k), np.int32), np.cumsum(M, 0, dtype=np.int32)])
        cnt = cs[idx + 1, cls[idx]] - cs[np.maximum(idx + 1 - n, 0), cls[idx]]
        idx = _refractory(idx[cnt >= kk], st, r)
        return st[np.maximum(idx - n + 1, 0)], st[idx] + W, cls[idx]
    if fam == 'hyst':                                   # demo live rule
        cool, refr, low = 1.5 * SR, 0.8 * SR, 0.15
        cls, val, _ = ctx.smoothed(1); cum = ctx.below()
        idx = np.where((cls < k) & (val > thr) & ctx.gate)[0]
        out, last_any, lastfire = [], -10 ** 12, {}
        for t in idx:
            s, c = st[t], cls[t]
            if s - last_any < cool: continue
            tf = lastfire.get(c)
            if tf is not None and (s - st[tf] < refr or cum[t + 1, c] - cum[tf + 1, c] <= 0): continue
            out.append(t); lastfire[c] = t; last_any = s
        idx = np.array(out, dtype=np.int64)
        return st[idx], st[idx] + W, cls[idx]
    if fam == 'merge':                                  # demo batch rule
        cls, val, _ = ctx.smoothed(1)
        if 'merge' not in ctx._sm:
            idx = np.where((cls < k) & (val >= 0.20) & ctx.gate)[0]
            if len(idx) == 0:
                ctx._sm['merge'] = (idx, idx, idx)
            else:
                c, s = cls[idx], st[idx]
                new = np.ones(len(idx), bool)
                new[1:] = (c[1:] != c[:-1]) | (s[1:] - (s[:-1] + W) > 0.5 * SR)
                gid = np.cumsum(new)
                order = np.lexsort((-val[idx], gid))
                first = order[np.r_[True, np.diff(gid[order]) > 0]]
                best = idx[first]; best = best[np.argsort(st[best])]
                ctx._sm['merge'] = (best, val[best], cls[best])
        best, bv, bc = ctx._sm['merge']
        keep = bv >= thr
        return st[best[keep]], st[best[keep]] + W, bc[keep]
    raise ValueError(rule)


RULES = ([('thr', r) for r in (0.5, 1.0)] + [('avg', m, r) for m in (2, 3, 4) for r in (0.5, 1.0)]
         + [('kofn', kk, n, r) for kk, n in ((2, 3), (2, 4), (3, 4)) for r in (0.5, 1.0)] + [('hyst',), ('merge',)])


def rule_name(r): return '-'.join(str(x) for x in r)


# ==================================================================================================== scoring
class Stream:
    def __init__(self, name):
        ev = np.load(os.path.join(TC, f'events_{name}.npy'))
        self.hours = json.load(open(os.path.join(TC, f'meta_{name}.json')))['hours']
        kw = ev[ev[:, 2] >= 0]; oov = ev[ev[:, 2] < 0]
        self.ks, self.ke, self.kl = kw[:, 0], kw[:, 1], kw[:, 2]
        self.os_, self.oe = oov[:, 0], oov[:, 1]
        self.nkw, self.noov = len(kw), len(oov)


def score(S, es, ee, c, mode='window'):
    if mode == 'centre':                                # fixed 1 s span around the centre of the evidence
        ctr = (es + ee) / 2; es, ee = ctr - 8000, ctr + 8000
    n = len(es)
    if n == 0: return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    lo = np.searchsorted(S.ke, es, 'right'); hi = np.searchsorted(S.ks, ee, 'left')
    nov = np.maximum(hi - lo, 0)
    hit = np.zeros(S.nkw, bool); good = np.zeros(n, bool)
    for j in range(int(nov.max())):
        v = np.where(nov > j)[0]
        ii = lo[v] + j; m = S.kl[ii] == c[v]
        hit[ii[m]] = True; good[v[m]] = True
    subs = int(((nov > 0) & ~good).sum())
    rest = nov == 0
    inoov = np.searchsorted(S.oe, es, 'right') < np.searchsorted(S.os_, ee, 'left')
    fa_oov = int((rest & inoov).sum()); fa_noise = int((rest & ~inoov).sum())
    h = S.hours
    return (hit.mean(), subs / S.nkw, fa_oov / max(1, S.noov), fa_noise / h, (fa_oov + fa_noise) / h,
            (subs + fa_oov + fa_noise) / h, n, subs / h)


# ===================================================================================================== search
def load_post(run, stream, W, norm):
    return np.load(os.path.join(TC, 'post', f'{run}__{stream}__w{W}__{post_norm(run, norm)}.npy'), mmap_mode='r')


def n_kw(run): return 30


def search_run(run):
    S = Stream('moore_dev'); G = Gates('moore_dev'); rows = []
    for norm in norms_for(run):
        for W in WINDOWS:
            P = np.asarray(load_post(run, 'moore_dev', W, norm))
            for hop in HOPS:
                for gate in GATES:
                    g = G.get(gate, W, len(P))[::int(round(hop / 0.25))]
                    ctx = Ctx(P, W, hop, n_kw(run), g)
                    for rule in RULES:
                        res = []
                        for thr in THRS:
                            res.append(score(S, *rule_emit(rule, ctx, thr), mode=MODE))
                        res = np.array(res)
                        for bud, wcap in BUDGETS:
                            ok = np.where((res[:, 4] <= bud) & (res[:, 1] <= wcap))[0]
                            if len(ok) == 0:
                                rows.append((run, norm, gate, W, hop, rule_name(rule), bud, wcap, np.nan) + (np.nan,) * 8); continue
                            j = ok[np.lexsort((-np.array(THRS)[ok], res[ok, 1], -res[ok, 0]))[0]]
                            rows.append((run, norm, gate, W, hop, rule_name(rule), bud, wcap, THRS[j]) + tuple(res[j]))
    print('done', run, flush=True)
    return rows


COLS = ['run', 'norm', 'gate', 'window', 'hop', 'rule', 'budget', 'wcap', 'thr'] + METRICS


def search_stage(a):
    runs = a.runs.split(',') if a.runs else RUNS
    with mp.Pool(min(len(runs), a.procs)) as p:
        out = p.map(search_run, runs, chunksize=1)
    df = pd.DataFrame([r for o in out for r in o], columns=COLS)
    f0 = os.path.join(DATA, f'streaming_tuned_dev_ops{SUF()}.csv')
    if a.groups and os.path.exists(f0): df = pd.concat([pd.read_csv(f0).query('run not in @runs'), df])
    df.to_csv(os.path.join(DATA, f'streaming_tuned_dev_ops{SUF()}.csv'), index=False)
    print(len(df), 'rows')


# ======================================================================================================= final
def evaluate(run, stream, norm, gate, W, hop, rule, thr, mode=None, _cache={}):
    mode = mode or MODE
    key = (run, stream, W, norm)
    if key not in _cache:
        if len(_cache) > 12: _cache.clear()
        _cache[key] = np.asarray(load_post(run, stream, W, norm))
    P = _cache[key]
    if ('G', stream) not in _cache: _cache[('G', stream)] = Gates(stream)
    if ('S', stream) not in _cache: _cache[('S', stream)] = Stream(stream)
    g = _cache[('G', stream)].get(gate, W, len(P))[::int(round(hop / 0.25))]
    ctx = Ctx(P, W, hop, n_kw(run), g)
    return dict(zip(METRICS, score(_cache[('S', stream)], *rule_emit(parse_rule(rule), ctx, thr), mode=mode)))


def parse_rule(s):
    p = s.split('-'); return tuple([p[0]] + [float(x) if '.' in x else int(x) for x in p[1:]])


CFG = ['norm', 'gate', 'window', 'hop', 'rule']
DEMO = [  # fixed rules, no tuning: name, norm, gate, window, hop, rule, threshold (None = swept on dev for the batch-rule keep)
    ('demo-live W1.5', 'rms', 'vad0.5', 1.5, 0.5, 'hyst', 0.90), ('demo-live W1.0', 'rms', 'vad0.5', 1.0, 0.5, 'hyst', 0.90),
    ('demo-batch W1.5', 'rms', 'vad0.5', 1.5, 0.5, 'merge', 0.95), ('demo-batch W1.0', 'rms', 'vad0.5', 1.0, 0.5, 'merge', 0.95)]
OLD = [('old w2.0-h1.0', 'minmax', 'none', 2.0, 1.0, 'thr-0.5'), ('old w1.5-h0.5', 'minmax', 'none', 1.5, 0.5, 'thr-0.5')]


def tune_thr(run, c, bud):
    """dev threshold for a config not in the search grid (same criterion as the search). bud = (fa_max, wrong_max)."""
    res = np.array([[evaluate(run, 'moore_dev', c['norm'], c['gate'], c['window'], c['hop'], c['rule'], t)[m] for m in METRICS]
                    for t in THRS])
    ok = np.where((res[:, 4] <= bud[0]) & (res[:, 1] <= bud[1]))[0]
    if len(ok) == 0: return np.nan
    return THRS[ok[np.lexsort((-np.array(THRS)[ok], res[ok, 1], -res[ok, 0]))[0]]]


def pick(ops, runs, bud):
    """config maximising the mean (over runs) dev detection at the budget (threshold chosen per run on dev)."""
    d = ops[(ops.budget == bud[0]) & (ops.wcap == bud[1]) & ops.run.isin(runs)]
    g = d.groupby(CFG).agg(det=('det', 'mean'), wrong=('wrong_kw', 'mean'), fa=('fa_nonkw_h', 'mean'), n=('det', 'count'),
                           nan=('thr', lambda x: x.isna().sum())).reset_index()
    g = g[(g.n == len(runs)) & (g.nan == 0)]
    return g.sort_values(['det', 'wrong', 'fa'], ascending=[False, True, True])


A, B, C = BUDGETS     # A: <=5 FA/h (spec); B: <=1 FA/h; C: <=5 FA/h and <=0.05 wrong keywords per keyword


def final_stage(a):
    ops = pd.read_csv(os.path.join(DATA, f'streaming_tuned_dev_ops{SUF()}.csv'))
    sel, tst = [], []

    def run_cfg(group, label, cfgrow, bud, kind, mode=None, fixed_thr=None):
        """evaluate a config for every run of the group at the dev-chosen threshold; append to tst."""
        for run in GROUPS[group]:
            c = cfgrow
            if fixed_thr is None:
                o = ops[(ops.run == run) & (ops.budget == bud[0]) & (ops.wcap == bud[1]) & (ops.norm == c['norm']) &
                        (ops.gate == c['gate']) & (ops.window == c['window']) & (ops.hop == c['hop']) & (ops.rule == c['rule'])]
                thr = o.thr.iloc[0] if len(o) else tune_thr(run, c, bud)
            else: thr = fixed_thr
            for st in SNAMES:
                if np.isnan(thr): r = {m: np.nan for m in METRICS}
                else: r = evaluate(run, st, c['norm'], c['gate'], c['window'], c['hop'], c['rule'], thr, mode)
                tst.append(dict(group=group, run=run, label=label, kind=kind, budget=bud[0], wcap=bud[1], mode=mode or MODE,
                                stream=st, thr=thr, **{k: c[k] for k in CFG}, **r))

    for group, runs in GROUPS.items():
        chosen = {}
        for bud in BUDGETS:
            g = pick(ops, runs, bud)
            best = g.iloc[0]
            sel.append(dict(group=group, budget=bud[0], wcap=bud[1], **best.to_dict()))
            chosen[bud] = {k: best[k] for k in CFG}
            run_cfg(group, 'tuned', chosen[bud], bud, 'tuned')
            run_cfg(group, 'tuned', chosen[bud], bud, 'tuned-xmode', 'centre' if MODE == 'window' else 'window')
            if bud != B:                        # best config per window length (dev-selected)
                for W in WINDOWS:
                    gw = g[g.window == W]
                    if len(gw): run_cfg(group, f'tuned W={W}', {k: gw.iloc[0][k] for k in CFG}, bud, 'tuned-per-window')
        # old detector and demo defaults (old: threshold tuned on dev at 5 FA/h; demo: fixed rule, fixed threshold)
        for name, norm, gate, W, hop, rule in OLD:
            run_cfg(group, name, dict(norm=norm, gate=gate, window=W, hop=hop, rule=rule), A, 'old')
        for name, norm, gate, W, hop, rule, thr in DEMO:
            run_cfg(group, name, dict(norm=norm, gate=gate, window=W, hop=hop, rule=rule), A, 'demo', fixed_thr=thr)
            if W == 1.5:                        # same demo rule with a dev-tuned threshold
                run_cfg(group, name + ' (thr tuned)', dict(norm=norm, gate=gate, window=W, hop=hop, rule=rule), A, 'demo-tuned')
        # ablations (thresholds re-tuned on dev for every variant): add one chosen factor to the old detector,
        # drop one chosen factor from the chosen config, and best config within each factor level
        old = dict(norm='minmax', gate='none', window=1.5, hop=0.5, rule='thr-0.5')
        for bud in (A, C):
            ch = chosen[bud]; tag = 'A' if bud == A else 'C'
            for fac in ['norm', 'gate', 'window', 'rule']:
                c = dict(old); c[fac] = ch[fac]
                if fac == 'window': c['hop'] = ch['hop']
                run_cfg(group, f'old+{fac}', c, bud, 'abl-add')
                c = dict(ch); c[fac] = old[fac]
                if fac == 'window': c['hop'] = old['hop']
                run_cfg(group, f'chosen-{fac}', c, bud, 'abl-drop')
            g = pick(ops, runs, bud)
            g['fam'] = g.rule.str.split('-').str[0]; g['gfam'] = g.gate.str.replace(r'[-0-9.]+$', '', regex=True)
            for col, lv in [('norm', NORMS), ('gfam', ['none', 'rms', 'vad']), ('window', WINDOWS), ('hop', HOPS),
                            ('fam', ['thr', 'avg', 'kofn', 'hyst', 'merge'])]:
                for v in lv:
                    gg = g[g[col] == v]
                    if len(gg): run_cfg(group, f'best | {col}={v}', {k: gg.iloc[0][k] for k in CFG}, bud, 'abl-level')
        print('done', group, flush=True)
    for nm, rows in (('selected', sel), ('test', tst)):
        f0 = os.path.join(DATA, f'streaming_tuned_{nm}{SUF()}.csv'); df = pd.DataFrame(rows)
        if a.groups and os.path.exists(f0): df = pd.concat([pd.read_csv(f0).query('group not in @GROUPS'), df])
        df.to_csv(f0, index=False)


def check_stage(a):
    """reproduce streaming_v3 old-detector numbers with this scorer (dev + test)."""
    old = pd.read_csv(os.path.join(DATA, 'streaming_v3.csv'))
    bad = 0
    for run in [LETR_CLOSED, LETR_OPEN[0], W2V_CLOSED]:
        for st in SNAMES:
            for cfg, W, hop, thr in [('w1.5-h0.5', 1.5, 0.5, 0.9), ('w2.0-h1.0', 2.0, 1.0, 0.99)]:
                r = old[(old.run == run) & (old.stream == st) & (old.config == cfg) & (old.threshold == thr)].iloc[0]
                m = evaluate(run, st, 'minmax', 'none', W, hop, 'thr-0.5', thr)
                print(run[:22], st, cfg, thr, round(r.detection_rate, 4), round(m['det'], 4), round(r.fa_nonkw_per_h, 2),
                      round(m['fa_nonkw_h'], 2), round(r.wrong_kw_per_kw, 4), round(m['wrong_kw'], 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--runs', default='')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--mode', default='window')
    ap.add_argument('--groups', default='', help="comma-separated names of v4 groups (GROUPS_NEW): restrict to them and merge into the CSVs")
    a = ap.parse_args()
    MODE = a.mode
    torch_threads = os.environ.get('TORCH_NUM_THREADS')
    if a.groups:
        GROUPS = {g: GROUPS_NEW[g] for g in a.groups.split(',')}; RUNS = [r for g in GROUPS.values() for r in g]
        if not a.runs: a.runs = ','.join(RUNS)
    {'prep': prep, 'posteriors': posteriors_stage, 'search': search_stage, 'final': final_stage, 'check': check_stage}[a.stage](a)
