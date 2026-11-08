"""Shared helpers of the v4 analysis scripts (12-15): run loading, validation-based model selection, cluster bootstrap,
deployment measurements, number formatting (English / French) and the figure style.

Rules (as in 06_results.py): seed 10 = deployment checkpoint, excluded; a run with best_val_acc < 0.1 is a training failure,
excluded; selection is always by mean validation accuracy over seeds, never by test accuracy.
"""
import glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
RUNS = os.path.join(ROOT, 'experiments', 'runs')
RUNS32 = os.path.join(ROOT, 'experiments', 'v3', 'runs')
DATA, FIG = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'figures')
TAB = {'en': os.path.join(ROOT, 'manuscript', 'tables'), 'fr': os.path.join(ROOT, 'manuscript', 'tables_fr')}
C1, C2, C3, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
MIN_SEEDS_FOR_SELECTION = 3        # a configuration confirmed on >= 3 seeds is preferred to single-seed sweep points


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})
    return plt


# ------------------------------------------------------------------ closed-set runs
def load_runs(root=RUNS):
    rows = []
    for f in glob.glob(os.path.join(root, '*', 'summary.json')):
        s = json.load(open(f)); a = s['args']
        d = os.path.dirname(f)
        rows.append(dict(run=os.path.basename(d), dir=d, model=a['model'], init=a['init'], split=a['split'],
                         subset=a.get('train_subset') or 'full', lr=float(a['lr']), seed=int(a['seed']),
                         n_params=s['n_params'], n_train=s.get('n_train', np.nan), val=s['best_val_acc'],
                         test=s['test']['acc'], f1=s['test'].get('macro_f1', np.nan),
                         ood=s.get('ood', {}).get('acc', np.nan)))
    R = pd.DataFrame(rows).sort_values(['model', 'init', 'split', 'subset', 'lr', 'seed']).reset_index(drop=True)
    R = R[R.seed != 10]
    R['failed'] = R.val < 0.1
    return R


R_ALL = load_runs()
R_FAILED = R_ALL[R_ALL.failed]
R = R_ALL[~R_ALL.failed]


def get(model, init=None, lr=None, split='speaker', subset='full', seeds=None, runs=None):
    d = (R if runs is None else runs)
    d = d[(d.model == model) & (d.split == split) & (d.subset == subset)]
    if init is not None: d = d[d.init == init]
    if lr is not None: d = d[np.isclose(d.lr, lr, rtol=1e-6)]
    if seeds is not None: d = d[d.seed.isin(seeds)]
    return d


def configs(model, init=None):
    """Mean validation accuracy per (init, lr) on the speaker split, full data."""
    d = get(model, init)
    return (d.groupby(['init', 'lr']).agg(n=('seed', 'size'), val=('val', 'mean'), test=('test', 'mean'))
            .reset_index().sort_values(['val'], ascending=False))


def select(model, inits):
    """(init, lr) with the highest mean validation accuracy over seeds among `inits`. Configurations with at least
    MIN_SEEDS_FOR_SELECTION seeds are preferred (single-seed LR-sweep points are noisy); `literal` is the choice
    without that restriction, reported when it differs."""
    c = configs(model)
    c = c[c.init.isin(inits)]
    if not len(c): return None
    lit = c.iloc[0]
    cc = c[c.n >= MIN_SEEDS_FOR_SELECTION]
    best = (cc if len(cc) else c).iloc[0]
    return dict(init=best.init, lr=float(best.lr), val=float(best.val), n=int(best.n),
                literal=dict(init=lit.init, lr=float(lit.lr), val=float(lit.val), n=int(lit.n)),
                differs=(lit.init, float(lit.lr)) != (best.init, float(best.lr)),
                edge=float(best.lr) >= c.lr.max())


def select_lr(model, init):
    c = configs(model, init)
    if not len(c): return None
    cc = c[c.n >= MIN_SEEDS_FOR_SELECTION]
    return float((cc if len(cc) else c).iloc[0].lr)


BC_SEL = {m: select(m, ['scratch', 'mswc']) for m in ('bcresnet1', 'bcresnet8')}


def spec_runs():
    """The configurations of the article: key -> (display name, run DataFrame)."""
    out = {}
    for m in ('bcresnet1', 'bcresnet8'):
        s = BC_SEL[m]
        out[m] = get(m, s['init'], s['lr']) if s else get(m, 'none')
    out['letr96'] = get('letr96', 'gsc', 1e-3)
    out['letr128t'] = get('letr128t', 'gsc', 1e-3, seeds=[0, 1, 2])
    for m in ('w2v2', 'mhubert', 'mms300m'): out[m] = get(m, 'ssl', 3e-5)
    return out


SPEC = spec_runs()
NAME = {'bcresnet1': 'BC-ResNet-1', 'bcresnet8': 'BC-ResNet-8', 'letr96': 'LeTR-96', 'letr128t': 'LeTR-128',
        'w2v2': 'wav2vec\\,2.0 Base', 'mhubert': 'mHuBERT-147', 'mms300m': 'MMS-300M'}
NAME_TXT = {k: v.replace('\\,', ' ') for k, v in NAME.items()}
INIT = {'en': {'scratch': 'scratch', 'gsc': 'GSC init.', 'mswc': 'multiling. init.', 'ssl': None},
        'fr': {'scratch': 'sans init.', 'gsc': 'init. GSC', 'mswc': 'init. multiling.', 'ssl': None}}


def ms(x): return 'nan' if pd.isna(x) else f'{x:.1f}'


# ------------------------------------------------------------------ predictions and cluster bootstrap
META = pd.read_csv(os.path.join(DATA, 'splits', 'splits_usable.csv'))
META['verified'] = META.human_label.fillna('') == 'accepted'
VOL = pd.read_csv(os.path.join(DATA, 'volunteer_recordings.csv'))


def preds(run_dir, kind='test'):
    p = pd.read_csv(os.path.join(run_dir, f'{kind}_predictions.csv'))
    if kind == 'test': return p.merge(META, on='oid', how='left')
    return p.merge(VOL[['oid', 'user_id', 'gender', 'age']], on='oid', how='left')


def pooled(runs, kind='test'):
    """Predictions of all seeds, stacked, with a `seed` column and `ok`."""
    P = pd.concat([preds(d, kind).assign(seed=s) for d, s in zip(runs.dir, runs.seed)], ignore_index=True)
    P['ok'] = P.true == P.pred
    return P


def boot_ci(runs, kind='test', n=2000):
    """95 % CI of the accuracy averaged over seeds, resampling groups (test: 71 speaker groups; OOD: volunteers)."""
    P = pooled(runs, kind)
    key = 'group_id' if kind == 'test' else 'user_id'
    g = P.groupby(['oid', key]).ok.mean().reset_index().groupby(key).ok.agg(['sum', 'size'])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    v = g['sum'].to_numpy()[idx].sum(1) / g['size'].to_numpy()[idx].sum(1)
    return tuple(np.percentile(v, [2.5, 97.5]))


def mean_sd(x):
    x = pd.Series(x).dropna()
    return (float(x.mean()) if len(x) else np.nan, float(x.std()) if len(x) > 1 else np.nan, len(x))


# ------------------------------------------------------------------ subgroups, per-keyword accuracy, confusion
def subgroup_acc(runs):
    """Accuracy (mean over seeds) on field-test subsets and on OOD subsets."""
    P = pooled(runs)
    out = {}
    for col in ('gender', 'age', 'agent_id', 'verified'):
        t = P.groupby([col, 'seed']).ok.mean().groupby(col).mean()
        n = P[P.seed == P.seed.iloc[0]].groupby(col).size()
        out[col] = {str(k): (float(t[k]), int(n[k])) for k in t.index}
    out['all'] = (float(P.groupby('seed').ok.mean().mean()), int(P.oid.nunique()))
    ood = {}
    if all(os.path.exists(os.path.join(d, 'ood_predictions.csv')) for d in runs.dir):
        O = pooled(runs, 'ood')
        for col in ('gender', 'age'):
            t = O.groupby([col, 'seed']).ok.mean().groupby(col).mean()
            n = O[O.seed == O.seed.iloc[0]].groupby(col).size()
            ood[col] = {str(k): (float(t[k]), int(n[k])) for k in t.index}
        ood['all'] = (float(O.groupby('seed').ok.mean().mean()), int(O.oid.nunique()))
    out['ood'] = ood
    return out


def per_keyword(P):
    return P.groupby(['true', 'seed']).ok.mean().groupby('true').mean()


def confusion(P, kw):
    """Row-normalised confusion matrix (% of the clips of the true keyword, seeds pooled)."""
    C = pd.crosstab(P.true, P.pred).reindex(index=kw, columns=kw, fill_value=0)
    return 100 * C.div(C.sum(axis=1), axis=0)


def pair_stats(P, C, a='Biiga', b='Piiga'):
    err = P[~P.ok]
    share = 100 * ((err.true.isin([a, b])) & (err.pred.isin([a, b]))).sum() / max(len(err), 1)
    return dict(a_to_b=float(C.loc[a, b]), b_to_a=float(C.loc[b, a]), share_of_errors=float(share))


# ------------------------------------------------------------------ deployment measurements
def token(name):
    n = os.path.basename(str(name))
    for p in ('open32-', 'closed30-', 'random-'):
        if n.startswith(p): n = n[len(p):]
    return n.split('-')[0].replace('_', '').lower()


def is32(name): return os.path.basename(str(name)).startswith('open32-')


def _phone():
    f = os.path.join(DATA, 'deploy_phone.csv')
    if not os.path.exists(f): return pd.DataFrame()
    d = pd.read_csv(f)
    d = d[(d.status == 'ok') & (d.frontend_in_graph.astype(str) == 'True')].copy()
    d['tok'] = d.model.map(token); d['o32'] = d.model.map(is32); d['rnd'] = d.model.astype(str).str.startswith('random-')
    return d


def _desktop():
    f = os.path.join(DATA, 'deploy_desktop.csv')
    if not os.path.exists(f): return pd.DataFrame()
    d = pd.read_csv(f)
    rnd = d.ckpt.astype(str).str.startswith('random:')
    d['tok'] = [c.split(':')[1].split(':')[0].replace('-', '').replace('_', '').lower() if r else token(os.path.dirname(c))
                for c, r in zip(d.ckpt.astype(str), rnd)]
    d['o32'] = d.ckpt.astype(str).map(lambda c: is32(os.path.dirname(c))); d['rnd'] = rnd
    return d


PHONE, DESK = _phone(), _desktop()
FOOT = json.load(open(os.path.join(ROOT, 'experiments', 'footprint.json')))
_fb = os.path.join(ROOT, 'experiments', 'v3', 'footprint_big.json')
FOOT_BIG = json.load(open(_fb)) if os.path.exists(_fb) else {}
FOOT_KEY = {'letr96': 'letr96', 'letr128t': 'letr128t', 'w2v2': 'w2v2', 'mms300m': 'mms300m'}


def _pick(d, tok, prec, thr):
    if not len(d): return None
    x = d[(d.tok == tok) & (d.precision == prec) & (d.threads == thr)]
    if not len(x): return None
    sort = ['o32'] + (['rnd'] if 'rnd' in x else [])
    return x.sort_values(sort, kind='stable').iloc[0]


def phone(tok, prec, thr, col='median_ms'):
    r = _pick(PHONE, tok, prec, thr)
    return float(r[col]) if r is not None and pd.notna(r[col]) else np.nan


def desktop(tok, prec, thr, col='latency_ms_median'):
    """Desktop CPU measurement: data/deploy_desktop.csv if present, else (fp32 only) the footprint files."""
    r = _pick(DESK, tok, prec, thr)
    if r is not None and col in r and pd.notna(r[col]):
        try: return float(r[col])
        except ValueError: pass
    if prec == 'fp32' and col == 'latency_ms_median' and thr in (1, 4):
        k = FOOT_KEY.get(tok)
        for src in (FOOT_BIG, FOOT):
            if k in src: return float(src[k][f'latency_ms_{thr}thr_median'])
    return np.nan


def size_mb(tok, prec):
    s = phone(tok, prec, 4, 'size_mb')
    if np.isnan(s) and prec == 'fp32':
        k = FOOT_KEY.get(tok)
        for src in (FOOT_BIG, FOOT):
            if k in src: return float(src[k]['size_mb_fp32'])
    return s


def int8_acc():
    f = os.path.join(DATA, 'deploy_desktop_accuracy.csv')
    if not os.path.exists(f): return pd.DataFrame()
    d = pd.read_csv(f)
    d['tok'] = d.ckpt.map(lambda c: token(os.path.dirname(c)))
    return d


# ------------------------------------------------------------------ number formatting
class Fmt:
    """lang 'en' or 'fr' (decimal comma as {,}, ' ; ' in intervals)."""
    def __init__(self, lang): self.lang = lang

    def d(self, x, nd=1):
        if x is None or pd.isna(x): return '--'
        s = f'{x:.{nd}f}'
        return s.replace('.', '{,}') if self.lang == 'fr' else s

    def pct(self, x, nd=1): return '--' if x is None or pd.isna(x) else self.d(100 * x, nd)

    def pm(self, m, sd, n, nd=1):
        if m is None or pd.isna(m): return '--'
        s = self.pct(m, nd)
        return s + (f'\\,$\\pm$\\,{self.pct(sd, nd)}' if n > 1 and pd.notna(sd) else '')

    def ci(self, c):
        if c is None: return '--'
        sep = ' ; ' if self.lang == 'fr' else ', '
        return f'[{self.pct(c[0])}{sep}{self.pct(c[1])}]'

    def lat(self, x):
        if x is None or pd.isna(x): return '--'
        return self.d(x, 0) if x >= 10 else self.d(x, 1)

    def size(self, x):
        if x is None or pd.isna(x): return '--'
        return self.d(x, 0) if x >= 10 else self.d(x, 1)

    def params(self, p):
        return f'{self.d(p / 1e6, 1)}\\,M' if p > 1e6 else (f'{p / 1e3:.0f}\\,k' if p >= 1e5 else f'{self.d(p / 1e3, 1)}\\,k')

    def lr(self, lr):
        m, e = f'{lr:.0e}'.split('e')
        return f'${m}\\times10^{{{int(e)}}}$'
