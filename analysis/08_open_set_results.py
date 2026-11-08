"""Aggregate the open-set experiments (experiments/v3/runs) into tables and a figure.

Inputs  experiments/v3/runs/open32-*/      32-class models: BC-ResNet-8, LeTR-128, wav2vec 2.0, mHuBERT-147, MMS-300M
                                           (summary.json, test/ood_predictions.csv, streaming_v3.json); failed runs (val < 0.1) are skipped
        experiments/v3/runs/closed30-*/    closed-set baselines of the article (predictions.csv, streaming_v3.json)
Outputs data/results_open_set.csv            one row per run x rejection threshold (isolated evaluation)
        data/results_open_set_summary.csv    mean / sd over seeds
        data/streaming_v3.csv                one row per run x stream x detector configuration x threshold
        data/streaming_v3_operating_points.csv  operating point per run x configuration: threshold chosen on the
                                             moore_dev stream (max detection with <= 5 FA/h outside keywords),
                                             then applied to the moore and english test streams
        figures/fig_streaming_v3.pdf         detection rate vs false alarms per hour (Moore stream)
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RUNS = os.path.join(ROOT, 'experiments', 'v3', 'runs')
OLD = os.path.join(ROOT, 'experiments', 'runs')
DATA, FIG = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'figures')
C1, C2, C3, C4, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})
EXTRA = ['_background_', '_unknown_']
FA_MAX = 5.0                         # false alarms per hour outside keywords (noise + OOV) at the operating point
LABEL = {('closed30', 'letr128t', 'gsc'): 'LeTR-128 GSC, closed set', ('closed30', 'w2v2', 'ssl'): 'wav2vec 2.0, closed set',
         ('open32', 'letr128t', 'gsc'): 'LeTR-128 GSC, 32 classes', ('open32', 'letr128t', 'scratch'): 'LeTR-128 scratch, 32 classes',
         ('open32', 'w2v2', 'ssl'): 'wav2vec 2.0, 32 classes', ('open32', 'bcresnet8', 'scratch'): 'BC-ResNet-8, 32 classes',
         ('open32', 'bcresnet8', 'mswc'): 'BC-ResNet-8 (multiling. init.), 32 classes', ('open32', 'mhubert', 'ssl'): 'mHuBERT-147, 32 classes',
         ('open32', 'mms300m', 'ssl'): 'MMS-300M, 32 classes'}


def run_info(d):
    name = os.path.basename(d)
    system = name.split('-')[0]
    f = os.path.join(OLD, name.split('-', 1)[1], 'summary.json') if system == 'closed30' else os.path.join(RUNS, name, 'summary.json')
    if not os.path.exists(f): return None
    s = json.load(open(f))
    if s['best_val_acc'] < 0.1: return None                    # training failure (validation accuracy at chance): excluded
    return dict(run=name, system=system, model=s['args']['model'], init=s['args']['init'], seed=s['args']['seed'], summary=s)


def iso_metrics(P, thr):
    """P: oid, subsource, true, pred, confidence. A keyword is accepted if pred is a keyword and confidence > thr."""
    acc = ~P.pred.isin(EXTRA) & (P.confidence > thr)
    kw = P.subsource == 'keyword'
    r = dict(threshold=thr, kw_acc=float((acc & (P.true == P.pred))[kw].mean()), kw_reject=float((~acc)[kw].mean()))
    for s in ('moore_broadcast', 'gsc_english', 'field_noise', 'gsc_noise'):
        r[f'far_{s}'] = float(acc[P.subsource == s].mean())
    r['far_unknown'] = float(acc[P.true == '_unknown_'].mean()); r['far_background'] = float(acc[P.true == '_background_'].mean())
    return r


# ------------------------------------------------------------------ isolated evaluation
rows = []
for d in sorted(glob.glob(os.path.join(RUNS, 'open32-*')) + glob.glob(os.path.join(RUNS, 'closed30-*'))):
    if not os.path.isdir(d): continue
    i = run_info(d)
    if i is None: continue
    s = i.pop('summary')
    if i['system'] == 'open32':
        if not os.path.exists(os.path.join(d, 'summary.json')): continue
        s = json.load(open(os.path.join(d, 'summary.json')))
        P = pd.read_csv(os.path.join(d, 'test_predictions.csv'))
        O = pd.read_csv(os.path.join(d, 'ood_predictions.csv'))
        extra = dict(val_acc=s['best_val_acc'], epochs=len(s['history']), kw_acc_kw_only=s['test']['acc_kw_only'],
                     kw_macro_f1=s['test']['macro_f1'])
    else:
        if not os.path.exists(os.path.join(d, 'predictions.csv')): continue
        P = pd.read_csv(os.path.join(d, 'predictions.csv'))
        O = pd.read_csv(os.path.join(OLD, i['run'].split('-', 1)[1], 'ood_predictions.csv'))
        extra = dict(val_acc=s['best_val_acc'], epochs=len(s['history']), kw_acc_kw_only=s['test']['acc'],
                     kw_macro_f1=s['test']['macro_f1'])
    for thr in (0.0, 0.5, 0.7, 0.9, 0.99):
        r = dict(**i, **iso_metrics(P, thr), **extra)
        oa = ~O.pred.isin(EXTRA) & (O.confidence > thr)
        r['ood_acc'] = float((oa & (O.true == O.pred)).mean()); r['ood_reject'] = float((~oa).mean())
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(DATA, 'results_open_set.csv'), index=False)
num = [c for c in R.columns if c not in ('run', 'system', 'model', 'init', 'seed', 'threshold')]
g = R.groupby(['system', 'model', 'init', 'threshold'])
Sm = g[num].mean().add_suffix('_mean').join(g[num].std().add_suffix('_sd')).join(g.size().rename('n_seeds')).reset_index()
Sm.to_csv(os.path.join(DATA, 'results_open_set_summary.csv'), index=False)
cols = ['kw_acc', 'kw_reject', 'far_moore_broadcast', 'far_gsc_english', 'far_field_noise', 'far_gsc_noise', 'ood_acc', 'ood_reject']
print((Sm.set_index(['system', 'model', 'init', 'threshold'])[[c + '_mean' for c in cols] + ['n_seeds']] * 1).round(3).to_string())

# ------------------------------------------------------------------ streaming
srows = []
for f in sorted(glob.glob(os.path.join(RUNS, '*', 'streaming_v3.json'))):
    d = os.path.dirname(f)
    if not (os.path.basename(d).startswith('open32-') or os.path.basename(d).startswith('closed30-')): continue
    i = run_info(d)
    if i is None: continue
    i.pop('summary')
    z = json.load(open(f))
    for r in z['results']:
        st = z['streams'][r['stream']]
        srows.append(dict(**i, n_keywords=st['keywords'], n_oov=st['oov_words'], hours=st['hours'], **r))
S = pd.DataFrame(srows)
S.to_csv(os.path.join(DATA, 'streaming_v3.csv'), index=False)

ops = []
for (run, cfg), g in S.groupby(['run', 'config']):
    dev = g[g.stream == 'moore_dev'].sort_values('threshold')
    ok = dev[dev.fa_nonkw_per_h <= FA_MAX]
    thr = ok.sort_values(['detection_rate', 'threshold'], ascending=[False, True]).threshold.iloc[0] if len(ok) else dev.threshold.max()
    for stream in ('moore', 'english'):
        t = g[(g.stream == stream)]
        r = t[t.threshold == thr].iloc[0].to_dict()
        okt = t[t.fa_nonkw_per_h <= FA_MAX]                     # oracle: threshold chosen on the test stream
        r.update(feasible_on_dev=bool(len(ok)), oracle_threshold=okt.sort_values('detection_rate').threshold.iloc[-1] if len(okt) else np.nan,
                 oracle_detection_rate=okt.detection_rate.max() if len(okt) else np.nan)
        ops.append(r)
OP = pd.DataFrame(ops)
OP.to_csv(os.path.join(DATA, 'streaming_v3_operating_points.csv'), index=False)
k = ['system', 'model', 'init', 'stream', 'config']
m = ['threshold', 'detection_rate', 'wrong_kw_per_kw', 'fa_oov_per_oov_word', 'fa_noise_per_h', 'fa_nonkw_per_h', 'oracle_detection_rate']
print(OP.groupby(k)[m].agg(['mean', 'std']).round(3).to_string())

# ------------------------------------------------------------------ figure
STY = {('closed30', 'letr128t', 'gsc'): (C1, '--', 'o'), ('closed30', 'w2v2', 'ssl'): (C2, '--', 's'),
       ('open32', 'letr128t', 'gsc'): (C1, '-', 'o'), ('open32', 'letr128t', 'scratch'): (C3, '-', '^'),
       ('open32', 'w2v2', 'ssl'): (C2, '-', 's')}
panels = [('w2.0-h1.0', '(a) window 2 s, hop 1 s (demonstrator detector)'), ('w1.5-h0.25-avg4', '(b) hop 0.25 s, posteriors averaged over 1 s')]
if len(S):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)
    for ax, (cfg, title) in zip(axes, panels):
        T = S[(S.stream == 'moore') & (S.config == cfg)]
        for key, (c, ls, mk) in STY.items():
            g = T[(T.system == key[0]) & (T.model == key[1]) & (T.init == key[2])]
            if not len(g): continue
            c_ = g.groupby('threshold')[['fa_nonkw_per_h', 'detection_rate']].mean().reset_index()
            ax.plot(c_.fa_nonkw_per_h.clip(lower=0.3), 100 * c_.detection_rate, color=c, ls=ls, lw=1.3, marker=mk, ms=2.2,
                    label=f"{LABEL[key]} ({g.seed.nunique()} seed{'s' if g.seed.nunique() > 1 else ''})")
        ax.axvline(FA_MAX, color=INK2, lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_xlim(0.25, 1500)
        ax.set_xticks([0.3, 1, 5, 10, 100, 1000], ['0', '1', '5', '10', '100', '1000'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('False alarms per hour (noise + out-of-vocabulary speech)')
        ax.set_title(title, fontsize=8, loc='left', color=INK)
    axes[0].set_ylabel('Keywords detected (%)'); axes[0].set_ylim(0, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5, handlelength=2.2, columnspacing=1.0)
    fig.savefig(os.path.join(FIG, 'fig_streaming_v3.pdf'))
