"""Aggregate the benchmark runs (experiments/runs/*/summary.json) into tables and figures.

Outputs: data/results_runs.csv (one row per run), data/results_summary.json, figures/fig_ssl_layers.pdf
(figures/fig_scaling.pdf -> analysis/14_fig_scaling.py)
"""
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RUNS = os.path.join(ROOT, 'experiments', 'runs')
DATA, FIG = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'figures')
C1, C2, C3, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})

# ------------------------------------------------------------------ collect runs
rows = []
for f in glob.glob(os.path.join(RUNS, '*', 'summary.json')):
    s = json.load(open(f)); a = s['args']
    rows.append(dict(run=os.path.basename(os.path.dirname(f)), model=a['model'], init=a['init'], split=a['split'],
                     subset=a.get('train_subset') or 'full', lr=a['lr'], seed=a['seed'], n_params=s['n_params'],
                     n_train=s['n_train'], epochs=len(s['history']), val_acc=s['best_val_acc'],
                     test_acc=s['test']['acc'], test_f1=s['test']['macro_f1'],
                     ood_acc=s.get('ood', {}).get('acc', np.nan), ood_f1=s.get('ood', {}).get('macro_f1', np.nan),
                     train_time_s=s['train_time_s'], peak_mem_mib=s.get('peak_mem_mib', np.nan), gpu=s.get('gpu', '')))
R = pd.DataFrame(rows).sort_values(['model', 'init', 'split', 'subset', 'lr', 'seed'])
R = R[R.seed != 10]                   # seed 10 = extra model saved for the streaming evaluation only
R['failed'] = R.val_acc < 0.1          # training failure: validation accuracy stays near chance (1/30); replaced by seeds 3-4
R.to_csv(os.path.join(DATA, 'results_runs.csv'), index=False)
FAILED = R[R.failed].run.tolist()
R = R[~R.failed]

# learning rate chosen on validation (speaker split, seed 0)
sweep = R[(R.split == 'speaker') & (R.subset == 'full') & (R.seed == 0) & (R.model == 'letr128t')]
best_lr = sweep.loc[sweep.groupby('init').val_acc.idxmax()].set_index('init').lr.to_dict() if len(sweep) else {}
best_lr['ssl'] = 3e-5
# MMS-300M: learning rate chosen on validation by a short check (experiments/v3/mms_lr_check, seed 0, 5 of 20 epochs)
lrc = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'v3', 'mms_lr_check', '*', 'summary.json')))]
best_lr_model = {'mms300m': max(lrc, key=lambda s: s['best_val_acc'])['args']['lr']} if lrc else {}
NEW = list(best_lr_model)              # models added after the v2 manuscript (CIs drawn after the existing ones)
sel = R[R.apply(lambda r: r.lr == best_lr_model.get(r.model, best_lr.get(r.init, r.lr)), axis=1)]

def agg(df):
    g = df.groupby(['model', 'init', 'split', 'subset'])
    out = g.agg(n=('seed', 'size'), n_train=('n_train', 'first'), params=('n_params', 'first'),
                val=('val_acc', 'mean'), test=('test_acc', 'mean'), test_sd=('test_acc', 'std'),
                f1=('test_f1', 'mean'), f1_sd=('test_f1', 'std'), ood=('ood_acc', 'mean'), ood_sd=('ood_acc', 'std'))
    return (out * 1).reset_index()
A = agg(sel)
S = dict(best_lr=best_lr, summary=A.round(6).to_dict(orient='records'), sweep=sweep[['init', 'lr', 'val_acc', 'test_acc']].to_dict(orient='records'))
if FAILED: S['failed_runs'] = FAILED
if lrc:
    S['best_lr_model'] = best_lr_model
    S['lr_check_mms300m'] = [dict(lr=s['args']['lr'], epochs=len(s['history']), val_acc=s['best_val_acc'],
                                  val_curve=[round(h['val_acc'], 6) for h in s['history']]) for s in lrc]

# ------------------------------------------------------------------ subgroup / error analysis
meta = pd.read_csv(os.path.join(DATA, 'splits', 'splits_usable.csv'))
meta['verified'] = meta.human_label.fillna('') == 'accepted'
def preds(run):
    p = pd.read_csv(os.path.join(RUNS, run, 'test_predictions.csv'))
    return p.merge(meta, on='oid', how='left')
sub = {}
for (model, init), g in sel[(sel.split == 'speaker') & (sel.subset == 'full')].groupby(['model', 'init']):
    P = pd.concat([preds(r).assign(seed=s) for r, s in zip(g.run, g.seed)])
    P['ok'] = P.true == P.pred
    d = {}
    for col in ('gender', 'age', 'agent_id', 'verified', 'auto_triage'):
        t = P.groupby([col, 'seed']).ok.mean().groupby(col).agg(['mean', 'std'])
        t['n'] = P[P.seed == P.seed.iloc[0]].groupby(col).size()
        d[col] = t.round(6).reset_index().to_dict(orient='records')
    pk = P.groupby(['true', 'seed']).ok.mean().groupby('true').mean().sort_values()
    d['per_keyword'] = pk.round(6).to_dict()
    conf = P[~P.ok].groupby(['true', 'pred']).size().sort_values(ascending=False) / P.seed.nunique()
    d['top_confusions'] = [dict(true=t, pred=p, n=round(n, 1)) for (t, p), n in conf.head(12).items()]
    sub[f'{model}-{init}'] = d
S['subgroups'] = sub

# volunteer (OOD) breakdown by gender / age for the best runs
vol = pd.read_csv(os.path.join(DATA, 'volunteer_recordings.csv'))
oodsub = {}
for (model, init), g in sel[(sel.split == 'speaker') & (sel.subset == 'full')].groupby(['model', 'init']):
    fs = [os.path.join(RUNS, r, 'ood_predictions.csv') for r in g.run]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs: continue
    P = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(fs)]).merge(vol[['oid', 'gender', 'age']], on='oid', how='left')
    P['ok'] = P.true == P.pred
    oodsub[f'{model}-{init}'] = {c: P.groupby(c).ok.mean().round(6).to_dict() for c in ('gender', 'age')}
S['ood_subgroups'] = oodsub
# ------------------------------------------------------------------ group bootstrap CIs
rng = np.random.default_rng(0)
def ok_per_utt(runs):
    P = pd.concat([preds(r) for r in runs])
    P['ok'] = (P.true == P.pred).astype(float)
    return P.groupby(['oid', 'group_id']).ok.mean().reset_index()
def boot(df, n=2000):
    g = df.groupby('group_id').ok.agg(['sum', 'size'])
    idx = rng.integers(0, len(g), size=(n, len(g)))
    s, c = g['sum'].to_numpy()[idx].sum(1), g['size'].to_numpy()[idx].sum(1)
    return np.percentile(s / c, [2.5, 97.5]).round(6).tolist()
ci = {}
for (model, init, split), g in sel[(sel.subset == 'full') & ~sel.model.isin(NEW)].groupby(['model', 'init', 'split']):
    if split in ('speaker', 'random'):
        ci[f'{model}-{init}-{split}'] = boot(ok_per_utt(g.run))
S['group_bootstrap_ci95'] = ci

# ------------------------------------------------------------------ controlled seen vs unseen speakers
ctrl = {}
for init in ('gsc', 'scratch'):
    gu = sel[(sel.split == 'ctrl_unseen') & (sel.init == init)]
    gs = sel[(sel.split == 'ctrl_seen') & (sel.init == init)]
    if not len(gu) or not len(gs): continue
    U, Sn = ok_per_utt(gu.run), ok_per_utt(gs.run)
    M = U.merge(Sn, on=['oid', 'group_id'], suffixes=('_u', '_s'))
    M['ok'] = M.ok_s - M.ok_u
    gsum = M.groupby('group_id').ok.agg(['sum', 'size'])
    idx = rng.integers(0, len(gsum), size=(2000, len(gsum)))
    d = gsum['sum'].to_numpy()[idx].sum(1) / gsum['size'].to_numpy()[idx].sum(1)
    ctrl[init] = dict(unseen=float(gu.test_acc.mean()), unseen_sd=float(gu.test_acc.std()),
                      seen=float(gs.test_acc.mean()), seen_sd=float(gs.test_acc.std()),
                      diff=float(M.ok.mean()), diff_ci95=np.percentile(d, [2.5, 97.5]).round(6).tolist(),
                      n_test=len(M), n_seeds=int(min(len(gu), len(gs))))
S['controlled'] = ctrl
for (model, init, split), g in sel[(sel.subset == 'full') & sel.model.isin(NEW)].groupby(['model', 'init', 'split']):
    if split in ('speaker', 'random'):
        ci[f'{model}-{init}-{split}'] = boot(ok_per_utt(g.run))

# ------------------------------------------------------------------ training cost (wall-clock on one GPU)
cost = sel[(sel.subset == 'full') & sel.split.isin(['speaker', 'random'])].groupby(['model', 'init', 'split'])
S['training_cost'] = cost.agg(n=('seed', 'size'), epochs=('epochs', 'mean'), train_h=('train_time_s', lambda t: t.mean() / 3600),
                              gpu_h_total=('train_time_s', lambda t: t.sum() / 3600), peak_mem_mib=('peak_mem_mib', 'max'),
                              gpu=('gpu', 'last')).round(4).reset_index().to_dict(orient='records')
json.dump(S, open(os.path.join(DATA, 'results_summary.json'), 'w'), indent=1, default=float)

# figure: data scaling -> analysis/14_fig_scaling.py
# figure: per-keyword accuracy and confusion matrix -> analysis/09_confusion.py
print(json.dumps(dict(best_lr=best_lr), default=float))
print(A.round(4).to_string())

# ------------------------------------------------------------------ figure: frozen SSL layer-wise probes
pf = os.path.join(ROOT, 'experiments', 'ssl_probe.json')
if os.path.exists(pf):
    pr = json.load(open(pf))
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    for name, c, lab in (('mms-300m', C1, 'MMS-300M (1,400 languages)'), ('w2v2-base', C2, 'wav2vec 2.0 Base (English)')):
        if name not in pr or 'speaker' not in pr[name]: continue
        pl = pr[name]['speaker']['per_layer']
        L = np.array([r['layer'] for r in pl]); depth = L / L.max()
        ax.plot(depth, [100 * r['test'] for r in pl], color=c, lw=1.6, label=lab)
        ax.plot(depth, [100 * r['ood'] for r in pl], color=c, lw=1.2, ls='--')
    ax.set_xlabel('Relative layer depth'); ax.set_ylabel('Linear-probe accuracy (%)')
    ax.plot([], [], color=INK2, lw=1.2, label='field test (solid)'); ax.plot([], [], color=INK2, lw=1.2, ls='--', label='out-of-domain (dashed)')
    ax.legend(loc='lower center', ncol=2, fontsize=6.3, handlelength=1.6, columnspacing=0.8)
    ax.set_ylim(50, 95)
    fig.savefig(os.path.join(FIG, 'fig_ssl_layers.pdf'))
