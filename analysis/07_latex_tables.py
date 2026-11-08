"""Generate the LaTeX result tables (manuscript/tables/*.tex) from data/results_summary.json and the
streaming evaluations, so that no number is copied by hand into the manuscript."""
import glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(ROOT, 'manuscript', 'tables'); os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(ROOT, 'data', 'results_summary.json')))
A = pd.DataFrame(S['summary'])
A = A[A.subset == 'full']

NAMES = [('letr96', 'scratch', 'LeTR-96, scratch'), ('letr96', 'gsc', 'LeTR-96, GSC init.'),
         ('letr128t', 'scratch', 'LeTR-128, scratch'), ('letr128t', 'gsc', 'LeTR-128, GSC init.'),
         ('w2v2', 'ssl', 'wav2vec\\,2.0 Base, fine-tuned'), ('mms300m', 'ssl', 'MMS-300M, fine-tuned')]

def pm(m, sd, n):
    if pd.isna(m): return '--'
    return f'{100*m:.1f}' + (f'\\,$\\pm$\\,{100*sd:.1f}' if n > 1 and not pd.isna(sd) else '')

def cell(model, init, split, col):
    r = A[(A.model == model) & (A.init == init) & (A.split == split)]
    if not len(r): return '--', None
    r = r.iloc[0]
    return pm(r[col], r[col + '_sd'] if col + '_sd' in r else float('nan'), r.n), r

FP = json.load(open(os.path.join(ROOT, 'experiments', 'footprint.json')))
fb = os.path.join(ROOT, 'experiments', 'v3', 'footprint_big.json')   # fine-tuned MMS-300M classifier
if os.path.exists(fb): FP['mms300m'] = json.load(open(fb))['mms300m']
lines, full = [], []
for model, init, name in NAMES:
    rnd, rr = cell(model, init, 'random', 'test')
    spk, rs = cell(model, init, 'speaker', 'test')
    f1, _ = cell(model, init, 'speaker', 'f1')
    ood, _ = cell(model, init, 'speaker', 'ood')
    if rr is None and rs is None: continue
    ref = rs if rs is not None else rr
    gap = f'{100*(rr.test - rs.test):+.1f}'.replace('-', '$-$') if rr is not None and rs is not None else '--'
    ci = S.get('group_bootstrap_ci95', {}).get(f'{model}-{init}-speaker')
    cis = f'[{100*ci[0]:.1f}, {100*ci[1]:.1f}]' if ci else '--'
    p = ref.params
    ps = f'{p/1e6:.1f}\\,M' if p > 1e6 else f'{p/1e3:.0f}\\,k'
    lat = FP[model]['latency_ms_1thr_median']
    lats = f'{lat:.0f}' if lat >= 10 else f'{lat:.1f}'
    lines.append(f'{name} & {ps} & {lats} & {spk} & {cis} & {rnd} & {ood} \\\\')
    full.append(f'{name} & {ps} & {rnd} & {spk} & {cis} & {gap} & {f1} & {ood} \\\\')
# main-text table: accuracy and cost side by side
tab = r'''\begin{table}[t]
\centering\small
\caption{Test accuracy (\%%, mean $\pm$ standard deviation over training seeds) and cost of the baselines. The speaker-disjoint split is the reference protocol; the random split has the same set sizes; the 95\,\%% CI is a group-bootstrap interval of the speaker-disjoint accuracy. The out-of-domain (OOD) set contains %s volunteer clips; models were trained on the speaker-disjoint split. Latency: median CPU time to classify one 1.5\,s window on one thread, feature extraction included}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrcccc@{}}
\toprule
 & & Latency & \multicolumn{3}{c}{Field test set (\num{1658} clips)} & OOD \\
\cmidrule(lr){4-6}\cmidrule(l){7-7}
Model & Params & (ms) & Speaker-disjoint & 95\,\%% CI & Random & set \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % ('\\num{2921}', '\n'.join(lines))
open(os.path.join(OUT, 'tab_main.tex'), 'w').write(tab)
# supplementary table: all metrics
tab = r'''\begin{table}[ht]
\centering\small
\caption{Complete test results (\%%, mean $\pm$ s.d.\ over training seeds). $\Delta$: random minus speaker-disjoint accuracy; F1: macro-averaged F1 score on the speaker-disjoint split; OOD: %s volunteer clips}
\label{tab:si-main}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrcccrcc@{}}
\toprule
 & & \multicolumn{5}{c}{Field test set (\num{1658} clips)} & OOD \\
\cmidrule(lr){3-7}\cmidrule(l){8-8}
Model & Params & Random & Speaker-disjoint & 95\,\%% CI & $\Delta$ & F1 & \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % ('\\num{2921}', '\n'.join(full))
open(os.path.join(OUT, 'si_tab_main_full.tex'), 'w').write(tab)

# ------------------------------------------------------------------ subgroup table
sub = S.get('subgroups', {})
cols = [k for k in ('letr128t-gsc', 'letr128t-scratch', 'w2v2-ssl', 'mms300m-ssl') if k in sub]
hdr = {'letr128t-gsc': 'LeTR-128 (GSC)', 'letr128t-scratch': 'LeTR-128 (scratch)', 'w2v2-ssl': 'wav2vec\\,2.0', 'mms300m-ssl': 'MMS-300M'}
rows = []
def grp(name, key, order, lab=lambda x: x):
    rows.append(f'\\multicolumn{{{len(cols)+2}}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\')
    for v in order:
        vals, n = [], None
        for c in cols:
            d = {str(r[key]): r for r in sub[c][key]}
            r = d.get(str(v))
            vals.append(f'{100*r["mean"]:.1f}' if r else '--'); n = n or (r['n'] if r else None)
        rows.append(f'\\quad {lab(v)} & {int(n) if n else "--"} & ' + ' & '.join(vals) + ' \\\\')
if cols:
    grp('Declared gender', 'gender', ['Femme', 'Homme'], {'Femme': 'female', 'Homme': 'male'}.get)
    grp('Declared age range', 'age', ['15-18', '18-35', '35-65'], lambda x: x.replace('-', '--'))
    grp('Field agent', 'agent_id', ['A1', 'A2', 'A3'])
    grp('Label source', 'verified', ['True', 'False'], {'True': 'human-verified', 'False': 'automatic triage only'}.get)
    tab = r'''\begin{table}[t]
\centering\small
\caption{Accuracy (\%%) on subsets of the speaker-disjoint test set (mean over seeds); $n$: number of test clips}
\label{tab:subgroups}
\begin{tabular}{@{}lr%s@{}}
\toprule
Subset & $n$ & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % ('c' * len(cols), ' & '.join(hdr[c] for c in cols), '\n'.join(rows))
    open(os.path.join(OUT, 'tab_subgroups.tex'), 'w').write(tab)

# ------------------------------------------------------------------ streaming tables
rows_main, rows_full = [], []
for f in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*', 'streaming_eval.json'))):
    st = json.load(open(f))
    name = 'LeTR-128 (GSC)' if 'letr128t-gsc' in st['run'] else ('wav2vec\\,2.0' if 'w2v2' in st['run'] else
                                                                ('MMS-300M' if 'mms300m' in st['run'] else st['run']))
    H, K, O = st['hours'], st['keywords'], st['oov_words']
    for r in st['results']:
        row = (f"{r['threshold']:.2f} & {100*r['detection_rate']:.1f} & "
               f"{r['substitutions_per_h']*H/K:.2f} & {r['fa_oov_per_h']*H/O:.2f} & {r['fa_noise_per_h']:.0f} \\\\")
        rows_full.append(f"{name} & {r['window_s']:.1f}, {r['hop_s']:.1f} & " + row)
        if r['window_s'] == 2.0 and r['threshold'] in (0.7, 0.95):
            rows_main.append(f"{name} & " + row)
    hours, kw, oov = H, K, O
if rows_main:
    defs = (r"Detection: share of keywords for which the correct word was emitted; substitutions: wrong keywords "
            r"emitted per keyword occurrence; FA: false alarms; FA per OOV word: keywords emitted per out-of-vocabulary word; FA per hour "
            r"of noise: keywords emitted per hour of background noise only")
    tab = r"""\begin{table}[t]
\centering\small
\caption{Simulated continuous detection with the sliding-window detector of the demonstrator (2\,s window, 1\,s hop; %.1f\,h of audio with %d speaker-disjoint test keywords and %d out-of-vocabulary English words). %s}
\label{tab:streaming}
\begin{tabular}{@{}lccccc@{}}
\toprule
 & & Detection & Substitutions & FA per & FA per hour \\
Model & Threshold & (\%%) & per keyword & OOV word & of noise \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (hours, kw, oov, defs, '\n'.join(rows_main))
    open(os.path.join(OUT, 'tab_streaming.tex'), 'w').write(tab)
    tab = r"""\begin{table}[ht]
\centering\small
\caption{Simulated continuous detection for all window settings and thresholds (same streams as Table~6 of the article). %s}
\label{tab:si-streaming}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & Window, hop & & Detection & Substitutions & FA per & FA per hour \\
Model & (s) & Threshold & (\%%) & per keyword & OOV word & of noise \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (defs, '\n'.join(rows_full))
    open(os.path.join(OUT, 'si_tab_streaming_full.tex'), 'w').write(tab)
print('tables written:', os.listdir(OUT))

# ------------------------------------------------------------------ frozen SSL probe table
pf = os.path.join(ROOT, 'experiments', 'ssl_probe.json')
if os.path.exists(pf):
    pr = json.load(open(pf))
    names = {'w2v2-base': ('wav2vec\\,2.0 Base', 'English (960\\,h)', '95\\,M'), 'mms-300m': ('MMS-300M', '1,400 languages', '315\\,M')}
    rows = []
    for k, (nm, data, par) in names.items():
        if k not in pr or 'speaker' not in pr[k] or 'random' not in pr[k]: continue
        b, br = pr[k]['speaker']['best'], pr[k]['random']['best']
        nl = len(pr[k]['speaker']['per_layer']) - 1
        rows.append(f"{nm} & {data} & {par} & {b['layer']}/{nl} & {100*br['test']:.1f} & {100*b['test']:.1f} & {100*b['ood']:.1f} \\\\")
    tab = r"""\begin{table}[t]
\centering\small
\caption{Frozen self-supervised encoders with a linear probe (logistic regression on mean-pooled hidden states; layer selected on the validation set of each split). Accuracies in \%%}
\label{tab:ssl}
\begin{tabular}{@{}lllcccc@{}}
\toprule
Encoder & Pre-training data & Params & Layer & Random & Speaker & OOD \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % '\n'.join(rows)
    open(os.path.join(OUT, 'tab_ssl.tex'), 'w').write(tab)
    print('ssl table written')
