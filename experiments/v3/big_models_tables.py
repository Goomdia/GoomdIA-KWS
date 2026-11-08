"""Markdown tables for REPORT_big_models.md from the run summaries, data/results_summary.json,
v3/footprint_big.json and the streaming evaluations (no new computation)."""
import glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP, ROOT = os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))
S = json.load(open(os.path.join(ROOT, 'data', 'results_summary.json')))
md = lambda df: '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * df.shape[1]] +
                          ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.itertuples(index=False)])
pct = lambda v: f'{100 * v:.2f}'


def runs(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(pattern, 'summary.json'))):
        s = json.load(open(f)); a = s['args']
        rows.append(dict(run=os.path.basename(os.path.dirname(f)), split=a['split'], seed=a['seed'], lr=f"{a['lr']:g}",
                         bs=f"{a['bs']}x{a.get('accum', 1)}", epochs=len(s['history']),
                         best_epoch=1 + max(range(len(s['history'])), key=lambda i: (s['history'][i]['val_acc'], -i)),
                         wall_h=round(s['train_time_s'] / 3600, 3), s_per_epoch=round(s['history'][-1]['time_s'] / len(s['history']), 1),
                         peak_alloc_mib=round(s['peak_mem_mib']) if 'peak_mem_mib' in s else '-',
                         peak_reserved_mib=round(s['peak_reserved_mib']) if 'peak_reserved_mib' in s else '-',
                         gpu=s.get('gpu', '-'), val=pct(s['best_val_acc']), test=pct(s['test']['acc']),
                         ood=pct(s['ood']['acc']) if 'ood' in s else '-'))
    return pd.DataFrame(rows)


print('## Per-run table\n')
print(md(pd.concat([runs(os.path.join(EXP, 'runs', 'w2v2-*')), runs(os.path.join(EXP, 'runs', 'mms300m-*'))])))
print('\n## MMS-300M learning-rate check\n')
print(md(runs(os.path.join(HERE, 'mms_lr_check', '*'))))
for c in S.get('lr_check_mms300m', []):
    print(f"- lr {c['lr']:g}: val acc per epoch {[round(100 * v, 1) for v in c['val_curve']]}")
print('\n## LeTR timing reference (exclusive GPU)\n')
print(md(runs(os.path.join(HERE, 'letr_timing', '*'))))

A = pd.DataFrame(S['summary']); A = A[A.subset == 'full']
print('\n## Means and sd over seeds (data/results_summary.json)\n')
T = A[A.split.isin(['speaker', 'random'])].copy()
T['ci95_group_bootstrap'] = [str([round(100 * v, 2) for v in S['group_bootstrap_ci95'].get(f'{m}-{i}-{s}', [])])
                             for m, i, s in zip(T.model, T.init, T.split)]
for c in ('val', 'test', 'test_sd', 'f1', 'f1_sd', 'ood', 'ood_sd'): T[c] = (100 * T[c]).round(2)
print(md(T[['model', 'init', 'split', 'n', 'params', 'val', 'test', 'test_sd', 'f1', 'f1_sd', 'ood', 'ood_sd', 'ci95_group_bootstrap']]))

print('\n## Training cost (results_summary.json: training_cost)\n')
C = pd.DataFrame(S.get('training_cost', []))
if len(C): print(md(C))

print('\n## Subgroups (speaker-disjoint test, mean over seeds; OOD pooled over seeds)\n')
keys = [k for k in ('letr128t-gsc', 'w2v2-ssl', 'mms300m-ssl') if k in S['subgroups']]
rows = []
for col in ('gender', 'age'):
    vals = sorted({str(r[col]) for k in keys for r in S['subgroups'][k][col]})
    for v in vals:
        r = dict(subset=f'test {col}={v}')
        for k in keys:
            d = {str(x[col]): x for x in S['subgroups'][k][col]}
            r['n'] = int(d[v]['n']) if v in d else r.get('n'); r[k] = pct(d[v]['mean']) if v in d else '-'
        rows.append(r)
    for v in sorted({str(x) for k in keys for x in S['ood_subgroups'].get(k, {}).get(col, {})}):
        r = dict(subset=f'OOD {col}={v}', n='')
        for k in keys:
            d = {str(x): y for x, y in S['ood_subgroups'].get(k, {}).get(col, {}).items()}
            r[k] = pct(d[v]) if v in d else '-'
        rows.append(r)
print(md(pd.DataFrame(rows)))

print('\n## Hardest keywords (speaker-disjoint test, mean over seeds)\n')
pk = pd.DataFrame({k: S['subgroups'][k]['per_keyword'] for k in keys})
pk = (100 * pk).round(1).sort_values(keys[-1])
print(md(pk.head(8).reset_index().rename(columns={'index': 'keyword'})))
for k in keys[1:]:
    print(f"\nTop confusions {k}: " + '; '.join(f"{c['true']}->{c['pred']} ({c['n']})" for c in S['subgroups'][k]['top_confusions'][:6]))

fb = os.path.join(HERE, 'footprint_big.json')
if os.path.exists(fb):
    print('\n## Footprint (v3/footprint_big.json vs experiments/footprint.json)\n')
    F, F0 = json.load(open(fb)), json.load(open(os.path.join(EXP, 'footprint.json')))
    rows = [dict(model=k, source='v3/footprint_big.json', **F[k]) for k in ('w2v2', 'mms300m')]
    rows += [dict(model=k, source='footprint.json (existing)', **F0[k]) for k in ('letr128t', 'w2v2', 'mms300m')]
    print(md(pd.DataFrame(rows)))

print('\n## Streaming (window 2.0 s / hop 1.0 s and 1.5 / 0.5)\n')
rows = []
for f in sorted(glob.glob(os.path.join(EXP, 'runs', '*', 'streaming_eval.json'))):
    st = json.load(open(f)); H, K, O = st['hours'], st['keywords'], st['oov_words']
    for r in st['results']:
        rows.append(dict(run=st['run'], win_hop=f"{r['window_s']}/{r['hop_s']}", thr=r['threshold'], detection=pct(r['detection_rate']),
                         subs_per_kw=round(r['substitutions_per_h'] * H / K, 3), fa_per_oov=round(r['fa_oov_per_h'] * H / O, 3),
                         fa_noise_per_h=round(r['fa_noise_per_h']), hours=round(H, 4), keywords=K, oov=O))
print(md(pd.DataFrame(rows)))
