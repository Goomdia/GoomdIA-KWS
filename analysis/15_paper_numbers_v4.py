"""Numbers sheet of the v4 manuscript: experiments/v3/paper_numbers_v4.txt, one "key = value" per line, rounded as in the paper
(1 decimal for %, integers for ms). Anything not available is written as MISSING; lines starting with '#' are section titles or warnings.
Run after 08, 11, 12, 13, 09, 10, 14 (it reads the same data and run directories).
"""
import json, os
import numpy as np
import pandas as pd
import v4common as V

LINES = []
MISSING = []


def put(key, val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        val = 'MISSING'
    if val == 'MISSING' or 'nan' in str(val).lower().split() or str(val).endswith('nan'): MISSING.append(key); val = 'MISSING'
    LINES.append(f'{key} = {val}')


def sec(t): LINES.append(f'# {t}')


def p1(x): return None if x is None or pd.isna(x) else f'{100 * x:.1f}'
def pm(m, sd, n=2): return None if m is None or pd.isna(m) else p1(m) + (f' ± {p1(sd)}' if n > 1 and pd.notna(sd) else '')
def i0(x): return None if x is None or pd.isna(x) else f'{x:.0f}'
def f1(x): return None if x is None or pd.isna(x) else f'{x:.1f}'
def ci(c): return None if c is None else f'[{p1(c[0])}, {p1(c[1])}]'


MODELS = ['bcresnet1', 'bcresnet8', 'letr128t', 'w2v2', 'mhubert', 'mms300m']
ENC = ['w2v2', 'mhubert', 'mms300m']

sec('(a) Samsung Galaxy S8 latency, ms (TFLite, XNNPACK, median of one 1.5 s window, features included)')
for m in MODELS:
    for th in (1, 4): put(f's8_fp32_{th}thr_{m}', i0(V.phone(m, 'fp32', th)))
    if m in ENC:
        for th in (1, 4): put(f's8_int8_{th}thr_{m}', i0(V.phone(m, 'int8', th)))
    if m in ('bcresnet1', 'bcresnet8', 'letr128t'):
        pass
sec('(b) desktop CPU latency (i9-9960X), ms, 4 threads')
for m in MODELS:
    put(f'desktop_fp32_4thr_{m}', i0(V.desktop(m, 'fp32', 4)))
    if m in ENC: put(f'desktop_int8_4thr_{m}', i0(V.desktop(m, 'int8', 4)))
sec('(c) encoder size reduction fp32 / int8')
for m in ENC:
    a, b = V.size_mb(m, 'fp32'), V.size_mb(m, 'int8')
    put(f'tflite_size_fp32_mb_{m}', i0(a)); put(f'tflite_size_int8_mb_{m}', i0(b))
    put(f'tflite_size_factor_{m}', f1(a / b) if pd.notna(a) and pd.notna(b) else None)
    a, b = V.desktop(m, 'fp32', 4, 'size_mb_state_dict'), V.desktop(m, 'int8', 4, 'size_mb_state_dict')
    put(f'pytorch_size_fp32_mb_{m}', i0(a)); put(f'pytorch_size_int8_mb_{m}', i0(b))
    put(f'pytorch_size_factor_{m}', f1(a / b) if pd.notna(a) and pd.notna(b) else None)
acc = V.int8_acc()
for m in ENC:
    for prec in ('fp32', 'int8'):
        a = acc[(acc.tok == m) & (acc.precision == prec)] if len(acc) else []
        put(f'pytorch_{prec}_test_acc_{m}', p1(a.test_acc.iloc[0]) if len(a) else None)
        if prec == 'int8': put(f'pytorch_int8_agreement_{m}', p1(a.pred_agreement_with_fp32.iloc[0]) if len(a) else None)
    put(f'tflite_int8_argmax_agreement_phone_{m}', p1(V.phone(m, 'int8', 4, 'argmax_agreement')))

sec('(d) BC-ResNet: configuration selected on mean validation accuracy over seeds')
for m in ('bcresnet1', 'bcresnet8'):
    s = V.BC_SEL[m]
    if s is None:
        put(f'{m}_selected', None); continue
    runs = V.SPEC[m]
    mt, sd, n = V.mean_sd(runs.test); mo, sdo, no = V.mean_sd(runs.ood)
    put(f'{m}_selected_init', s['init']); put(f'{m}_selected_lr', f'{s["lr"]:g}'); put(f'{m}_selected_val_mean', p1(s['val'])); put(f'{m}_selected_n_seeds', n)
    put(f'{m}_test_mean_sd', pm(mt, sd, n)); put(f'{m}_test_ci95', ci(V.boot_ci(runs)))
    put(f'{m}_ood_mean_sd', pm(mo, sdo, no)); put(f'{m}_ood_ci95_volunteers', ci(V.boot_ci(runs, 'ood')))
    put(f'{m}_params', int(runs.n_params.iloc[0]))
    for init in ('scratch', 'mswc'):
        lr = V.select_lr(m, init)
        if lr is None: put(f'{m}_{init}_test', None); continue
        r = V.get(m, init, lr); a, b, k = V.mean_sd(r.test); c, d, _ = V.mean_sd(r.ood)
        put(f'{m}_{init}_lr', f'{lr:g}'); put(f'{m}_{init}_n_seeds', k); put(f'{m}_{init}_test_mean_sd', pm(a, b, k)); put(f'{m}_{init}_ood_mean_sd', pm(c, d, k))
    sc, mw = V.select_lr(m, 'scratch'), V.select_lr(m, 'mswc')
    if sc and mw:
        a = V.get(m, 'mswc', mw).test.mean() - V.get(m, 'scratch', sc).test.mean()
        put(f'{m}_mswc_minus_scratch_pp', f1(100 * a))
    put(f'{m}_lr_1e-2_runs', ', '.join(f'{r.init}/s{r.seed}: val {100 * r.val:.1f} test {100 * r.test:.1f}' for r in V.R[(V.R.model == m) & np.isclose(V.R.lr, 0.01) & (V.R.subset == 'full') & (V.R.split == 'speaker')].itertuples()) or None)

sec('(e) scaling: test accuracy at each subset size (mean ± sd over seeds)')
bc_sel = V.BC_SEL['bcresnet8']
sub_defs = [('bcresnet8', bc_sel['init'], bc_sel['lr'], None), ('letr128t', 'gsc', 1e-3, [0, 1, 2]), ('mms300m', 'ssl', 3e-5, None)]
alt = V.R[(V.R.model == 'bcresnet8') & (V.R.subset == 'train_frac_10')]
if len(alt) and not len(V.get('bcresnet8', bc_sel['init'], bc_sel['lr'], subset='train_frac_10')):
    sub_defs[0] = ('bcresnet8', alt.init.iloc[0], float(alt.lr.iloc[0]), None)
    put('bcresnet8_subsets_run_with', f'{alt.init.iloc[0]} lr {alt.lr.iloc[0]:g} (differs from selected {bc_sel["init"]} {bc_sel["lr"]:g})')
for m, init, lr, seeds in sub_defs:
    res = {}
    for sub, lab in (('train_frac_10', '10pct'), ('train_frac_25', '25pct'), ('train_frac_50', '50pct'), ('full', '100pct')):
        d = V.get(m, init, lr, subset=sub, seeds=seeds)
        res[lab] = d.test.mean() if len(d) else np.nan
        n_tr = int(d.n_train.iloc[0]) if len(d) else None
        put(f'scaling_{m}_{lab}', f'{pm(d.test.mean(), d.test.std(), len(d))} (n_train {n_tr}, {len(d)} seeds)' if len(d) else None)
    put(f'scaling_{m}_gain_50_to_100_pp', f1(100 * (res['100pct'] - res['50pct'])))
    put(f'scaling_{m}_gain_25_to_50_pp', f1(100 * (res['50pct'] - res['25pct'])))

sec('(f) per-keyword accuracy and confusion')
PB, PM = V.pooled(V.SPEC['bcresnet8']), V.pooled(V.SPEC['mms300m'])
if len(PB) and len(PM):
    ab, am = V.per_keyword(PB).sort_values(), V.per_keyword(PM)
    kw = list(ab.index)
    CB, CM = V.confusion(PB, kw), V.confusion(PM, kw)
    put('bcresnet8_keyword_acc_min', f'{p1(ab.iloc[0])} ({ab.index[0]})'); put('bcresnet8_keyword_acc_max', f'{p1(ab.iloc[-1])} ({ab.index[-1]})')
    put('mms300m_keyword_acc_min', f'{p1(am.min())} ({am.idxmin()})'); put('mms300m_keyword_acc_max', f'{p1(am.max())} ({am.idxmax()})')
    for nm, P, C in (('bcresnet8', PB, CB), ('mms300m', PM, CM)):
        ps = V.pair_stats(P, C)
        put(f'{nm}_Biiga_to_Piiga_pct', f1(ps['a_to_b'])); put(f'{nm}_Piiga_to_Biiga_pct', f1(ps['b_to_a']))
        put(f'{nm}_pair_share_of_errors_pct', f1(ps['share_of_errors']))
        put(f'{nm}_total_error_rate_pct', f1(100 * (~P.ok).mean()))
    for i, k in enumerate(ab.index[:5], 1):
        row = CB.loc[k].drop(k)
        put(f'bcresnet8_hardest_{i}', f'{k} {p1(ab[k])} %; main confusion -> {row.idxmax()} ({row.max():.1f} %)')
else:
    put('per_keyword', None)

sec('(g) subgroup accuracies on the speaker-disjoint field test set (mean over seeds) and OOD')
AGE = [('15-18', 'age_15_18'), ('18-35', 'age_18_35'), ('35-65', 'age_over_35')]
for m in ('bcresnet8', 'mhubert', 'mms300m'):
    if not len(V.SPEC[m]):
        put(f'{m}_subgroups', None); continue
    g = V.subgroup_acc(V.SPEC[m])
    put(f'{m}_field_all', p1(g['all'][0])); put(f'{m}_ood_all', p1(g['ood'].get('all', (np.nan,))[0]))
    for k, lab in (('Femme', 'women'), ('Homme', 'men')):
        put(f'{m}_field_{lab}', p1(g['gender'][k][0]))
        put(f'{m}_ood_{lab}', p1(g['ood'].get('gender', {}).get(k, (np.nan,))[0]))
    for k, lab in AGE:
        put(f'{m}_field_{lab}', p1(g['age'][k][0])); put(f'{m}_ood_{lab}', p1(g['ood'].get('age', {}).get(k, (np.nan,))[0]))
    put(f'{m}_field_gap_men_minus_women_pp', f1(100 * (g['gender']['Homme'][0] - g['gender']['Femme'][0])))
    put(f'{m}_field_gap_18_35_minus_15_18_pp', f1(100 * (g['age']['18-35'][0] - g['age']['15-18'][0])))
    put(f'{m}_ood_gap_men_minus_women_pp', f1(100 * (g['ood']['gender']['Homme'][0] - g['ood']['gender']['Femme'][0])) if g['ood'] else None)
    put(f'{m}_ood_gap_18_35_minus_15_18_pp', f1(100 * (g['ood']['age']['18-35'][0] - g['ood']['age']['15-18'][0])) if g['ood'] else None)

sec('(h) 32-class isolated clips at threshold 0.7: keyword accuracy and false acceptance (%), mean ± sd over seeds')
iso = pd.read_csv(os.path.join(V.DATA, 'results_open_set_summary.csv'))
ISO = [('letr128t_closed30', ('closed30', 'letr128t', 'gsc')), ('bcresnet8_32cl', ('open32', 'bcresnet8', None)), ('letr128t_gsc_32cl', ('open32', 'letr128t', 'gsc')),
       ('w2v2_32cl', ('open32', 'w2v2', 'ssl')), ('mhubert_32cl', ('open32', 'mhubert', 'ssl')), ('mms300m_32cl', ('open32', 'mms300m', 'ssl'))]
for nm, (sy, mo, ini) in ISO:
    d = iso[(iso.system == sy) & (iso.model == mo) & np.isclose(iso.threshold, 0.7)]
    if ini: d = d[d.init == ini]
    if not len(d):
        put(f'iso_{nm}', None); continue
    r = d.iloc[0]
    put(f'iso_{nm}_n_seeds', int(r.n_seeds)); put(f'iso_{nm}_init', r.init)
    put(f'iso_{nm}_keyword_acc', pm(r.kw_acc_mean, r.kw_acc_sd, int(r.n_seeds)))
    for k, lab in (('far_moore_broadcast', 'far_moore_speech'), ('far_gsc_english', 'far_english'), ('far_field_noise', 'far_field_noise')):
        put(f'iso_{nm}_{lab}', pm(r[k + '_mean'], r[k + '_sd'], int(r.n_seeds)))

sec('(i) tuned streaming, moore test stream, centre scoring, budget 5 FA/h, wcap 0.05 (strict: keyword within 0.5 s of its centre)')
st = pd.read_csv(os.path.join(V.DATA, 'streaming_tuned_test_centre.csv'))
st = st[(st.kind == 'tuned') & (st.label == 'tuned') & (st['mode'] == 'centre') & (st.stream == 'moore') & np.isclose(st.budget, 5) & np.isclose(st.wcap, 0.05)]
sv = pd.read_csv(os.path.join(V.DATA, 'streaming_v3.csv'))
nkw = sv[sv.stream == 'moore'].n_keywords.iloc[0]
put('moore_test_stream_keyword_occurrences', int(nkw)); put('moore_test_stream_hours', f'{sv[sv.stream == "moore"].hours.iloc[0]:.2f}')
GROUPS = [('LeTR_closed', ['LeTR closed']), ('bcresnet8_32cl', ['BC-ResNet-8 32 cl.']), ('letr128t_gsc_32cl', ['LeTR 32 cl. GSC']),
          ('w2v2_32cl', ['w2v2 32 cl. (2 seeds)', 'w2v2 32 cl.']), ('mhubert_32cl', ['mHuBERT-147 32 cl.']), ('mms300m_32cl', ['MMS-300M 32 cl.'])]
for nm, names in GROUPS:
    d = next((st[st.group == g] for g in names if len(st[st.group == g])), None)
    if d is None:
        for k in ('n_runs', 'detection_pct', 'wrong_kw_per_kw', 'fa_nonkw_per_h'): put(f'stream_{nm}_{k}', None)
        continue
    n = d.run.nunique()
    put(f'stream_{nm}_n_runs', n)
    put(f'stream_{nm}_detection_pct', pm(d.det.mean(), d.det.std(), n))
    put(f'stream_{nm}_wrong_kw_per_kw', f'{d.wrong_kw.mean():.3f}' + (f' ± {d.wrong_kw.std():.3f}' if n > 1 else ''))
    put(f'stream_{nm}_fa_nonkw_per_h', f'{d.fa_nonkw_h.mean():.1f}' + (f' ± {d.fa_nonkw_h.std():.1f}' if n > 1 else ''))
    put(f'stream_{nm}_detected_keywords_of_{int(nkw)}', f'{d.det.mean() * nkw:.0f}')

sec('(j) mHuBERT-147 OOD and frozen-probe numbers already in the paper (copied from experiments/ssl_probe.json)')
mh = V.SPEC['mhubert']
put('mhubert_ood_mean_sd', pm(*(lambda t: (t[0], t[1], t[2]))(V.mean_sd(mh.ood))) if len(mh) else None)
put('mhubert_test_mean_sd', pm(*V.mean_sd(mh.test)) if len(mh) else None)
pr = json.load(open(os.path.join(V.ROOT, 'experiments', 'ssl_probe.json')))
for k, nm in (('w2v2-base', 'w2v2'), ('mms-300m', 'mms300m')):
    for sp in ('speaker', 'random'):
        b = pr[k][sp]['best']
        put(f'probe_{nm}_{sp}_best_layer', b['layer']); put(f'probe_{nm}_{sp}_test', p1(b['test'])); put(f'probe_{nm}_{sp}_ood', p1(b['ood']))

sec('checks and warnings')
for m, s in V.BC_SEL.items():
    if s and s['differs']: LINES.append(f'# WARNING {m}: literal best mean validation accuracy is {s["literal"]["init"]} lr {s["literal"]["lr"]:g} ({s["literal"]["n"]} seed, val {p1(s["literal"]["val"])}); the selected configuration has {s["n"]} seeds')
    if s and s['edge']: LINES.append(f'# WARNING {m}: selected LR {s["lr"]:g} is the largest LR tried (1e-2 not yet run or not better)')
    if s and np.isclose(s['lr'], 0.01): LINES.append(f'# WARNING {m}: LR 1e-2 won on validation; extend the grid?')
if len(V.R_FAILED): LINES.append('# failed runs (best_val_acc < 0.1, excluded): ' + ', '.join(V.R_FAILED.run))
for nm, runs in V.SPEC.items():
    if len(runs) > 1 and runs.test.std() > 0.015: LINES.append(f'# WARNING large seed sd: {nm} {pm(runs.test.mean(), runs.test.std())}')
if MISSING: LINES.append(f'# {len(MISSING)} MISSING values: ' + ', '.join(MISSING))
out = os.path.join(V.ROOT, 'experiments', 'v3', 'paper_numbers_v4.txt')
open(out, 'w').write('\n'.join(LINES) + '\n')
print('\n'.join(LINES))
