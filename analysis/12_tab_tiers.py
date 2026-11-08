"""Main results table grouped by hardware tier (tab:main) and the two supplementary tables, English and French.

Outputs  manuscript/tables/tab_main.tex, manuscript/tables_fr/tab_main.tex      (phone / laptop / workstation)
         manuscript/tables/si_tab_main_full.tex                                  (label tab:si-main; every configuration)
         manuscript/tables/si_tab_deploy.tex                                     (label si:deploy; deployment cost)
         manuscript/tables_fr/si_tab_main_full.tex, si_tab_deploy.tex            only if tables_fr/si_tab_main_full.tex already exists
Models selected on mean validation accuracy over seeds (v4common.select); CIs: cluster bootstrap over the 71 test groups.
"""
import os
import numpy as np
import pandas as pd
import v4common as V

TIERS = [('Phone', 'Téléphone', [('bcresnet1', None), ('bcresnet8', None), ('letr96', 'gsc'), ('letr128t', 'gsc')]),
         ('Laptop', 'Ordinateur portable', [('w2v2', 'ssl'), ('mhubert', 'ssl')]),
         ('Workstation', 'Station de travail', [('mms300m', 'ssl')])]
ENCODERS = ('w2v2', 'mhubert', 'mms300m')
CI = {}


def ci_of(key, runs):
    if key not in CI: CI[key] = V.boot_ci(runs) if len(runs) else None
    return CI[key]


def model_label(m, init, lang):
    nm = V.NAME[m]
    i = V.INIT[lang].get(init)
    return nm + (f' ({i})' if i else '')


def main_row(m, lang):
    F = V.Fmt(lang)
    runs = V.SPEC[m]
    init = runs.init.iloc[0] if len(runs) else (V.BC_SEL.get(m) or {}).get('init', 'scratch')
    tok = m.replace('_', '')
    if not len(runs):
        return f'{V.NAME[m]} & -- & -- & -- & -- & -- & -- & -- \\\\'
    mt, sd, n = V.mean_sd(runs.test); mo, sdo, no = V.mean_sd(runs.ood)
    s8 = V.phone(tok, 'fp32', 4)
    s8s = F.lat(s8)
    if m in ENCODERS:
        s8s += f' ({F.lat(V.phone(tok, "int8", 4))})'
    return ' & '.join([model_label(m, init, lang), F.params(int(runs.n_params.iloc[0])), F.pm(mt, sd, n), F.ci(ci_of(m, runs)),
                       F.pm(mo, sdo, no), s8s, F.lat(V.desktop(tok, 'fp32', 4)), F.size(V.size_mb(tok, 'fp32'))]) + ' \\\\'


def main_table(lang):
    F = V.Fmt(lang)
    body = []
    for i, (en, fr, ms_) in enumerate(TIERS):
        if i: body.append('\\midrule')
        body.append(f'\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{en if lang == "en" else fr}}}}} \\\\')
        for m, _ in ms_: body.append('\\quad ' + main_row(m, lang))
    if lang == 'en':
        cap = ('Test accuracy (\\%, mean $\\pm$ standard deviation over three seeds) and latency of the models, grouped by the hardware '
               'they are intended for. Latency: median time to classify one 1.5\\,s window, feature extraction included, on a Samsung Galaxy S8 (2017) '
               'and on four threads of a desktop CPU. All other configurations, the random split and the quantised models are given in Online Resource~1')
        h1 = ('Model & Params & \\multicolumn{2}{c}{Speaker-disjoint test set} & OOD & \\multicolumn{2}{c}{Latency (ms)} & Size \\\\\n'
              '\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}')
        h2 = ' & & Accuracy (\\%) & 95\\,\\% CI & (\\%) & S8 & Desktop & (MB) \\\\'
    else:
        cap = ('Exactitude sur le jeu de test (\\%, moyenne $\\pm$ écart-type sur trois graines) et latence des modèles, regroupés selon le matériel '
               'visé. Latence : temps médian pour classer une fenêtre de 1{,}5\\,s, extraction des caractéristiques comprise, '
               'sur un Samsung Galaxy S8 (2017) et sur quatre fils d\'exécution d\'un processeur de bureau. '
               'Les autres configurations, le découpage aléatoire et les modèles quantifiés figurent dans la ressource en ligne~1')
        h1 = ('Modèle & Param. & \\multicolumn{2}{c}{Jeu de test disjoint en locuteurs} & OOD & \\multicolumn{2}{c}{Latence (ms)} & Taille \\\\\n'
              '\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}')
        h2 = ' & & Exactitude (\\%) & IC à 95\\,\\% & (\\%) & S8 & Bureau & (Mo) \\\\'
    return '\n'.join(['\\begin{table}[t]', '\\centering\\small', f'\\caption{{{cap}}}', '\\label{tab:main}',
                      '\\setlength{\\tabcolsep}{' + ('3.5pt' if lang == 'en' else '3pt') + '}', '\\begin{tabular}{@{}lrccccrr@{}}', '\\toprule', h1, h2, '\\midrule',
                      *body, '\\bottomrule', '\\end{tabular}', '\\end{table}', ''])


# ------------------------------------------------------------------ supplementary: every configuration
def full_rows(lang):
    F = V.Fmt(lang)
    out = []
    combos = []
    for m in ('letr96', 'letr128t', 'bcresnet1', 'bcresnet8', 'w2v2', 'mhubert', 'mms300m'):
        inits = [i for i in ('scratch', 'gsc', 'mswc', 'ssl') if len(V.get(m, i))]
        for i in inits:
            lrs = sorted(V.get(m, i).lr.unique()) if m.startswith('bcresnet') else [V.select_lr(m, i)]
            for lr in lrs: combos.append((m, i, lr))
    prev = None
    for m, i, lr in combos:
        sp = V.get(m, i, lr)
        if m == 'letr128t' and i == 'gsc': sp = V.get(m, i, lr, seeds=[0, 1, 2])
        rd = V.get(m, i, lr, split='random')
        if not len(sp): continue
        mt, sd, n = V.mean_sd(sp.test); mr, sdr, nr = V.mean_sd(rd.test) if len(rd) else (np.nan, np.nan, 0)
        mo, sdo, no = V.mean_sd(sp.ood); mf, sdf, nf = V.mean_sd(sp.f1)
        sel = m in V.BC_SEL and V.BC_SEL[m] and V.BC_SEL[m]['init'] == i and np.isclose(V.BC_SEL[m]['lr'], lr)
        delta = f'{100 * (mr - mt):+.1f}'.replace('-', '$-$') if not np.isnan(mr) else '--'
        if lang == 'fr': delta = delta.replace('.', '{,}')
        name = model_label(m, i, lang) + ('$^\\star$' if sel else '')
        row = ' & '.join([name, F.lr(lr), str(n), F.params(int(sp.n_params.iloc[0])), F.pm(mr, sdr, nr),
                          F.pm(mt, sd, n), F.ci(ci_of(f'{m}-{i}-{lr}', sp)) if n >= 1 else '--', delta, F.pm(mf, sdf, nf),
                          F.pm(mo, sdo, no)]) + ' \\\\'
        if prev is not None and prev != m: out.append('\\addlinespace')
        out.append(row); prev = m
    return out


def full_table(lang='en'):
    rows = full_rows(lang)
    cap = ('Complete test results (\\%, mean $\\pm$ s.d.\\ over training seeds; no s.d.\\ for a single seed). One row per initialisation at its '
           'learning rate (LR) selected on validation accuracy; for BC-ResNet, every learning rate that was run. $^\\star$: configuration selected '
           'for the article (highest mean validation accuracy over the BC-ResNet initialisations and learning rates). $\\Delta$: random minus '
           'speaker-disjoint accuracy; F1: macro-averaged F1 score on the speaker-disjoint split; OOD: \\num{2921} volunteer clips')
    return '\n'.join(['\\begin{table}[ht]', '\\centering\\footnotesize', f'\\caption{{{cap}}}', '\\label{tab:si-main}',
                      '\\setlength{\\tabcolsep}{2pt}', '\\begin{tabular}{@{}lcrrcccrcc@{}}',
                      '\\toprule',
                      ' & & & & \\multicolumn{5}{c}{Field test set (\\num{1658} clips)} & \\\\',
                      '\\cmidrule(lr){5-9}',
                      'Model & LR & Seeds & Params & Random & Speaker-disjoint & 95\\,\\% CI & $\\Delta$ & F1 & OOD \\\\',
                      '\\midrule', *rows, '\\bottomrule', '\\end{tabular}', '\\end{table}', ''])


# ------------------------------------------------------------------ supplementary: deployment
def deploy_table(lang='en'):
    F = V.Fmt(lang)
    acc = V.int8_acc()
    rows = []
    for m in ('bcresnet1', 'bcresnet8', 'letr96', 'letr128t', 'w2v2', 'mhubert', 'mms300m'):
        for prec in ('fp32', 'int8'):
            vals = [V.size_mb(m, prec), V.phone(m, prec, 1), V.phone(m, prec, 4), V.desktop(m, prec, 1), V.desktop(m, prec, 4),
                    V.phone(m, prec, 4, 'peak_mem_mb')]
            a = acc[(acc.tok == m) & (acc.precision == prec)] if len(acc) else []
            ta = float(a.test_acc.iloc[0]) if len(a) else np.nan
            ag = float(a.pred_agreement_with_fp32.iloc[0]) if len(a) else np.nan
            if all(pd.isna(v) for v in vals) and pd.isna(ta): continue
            rows.append(' & '.join([V.NAME[m], prec, F.size(vals[0]), F.lat(vals[1]), F.lat(vals[2]),
                                    F.lat(vals[3]), F.lat(vals[4]), F.d(vals[5], 0) if pd.notna(vals[5]) else '--',
                                    F.pct(ta), F.pct(ag)]) + ' \\\\')
    cap = ('Deployment cost of the models. Size: TFLite file (fp32 or int8) of the model including the feature extractor; latency: median time '
           'to classify one 1.5\\,s window on a Samsung Galaxy S8 (TFLite, XNNPACK) and on a desktop CPU (PyTorch), with 1 and 4 threads; peak memory on the phone, 4 threads. '
           'Accuracy and agreement: PyTorch model with dynamically quantised linear layers on the speaker-disjoint test set (agreement: share of clips '
           'with the same prediction as the fp32 model). \\textquotedblleft--\\textquotedblright: not measured')
    return '\n'.join(['\\begin{table}[ht]', '\\centering\\small', f'\\caption{{{cap}}}', '\\label{si:deploy}',
                      '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}llrrrrrrcc@{}}', '\\toprule',
                      ' & & & \\multicolumn{2}{c}{S8 (ms)} & \\multicolumn{2}{c}{Desktop (ms)} & Peak mem. & & \\\\',
                      '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}',
                      'Model & Precision & Size (MB) & 1 thr. & 4 thr. & 1 thr. & 4 thr. & S8 (MB) & Acc. (\\%) & Agr. (\\%) \\\\',
                      '\\midrule', *rows, '\\bottomrule', '\\end{tabular}', '\\end{table}', ''])


if __name__ == '__main__':
    for m, s in V.BC_SEL.items():
        if s and s['differs']:
            print(f'WARNING {m}: literal best validation mean is {s["literal"]} (n={s["literal"]["n"]}), selected among >=3-seed configs: {s}')
        if s and s['edge']: print(f'WARNING {m}: selected LR {s["lr"]} is the largest LR tried')
    for lang in ('en', 'fr'):
        open(os.path.join(V.TAB[lang], 'tab_main.tex'), 'w').write(main_table(lang))
    open(os.path.join(V.TAB['en'], 'si_tab_main_full.tex'), 'w').write(full_table())
    open(os.path.join(V.TAB['en'], 'si_tab_deploy.tex'), 'w').write(deploy_table())
    if os.path.exists(os.path.join(V.TAB['fr'], 'si_tab_main_full.tex')):
        print('NOTE: tables_fr/si_tab_main_full.tex exists, but French SI tables are not generated')
    print(open(os.path.join(V.TAB['en'], 'tab_main.tex')).read())
