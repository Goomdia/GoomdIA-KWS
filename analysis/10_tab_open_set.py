"""LaTeX table of the open-set experiments (closed 30-class LeTR-128 vs 32-class models), English and French.

Inputs  data/results_open_set_summary.csv   isolated clips at threshold 0.7 (analysis/08_open_set_results.py, from each run's test_predictions.csv)
        data/streaming_tuned_test_centre.csv   tuned detector, centre scoring, 'moore' test stream (experiments/v3/streaming_tune.py):
                                                budget 5 false alarms/h outside keywords, at most 0.05 wrong keywords per keyword (wcap 0.05)
Outputs manuscript/tables/tab_openset.tex, manuscript/tables_fr/tab_openset.tex
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
iso = pd.read_csv(os.path.join(ROOT, 'data', 'results_open_set_summary.csv'))
st = pd.read_csv(os.path.join(ROOT, 'data', 'streaming_tuned_test_centre.csv'))
THR, BUDGET, WCAP = 0.7, 5.0, 0.05
st = st[(st.kind == 'tuned') & (st.label == 'tuned') & (st['mode'] == 'centre') & (st.stream == 'moore')
        & np.isclose(st.budget, BUDGET) & np.isclose(st.wcap, WCAP)]
# (system, model, init) of the isolated evaluation, streaming group names (first one found is used), display name
ROWS = [(('closed30', 'letr128t', 'gsc'), ['LeTR closed'], 'LeTR-128', 'GSC'),
        (('open32', 'bcresnet8', None), ['BC-ResNet-8 32 cl.'], 'BC-ResNet-8', None),
        (('open32', 'letr128t', 'gsc'), ['LeTR 32 cl. GSC'], 'LeTR-128', 'GSC'),
        (('open32', 'w2v2', 'ssl'), ['w2v2 32 cl. (2 seeds)', 'w2v2 32 cl.'], 'wav2vec\\,2.0 Base', None),
        (('open32', 'mhubert', 'ssl'), ['mHuBERT-147 32 cl.'], 'mHuBERT-147', None),
        (('open32', 'mms300m', 'ssl'), ['MMS-300M 32 cl.'], 'MMS-300M', None)]
INIT = {'en': {'GSC': 'GSC init.', 'scratch': 'scratch', 'mswc': 'multiling. init.'},
        'fr': {'GSC': 'init. GSC', 'scratch': 'sans init.', 'mswc': 'init. multiling.'}}


def isolated(key):
    system, model, init = key
    d = iso[(iso.system == system) & (iso.model == model) & np.isclose(iso.threshold, THR)]
    if init: d = d[d.init == init]
    return d.iloc[0] if len(d) else None


def stream(groups):
    for g in groups:
        d = st[st.group == g]
        if len(d): return d
    return None


def build(lang):
    fr = lang == 'fr'
    def n(x, nd=1): return ('--' if pd.isna(x) else f'{x:.{nd}f}').replace('.', '{,}' if fr else '.')
    def pm(m, sd, nd=1, k=1):
        if pd.isna(m): return '--'
        return n(100 * m, nd) + (f'\\,$\\pm$\\,{n(100 * sd, nd)}' if k > 1 and pd.notna(sd) else '')
    body, notes = [], []
    for i, (key, groups, name, ini) in enumerate(ROWS):
        r, s = isolated(key), stream(groups)
        if i == 1: body.append('\\midrule')
        label = name
        init = ini or (r.init if r is not None and key[1] in ('bcresnet8',) else None)
        if init: label += f' ({INIT[lang][init]})'
        cls = '30' if key[0] == 'closed30' else '32'
        if r is None: iso_c = ['--'] * 4
        else: iso_c = [pm(r.kw_acc_mean, r.kw_acc_sd, 1, r.n_seeds), pm(r.far_moore_broadcast_mean, np.nan), pm(r.far_gsc_english_mean, np.nan),
                       pm(r.far_field_noise_mean, np.nan)]
        if s is None: det, fah = '--', '--'; ns = 0
        else:
            ns = s.run.nunique()
            det = pm(s.det.mean(), s.det.std(), 1, ns); fah = n(s.fa_nonkw_h.mean(), 1) + (f'\\,$\\pm$\\,{n(s.fa_nonkw_h.std(), 1)}' if ns > 1 else '')
        body.append(' & '.join([label, cls] + iso_c + [det, fah]) + ' \\\\')
        if key[0] != 'closed30': notes.append((label, 0 if r is None else int(r.n_seeds), ns))
    # seeds note: only where the number differs from three
    odd = [f'{nm}: {a} / {b}' for nm, a, b in notes if (a, b) != (3, 3)]
    if lang == 'en':
        cap = ('Closed-set (30 classes) and open-set (32 classes) models on the speaker-disjoint test set. Isolated clips: keyword accuracy and false acceptance of held-out '
               'non-keyword clips at the demonstrator threshold of 0.7. Continuous detection on a one-hour \\Moore{} stream with the strict criterion that a keyword is '
               'emitted within 0.5\\,s of its centre: detector tuned on a development stream for at most 5 false alarms per hour and at most 0.05 wrong keywords per '
               'keyword occurrence (Sect.~\\ref{sec:streaming}); false alarms (FA) per hour are keywords emitted outside keyword occurrences on the test stream. '
               'Mean $\\pm$ standard deviation over three seeds unless stated below the table')
        h1 = 'Model & Classes & Keyword & \\multicolumn{3}{c}{False acceptance (\\%)} & Detection & FA \\\\'
        h2 = ' & & acc. (\\%) & \\Moore{} speech & English words & Field noise & (\\%) & per hour \\\\'
        foot = ('Seeds (isolated clips / detection) where not three: ' + '; '.join(odd)) if odd else ''
    else:
        cap = ('Modèles à vocabulaire fermé (30 classes) et ouvert (32 classes) sur l\'ensemble de test disjoint en locuteurs. Clips isolés : exactitude sur les mots-clés et taux '
               'd\'acceptation à tort de clips hors mots-clés réservés, au seuil de 0,7 du démonstrateur. Détection en continu dans un flux \\Moore{} d\'une heure, avec le critère strict '
               'qu\'un mot-clé est émis à moins de 0,5\\,s de son centre : détecteur réglé sur un flux de développement pour au plus 5 fausses alarmes par heure et au plus 0,05 mot-clé '
               'erroné par occurrence de mot-clé (sect.~\\ref{sec:streaming}) ; les fausses alarmes (FA) par heure sont les mots-clés émis hors des occurrences de mots-clés '
               'dans le flux de test. Moyenne $\\pm$ écart-type sur trois graines, sauf indication sous le tableau')
        h1 = 'Modèle & Classes & Exactitude & \\multicolumn{3}{c}{Acceptation à tort (\\%)} & Détection & FA \\\\'
        h2 = ' & & mots-clés (\\%) & Parole \\Moore{} & Mots anglais & Bruit de terrain & (\\%) & par heure \\\\'
        foot = ('Graines (clips isolés / détection) lorsque ce n\'est pas trois : ' + '; '.join(odd)) if odd else ''
    lines = ['\\begin{table}[t]', '\\centering\\small', f'\\caption{{{cap}}}', '\\label{tab:openset}',
             '\\setlength{\\tabcolsep}{' + ('3.5pt' if lang == 'en' else '2pt') + '}', '\\begin{tabular}{@{}lcrrrrrr@{}}', '\\toprule', h1,
             '\\cmidrule(lr){4-6}', h2, '\\midrule', *body, '\\bottomrule', '\\end{tabular}', '']
    if foot: lines += [f'\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\footnotesize {foot}}}']
    lines += ['\\end{table}', '']
    return '\n'.join(lines), body


if __name__ == '__main__':
    for lang in ('en', 'fr'):
        tex, body = build(lang)
        open(os.path.join(ROOT, 'manuscript', 'tables' if lang == 'en' else 'tables_fr', 'tab_openset.tex'), 'w').write(tex)
    print('\n'.join(body))
