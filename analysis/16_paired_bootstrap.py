"""Paired cluster bootstrap of accuracy differences between two configurations on the same speaker-disjoint test set.

For each configuration, per-clip correctness is averaged over training seeds. The 71 test recording groups are resampled
with replacement (2000 replicates, seed 0, the same group draws for both configurations), and the difference of the two
accuracies is recomputed on each replicate. Reports the observed difference (points), its 95 % percentile interval and
the share of replicates with a difference <= 0 (one-sided bootstrap p-value).

Outputs data/paired_bootstrap.csv
        manuscript/tables/si_tab_paired.tex, manuscript/tables_fr/si_tab_paired.tex   (label tab:si-paired)
"""
import os
import numpy as np
import pandas as pd
import v4common as V

N_BOOT = 2000


def clip_acc(runs):
    P = V.pooled(runs)
    return P.groupby(['oid', 'group_id']).ok.mean()


def paired(a_runs, b_runs):
    a, b = clip_acc(a_runs), clip_acc(b_runs)
    d = pd.concat({'a': a, 'b': b}, axis=1, join='inner').reset_index()
    g = d.groupby('group_id').agg(a=('a', 'sum'), b=('b', 'sum'), n=('a', 'size'))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    A, B, N = (g[c].to_numpy()[idx].sum(1) for c in ('a', 'b', 'n'))
    diff = 100 * (A - B) / N
    obs = 100 * (g.a.sum() - g.b.sum()) / g.n.sum()
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return dict(n_clips=int(g.n.sum()), n_groups=len(g), acc_a=100 * g.a.sum() / g.n.sum(), acc_b=100 * g.b.sum() / g.n.sum(),
                diff=obs, ci_lo=lo, ci_hi=hi, p_le0=float((diff <= 0).mean()))


G = V.get
FR = {'mHuBERT-147 vs wav2vec 2.0 Base': 'mHuBERT-147 / wav2vec 2.0 Base',
      'MMS-300M vs mHuBERT-147': 'MMS-300M / mHuBERT-147',
      'BC-ResNet-8 multiling. vs scratch': 'BC-ResNet-8, init. multiling. / sans init.',
      'BC-ResNet-1 multiling. vs scratch': 'BC-ResNet-1, init. multiling. / sans init.',
      'LeTR-128 multiling. vs scratch': 'LeTR-128, init. multiling. / sans init.',
      'BC-ResNet-1 (multiling.) vs LeTR-128 (GSC)': 'BC-ResNet-1 (init. multiling.) / LeTR-128 (init. GSC)',
      'BC-ResNet-8 100 % vs 50 % of training groups': 'BC-ResNet-8, 100 / 50\\,\\% des groupes',
      'LeTR-128 100 % vs 50 %': 'LeTR-128, 100 / 50\\,\\% des groupes',
      'MMS-300M 100 % vs 50 %': 'MMS-300M, 100 / 50\\,\\% des groupes'}
EN = {'BC-ResNet-8 multiling. vs scratch': 'BC-ResNet-8, multiling. init. / scratch',
      'BC-ResNet-1 multiling. vs scratch': 'BC-ResNet-1, multiling. init. / scratch',
      'LeTR-128 multiling. vs scratch': 'LeTR-128, multiling. init. / scratch',
      'BC-ResNet-1 (multiling.) vs LeTR-128 (GSC)': 'BC-ResNet-1 (multiling. init.) / LeTR-128 (GSC init.)',
      'BC-ResNet-8 100 % vs 50 % of training groups': 'BC-ResNet-8, 100 / 50\\,\\% of the groups',
      'LeTR-128 100 % vs 50 %': 'LeTR-128, 100 / 50\\,\\% of the groups',
      'MMS-300M 100 % vs 50 %': 'MMS-300M, 100 / 50\\,\\% of the groups',
      'mHuBERT-147 vs wav2vec 2.0 Base': 'mHuBERT-147 / wav2vec\\,2.0 Base', 'MMS-300M vs mHuBERT-147': 'MMS-300M / mHuBERT-147'}


def table(rows, lang):
    F = V.Fmt(lang)
    d = lambda x: F.d(0.0 if round(x, 1) == 0 else x, 1).replace('-', '$-$')
    if lang == 'en':
        cap = ('Paired comparisons on the speaker-disjoint test set (accuracy in \\%, mean over seeds). The difference A $-$ B is in '
               'percentage points; its 95\\,\\% CI comes from a paired cluster bootstrap over the 71 test groups (2000 replicates, the '
               'same groups for both configurations); $p$ is the share of replicates with a difference $\\le 0$')
        head = 'Comparison A / B & A & B & A $-$ B & 95\\,\\% CI & $p$ \\\\'
        names = EN
    else:
        cap = ('Comparaisons appariées sur le jeu de test disjoint en locuteurs (exactitude en \\%, moyenne sur les graines). La différence '
               'A $-$ B est en points de pourcentage ; son IC à 95\\,\\% provient d’un bootstrap apparié par grappes sur les 71 groupes '
               'de test (2000 réplications, les mêmes groupes pour les deux configurations) ; $p$ est la part des réplications où la '
               'différence est $\\le 0$')
        head = 'Comparaison A / B & A & B & A $-$ B & IC à 95\\,\\% & $p$ \\\\'
        names = FR
    sep = ' ; ' if lang == 'fr' else ', '
    body = []
    for r in rows:
        pv = '$<$\\,' + F.d(0.001, 3) if r['p_le0'] < 0.001 else F.d(r['p_le0'], 3)
        body.append(f"{names.get(r['comparison'], r['comparison'])} & {F.d(r['acc_a'])} & {F.d(r['acc_b'])} & "
                    f"{d(r['diff'])} & [{d(r['ci_lo'])}{sep}{d(r['ci_hi'])}] & {pv} \\\\")
    return ('\\begin{table}[ht]\n\\centering\\small\n\\caption{' + cap + '}\n\\label{tab:si-paired}\n'
            '\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n' + head + '\n\\midrule\n' + '\n'.join(body) +
            '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


COMPARISONS = [
    # (label, a runs, b runs)  -- difference = a - b
    ('mHuBERT-147 vs wav2vec 2.0 Base', V.SPEC['mhubert'], V.SPEC['w2v2']),
    ('MMS-300M vs mHuBERT-147', V.SPEC['mms300m'], V.SPEC['mhubert']),
    ('BC-ResNet-8 multiling. vs scratch', G('bcresnet8', 'mswc', 3e-3), G('bcresnet8', 'scratch', 3e-3)),
    ('BC-ResNet-1 multiling. vs scratch', G('bcresnet1', 'mswc', 1e-2), G('bcresnet1', 'scratch', 1e-2)),
    ('LeTR-128 multiling. vs scratch', G('letr128t', 'mswc', 1e-3), G('letr128t', 'scratch', 1e-3)),
    ('BC-ResNet-1 (multiling.) vs LeTR-128 (GSC)', V.SPEC['bcresnet1'], V.SPEC['letr128t']),
    ('BC-ResNet-8 100 % vs 50 % of training groups', G('bcresnet8', 'mswc', 3e-3), G('bcresnet8', 'mswc', 3e-3, subset='train_frac_50')),
    ('LeTR-128 100 % vs 50 %', V.SPEC['letr128t'], G('letr128t', 'gsc', 1e-3, subset='train_frac_50')),
    ('MMS-300M 100 % vs 50 %', V.SPEC['mms300m'], G('mms300m', 'ssl', 3e-5, subset='train_frac_50')),
]

if __name__ == '__main__':
    rows = []
    for lab, a, b in COMPARISONS:
        r = paired(a, b); r['comparison'] = lab; r['seeds_a'] = len(a); r['seeds_b'] = len(b)
        rows.append(r)
        print(f'{lab:48s} {r["acc_a"]:5.1f} vs {r["acc_b"]:5.1f}  diff {r["diff"]:+5.2f}  95% CI [{r["ci_lo"]:+5.2f}, {r["ci_hi"]:+5.2f}]'
              f'  p(<=0)={r["p_le0"]:.3f}  seeds {len(a)}/{len(b)}')
    cols = ['comparison', 'seeds_a', 'seeds_b', 'n_clips', 'n_groups', 'acc_a', 'acc_b', 'diff', 'ci_lo', 'ci_hi', 'p_le0']
    pd.DataFrame(rows)[cols].round(4).to_csv(os.path.join(V.DATA, 'paired_bootstrap.csv'), index=False)
    for lang in ('en', 'fr'):
        open(os.path.join(V.TAB[lang], 'si_tab_paired.tex'), 'w').write(table(rows, lang))
