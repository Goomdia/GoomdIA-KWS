"""Subgroup table of the v4 manuscript (tab:subgroups), English and French: BC-ResNet-8 (selected configuration),
mHuBERT-147 and MMS-300M on subsets of the speaker-disjoint field test set, plus the field test set and the OOD set as a whole.
Output: manuscript/tables/tab_subgroups.tex, manuscript/tables_fr/tab_subgroups.tex
"""
import os
import v4common as V

MODELS = ['bcresnet8', 'mhubert', 'mms300m']
G = {s: V.subgroup_acc(V.SPEC[s]) if len(V.SPEC[s]) else None for s in MODELS}

TXT = dict(
    en=dict(cap='Accuracy (\\%) on subsets of the speaker-disjoint test set (mean over three seeds); $n$: number of test clips. '
                   'BC-ResNet-8: configuration selected on validation accuracy. The last two rows give the accuracy on the whole field test set and on the '
                   'out-of-domain (OOD) set of volunteer recordings',
               sub='Subset', hdr=['BC-ResNet-8', 'mHuBERT-147', 'MMS-300M'],
               groups=[('Declared gender', 'gender', ['Femme', 'Homme'], {'Femme': 'female', 'Homme': 'male'}),
                       ('Declared age range', 'age', ['15-18', '18-35', '35-65'], {}),
                       ('Field agent', 'agent_id', ['A1', 'A2', 'A3'], {}),
                       ('Label source', 'verified', ['True', 'False'], {'True': 'human-verified', 'False': 'automatic triage only'})],
               overall=('Whole test sets', 'field test set', 'OOD set (volunteers)')),
    fr=dict(cap='Exactitude (\\%) sur des sous-ensembles du jeu de test disjoint en locuteurs (moyenne sur trois graines) ; $n$ : nombre de clips de test. '
                   'BC-ResNet-8 : configuration choisie sur l\'exactitude en validation. Les deux dernières lignes donnent l\'exactitude sur l\'ensemble du jeu de test '
                   'de terrain et sur le jeu hors domaine (OOD) des enregistrements de bénévoles',
               sub='Sous-ensemble', hdr=['BC-ResNet-8', 'mHuBERT-147', 'MMS-300M'],
               groups=[('Genre déclaré', 'gender', ['Femme', 'Homme'], {'Femme': 'femmes', 'Homme': 'hommes'}),
                       ('Tranche d\'âge déclarée', 'age', ['15-18', '18-35', '35-65'], {}),
                       ('Agent de terrain', 'agent_id', ['A1', 'A2', 'A3'], {}),
                       ('Origine de l\'étiquette', 'verified', ['True', 'False'], {'True': 'vérifiée par un humain', 'False': 'tri automatique seulement'})],
               overall=('Jeux de test complets', 'jeu de test de terrain', 'jeu OOD (bénévoles)')))


def table(lang):
    t, F = TXT[lang], V.Fmt(lang)
    ncol = len(MODELS) + 2
    rows = []
    for title, key, order, lab in t['groups']:
        rows.append(f'\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
        for v in order:
            vals = [(G[m][key].get(v) if G[m] else None) for m in MODELS]
            n = next((x[1] for x in vals if x), None)
            rows.append(f'\\quad {lab.get(v, v.replace("-", "--"))} & {n if n else "--"} & ' +
                        ' & '.join(F.pct(x[0]) if x else '--' for x in vals) + ' \\\\')
    rows.append(f'\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{t["overall"][0]}}}}} \\\\')
    allf = [(G[m]['all'] if G[m] else None) for m in MODELS]
    allo = [(G[m]['ood'].get('all') if G[m] else None) for m in MODELS]
    for lab, vals in ((t['overall'][1], allf), (t['overall'][2], allo)):
        n = next((x[1] for x in vals if x), None)
        rows.append(f'\\quad {lab} & {n if n else "--"} & ' + ' & '.join(F.pct(x[0]) if x else '--' for x in vals) + ' \\\\')
    return '\n'.join(['\\begin{table}[t]', '\\centering\\small', f'\\caption{{{t["cap"]}}}', '\\label{tab:subgroups}',
                      f'\\begin{{tabular}}{{@{{}}lr{"c" * len(MODELS)}@{{}}}}', '\\toprule',
                      f'{t["sub"]} & $n$ & ' + ' & '.join(t['hdr']) + ' \\\\', '\\midrule', *rows, '\\bottomrule', '\\end{tabular}', '\\end{table}', ''])


if __name__ == '__main__':
    for lang in ('en', 'fr'):
        open(os.path.join(V.TAB[lang], 'tab_subgroups.tex'), 'w').write(table(lang))
    print(table('en'))
