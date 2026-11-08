"""Figure: test accuracy against the number of training clips (nested subsets of the training groups), mean +- sd over seeds.

Models: BC-ResNet-8 (init and LR selected on validation, subsets run with them), LeTR-128 (GSC init., lr 1e-3, seeds 0-2),
MMS-300M (full data: 3 seeds; subsets: the seeds that were run, 2). Output: figures/fig_scaling.pdf (included at 0.62 \\textwidth).
"""
import os
import numpy as np
import pandas as pd
import v4common as V

plt = V.style()
import matplotlib
SUBS = ['train_frac_10', 'train_frac_25', 'train_frac_50', 'full']


def curve(model, init, lr, seeds=None):
    rows = []
    for sub in SUBS:
        d = V.get(model, init, lr, subset=sub, seeds=seeds)
        if len(d): rows.append(dict(subset=sub, n_train=int(d.n_train.iloc[0]), mean=d.test.mean(), sd=d.test.std() if len(d) > 1 else np.nan, n=len(d)))
    return pd.DataFrame(rows)


sel = V.BC_SEL['bcresnet8']
bc_init, bc_lr = sel['init'], sel['lr']
if not len(V.get('bcresnet8', bc_init, bc_lr, subset='train_frac_10')):      # subsets were run with another configuration: use it
    alt = V.R[(V.R.model == 'bcresnet8') & (V.R.subset == 'train_frac_10')]
    if len(alt):
        bc_init, bc_lr = alt.init.iloc[0], float(alt.lr.iloc[0])
        print(f'WARNING: BC-ResNet-8 subsets were run with {bc_init} lr {bc_lr:g}, not with the selected configuration {sel["init"]} {sel["lr"]:g}')
C = {'BC-ResNet-8': (V.C2, 's', curve('bcresnet8', bc_init, bc_lr)),
     'LeTR-128 (GSC init.)': (V.C1, 'o', curve('letr128t', 'gsc', 1e-3, seeds=[0, 1, 2])),
     'MMS-300M': (V.C3, 'D', curve('mms300m', 'ssl', 3e-5))}
fig, ax = plt.subplots(figsize=(3.35, 1.9))
for lab, (c, mk, g) in C.items():
    if len(g): ax.errorbar(g.n_train, 100 * g['mean'], yerr=100 * g['sd'].fillna(0), color=c, lw=1.6, marker=mk, ms=4, capsize=2, label=lab)
ax.set_xscale('log'); ax.set_xlabel('Training clips (log scale)'); ax.set_ylabel('Test accuracy (%)')
ax.set_xticks([2174, 4501, 7553, 13376], ['2174', '4501', '7553', '13376'], fontsize=6)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(loc='lower right')
fig.savefig(os.path.join(V.FIG, 'fig_scaling.pdf'))
for lab, (c, mk, g) in C.items():
    print(lab); print(g.round(4).to_string(index=False) if len(g) else '  MISSING')
