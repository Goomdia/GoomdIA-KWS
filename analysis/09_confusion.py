"""Per-keyword accuracy and confusion matrices of the selected BC-ResNet-8 and of MMS-300M on the speaker-disjoint test set.

Outputs: figures/fig_per_keyword.pdf   (a: per-keyword accuracy, BC-ResNet-8 bars and MMS-300M dots; b: off-diagonal confusion of BC-ResNet-8)
         figures/fig_confusion_full.pdf (Online Resource 1, Fig. S2: both full matrices, every non-zero cell annotated)
         data/confusion_bcresnet8.csv, data/confusion_mms300m.csv (row-normalised, %, seeds pooled; same keyword order in both:
         hardest to easiest keyword of BC-ResNet-8)
BC-ResNet-8: init and LR selected on validation (v4common.BC_SEL); MMS-300M: lr 3e-5; seeds pooled.
"""
import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import v4common as V

plt = V.style()
C1, C2, INK, INK2, GRID = V.C1, V.C2, V.INK, V.INK2, V.GRID
B, M = V.SPEC['bcresnet8'], V.SPEC['mms300m']
sel = V.BC_SEL['bcresnet8']
PB, PM = V.pooled(B), V.pooled(M)
acc_b, acc_m = V.per_keyword(PB).sort_values(), V.per_keyword(PM)
kw = list(acc_b.index)                                 # rows and columns ordered from hardest to easiest keyword of BC-ResNet-8
CB, CM = V.confusion(PB, kw), V.confusion(PM, kw)
CB.round(2).to_csv(os.path.join(V.DATA, 'confusion_bcresnet8.csv'))
CM.round(2).to_csv(os.path.join(V.DATA, 'confusion_mms300m.csv'))
cmap = LinearSegmentedColormap.from_list('err', ['#ffffff', '#f7c9b3', C2, '#8a2a08'])
vmax = 12


def matrix(ax, C, thr, fs, labels_fs=None):
    E = C.values.copy(); np.fill_diagonal(E, np.nan)    # errors only: the diagonal is the accuracy of panel (a)
    ax.imshow(np.where(np.isnan(E), np.nan, np.clip(E, 0, vmax)), cmap=cmap, vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
    for i in range(len(kw)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=GRID, lw=0))
        for j in range(len(kw)):
            if i != j and E[i, j] >= thr:
                ax.text(j, i, f'{E[i, j]:.0f}' if E[i, j] >= 1 else f'{E[i, j]:.1f}'.lstrip('0'), ha='center', va='center',
                        fontsize=fs, color='white' if E[i, j] > 0.6 * vmax else INK)
    kws = dict(fontsize=labels_fs) if labels_fs else {}
    ax.set_xticks(range(len(kw)), kw, rotation=90, **kws); ax.set_yticks(range(len(kw)), kw, **kws)
    ax.set_xlabel('Predicted keyword'); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(True); s.set_color(GRID)
    ax.set_xlim(-0.5, len(kw) - 0.5); ax.set_ylim(len(kw) - 0.5, -0.5)


# ------------------------------------------------------------------ main figure: (a) accuracy, (b) errors
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 4.3), gridspec_kw=dict(width_ratios=[1, 1.75], wspace=0.04))
y = np.arange(len(kw))
a.barh(y, 100 * acc_b.values, color=C1, height=0.62, label='BC-ResNet-8')
a.scatter(100 * acc_m.reindex(kw).values, y, color=INK, s=8, zorder=3, label='MMS-300M')
lo = 10 * np.floor(min(acc_b.min(), acc_m.min()) * 10)   # axis starts at a multiple of 10 below the smallest accuracy
a.set_yticks(y, kw); a.set_ylim(len(kw) - 0.5, -0.5); a.set_xlim(lo, 102); a.set_xticks(range(int(lo), 101, 10 if 100 - lo <= 40 else 20))
a.set_xlabel('Test accuracy (%)'); a.grid(axis='y', visible=False)
a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=1, handletextpad=0.3)
a.set_title('a', loc='left', fontweight='bold', x=-0.35)
matrix(b, CB, thr=2, fs=6)
b.set_yticks(y, []); b.tick_params(axis='y', length=0)
b.set_title('b', loc='left', fontweight='bold')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
cb = fig.colorbar(sm, ax=b, fraction=0.035, pad=0.015); cb.set_label('Clips of the true keyword (%)'); cb.outline.set_visible(False)
cb.set_ticks([0, 4, 8, 12], labels=['0', '4', '8', '≥12'])
fig.savefig(os.path.join(V.FIG, 'fig_per_keyword.pdf'))

# ------------------------------------------------------------------ Online Resource 1: both full matrices, every non-zero cell annotated
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw=dict(wspace=0.08))
for ax, C, lab in zip(axes, (CB, CM), ('a', 'b')):
    matrix(ax, C, thr=0.05, fs=3.6, labels_fs=5.5)
    ax.set_title(lab, loc='left', fontweight='bold')
axes[0].set_ylabel('True keyword'); axes[1].set_yticklabels([]); axes[1].tick_params(axis='y', length=0)
fig.savefig(os.path.join(V.FIG, 'fig_confusion_full.pdf'))

print(f'BC-ResNet-8 {sel["init"]} lr {sel["lr"]:g}: {B.seed.nunique()} seeds; MMS-300M: {M.seed.nunique()} seeds')
for nm, P, C in (('BC-ResNet-8', PB, CB), ('MMS-300M', PM, CM)):
    top = C.where(~np.eye(len(kw), dtype=bool)).stack().sort_values(ascending=False).head(6)
    print(nm, 'top confusions (% of true-keyword clips):'); print(top.round(1).to_string())
    print(nm, 'Biiga/Piiga', {k: round(v, 1) for k, v in V.pair_stats(P, C).items()})
