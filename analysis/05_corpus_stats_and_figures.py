"""Corpus statistics (data/stats_corpus.json) and descriptive figures (figures/*.pdf)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, FIG = os.path.join(HERE, '..', 'data'), os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

# ---- style (reference palette, light/print) ---------------------------------------------
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})
COL1 = 3.35  # single-column width (in)

f = pd.read_csv(os.path.join(DATA, 'field_recordings.csv'), parse_dates=['ts'])
f['human_label'] = f.human_label.fillna('')
v = pd.read_csv(os.path.join(DATA, 'volunteer_recordings.csv'), parse_dates=['ts'])
vp = pd.read_csv(os.path.join(DATA, 'volunteer_profiles_summary.csv'))
S = {}

# ---- volunteer phase -------------------------------------------------------------------
S['volunteer'] = dict(
    profiles=int(vp.n.sum()), contributors=int(v.user_id.nunique()), recordings=len(v),
    hours=round(v.duration_s.sum() / 3600, 2), start=str(v.ts.min().date()), end=str(v.ts.max().date()),
    days=int((v.ts.max() - v.ts.min()).days) + 1,
    recs_per_contributor_median=float(v.groupby('user_id').size().median()),
    contributors_full_list=int((v.groupby('user_id').word_fr.nunique() >= 30).sum()),
    gender_recs=v.gender.value_counts().to_dict(), age_recs=v.age.value_counts().to_dict(),
    profiles_gender=vp.groupby('gender').n.sum().to_dict(), profiles_age=vp.groupby('ageRange').n.sum().to_dict(),
    empty=int((v.duration_s.fillna(0) < 0.05).sum()),
    sample_rates=v.sample_rate.value_counts().to_dict())

# ---- field phase -----------------------------------------------------------------------
ne = f[~f['empty']]
grp = ne.groupby('group_id').size()
S['field'] = dict(
    recordings=len(f), agent_accounts=int(f.agent_id.nunique()),
    recordings_per_agent=f.agent_id.value_counts().to_dict(),
    top3_share=round(f.agent_id.value_counts().iloc[:3].sum() / len(f), 4),
    start=str(f.ts.min().date()), end=str(f.ts.max().date()), days_active=int(f.day.nunique()),
    span_days=int((f.ts.max().normalize() - f.ts.min().normalize()).days) + 1,
    hours_raw=round(f.duration_s.sum() / 3600, 2), empty=int(f['empty'].sum()), empty_pct=round(100 * f['empty'].mean(), 1),
    nonempty=len(ne), dur_nonempty_median=round(ne.duration_s.median(), 2), dur_nonempty_mean=round(ne.duration_s.mean(), 2),
    sample_rate=f.sample_rate.value_counts().to_dict(),
    gender=f.gender.value_counts().to_dict(), age=f.age.value_counts().to_dict(),
    words=int(f.word_fr.nunique()), per_word_min=int(f.word_fr.value_counts().min()), per_word_max=int(f.word_fr.value_counts().max()),
    groups=int(f.group_id.nunique()),
    est_speakers_from_groups=int(np.ceil(grp / 30).sum()),       # >= one participant per 30 prompts
    est_speakers_from_prompts=int(round(len(ne) / 30)),
    max_recs_per_day=int(f.groupby('day').size().max()), mean_recs_per_active_day=round(f.groupby('day').size().mean(), 0),
    agent_empty_pct={a: round(100 * g['empty'].mean(), 1) for a, g in f.groupby('agent_id')})

# ---- triage vs human review (exclusive bins) ------------------------------------------
rows = []
for b in ['clean', 'review', 'error', 'clean+error']:
    g = f[f.auto_triage == b]
    rev = g[g.human_label != '']
    rows.append(dict(bin=b, n=len(g), empty=int(g['empty'].sum()), reviewed=len(rev),
                     accepted=int((rev.human_label == 'accepted').sum()),
                     acc_rate=round(float((rev.human_label == 'accepted').mean()), 3) if len(rev) else None))
S['triage'] = rows
S['human_review'] = dict(reviewed=int((f.human_label != '').sum()), accepted=int((f.human_label == 'accepted').sum()),
                         rejected=int((f.human_label == 'rejected').sum()))
u = f[f['usable']]
S['usable'] = dict(n=len(u), verified=int((u.human_label == 'accepted').sum()),
                   auto_only=int((u.human_label != 'accepted').sum()),
                   gender=u.gender.value_counts().to_dict(), age=u.age.value_counts().to_dict(),
                   agents=u.agent_id.value_counts().to_dict(), per_word_min=int(u.word_fr.value_counts().min()),
                   per_word_max=int(u.word_fr.value_counts().max()), groups=int(u.group_id.nunique()))
z = np.load(os.environ['POOL'])
S['usable']['hours'] = round(float(z['lengths'].sum()) / 16000 / 3600, 2)
S['usable']['dur_median'] = round(float(np.median(z['lengths'])) / 16000, 2)
S['usable']['dur_p95'] = round(float(np.quantile(z['lengths'], .95)) / 16000, 2)
sp = pd.read_csv(os.path.join(DATA, 'splits', 'splits_usable.csv'))
S['splits'] = {s: dict(n=int((sp.speaker == s).sum()), groups=int(sp.group_id[sp.speaker == s].nunique()),
                       verified=int(((sp.speaker == s) & (sp.human_label == 'accepted')).sum()))
               for s in ('train', 'valid', 'test')}
json.dump(S, open(os.path.join(DATA, 'stats_corpus.json'), 'w'), indent=1, default=str)
print(json.dumps(S, indent=1, default=str))

# ---- Figure: cumulative collection -----------------------------------------------------
fig, ax = plt.subplots(figsize=(COL1, 1.9))
for d, lab, c in ((v, 'Volunteer web platform', C1), (f, 'Paid field agents', C2)):
    s = d.set_index('ts').sort_index().assign(n=1).n.cumsum()
    ax.step(s.index, s.values, where='post', color=c, lw=1.6, label=lab)
    ax.annotate(f'{s.values[-1]:,}', (s.index[-1], s.values[-1]), xytext=(3, 0), textcoords='offset points',
                va='center', fontsize=7, color=INK)
ax.set_yscale('log'); ax.set_ylim(1, 6e4)
ax.set_ylabel('Cumulative recordings (log)')
ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.set_xlim(pd.Timestamp('2024-03-25'), pd.Timestamp('2024-08-31'))
ax.legend(loc='upper left')
fig.savefig(os.path.join(FIG, 'fig_collection_timeline.pdf'))

# ---- Figure: corpus composition (3 panels) -------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.75), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.38))
ax = axs[0]
bins = np.arange(0, 4.01, 0.1)
ax.hist(np.clip(z['lengths'] / 16000, 0, 4), bins=bins, color=C2, rwidth=0.85, label='Usable (trimmed)')
ax.hist(ne.duration_s.clip(upper=4), bins=bins, histtype='step', color=C1, lw=1.2, label='Raw field clips')
ax.set_xlabel('Duration (s)'); ax.set_ylabel('Recordings'); ax.legend(loc='upper right'); ax.set_title('(a) Clip duration', loc='left')
ax = axs[1]
ages = ['15-18', '18-35', '35-65', '65+']
gm = u[u.gender == 'Homme'].age.value_counts().reindex(ages, fill_value=0)
gf = u[u.gender == 'Femme'].age.value_counts().reindex(ages, fill_value=0)
x = np.arange(len(ages)); w = 0.38
ax.bar(x - w / 2 - 0.01, gm.values, w, color=C1, label='Male')
ax.bar(x + w / 2 + 0.01, gf.values, w, color=C2, label='Female')
ax.set_xticks(x, ages); ax.set_xlabel('Declared age range'); ax.set_ylabel('Usable recordings')
ax.legend(loc='upper right'); ax.set_title('(b) Speaker demographics', loc='left'); ax.grid(axis='x', visible=False)
ax = axs[2]
ag = f.groupby('agent_id').agg(n=('oid', 'size'), empty=('empty', 'sum'), usable=('usable', 'sum'))
ag = ag.sort_values('n', ascending=False).iloc[:3]
other = ag.n - ag.usable - ag['empty']
y = np.arange(len(ag))[::-1]
ax.barh(y, ag.usable, color=C3, label='Usable', height=0.6)
ax.barh(y, other, left=ag.usable + 40, color=C4, label='Discarded', height=0.6)
ax.barh(y, ag['empty'], left=ag.usable + other + 80, color='#c3c2b7', label='Empty', height=0.6)
ax.set_yticks(y, [f'Agent {i}' for i in ag.index]); ax.set_xlabel('Recordings'); ax.grid(axis='y', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=6.5, handlelength=1, columnspacing=0.8)
ax.set_title('(c) Yield per field agent', loc='left')
fig.savefig(os.path.join(FIG, 'fig_corpus_composition.pdf'))

# ---- Figure: triage vs human review ----------------------------------------------------
t = pd.DataFrame(rows).set_index('bin').loc[['clean', 'review', 'error']]
fig, ax = plt.subplots(figsize=(COL1, 1.35))
y = np.arange(3)[::-1]
ax.barh(y, t.acc_rate * 100, color=C1, height=0.55)
for yi, (b, r) in zip(y, t.iterrows()):
    ax.text(r.acc_rate * 100 + 1.5, yi, f'{100 * r.acc_rate:.0f}%  ({int(r.accepted):,}/{int(r.reviewed):,})', va='center', fontsize=7, color=INK)
ax.set_yticks(y, ['Clean (1 speech turn)', 'Review (2-4 turns)', 'Error (other)'])
ax.set_xlim(0, 115); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Accepted by human reviewers (%)')
ax.grid(axis='y', visible=False)
fig.savefig(os.path.join(FIG, 'fig_triage_vs_human.pdf'))
print('figures written')
