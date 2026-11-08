"""Verification of manuscript claims 1, 2, 3 (metadata part) and 4. Read-only, CPU."""
import os, json, glob, re
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
m = pd.read_csv(f'{R}/data/field_recordings.csv', parse_dates=['ts'])
m = m.sort_values(['group_id', 'ts', 'oid']).reset_index(drop=True)
m['pos'] = m.groupby('group_id').cumcount()
m['emp'] = m['empty']
print('== general', len(m), 'groups', m.group_id.nunique(), 'empty', m.emp.sum(), round(100*m.emp.mean(), 2), 'usable', m.usable.sum())

# ---- claim 2
ne = (~m.emp).sum()
gs = m.groupby('group_id').size()
print('== claim2 non-empty/30 =', ne, ne/30, ' | sum ceil(n/30) =', int(np.ceil(gs/30).sum()), ' all/30=', len(m)/30,
      ' groups=', len(gs), ' group size dist:', gs.describe().round(1).to_dict(),
      ' groups with n%30==0:', int((gs % 30 == 0).sum()), ' n<=30:', int((gs <= 30).sum()), ' n>30:', int((gs > 30).sum()))
print('   sum floor(n/30) =', int((gs//30).sum()), ' groups with remainder', int((gs % 30 > 0).sum()), ' remainder<10:', int(((gs%30>0)&(gs%30<10)).sum()))
# per agent/day sessions
print('   agents x days:', m.groupby(['agent_id', 'day']).ngroups if False else m.groupby(['agent_id','day']).size().shape[0])

# ---- claim 1
def repeats(df, label):
    out = {}
    g = df.groupby(['group_id', 'word_fr'])
    sz = g.size()
    out['pairs'] = len(sz); out['pairs_with_repeat'] = int((sz > 1).sum()); out['clips_in_repeat_pairs'] = int(sz[sz > 1].sum())
    out['extra_clips(n-1)'] = int((sz - 1).sum())
    # consecutive occurrences
    df = df.sort_values(['group_id', 'word_fr', 'ts'])
    dt = df.groupby(['group_id', 'word_fr']).ts.diff().dt.total_seconds()
    dp = df.groupby(['group_id', 'word_fr']).pos.diff()
    k = dt.notna()
    out['consecutive_repeat_pairs'] = int(k.sum())
    out['dt<60s'] = int((dt < 60).sum()); out['dt<60s&gap<30prompts'] = int(((dt < 60) & (dp < 30)).sum())
    out['dt<30s'] = int((dt < 30).sum())
    out['dt>=300s'] = int((dt >= 300).sum())
    out['median_dt_s'] = float(dt.median())
    print('==', label, out)
    return df.assign(dt=dt, dp=dp)
allr = repeats(m, 'claim1 ALL clips (by group)')
us = m[m.usable]
usr = repeats(us, 'claim1 USABLE clips')
# histogram of dt for all repeats
bins = [0, 10, 30, 60, 120, 300, 900, 3600, 1e9]
print(' dt histogram (all, consecutive occurrences):', pd.cut(allr.dt.dropna(), bins).value_counts().sort_index().to_dict())
# size of groups with repeats: groups with n>30 explained by several participants
gsz = m.groupby('group_id').size()
rep_groups = allr[allr.dt.notna()].group_id.unique()
print(' groups with any repeat:', len(rep_groups), 'of', len(gsz), '; among groups with n<=30:', int((gsz.loc[rep_groups] <= 30).sum()))
# block-aware: for groups with n>30, split by position into blocks of 30; repeats within same 30-block position
m['blk'] = m.pos // 30
b = m.groupby(['group_id', 'blk', 'word_fr']).size()
print(' repeats WITHIN a positional 30-block (impossible if block = permutation):', int((b > 1).sum()), 'extra clips', int((b-1).sum()))

# splits
s = pd.read_csv(f'{R}/data/splits/splits_usable.csv')
s = s.merge(m[['oid', 'ts', 'pos']], on='oid')
for split in ('speaker', 'random'):
    tr = s[s[split] == 'train']; te = s[s[split] == 'test']
    key = lambda d: set(zip(d.group_id, d.word_fr))
    trk = key(tr)
    shared = te.apply(lambda r: (r.group_id, r.word_fr) in trk, axis=1)
    # likely retake: train clip same group/word within 60 s (and <30 prompts apart)
    trg = {k: v for k, v in tr.groupby(['group_id', 'word_fr'])}
    def close(r, T=60, P=30):
        g = trg.get((r.group_id, r.word_fr))
        if g is None: return False
        dt = (g.ts - r.ts).abs().dt.total_seconds(); dp = (g.pos - r.pos).abs()
        return bool(((dt < T) & (dp < P)).any())
    c60 = te.apply(close, axis=1)
    c300 = te.apply(lambda r: close(r, 300, 10**9), axis=1)
    c10 = te.apply(lambda r: close(r, 10, 10**9), axis=1)
    print(f'== split={split}: test={len(te)} train={len(tr)} groups shared train/test={len(set(tr.group_id)&set(te.group_id))}'
          f' | test clips sharing (group,keyword) with a train clip: {int(shared.sum())} ({100*shared.mean():.2f}%)'
          f' | with train clip <60s&<30 prompts: {int(c60.sum())} ({100*c60.mean():.2f}%) | <300s: {int(c300.sum())} | <10s: {int(c10.sum())}')
    # candidate retakes among all test-train pairs
    # distinct-participant plausibility: test groups with n>30
    print('   shared clips whose group has >30 clips (multi-participant plausible):', int((shared & te.group_id.map(gsz).gt(30)).sum()),
          ' ; group size<=30:', int((shared & te.group_id.map(gsz).le(30)).sum()))

# ---- claim 3 (metadata part)
e = m[m.emp]
print('== claim3: empty n', len(e), 'dur0', int((e.duration_s == 0).sum()), 'usable among empty', int(e.usable.sum()), 'triage', e.auto_triage.value_counts().to_dict(),
      'sample_rate', e.sample_rate.value_counts().head().to_dict(), 'nonempty sr', m[~m.emp].sample_rate.value_counts().head().to_dict())
print(' by agent (n, empty, %):'); t = m.groupby('agent_id').emp.agg(['size', 'sum', 'mean']); t['mean'] = (100*t['mean']).round(1); print(t.to_string())
t = m.groupby('day').emp.agg(['size', 'sum', 'mean']); t['mean'] = (100*t['mean']).round(1); print(' by day:\n', t.to_string())
m['hour'] = m.ts.dt.hour
t = m.groupby('hour').emp.agg(['size', 'sum', 'mean']); t['mean'] = (100*t['mean']).round(1); print(' by hour:\n', t.to_string())
t = m.groupby('word_fr').emp.mean().mul(100).round(1); print(' by word: min/max', t.min(), t.max())
# per group
gg = m.groupby('group_id').emp.agg(['size', 'sum']); gg['f'] = gg['sum']/gg['size']
print(' groups with no empty:', int((gg['sum'] == 0).sum()), ' all-empty groups:', int((gg.f == 1).sum()), ' groups>50% empty:', int((gg.f > .5).sum()),
      ' empties in groups>50%:', int(gg.loc[gg.f > .5, 'sum'].sum()), ' dispersion: var/mean of per-group empty count vs binomial')
p = m.emp.mean(); exp_var = (gg['size']*p*(1-p)).sum(); print('   obs var of per-group empties around expected: ', float(((gg['sum']-gg['size']*p)**2).sum()), 'vs binomial expectation', float(exp_var))
# runs
m['run'] = (m.emp != m.emp.shift()) | (m.group_id != m.group_id.shift())
m['rid'] = m.run.cumsum()
runs = m[m.emp].groupby('rid').size()
print(' runs of consecutive empties: n runs', len(runs), ' length dist', runs.value_counts().sort_index().head(12).to_dict(), ' max', runs.max())
# expected number of runs under independence
n = len(m); print('   P(empty)=%.4f, P(empty|prev empty)=%.4f' % (p, (m.emp & m.emp.shift(fill_value=False) & (m.group_id == m.group_id.shift())).sum()/max(1, (m.emp.shift(fill_value=False) & (m.group_id == m.group_id.shift())).sum())))
# position in 30-block and in group
m['bpos'] = m.pos % 30
t = m.groupby('bpos').emp.mean().mul(100).round(1); print(' empty % by position in 30-block:', t.to_dict())
last = m.groupby('group_id').tail(1); print(' last clip of group empty %: ', round(100*last.emp.mean(), 1), ' first clip:', round(100*m.groupby('group_id').head(1).emp.mean(), 1))
m['gsize'] = m.group_id.map(gsz); m['rev'] = m.gsize - 1 - m.pos
print(' empty % by distance from group end (0..5):', m[m.rev < 6].groupby('rev').emp.mean().mul(100).round(1).to_dict())
# gap to previous clip
m['gap'] = m.groupby('group_id').ts.diff().dt.total_seconds()
print(' median gap to previous clip, empty vs non-empty:', m[m.emp].gap.median(), m[~m.emp].gap.median())
# gap to next clip
m['gapn'] = -m.groupby('group_id').ts.diff(-1).dt.total_seconds()
print(' median gap to next clip, empty vs non-empty:', m[m.emp].gapn.median(), m[~m.emp].gapn.median())
print(' duplicate timestamps within agent: ', int(m.duplicated(['agent_id', 'ts']).sum()))
m[['oid','group_id','pos','emp','hour','gap']].to_csv(os.path.join(os.path.dirname(__file__), 'empty_meta.csv'), index=False)

# ---- claim 4
rows = []
for d in sorted(glob.glob(f'{R}/experiments/runs/*/summary.json')):
    S = json.load(open(d)); name = S['args']['name']
    h = S.get('history', [])
    tt = h[-1]['time_s'] if h else np.nan
    cfg = re.sub(r'-s\d+$', '', name)
    rows.append(dict(name=name, cfg=cfg, model=S['args']['model'], init=S['args'].get('init'), split=S['args']['split'], subset=S['args'].get('train_subset'),
                     epochs_run=len(h), epochs_max=S['args'].get('epochs'), best_ep=int(np.argmax([x['val_acc'] for x in h]))+1 if h else np.nan,
                     train_s=tt, per_epoch=tt/len(h) if h else np.nan, n_train=S.get('n_train'), n_params=S.get('n_params'), bs=S['args'].get('bs'), extra=','.join(k for k in S if k not in ('args','history','test','ood'))))
r = pd.DataFrame(rows); r.to_csv(os.path.join(os.path.dirname(__file__), 'training_cost_per_run.csv'), index=False)
print('== claim4 runs with summary:', len(r), ' summary keys extra:', r.extra.iloc[0])
pd.set_option('display.width', 250)
g = r.groupby('cfg').agg(runs=('name', 'size'), epochs=('epochs_run', 'mean'), time_min=('train_s', lambda x: x.mean()/60), time_s_sd=('train_s', 'std'), s_per_epoch=('per_epoch', 'mean'), n_train=('n_train', 'mean'))
print(g.round(2).to_string())
r['fam'] = np.where(r.model.str.startswith('w2v2'), r.model, r.model + '-' + r.init.astype(str))
print(r.groupby(['model', 'init']).agg(runs=('name', 'size'), epochs=('epochs_run', 'mean'), time_min=('train_s', lambda x: x.mean()/60), min_min=('train_s', lambda x: x.min()/60), max_min=('train_s', lambda x: x.max()/60), s_per_epoch=('per_epoch', 'mean')).round(2).to_string())
print(' total training time (h):', round(r.train_s.sum()/3600, 2), ' main speaker/random only (lr 1e-3):')
print(r[r.cfg.str.contains('lr0.001$|lr3e-05$') & ~r.cfg.str.contains('ctrl|frac')].groupby(['model','init','split']).agg(runs=('name','size'),ep=('epochs_run','mean'),min=('train_s',lambda x:x.mean()/60),spe=('per_epoch','mean')).round(2).to_string())

# ---- claim 1 supplement: prompt-distance histogram of consecutive repeats, and block-simulation
print('== claim1 supplement: prompt distance dp of consecutive same-keyword occurrences (all clips)')
ad = allr[allr.dp.notna()]
print(' dp histogram:', pd.cut(ad.dp, [0,1,2,5,10,20,29,30,40,59,100,10**5]).value_counts().sort_index().to_dict())
print(' dp==1 (immediate retake):', int((ad.dp == 1).sum()), ' dp<=5:', int((ad.dp <= 5).sum()), ' dt<60 &dp<=5:', int(((ad.dt<60)&(ad.dp<=5)).sum()))
# simulated expectation: for random 30-permutation blocks, consecutive occurrence distance dp has a triangular law over 1..59; P(dp<=5)
rng = np.random.default_rng(0); d = []
prev = None
for _ in range(20000):
    a = rng.permutation(30); b = rng.permutation(30)
    d.append(30 - np.where(a == 0)[0][0] + np.where(b == 0)[0][0])
d = np.array(d); print(' simulated consecutive blocks: P(dp<=5)=%.3f P(dp==1)=%.3f P(dp<30)=%.3f' % ((d <= 5).mean(), (d == 1).mean(), (d < 30).mean()))
print(' observed:   P(dp<=5)=%.3f P(dp==1)=%.3f P(dp<30)=%.3f' % ((ad.dp <= 5).mean(), (ad.dp == 1).mean(), (ad.dp < 30).mean()))
# random split: test clips sharing with train clip under various retake definitions (dp-based)
s2 = pd.read_csv(f'{R}/data/splits/splits_usable.csv').merge(m[['oid','ts','pos']], on='oid')
tr = s2[s2.random=='train']; te = s2[s2.random=='test']
trg = {k: v for k, v in tr.groupby(['group_id','word_fr'])}
def mind(r):
    g = trg.get((r.group_id, r.word_fr)); 
    return np.nan if g is None else (g.pos - r.pos).abs().min()
te = te.assign(mdp=te.apply(mind, axis=1))
print(' random split: test clips with shared (group,kw):', int(te.mdp.notna().sum()), ' min prompt distance to a train twin: dp<=5:', int((te.mdp<=5).sum()), ' dp<=29:', int((te.mdp<30).sum()), ' dp>=30:', int((te.mdp>=30).sum()))
print(' (usable-only positions are clip positions among ALL clips of the group, so empties/rejected count as prompts)')
