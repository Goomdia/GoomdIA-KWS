"""Build an anonymised, per-recording metadata table for the GoomdIA field corpus.

Inputs
  - pickles produced by 01_parse_mongo_exports.py (metadata of the MongoDB exports, no audio)
  - folder listings of the automatic triage (Clean/Review/Error) and of the human review
    (Accepted/Rejected) under Goomdia_/05-machine_learning/Data
Outputs (in ../data)
  - field_recordings.csv   one row per field recording (29,128 rows), agents anonymised (A1..A6)
  - volunteer_recordings.csv  one row per volunteer web recording
No personal data (names, e-mails, audio) is written.
"""
import os, re, sys
import numpy as np
import pandas as pd

PKL = sys.argv[1]                      # folder with agent_w2.pkl, volunteer.pkl, profiles.pkl
RAW = sys.argv[2]                      # .../Goomdia_/05-machine_learning/Data
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
GAP_S = 300                            # max silence between two recordings of the same group

def oids(sub):
    d = os.path.join(RAW, sub)
    return {f[:24] for f in os.listdir(d) if f.endswith('.wav') and re.match(r'[0-9a-f]{24}_', f)}

a = pd.read_pickle(os.path.join(PKL, 'agent_w2.pkl')).sort_values(['agent', 'line']).reset_index(drop=True)

# anonymise agents by decreasing volume
order = a.agent.value_counts().index.tolist()
amap = {ag: f'A{i+1}' for i, ag in enumerate(order)}
a['agent_id'] = a.agent.map(amap)

# recording groups: contiguous recordings of one agent with constant declared gender/age and
# no pause longer than GAP_S. A group may merge several speakers but a speaker session is
# (almost surely) never split across groups -> group-disjoint splits are speaker-disjoint.
gap = a.groupby('agent').ts.diff().dt.total_seconds()
new = (a.agent != a.agent.shift()) | (a.gender != a.gender.shift()) | (a.age != a.age.shift()) | (gap > GAP_S)
a['group_id'] = new.cumsum().astype(int)
a['day'] = a.ts.dt.strftime('%Y-%m-%d')

# automatic triage (pyannote diarisation) and human review labels
au = '02 - Agents - 2eme semaine/Audios/'
clean, review, error = oids(au + 'Clean'), oids(au + 'Review'), oids(au + 'Error')
acc, rej, pool_clean = oids('AiData/raw/Accepted'), oids('AiData/raw/Rejected'), oids('AiData/raw/Clean')

def triage(o):
    c, r, e = o in clean, o in review, o in error
    if c and e: return 'clean+error'          # the two pipeline runs disagreed
    if c: return 'clean'
    if r: return 'review'
    if e: return 'error'
    return 'none'
a['auto_triage'] = a.oid.map(triage)
a['human_label'] = np.where(a.oid.isin(acc), 'accepted', np.where(a.oid.isin(rej), 'rejected', ''))
a['usable'] = a.oid.isin(acc | pool_clean)     # pool used to build the training corpora
a['empty'] = a.dur.fillna(0) < 0.05

cols = ['oid', 'agent_id', 'group_id', 'day', 'ts', 'gender', 'age', 'French', 'Moore', 'dur', 'sr',
        'empty', 'auto_triage', 'human_label', 'usable']
a[cols].rename(columns={'French': 'word_fr', 'Moore': 'word_mos', 'dur': 'duration_s', 'sr': 'sample_rate'}) \
    .to_csv(os.path.join(OUT, 'field_recordings.csv'), index=False)

v = pd.read_pickle(os.path.join(PKL, 'volunteer.pkl'))
p = pd.read_pickle(os.path.join(PKL, 'profiles.pkl'))
umap = {u: f'V{i+1:03d}' for i, u in enumerate(v.user.drop_duplicates())}
v = v.merge(p[['userID', 'gender', 'ageRange']], left_on='user', right_on='userID', how='left', suffixes=('_x', ''))
v['user_id'] = v.user.map(umap)
v[['oid', 'user_id', 'ts', 'gender', 'ageRange', 'French', 'Moore', 'dur', 'sr']] \
    .rename(columns={'French': 'word_fr', 'Moore': 'word_mos', 'dur': 'duration_s', 'sr': 'sample_rate', 'ageRange': 'age'}) \
    .to_csv(os.path.join(OUT, 'volunteer_recordings.csv'), index=False)
p[['gender', 'ageRange']].value_counts().rename('n').reset_index().to_csv(os.path.join(OUT, 'volunteer_profiles_summary.csv'), index=False)
print('field', len(a), 'groups', a.group_id.nunique(), 'usable', a.usable.sum(), '| volunteer', len(v), 'users', v.user_id.nunique())
