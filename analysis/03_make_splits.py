"""Create the evaluation splits of the 16,699 usable recordings.

speaker : recording groups (see 02_build_metadata.py) are assigned as a whole to train/valid/test,
          so that no speaker appears in two subsets (group-disjoint, ~80/10/10 of recordings).
random  : utterance-level random split with exactly the same subset sizes (the protocol used
          during development), to measure the effect of speaker overlap.
Also writes nested training subsets (10/25/50 % of the training groups) for the data-scaling study.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
os.makedirs(os.path.join(DATA, 'splits'), exist_ok=True)
SEED = 2024

m = pd.read_csv(os.path.join(DATA, 'field_recordings.csv'))
u = m[m.usable].copy().reset_index(drop=True)
rng = np.random.default_rng(SEED)

# --- speaker(group)-disjoint split -------------------------------------------------------
groups = u.groupby('group_id').size()
perm = rng.permutation(groups.index.to_numpy())
cum = groups.loc[perm].cumsum() / len(u)
test_g = set(perm[cum.to_numpy() <= 0.10])
valid_g = set(perm[(cum.to_numpy() > 0.10) & (cum.to_numpy() <= 0.20)])
u['speaker'] = np.where(u.group_id.isin(test_g), 'test', np.where(u.group_id.isin(valid_g), 'valid', 'train'))

# --- random split with identical sizes ---------------------------------------------------
sizes = u.speaker.value_counts()
idx = rng.permutation(len(u))
lab = np.empty(len(u), dtype=object)
lab[idx[:sizes['test']]] = 'test'
lab[idx[sizes['test']:sizes['test'] + sizes['valid']]] = 'valid'
lab[idx[sizes['test'] + sizes['valid']:]] = 'train'
u['random'] = lab

# --- nested training-group subsets for the scaling study (speaker split) ----------------
tr_groups = rng.permutation(np.array(sorted(set(u.group_id[u.speaker == 'train']))))
for frac in (0.10, 0.25, 0.50):
    keep = set(tr_groups[:max(1, int(round(frac * len(tr_groups))))])
    u[f'train_frac_{int(frac*100)}'] = (u.speaker == 'train') & u.group_id.isin(keep)

# --- controlled seen/unseen-speaker experiment (identical test utterances) ---------------
# Each test group is cut in two random halves: A (test) and B. ctrl_unseen trains on the full speaker
# training set; ctrl_seen replaces |B| random training utterances by B, i.e. by other utterances of the
# test speakers. Both training sets have the same size and both use the same validation and test set A.
rng2 = np.random.default_rng(SEED + 1)
half = np.zeros(len(u), dtype=bool)
for g, idx in u[u.speaker == 'test'].groupby('group_id').groups.items():
    idx = rng2.permutation(np.array(list(idx)))
    half[idx[:len(idx) // 2]] = True                       # half B (goes to training in ctrl_seen)
tr_idx = np.where(u.speaker == 'train')[0]
removed = set(rng2.choice(tr_idx, size=int(half.sum()), replace=False))
rem = np.isin(np.arange(len(u)), list(removed))
for name, add_b in (('ctrl_unseen', False), ('ctrl_seen', True)):
    lab = np.array(['none'] * len(u), dtype=object)
    lab[(u.speaker == 'train').to_numpy() & (~rem if add_b else True)] = 'train'
    lab[(u.speaker == 'valid').to_numpy()] = 'valid'
    lab[(u.speaker == 'test').to_numpy() & ~half] = 'test'
    if add_b: lab[(u.speaker == 'test').to_numpy() & half] = 'train'
    u[name] = lab

cols = ['oid', 'group_id', 'agent_id', 'word_fr', 'gender', 'age', 'auto_triage', 'human_label',
        'speaker', 'random', 'train_frac_10', 'train_frac_25', 'train_frac_50', 'ctrl_unseen', 'ctrl_seen']
tmp = os.path.join(DATA, 'splits', 'splits_usable.csv.tmp')
u[cols].to_csv(tmp, index=False)
os.replace(tmp, os.path.join(DATA, 'splits', 'splits_usable.csv'))     # atomic: runs may be reading it
for name in ('ctrl_unseen', 'ctrl_seen'):
    print(name, u[name].value_counts().to_dict())

print('subset sizes:', sizes.to_dict())
for s in ('train', 'valid', 'test'):
    g = u[u.speaker == s]
    print(f'{s}: groups={g.group_id.nunique()} recs={len(g)} agents={sorted(g.agent_id.unique())} '
          f'verified={(g.human_label == "accepted").sum()} min/max per class={g.word_fr.value_counts().min()}/{g.word_fr.value_counts().max()}')
for frac in (10, 25, 50):
    print(f'train_frac_{frac}:', u[f"train_frac_{frac}"].sum())
