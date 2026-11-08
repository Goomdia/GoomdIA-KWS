"""Isolated evaluation of a saved closed-set (30-class) model on the held-out _background_ / _unknown_ test clips
of open_set_clips.npz and on the speaker-disjoint keyword test set (for the comparison with the 32-class
models; a closed-set model can only reject through a threshold on its maximum posterior).
Output: <out>/predictions.csv (oid, subsource, true, pred, confidence) and <out>/summary.json.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from train_kws import LeTRKWS, W2V2KWS, Pool, WaveSet, collate, evaluate, SR

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--extra', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda')
a = ap.parse_args()

s = json.load(open(os.path.join(a.run, 'summary.json')))
pool = Pool(a.cache); k = len(pool.dictionary)
model = W2V2KWS(k) if s['args']['model'] == 'w2v2' else LeTRKWS(s['args']['model'], k)
model.load_state_dict(torch.load(os.path.join(a.run, 'model.pt'), map_location='cpu'))
dev = torch.device(a.device); model.to(dev).eval(); torch.set_num_threads(12)
amp = s['args']['model'] == 'w2v2' and dev.type == 'cuda'
ms = int(s['args']['max_seconds'] * SR)
sp = pd.read_csv(os.path.join(HERE, '..', '..', 'data', 'splits', 'splits_usable.csv'))
idx = [pool.index[o] for o in sp.oid[sp[s['args']['split']] == 'test']]
z = np.load(a.extra, allow_pickle=True)
ex = np.where(z['split'] == 'test')[0]
au, of, le = z['audio'], z['offsets'], z['lengths']
waves = [pool.wave(i) for i in idx] + [au[of[j]:of[j] + le[j]] for j in ex]
labels = np.concatenate([pool.labels[idx], np.zeros(len(ex), int)])      # dummy label for the extra clips
ids = np.arange(len(waves))
oids = np.concatenate([pool.oids[idx].astype(object), z['clip_id'][ex].astype(object)])
subs = np.array(['keyword'] * len(idx) + list(z['subsource'][ex]), dtype=object)
true = [pool.dictionary[pool.labels[i]] for i in idx] + [['_background_', '_unknown_'][z['labels'][j] - k] for j in ex]
dl = torch.utils.data.DataLoader(WaveSet(waves, labels, ids, ms), batch_size=8 if s['args']['model'] == 'w2v2' else 64,
                                 collate_fn=collate, num_workers=2)
_, y, p, c, i = evaluate(model, dl, dev, amp)      # same routine as the article (autocast is a no-op on CPU)
P = pd.DataFrame(dict(oid=oids[i], subsource=subs[i], true=np.array(true, dtype=object)[i],
                      pred=[pool.dictionary[j] for j in p], confidence=c))
os.makedirs(a.out, exist_ok=True)
P.to_csv(os.path.join(a.out, 'predictions.csv'), index=False)
res = dict(run=os.path.basename(a.run.rstrip('/')), keyword_test_acc=float((P.true == P.pred)[P.subsource == 'keyword'].mean()))
for thr in (0.0, 0.5, 0.7, 0.9, 0.99):
    d = {}
    for sub, g in P.groupby('subsource'):
        acc = g.confidence > thr
        d[sub] = float(((g.true == g.pred) & acc).mean()) if sub == 'keyword' else float(acc.mean())
    res[f'thr_{thr}'] = d          # keyword: correct-and-accepted rate; others: false acceptance rate
json.dump(res, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
