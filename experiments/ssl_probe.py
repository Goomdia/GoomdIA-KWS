"""Frozen self-supervised representations + linear probes (CPU).

For each encoder (English-only wav2vec 2.0 Base vs. massively multilingual MMS-300M), every clip of the usable
corpus and of the volunteer OOD set is encoded once; the hidden states of every layer are mean-pooled over time.
A multinomial logistic regression is trained per layer on the training set, the best layer is chosen on the
validation set, and accuracy is reported on the test set (speaker and random splits) and on the OOD set.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser('~/goomdia_kws_cache')
ENCODERS = {'w2v2-base': 'facebook/wav2vec2-base', 'mms-300m': 'facebook/mms-300m'}
torch.set_num_threads(int(os.environ.get('THREADS', '12')))


def clips():
    z = np.load(os.path.join(CACHE, 'pool.npz'))
    a, o, l = z['audio'], z['offsets'], z['lengths']
    pool = [a[s:s + n] for s, n in zip(o, l)]
    y = z['labels']; oids = z['oids']
    zo = np.load(os.path.join(CACHE, 'volunteer_ood.npz'))
    ao = zo['audio']
    ood = [ao[s:s + n] for s, n in zip(zo['offsets'], zo['lengths'])]
    return pool, y, oids, ood, zo['labels']


@torch.no_grad()
def encode(name, waves):
    from transformers import Wav2Vec2Model
    m = Wav2Vec2Model.from_pretrained(ENCODERS[name]).eval()
    out = None
    t0 = time.time()
    for i, w in enumerate(waves):
        w = np.asarray(w[:3 * 16000], dtype=np.float32)
        w = (w - w.mean()) / (w.std() + 1e-7)
        hs = m(torch.from_numpy(w)[None], output_hidden_states=True).hidden_states
        v = torch.stack([h[0].mean(0) for h in hs]).numpy().astype(np.float16)    # (layers+1, dim)
        if out is None: out = np.zeros((len(waves),) + v.shape, np.float16)
        out[i] = v
        if i % 2000 == 0: print(name, i, f'{time.time() - t0:.0f}s', flush=True)
    return out


def main():
    pool, y, oids, ood, yo = clips()
    sp = pd.read_csv(os.path.join(HERE, '..', 'data', 'splits', 'splits_usable.csv')).set_index('oid')
    res = {}
    for name in ENCODERS:
        f = os.path.join(CACHE, f'feats_{name}.npz')
        if not os.path.exists(f):
            np.savez(f, pool=encode(name, pool), ood=encode(name, ood))
        F = np.load(f); X, XO = F['pool'].astype(np.float32), F['ood'].astype(np.float32)
        res[name] = {}
        for split in ('speaker', 'random'):
            lab = sp.loc[oids, split].to_numpy()
            tr, va, te = (lab == 'train'), (lab == 'valid'), (lab == 'test')
            per_layer = []
            for L in range(X.shape[1]):
                sc = StandardScaler().fit(X[tr, L])
                clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr, L]), y[tr])
                per_layer.append(dict(layer=L, val=clf.score(sc.transform(X[va, L]), y[va]),
                                      test=clf.score(sc.transform(X[te, L]), y[te]),
                                      ood=clf.score(sc.transform(XO[:, L]), yo)))
                print(name, split, per_layer[-1], flush=True)
            best = max(per_layer, key=lambda r: r['val'])
            res[name][split] = dict(best=best, per_layer=per_layer)
        json.dump(res, open(os.path.join(HERE, 'ssl_probe.json'), 'w'), indent=1)
    print(json.dumps({n: {s: r[s]['best'] for s in r} for n, r in res.items()}, indent=1))


if __name__ == '__main__':
    main()
