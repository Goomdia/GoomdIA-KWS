"""Repair test_predictions.csv of runs produced before the oid-indexing fix in train_kws.py.

Those runs wrote pool.oids[j] where j was the position of the clip in the (unshuffled) test loader
instead of its pool index. The test order is deterministic (order of the split file), so the true oid
of row j is the j-th test oid. The repair is validated by checking that the stored true labels match
the labels of the recovered clips."""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
z = np.load(sys.argv[1])                      # pool.npz
pool_oids, labels, dic = z['oids'], z['labels'], list(z['dictionary'])
pidx = {o: i for i, o in enumerate(pool_oids)}
sp = pd.read_csv(os.path.join(ROOT, 'data', 'splits', 'splits_usable.csv'))
for f in glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*', 'test_predictions.csv')):
    run = os.path.dirname(f)
    if not os.path.exists(os.path.join(run, 'summary.json')):
        continue
    a = json.load(open(os.path.join(run, 'summary.json')))['args']
    test = sp.oid[sp[a['split']] == 'test'].tolist()
    p = pd.read_csv(f)
    if p.oid.isin(set(test)).all():
        continue
    j = p.oid.map(pidx).to_numpy()
    p['oid'] = [test[k] for k in j]
    ok = np.mean([dic[labels[pidx[o]]] == t for o, t in zip(p.oid, p.true)])
    assert ok == 1.0, (run, ok)
    p.to_csv(f, index=False)
    print('repaired', os.path.basename(run))
