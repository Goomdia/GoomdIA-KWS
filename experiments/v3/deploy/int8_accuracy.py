"""Test accuracy of PyTorch dynamic int8 quantisation (nn.Linear -> qint8) versus fp32, same evaluation as
train_kws.py (speaker-disjoint test split, per-utterance min-max scaling, up to 3 s, train_kws.evaluate), on CPU.

  python int8_accuracy.py --ckpt experiments/runs/w2v2-ssl-speaker-lr3e-05-s0/model.pt [--ckpt ...] \
      --out data/deploy_desktop_accuracy.csv
Rows are appended to --out: one per checkpoint x precision (fp32, int8), with the agreement of the predictions.
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
import common as C
T = C.T

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', action='append', required=True)
ap.add_argument('--arch', default=None); ap.add_argument('--factory', default=None)
ap.add_argument('--split', default='speaker')
ap.add_argument('--cache', default=C.CACHE)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--max_seconds', type=float, default=3.0)
ap.add_argument('--out', default=os.path.join(C.ROOT, 'data', 'deploy_desktop_accuracy.csv'))
a = ap.parse_args()
torch.set_num_threads(min(a.threads, 4))

pool = T.Pool(a.cache)
sp = pd.read_csv(os.path.join(C.EXP, '..', 'data', 'splits', 'splits_usable.csv'))
idx = [pool.index[o] for o in sp.oid[sp[a.split] == 'test']]
ms = int(a.max_seconds * C.SR)


def run(model, bs):
    ds = T.WaveSet([pool.wave(i) for i in idx], pool.labels[idx], idx, ms)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, collate_fn=T.collate, num_workers=2)
    t0 = time.time()
    m, y, p, c, ids = T.evaluate(model, dl, torch.device('cpu'), False)
    return m, y, p, c, ids, time.time() - t0


rows = []
for ck in a.ckpt:
    ck = os.path.abspath(ck)
    model, arch, k = C.load_model(ck, a.arch, a.factory)
    bs = 8 if arch in T.SSL else 64                      # eval batch sizes of train_kws.py
    res = {}
    for prec in ('fp32', 'int8'):
        mdl = model if prec == 'fp32' else torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        m, y, p, c, ids, dt = run(mdl, bs)
        res[prec] = (m, y, p, c, ids)
        print(f'{arch} {prec} acc={m["acc"]:.4f} macro_f1={T.macro_f1(y, p, k):.4f} ({dt:.0f} s)', flush=True)
    (m0, y, p0, c0, i0), (m1, _, p1, c1, i1) = res['fp32'], res['int8']
    assert (i0 == i1).all()
    for prec, (m, y, p, c, ids) in res.items():
        rows.append(dict(ckpt=os.path.relpath(ck, C.ROOT), arch=arch, precision=prec, n_test=len(y), test_acc=m['acc'],
                         test_macro_f1=T.macro_f1(y, p, k), delta_acc_vs_fp32_pp=100 * (m['acc'] - m0['acc']),
                         pred_agreement_with_fp32=float((p == p0).mean()), cpu_eval_threads=a.threads))
    pd.DataFrame(rows).to_csv(a.out, index=False)       # rewritten after each checkpoint
print(pd.DataFrame(rows).to_string())
