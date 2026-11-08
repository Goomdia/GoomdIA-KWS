"""Open-set (32-class) variant of experiments/train_kws.py: 30 keywords + _background_ + _unknown_.

The keyword data, the splits and the training recipe (optimiser, schedule, augmentation, early stopping on
validation accuracy, checkpoint selection) are those of train_kws.py, whose functions are imported unchanged;
the only differences are the two extra classes (clips of <cache>/v3/open_set_clips.npz, built by
prepare_open_set_data.py) appended to the train / valid / test sets, and the extra outputs below.
  <out>/<run>/summary.json            hyper-parameters, learning curves, keyword-test / extra-test / OOD metrics
  <out>/<run>/test_predictions.csv    keyword test + extra test clips (oid, subsource, true, pred, conf., p_bg, p_unk)
  <out>/<run>/ood_predictions.csv     volunteer recordings (same columns)
"""
import argparse, json, os, sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from train_kws import (LeTRKWS, W2V2KWS, Pool, WaveSet, collate, augment_wave, n_params, macro_f1, GSC_CKPT,
                       VARIANTS, SR, SSL, BC, BCResNetKWS, load_pretrained, pretrained_path)

EXTRA = ['_background_', '_unknown_']


@torch.no_grad()
def predict(model, loader, dev, amp):
    """Like train_kws.evaluate, but keeps the full posterior matrix."""
    model.eval()
    ys, P, ids, loss = [], [], [], 0.0
    for x, lens, y, i in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            logits = model(x, lens)
        logits = logits.float()
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        ys.append(y.cpu()); P.append(logits.softmax(-1).cpu()); ids.append(i)
    ys, P, ids = torch.cat(ys).numpy(), torch.cat(P).numpy(), torch.cat(ids).numpy()
    return dict(loss=loss / len(ys), acc=float((ys == P.argmax(1)).mean())), ys, P, ids


def kw_metrics(y, P, k=30):
    """Metrics on keyword utterances: 32-way accuracy, rejection (bg/unk) rate, keyword-restricted accuracy."""
    p = P.argmax(1)
    return dict(acc=float((p == y).mean()), macro_f1=macro_f1(y, p, k), reject_rate=float((p >= k).mean()),
                reject_bg=float((p == k).mean()), reject_unk=float((p == k + 1).mean()),
                acc_kw_only=float((P[:, :k].argmax(1) == y).mean()), n=int(len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='letr128t', choices=list(VARIANTS) + list(BC) + list(SSL))
    ap.add_argument('--init', default='gsc', choices=['gsc', 'scratch', 'ssl', 'mswc'])
    ap.add_argument('--split', default='speaker')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=8e-5)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--aug_p', type=float, default=0.25)
    ap.add_argument('--max_seconds', type=float, default=3.0)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--extra', required=True, help='open_set_clips.npz')
    ap.add_argument('--ood', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='')
    ap.add_argument('--save_model', action='store_true')
    ap.add_argument('--optim', default='', choices=['', 'adamw8bit'])
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', 4)))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device('cuda')
    pool = Pool(a.cache)
    k = len(pool.dictionary)                                   # 30 keywords
    classes = pool.dictionary + EXTRA                          # 32 classes (30 = _background_, 31 = _unknown_)
    z = np.load(a.extra, allow_pickle=True)
    ea, eo, el = z['audio'], z['offsets'], z['lengths']
    esplit, elab, esub, eid = z['split'], z['labels'], z['subsource'], z['clip_id']
    sp = pd.read_csv(os.path.join(HERE, '..', '..', 'data', 'splits', 'splits_usable.csv'))
    ms = int(a.max_seconds * SR)
    N0 = len(pool.oids)                                         # ids >= N0 index the extra clips
    oid_all = np.concatenate([pool.oids.astype(object), eid.astype(object)])
    sub_all = np.concatenate([np.array(['keyword'] * N0, dtype=object), esub.astype(object)])

    def make(s, shuffle):
        idx = [pool.index[o] for o in sp.oid[sp[a.split] == s]]
        ex = np.where(esplit == s)[0]
        waves = [pool.wave(i) for i in idx] + [ea[eo[j]:eo[j] + el[j]] for j in ex]
        labels = np.concatenate([pool.labels[idx], elab[ex]])
        ds = WaveSet(waves, labels, list(idx) + [N0 + j for j in ex], ms)
        eval_bs = a.bs if a.model in SSL else 64
        return torch.utils.data.DataLoader(ds, batch_size=a.bs if shuffle else eval_bs, shuffle=shuffle,
                                           collate_fn=collate, num_workers=2, drop_last=shuffle,
                                           pin_memory=True, persistent_workers=True)
    tr, va, te = make('train', True), make('valid', False), make('test', False)
    K = len(classes)

    if a.model in SSL:
        model = W2V2KWS(K, SSL[a.model]); amp = True
    elif a.model in BC:
        model = BCResNetKWS(a.model, K); amp = False
        if a.init in ('gsc', 'mswc'): print('loaded', load_pretrained(model, pretrained_path(a.model, a.init)), a.init, 'tensors')
    else:
        model = LeTRKWS(a.model, K); amp = False
        if a.init == 'gsc': print('loaded', model.load_gsc(GSC_CKPT[a.model]), 'GSC tensors')
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optim == 'adamw8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=a.wd)
    else:
        opt = (torch.optim.AdamW if a.model in SSL else torch.optim.Adam)(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr / 10)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    name = a.name or f'open32-{a.model}-{a.init}-{a.split}-lr{a.lr:g}-s{a.seed}'
    out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
    hist, best, best_state, bad, t0 = [], -1, None, 0, time.time()
    for ep in range(a.epochs):
        model.train(); tl, tc, tn = 0.0, 0, 0
        for x, lens, y, _ in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            if a.aug_p > 0: x = augment_wave(x, lens, a.aug_p)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                logits = model(x, lens, aug_p=a.aug_p)
                loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); tc += (logits.argmax(-1) == y).sum().item(); tn += len(y)
        sched.step()
        v, vy, vP, _ = predict(model, va, dev, amp)
        kwv = vy < k
        hist.append(dict(epoch=ep + 1, train_loss=tl / tn, train_acc=tc / tn, val_loss=v['loss'], val_acc=v['acc'],
                         val_acc_kw=float((vP[kwv].argmax(1) == vy[kwv]).mean()), time_s=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        if v['acc'] > best:
            best, bad = v['acc'], 0
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break

    model.load_state_dict(best_state)

    def table(y, P, ids, oids, subs):
        p = P.argmax(1)
        return pd.DataFrame(dict(oid=oids[ids], subsource=subs[ids], true=[classes[i] for i in y],
                                 pred=[classes[i] for i in p], confidence=P.max(1), p_kw_max=P[:, :k].max(1),
                                 pred_kw=[classes[i] for i in P[:, :k].argmax(1)], p_background=P[:, k],
                                 p_unknown=P[:, k + 1]))
    m, y, P, ids = predict(model, te, dev, amp)
    T = table(y, P, ids, oid_all, sub_all)
    T.to_csv(os.path.join(out, 'test_predictions.csv'), index=False)
    kw = y < k
    res = dict(args=vars(a), classes=classes, n_params=n_params(model), n_train=len(tr.dataset),
               n_valid=len(va.dataset), n_test=len(te.dataset), best_val_acc=best, test_all32=m,
               test=kw_metrics(y[kw], P[kw], k), history=hist, train_time_s=time.time() - t0)
    extra = {}
    for s in sorted(set(T.subsource[~kw])):
        sel = (T.subsource == s).to_numpy()
        pr = P[sel].argmax(1)
        extra[s] = dict(n=int(sel.sum()), false_accept=float((pr < k).mean()), as_background=float((pr == k).mean()),
                        as_unknown=float((pr == k + 1).mean()), correct=float((pr == y[sel]).mean()))
    res['extra_test'] = extra

    if a.ood:
        zo = np.load(a.ood)
        oa = zo['audio']
        ow = [oa[o:o + l] for o, l in zip(zo['offsets'], zo['lengths'])]
        od = WaveSet(ow, zo['labels'], list(range(len(ow))), ms)
        ol = torch.utils.data.DataLoader(od, batch_size=a.bs if a.model in SSL else 64, collate_fn=collate, num_workers=2)
        _, oy, oP, oi = predict(model, ol, dev, amp)
        res['ood'] = kw_metrics(oy, oP, k)
        table(oy, oP, oi, zo['oids'].astype(object), np.array(['volunteer'] * len(ow), dtype=object)) \
            .to_csv(os.path.join(out, 'ood_predictions.csv'), index=False)
    if a.save_model:
        torch.save(best_state, os.path.join(out, 'model.pt'))
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('TEST', json.dumps(res['test']), 'EXTRA', json.dumps(extra), 'OOD', json.dumps(res.get('ood')), flush=True)


if __name__ == '__main__':
    main()
