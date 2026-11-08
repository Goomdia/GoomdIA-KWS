"""Training / evaluation script for the paper experiments (GoomdIA Mooré keyword corpus).

Plain PyTorch re-implementation of the GoomdiaWordRecognizer recipe (LeTR on MFCC, optional
initialisation from the Google Speech Commands v2 checkpoints trained during development),
plus wav2vec 2.0 (base) and MMS-300M fine-tuning baselines. Every run writes:
  <out>/<run_name>/summary.json      hyper-parameters, learning curves and test metrics
  <out>/<run_name>/test_predictions.csv  one row per test utterance (oid, true, pred, confidence)
  <out>/<run_name>/ood_predictions.csv   (optional) predictions on the volunteer recordings
"""
import argparse, json, os, re, time, random, copy
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, os.path.join(HERE, 'src'))
from letr import LeTR
sys.path.insert(0, os.path.join(HERE, 'v3'))
from models_extra import BCResNetKWS, BC, load_pretrained, pretrained_path

SR = 16000
REPO = os.path.join(HERE, '..', '..', 'GoomdiaWordRecognizer')
POOL_H5_DIR = os.path.join(REPO, 'data', 'datasets', 'trainFromClean_testNvalidFromAccepted_80-10-10')
GSC_CKPT = {  # GSC-v2 (35 words) checkpoints trained with the development pipeline (MFCC-16, max_length 150)
    'letr128t': 'logs/trans-letr128t-scratch_-mfcc-30c_GSC-2aug25-b8-lr8e-5/version_0/checkpoints/epoch=59-val_acc=0.95.ckpt',
    'letr96': 'logs/trans-letr96__-scratch_-mfcc-30c_GSC-2aug25-b64-lr8e-5/version_0/checkpoints/epoch=94-val_acc=0.92.ckpt',
}
VARIANTS = {'letr96': (96, 3, 4), 'letr128t': (128, 4, 4), 'letr256': (512, 8, 8)}
SSL = {'w2v2': 'facebook/wav2vec2-base', 'mms300m': 'facebook/mms-300m', 'mhubert': 'utter-project/mHuBERT-147'}


# ----------------------------------------------------------------------------------------- data
def build_pool_cache(cache):
    """Concatenate the 16,699 usable waveforms (float32, 16 kHz) into one memory-friendly file."""
    waves, oids, labels, dictionary = [], [], [], None
    for sub in ('train', 'valid', 'test'):
        with h5py.File(os.path.join(POOL_H5_DIR, f'{sub}.h5'), 'r') as h:
            d = [x.decode() for x in h['dictionary']]
            dictionary = dictionary or d
            assert d == dictionary
            for fn, lab in zip(h['filenames'], h['labels']):
                fn = fn.decode(); lab = lab.decode()
                oids.append(re.search(r'[0-9a-f]{24}', fn).group(0))
                labels.append(dictionary.index(lab))
                waves.append(np.asarray(h['waveforms'][fn][()], dtype=np.float32).reshape(-1))
    lens = np.array([len(w) for w in waves]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    np.savez(cache, audio=np.concatenate(waves), offsets=offs, lengths=lens, oids=np.array(oids),
             labels=np.array(labels), dictionary=np.array(dictionary))


class Pool:
    def __init__(self, cache):
        z = np.load(cache)
        self.audio, self.offs, self.lens = z['audio'], z['offsets'], z['lengths']
        self.oids, self.labels, self.dictionary = z['oids'], z['labels'], list(z['dictionary'])
        self.index = {o: i for i, o in enumerate(self.oids)}

    def wave(self, i):
        return self.audio[self.offs[i]:self.offs[i] + self.lens[i]]


class WaveSet(torch.utils.data.Dataset):
    def __init__(self, waves, labels, ids, max_samples):
        self.waves, self.labels, self.ids, self.max_samples = waves, labels, ids, max_samples

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        w = torch.from_numpy(np.asarray(self.waves[i], dtype=np.float32))[:self.max_samples]
        lo, hi = w.min(), w.max()                           # per-utterance min-max scaling (dev. recipe)
        w = 2 * (w - lo) / (hi - lo) - 1 if hi > lo else torch.zeros_like(w)
        return w, int(self.labels[i]), int(self.ids[i])     # pool index (used to recover the oid)


def collate(batch):
    ws, ys, ids = zip(*batch)
    n = max(len(w) for w in ws)
    x = torch.zeros(len(ws), n)
    lens = torch.tensor([len(w) for w in ws])
    for k, w in enumerate(ws): x[k, :len(w)] = w
    return x, lens, torch.tensor(ys), torch.tensor(ids)


# ------------------------------------------------------------------------------- augmentation
def augment_wave(x, lens, p):
    """x: (B, N) on device. Random onset shift (<=100 ms) and white noise at 10-30 dB SNR."""
    B, N = x.shape
    out = x.clone()
    for b in range(B):
        if random.random() < p:
            s = random.randint(1, int(0.1 * SR))
            out[b] = F.pad(x[b], (s, 0))[:N]
        if random.random() < p:
            L = max(int(lens[b]), 1)
            pw = out[b, :L].pow(2).mean().clamp_min(1e-8)
            snr = random.uniform(10, 30)
            out[b, :L] += torch.randn(L, device=x.device) * torch.sqrt(pw / 10 ** (snr / 10))
    return out


def spec_augment(f, p, t_w=15, f_w=2, n=2):
    """f: (B, T, F). Two time masks (<= t_w frames) and two frequency masks (<= f_w bins)."""
    B, T, Fd = f.shape
    f = f.clone()
    for b in range(B):
        if random.random() >= p: continue
        for _ in range(n):
            w = random.randint(0, t_w); t0 = random.randint(0, max(0, T - w)); f[b, t0:t0 + w, :] = 0
            w = random.randint(0, f_w); f0 = random.randint(0, max(0, Fd - w)); f[b, :, f0:f0 + w] = 0
    return f


# ------------------------------------------------------------------------------------ models
class LeTRKWS(nn.Module):
    def __init__(self, variant, n_classes, n_mfcc=16, max_frames=150, dropout=0.1):
        super().__init__()
        e, h, l = VARIANTS[variant]
        self.mfcc = torchaudio.transforms.MFCC(SR, n_mfcc=n_mfcc,
                                               melkwargs=dict(n_fft=400, hop_length=160, f_min=20, f_max=8000))
        self.model = LeTR(n_mfcc, n_classes, embed_dim=e, num_heads=h, num_layers=l,
                          max_length=max_frames, dropout=dropout)

    def features(self, x):                                   # (B, N) -> (B, T, F)
        with torch.autocast(device_type='cuda', enabled=False):
            return self.mfcc(x.float()).permute(0, 2, 1)

    def forward(self, x, lens=None, aug_p=0.0):
        f = self.features(x)
        if self.training and aug_p > 0: f = spec_augment(f, aug_p)
        return self.model(f)[0]

    def load_gsc(self, path):
        sd = torch.load(os.path.join(REPO, path), map_location='cpu', weights_only=False)['state_dict']
        sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')
              and 'classifier_head' not in k and 'distillation_head' not in k}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected and all('head' in k for k in missing), (missing, unexpected)
        return len(sd)


class W2V2KWS(nn.Module):
    def __init__(self, n_classes, name='facebook/wav2vec2-base'):
        super().__init__()
        from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification
        cls = HubertForSequenceClassification if 'hubert' in name.lower() else Wav2Vec2ForSequenceClassification
        self.m = cls.from_pretrained(name, num_labels=n_classes)
        self.m.freeze_feature_encoder()
        self.m.gradient_checkpointing_enable()

    def forward(self, x, lens=None, aug_p=0.0):
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lens.to(x.device)[:, None]).float()
        mu = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        sd = (((x - mu) * mask).pow(2).sum(1, keepdim=True) / mask.sum(1, keepdim=True)).sqrt()
        x = (x - mu) / (sd + 1e-7) * mask                      # zero-mean / unit-variance, zero padding
        return self.m(input_values=x).logits


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, loader, dev, amp):
    model.eval()
    ys, ps, cs, ids, loss = [], [], [], [], 0.0
    for x, lens, y, i in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            logits = model(x, lens)
        logits = logits.float()
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        pr = logits.softmax(-1)
        c, p = pr.max(-1)
        ys.append(y.cpu()); ps.append(p.cpu()); cs.append(c.cpu()); ids.append(i)
    ys, ps, cs, ids = (torch.cat(t).numpy() for t in (ys, ps, cs, ids))
    return dict(loss=loss / len(ys), acc=float((ys == ps).mean())), ys, ps, cs, ids


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='letr128t', choices=list(VARIANTS) + list(BC) + list(SSL))
    ap.add_argument('--init', default='gsc', choices=['gsc', 'scratch', 'ssl', 'mswc'])
    ap.add_argument('--split', default='speaker', help='column of splits_usable.csv: speaker, random, ctrl_unseen, ctrl_seen')
    ap.add_argument('--train_subset', default='', help='column of splits_usable.csv restricting the train set')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=8e-5)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--aug_p', type=float, default=0.25)
    ap.add_argument('--max_seconds', type=float, default=3.0)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ood', default='', help='npz with volunteer recordings (audio/offsets/lengths/oids/labels)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='')
    ap.add_argument('--save_model', action='store_true')
    ap.add_argument('--optim', default='', choices=['', 'adamw8bit'], help='memory-lean optimiser (bitsandbytes)')
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation steps (effective batch = bs * accum)')
    ap.add_argument('--stop_after', type=int, default=0, help='stop after n epochs (schedule still set by --epochs)')
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', 4)))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device('cuda')
    if not os.path.exists(a.cache): build_pool_cache(a.cache)
    pool = Pool(a.cache)
    sp = pd.read_csv(os.path.join(HERE, '..', 'data', 'splits', 'splits_usable.csv'))
    sel = {s: sp.oid[sp[a.split] == s].tolist() for s in ('train', 'valid', 'test')}
    if a.train_subset:
        sel['train'] = sp.oid[sp[a.train_subset]].tolist()
    ms = int(a.max_seconds * SR)

    def make(s, shuffle):
        idx = [pool.index[o] for o in sel[s]]
        ds = WaveSet([pool.wave(i) for i in idx], pool.labels[idx], idx, ms)
        eval_bs = a.bs if a.model in SSL else 64          # large encoders: evaluate with small batches
        return torch.utils.data.DataLoader(ds, batch_size=a.bs if shuffle else eval_bs, shuffle=shuffle,
                                           collate_fn=collate, num_workers=2, drop_last=shuffle,
                                           pin_memory=True, persistent_workers=True)
    tr, va, te = make('train', True), make('valid', False), make('test', False)
    k = len(pool.dictionary)

    if a.model in SSL:
        model = W2V2KWS(k, SSL[a.model]); amp = True
    elif a.model in BC:
        model = BCResNetKWS(a.model, k); amp = False
        if a.init in ('gsc', 'mswc'): print('loaded', load_pretrained(model, pretrained_path(a.model, a.init)), a.init, 'tensors')
    else:
        model = LeTRKWS(a.model, k); amp = False
        if a.init == 'gsc': print('loaded', model.load_gsc(GSC_CKPT[a.model]), 'GSC tensors')
        if a.init == 'mswc': print('loaded', load_pretrained(model, pretrained_path(a.model, a.init)), 'MSWC tensors')
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optim == 'adamw8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=a.lr, weight_decay=a.wd)
    else:
        opt = (torch.optim.AdamW if a.model in SSL else torch.optim.Adam)(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr / 10)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    name = a.name or f'{a.model}-{a.init}-{a.split}{("-" + a.train_subset) if a.train_subset else ""}-lr{a.lr:g}-s{a.seed}'
    out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(dev)
    hist, best, best_state, bad, t0 = [], -1, None, 0, time.time()
    for ep in range(a.epochs):
        model.train(); tl, tc, tn = 0.0, 0, 0
        for it, (x, lens, y, _) in enumerate(tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            if a.aug_p > 0: x = augment_wave(x, lens, a.aug_p)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                logits = model(x, lens, aug_p=a.aug_p)
                loss = F.cross_entropy(logits.float(), y)
            if it % a.accum == 0: opt.zero_grad(set_to_none=True)
            scaler.scale(loss / a.accum).backward()
            tl += loss.item() * len(y); tc += (logits.argmax(-1) == y).sum().item(); tn += len(y)
            if (it + 1) % a.accum: continue
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        v, *_ = evaluate(model, va, dev, amp)
        hist.append(dict(epoch=ep + 1, train_loss=tl / tn, train_acc=tc / tn, val_loss=v['loss'], val_acc=v['acc'],
                         time_s=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        if v['acc'] > best:
            best, bad = v['acc'], 0
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
        if a.stop_after and ep + 1 >= a.stop_after: break

    model.load_state_dict(best_state)
    m, y, p, c, ids = evaluate(model, te, dev, amp)
    m['macro_f1'] = macro_f1(y, p, k)
    pd.DataFrame(dict(oid=pool.oids[ids], true=[pool.dictionary[i] for i in y], pred=[pool.dictionary[i] for i in p],
                      confidence=c)).to_csv(os.path.join(out, 'test_predictions.csv'), index=False)
    res = dict(args=vars(a), n_params=n_params(model), n_train=len(tr.dataset), n_valid=len(va.dataset),
               n_test=len(te.dataset), best_val_acc=best, test=m, history=hist, train_time_s=time.time() - t0,
               gpu=torch.cuda.get_device_name(dev), peak_mem_mib=torch.cuda.max_memory_allocated(dev) / 2 ** 20,
               peak_reserved_mib=torch.cuda.max_memory_reserved(dev) / 2 ** 20)

    if a.ood:
        z = np.load(a.ood)
        oa = z['audio']                                   # read once (npz members are re-read on access)
        ow = [oa[o:o + l] for o, l in zip(z['offsets'], z['lengths'])]
        od = WaveSet(ow, z['labels'], list(range(len(ow))), ms)
        ol = torch.utils.data.DataLoader(od, batch_size=a.bs if a.model in SSL else 64, collate_fn=collate, num_workers=2)
        om, oy, opr, oc, oi = evaluate(model, ol, dev, amp)
        om['macro_f1'] = macro_f1(oy, opr, k)
        res['ood'] = om
        pd.DataFrame(dict(oid=z['oids'][oi], true=[pool.dictionary[i] for i in oy], pred=[pool.dictionary[i] for i in opr],
                          confidence=oc)).to_csv(os.path.join(out, 'ood_predictions.csv'), index=False)
    if a.save_model:
        torch.save(best_state, os.path.join(out, 'model.pt'))
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('TEST', json.dumps(m), 'OOD', json.dumps(res.get('ood')), flush=True)


if __name__ == '__main__':
    main()
