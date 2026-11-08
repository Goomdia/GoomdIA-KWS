"""Pre-training of the small models on an external keyword corpus, same augmentation as train_kws.py (onset shift + white noise
p=0.25 on the waveform, SpecAugment on the features), Adam + cosine schedule, best-validation checkpoint.
  --data gsc  : Google Speech Commands v2, 35 words, official validation / testing lists   (-> pretrained/<model>-gsc.pt)
  --data mswc : multilingual subset built by prep_mswc.py (language:word classes), speaker-held-out validation (-> <model>-mswc.pt)
usage: pretrain.py --model {bcresnet1,bcresnet8,letr128t} --data {gsc,mswc} --cache FILE.npz [--epochs N --bs N --lr X]
"""
import argparse, glob, json, os, random, sys, time
import numpy as np, soundfile as sf, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
from train_kws import LeTRKWS, WaveSet, collate, augment_wave, evaluate, n_params, SR, REPO
from models_extra import BCResNetKWS, BC, pretrained_path, PRETRAINED_DIR

GSC_DIR = os.path.join(REPO, 'data', 'gsc', 'SpeechCommands', 'speech_commands_v0.02')


def build_gsc(cache):
    words = sorted(d for d in os.listdir(GSC_DIR) if os.path.isdir(os.path.join(GSC_DIR, d)) and not d.startswith('_'))
    assert len(words) == 35, words
    val = set(open(os.path.join(GSC_DIR, 'validation_list.txt')).read().split())
    tst = set(open(os.path.join(GSC_DIR, 'testing_list.txt')).read().split())
    files = [(w, f'{w}/{os.path.basename(p)}') for w in words for p in sorted(glob.glob(os.path.join(GSC_DIR, w, '*.wav')))]
    empty = [f for _, f in files if os.path.getsize(os.path.join(GSC_DIR, f)) < 100]   # the v0.02 archive ships one 0-byte file (one/dfdabe19_nohash_0.wav)
    files = [(w, f) for w, f in files if f not in set(empty)]; print('skipped empty files:', empty, 'kept', len(files), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    def rd(it):
        x, sr = sf.read(os.path.join(GSC_DIR, it[1]), dtype='int16'); assert sr == SR; return x
    with ThreadPoolExecutor(2) as ex: audio = list(ex.map(rd, files, chunksize=64))
    lens = np.array([len(a) for a in audio]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    split = np.array(['test' if f in tst else 'val' if f in val else 'train' for _, f in files])
    np.savez(cache + '.tmp.npz', audio=np.concatenate(audio), offsets=offs, lengths=lens,
             labels=np.array([words.index(w) for w, _ in files]), split=split, classes=np.array(words))
    os.replace(cache + '.tmp.npz', cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(BC) + ['letr128t'])
    ap.add_argument('--data', required=True, choices=['gsc', 'mswc'])
    ap.add_argument('--cache', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--aug_p', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_epochs_time', type=float, default=0, help='stop when elapsed training time (s) exceeds this')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', 4)))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device('cuda')
    z = np.load(a.cache, allow_pickle=True)
    audio, offs, lens, labels = z['audio'], z['offsets'], z['lengths'], z['labels']
    if a.data == 'gsc': sp = z['split']
    else:
        rng = np.random.RandomState(0)
        sp = np.where(z['val'], 'val', 'train'); sp[(sp == 'val') & (rng.rand(len(sp)) < 0.5)] = 'test'
    K = len(z['classes'])

    def make(s, shuffle):
        idx = np.where(sp == s)[0]
        ds = WaveSet([audio[offs[i]:offs[i] + lens[i]] for i in idx], labels[idx], list(idx), SR)
        return torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=shuffle, collate_fn=collate, num_workers=2,
                                           drop_last=shuffle, pin_memory=True, persistent_workers=True)
    tr, va, te = make('train', True), make('val', False), make('test', False)
    model = (BCResNetKWS(a.model, K) if a.model in BC else LeTRKWS(a.model, K)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr / 10)
    print(a.model, a.data, 'classes', K, 'train/val/test', len(tr.dataset), len(va.dataset), len(te.dataset), 'params', n_params(model), flush=True)
    hist, best, best_state, t0 = [], -1, None, time.time()
    for ep in range(a.epochs):
        model.train(); tl = tc = tn = 0
        for x, ln, y, _ in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            x = augment_wave(x, ln, a.aug_p)
            logits = model(x, ln, aug_p=a.aug_p); loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item() * len(y); tc += (logits.argmax(-1) == y).sum().item(); tn += len(y)
        sched.step()
        v, *_ = evaluate(model, va, dev, False)
        hist.append(dict(epoch=ep + 1, train_loss=tl / tn, train_acc=tc / tn, val_loss=v['loss'], val_acc=v['acc'], time_s=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        if v['acc'] > best: best = v['acc']; best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if a.max_epochs_time and time.time() - t0 > a.max_epochs_time: print('time budget reached'); break
    model.load_state_dict(best_state)
    t, *_ = evaluate(model, te, dev, False)
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    torch.save(best_state, pretrained_path(a.model, a.data))
    json.dump(dict(args=vars(a), n_classes=K, n_params=n_params(model), n_train=len(tr.dataset), best_val_acc=best, test=t, history=hist,
                   train_time_s=time.time() - t0, peak_mem_mib=torch.cuda.max_memory_allocated(dev) / 2 ** 20),
              open(os.path.join(PRETRAINED_DIR, f'{a.model}-{a.data}.json'), 'w'), indent=1)
    print('TEST', json.dumps(t), 'time', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
