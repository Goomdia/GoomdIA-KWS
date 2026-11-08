"""Build the multilingual pre-training subset of the Multilingual Spoken Words Corpus (MSWC, HF MLCommons/ml_spoken_words).

For each language: the ~50 most frequent keywords (VALID clips, all splits pooled), <= CAP clips per keyword (round-robin over
speakers). Shards are tar.gz files sorted alphabetically by word, so for the large languages the (much smaller) dev/test
splits are downloaded instead of train; splits are added (dev, test, train) until the cap is reached for most keywords.
Output  <MSWC>/mswc_subset.npz : audio (int16, 16 kHz, concatenated), offsets, lengths, labels, langs, words, val (speaker-held-out flag)
        <MSWC>/mswc_subset_stats.csv, <MSWC>/download_log.txt
usage: prep_mswc.py MSWC_DIR
"""
import io, os, sys, tarfile, time, hashlib
import numpy as np, pandas as pd, soundfile as sf, torch, torchaudio
from multiprocessing import Pool
torch.set_num_threads(1)

OUT = sys.argv[1]
os.environ['HF_HOME'] = os.path.join(OUT, 'hf')
from huggingface_hub import hf_hub_download

REPO = 'MLCommons/ml_spoken_words'
LANGS = ['rw', 'es', 'fa', 'ru', 'eu', 'tr', 'ar', 'id', 'zh-CN', 'ha']
TOPK, CAP, MIN_CLIPS = 50, 500, 20
SPLITS = ['dev', 'test', 'train']
LARGE = {'rw', 'es', 'fa', 'ru', 'eu'}   # train split too big for the download budget: dev + test only (all keywords, fewer clips)
log = open(os.path.join(OUT, 'download_log.txt'), 'a')


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


def get(path):
    return hf_hub_download(REPO, path, repo_type='dataset', cache_dir=os.path.join(OUT, 'hf'))


def decode(item):
    name, b = item
    try:
        x, sr = sf.read(io.BytesIO(b), dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        return name, np.clip(x * 32767, -32768, 32767).astype(np.int16)
    except Exception:
        return name, None


def select(lang):
    d = os.path.join(OUT, 'splits', lang); os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, 'train.csv')):
        tarfile.open(get(f'data/splits/{lang}/splits.tar.gz')).extractall(d)
    df = pd.concat([pd.read_csv(os.path.join(d, f'{s}.csv')).assign(split=s) for s in SPLITS])
    df = df[df.VALID.astype(str) == 'True'].reset_index(drop=True)
    top = df.WORD.value_counts().head(TOPK)
    top = top[top >= MIN_CLIPS]
    df = df[df.WORD.isin(top.index)]
    avail = df.groupby(['WORD', 'split']).size().unstack(fill_value=0).reindex(columns=SPLITS, fill_value=0)
    use, tot_target = [], np.minimum(top.values, CAP).sum()
    for s in (SPLITS[:2] if lang in LARGE else SPLITS):
        use.append(s)
        if np.minimum(avail[use].sum(1), CAP).sum() >= 0.95 * tot_target: break
    df = df[df.split.isin(use)].copy()
    df['r'] = np.random.RandomState(0).rand(len(df))
    df['rank_spk'] = df.sort_values('r').groupby(['WORD', 'SPEAKER']).cumcount().reindex(df.index)
    df = df.sort_values(['WORD', 'rank_spk', 'r']).groupby('WORD').head(CAP)
    df['name'] = df.LINK.str.replace('/', '_', regex=False)
    return df, use


def main():
    t0 = time.time(); sel = {}
    for lang in LANGS:
        df, use = select(lang)
        sel[lang] = (df, use)
        say(f'{lang}: words={df.WORD.nunique()} clips={len(df)} splits={use} speakers={df.SPEAKER.nunique()}')
    audio, meta, dl_bytes = [], [], 0
    pool = Pool(2)
    for lang in LANGS:
        df, use = sel[lang]
        want = dict(zip(df.name, zip(df.WORD, df.SPEAKER)))
        got = 0
        for s in use:
            shard = 0
            while True:
                try: p = get(f'data/opus/{lang}/{s}/audio/{shard}.tar.gz')
                except Exception as e:
                    if shard == 0: say('missing shard', lang, s, repr(e)[:80])
                    break
                sz = os.path.getsize(os.path.realpath(p)); dl_bytes += sz
                say(f'  {lang}/{s}/{shard} {sz / 1e6:.0f} MB, total {dl_bytes / 1e9:.2f} GB, {time.time() - t0:.0f}s')
                items = []
                with tarfile.open(p, 'r|gz') as tf:
                    for m in tf:
                        if m.isfile() and m.name in want:
                            items.append((m.name, tf.extractfile(m).read()))
                for name, x in pool.imap_unordered(decode, items, chunksize=32):
                    if x is None or len(x) < 1600: continue
                    audio.append(x); meta.append((lang,) + want[name]); got += 1
                shard += 1
        say(f'{lang}: decoded {got}/{len(want)} clips')
    pool.close()
    lens = np.array([len(a) for a in audio]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    langs = np.array([m[0] for m in meta]); words = np.array([m[1] for m in meta]); spk = [m[2] for m in meta]
    keys = sorted(set(zip(langs, words))); kid = {k: i for i, k in enumerate(keys)}
    labels = np.array([kid[(l, w)] for l, w in zip(langs, words)])
    val = np.array([int(hashlib.md5(f'{l}{s}'.encode()).hexdigest(), 16) % 20 == 0 for l, s in zip(langs, spk)])
    np.savez(os.path.join(OUT, 'mswc_subset.tmp.npz'), audio=np.concatenate(audio), offsets=offs, lengths=lens, labels=labels,
             langs=langs, words=words, val=val, classes=np.array([f'{l}:{w}' for l, w in keys]))
    os.replace(os.path.join(OUT, 'mswc_subset.tmp.npz'), os.path.join(OUT, 'mswc_subset.npz'))
    st = pd.DataFrame(dict(lang=langs, word=words)).groupby('lang').agg(clips=('word', 'size'), words=('word', 'nunique'))
    st['hours'] = pd.Series(lens / 16000 / 3600).groupby(langs).sum()
    st.to_csv(os.path.join(OUT, 'mswc_subset_stats.csv'))
    say(st.to_string())
    say(f'TOTAL clips={len(lens)} classes={len(keys)} hours={lens.sum() / 16000 / 3600:.1f} val={val.mean():.3f} '
        f'downloaded={dl_bytes / 1e9:.2f} GB time={time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
