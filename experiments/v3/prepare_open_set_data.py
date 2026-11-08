"""Build the two extra classes of the open-set (32-class) experiments: _background_ and _unknown_.

_unknown_  (label 31)
  - Moore non-keyword speech: 1-2 s excerpts of the ASR segments (20 s) of the broadcast collection of the
    sibling project Goomdia_Moore_Foundation (READ-ONLY; rights unknown -> local research use only).
    Only the `unassigned` partition of its transcription campaign is used (its dev/test reserves are not
    touched), restricted to the Moore sources listed in SOURCES. Segments whose ASR text contains one of the
    30 keywords (or a listed variant) as a word are excluded. Documents are split train/valid/test by
    leakage group (acoustic-duplicate groups of the sibling project, i.e. by broadcast/document).
  - English: GSC v2 words (digits excluded) from the GSC train / validation / test lists.
_background_ (label 30)
  - GSC _background_noise_ files, disjoint by file between train and test (no GSC file in validation).
  - in-domain field noise: the longest low-energy stretch (digital zeros excluded) of raw field recordings,
    split by recording group following the speaker-disjoint split of the keyword corpus.

Outputs
  <cache>/v3/open_set_clips.npz            audio (float32, 16 kHz) + labels, splits, subsources, clip ids
  data/open_set_clips.csv                   one row per clip (no audio)
  data/open_set_moore_documents.csv         one row per Moore document used (provenance, split, counts)
  data/open_set_prep.json                   counts, sizes and exclusion statistics
"""
import argparse, collections, glob, hashlib, json, os, random, re, subprocess, unicodedata
from multiprocessing import Pool as MPool
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
SR = 16000
BG, UNK = 30, 31
# Moore sources of the sibling collection (registry language `mos`, keyword rate in the ASR text consistent
# with Moore speech); small unverified YouTube search hits and low-keyword-rate channels are not used.
SOURCES = ['SRC-011', 'SRC-002', 'SRC-041', 'SRC-022', 'SRC-028', 'SRC-033', 'SRC-032']
# keyword forms searched in the ASR text (main forms of data/field_recordings.csv + listed variants)
KEYWORD_FORMS = ['Bedre', 'Beoogo', 'Biiga', 'Bilfu', 'Di', 'Gomé', 'Koom', 'Laafi', 'Laafi beeme', 'Naase',
                 'Ne-y yibeoogo', 'Ne-y yungo', 'Neda', 'Nii', 'Nu', 'Paga', 'Piiga', 'Rawa', 'Rounda', 'Taanbo',
                 'Tenga', 'Wae', 'Yaa wâna', 'Yeela', 'Yembre', 'Ye', 'Yiibu', 'Yoobe', 'Yopoe', 'Zaalm', 'dáaré',
                 'yungo']
GSC_NOISE_SPLIT = {'doing_the_dishes': 'train', 'dude_miaowing': 'train', 'exercise_bike': 'train',
                   'white_noise': 'train', 'running_tap': 'test', 'pink_noise': 'test'}
DIGITS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


def norm_text(s):
    s = s.lower().replace('ɛ', 'e').replace('ʋ', 'u').replace('ɩ', 'i').replace('ŋ', 'n').replace('ɔ', 'o')
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r"[^a-z' ]", ' ', s.replace('-', ' ')).split()


KW_TOKENS = [(k, norm_text(k)) for k in KEYWORD_FORMS]


def keywords_in(text):
    w = norm_text(text)
    return sorted({k for k, kk in KW_TOKENS if any(w[i:i + len(kk)] == kk for i in range(len(w) - len(kk) + 1))})


def h(s, seed):
    return hashlib.sha1(f'{seed}:{s}'.encode()).hexdigest()


def frame_energy(x, fl=400, hop=160):
    n = 1 + max(0, (len(x) - fl) // hop)
    idx = np.arange(fl)[None] + hop * np.arange(n)[:, None]
    return (x[np.minimum(idx, len(x) - 1)] ** 2).mean(1)


# ------------------------------------------------------------------------------------- Moore unknown
def decode(path, start, dur):
    r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', path,
                        '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def moore_clip(job):
    """Decode one ASR segment and cut a speech-dense excerpt of 1-2 s (None if none is found)."""
    path, start, dur, seed = job
    rng = random.Random(seed)
    x = decode(path, start, dur)
    if len(x) < 2 * SR: return None
    seg_rms = np.sqrt((x ** 2).mean())
    if seg_rms < 1e-3: return None
    for _ in range(20):
        n = int(rng.uniform(1.0, 2.0) * SR)
        st = rng.randint(0, len(x) - n)
        c = x[st:st + n]
        e = frame_energy(c)
        if np.sqrt((c ** 2).mean()) >= 0.5 * seg_rms and (e > e.max() / 100).mean() >= 0.6:
            return st / SR, c
    return None


def build_moore(a, rng):
    F = a.foundation
    man = {}
    for l in open(os.path.join(F, 'runs', 'ASR-FULL-v1', 'manifest.jsonl')):
        d = json.loads(l); man[d['sha256']] = d
    reg = json.load(open(os.path.join(F, 'sources', 'registry.json')))
    reg = reg if isinstance(reg, list) else reg.get('sources', reg)
    reg = {s['id']: s for s in (reg.values() if isinstance(reg, dict) else reg)}
    segs, excl = [], collections.Counter()
    n_seen = n_empty = n_kw = 0
    for f in sorted(glob.glob(os.path.join(F, 'runs', 'ASR-FULL-v1', 'transcripts', 'unassigned', '*.jsonl'))):
        for l in open(f):
            d = json.loads(l)
            if d['source_id'] not in SOURCES: continue
            m = man[d['asset_sha256']]
            if m['known_decode_issue'] or m['metadata_hold'] or d['decode_mode'] != 'strict': continue
            n_seen += 1
            if d['empty_output'] or not d['text'].strip():
                n_empty += 1; continue
            k = keywords_in(d['text'])
            if k:
                n_kw += 1; excl.update(k); continue
            if d['duration_seconds'] < 5: continue
            segs.append(dict(asset_sha256=d['asset_sha256'], asset_id=d['asset_id'], source_id=d['source_id'],
                             leakage_group=d['leakage_group'], start=d['start_seconds'], dur=d['duration_seconds'],
                             path=os.path.join(F, m['local_path'])))
    S = pd.DataFrame(segs)
    # document split by leakage group, stratified by source (70/10/20)
    gsplit = {}
    for src, g in S.groupby('source_id'):
        groups = sorted(g.leakage_group.unique(), key=lambda x: h(x, a.seed))
        n = len(groups); ntr, nva = int(round(0.7 * n)), max(1, int(round(0.1 * n)))
        for i, gg in enumerate(groups):
            gsplit[gg] = 'train' if i < ntr else 'valid' if i < ntr + nva else 'test'
    S['split'] = S.leakage_group.map(gsplit)
    clips, meta = [], []
    for split, target in (('train', a.n_moore[0]), ('valid', a.n_moore[1]), ('test', a.n_moore[2])):
        G = S[S.split == split]
        # round-robin over documents (in random order) so that clips are spread over many broadcasts
        per_doc = {d: g.sample(frac=1, random_state=rng.randint(0, 2 ** 31)).to_dict('records')
                   for d, g in G.groupby('asset_sha256')}
        docs = sorted(per_doc, key=lambda x: h(x, a.seed))
        order, r = [], 0
        while len(order) < 1.4 * target and any(per_doc.values()):
            for d in docs:
                if r < len(per_doc[d]): order.append(per_doc[d][r])
            r += 1
        jobs = [(o['path'], o['start'], o['dur'], rng.randint(0, 2 ** 31)) for o in order[:int(1.4 * target)]]
        with MPool(a.workers) as p:
            out = p.map(moore_clip, jobs, chunksize=4)
        n = 0
        for o, res in zip(order, out):
            if res is None or n >= target: continue
            off, c = res
            clips.append(c.astype(np.float32))
            meta.append(dict(clip_id=f"mos-{o['asset_id']}-{o['start'] + off:.2f}", label=UNK, cls='_unknown_',
                             subsource='moore_broadcast', split=split, origin=o['asset_id'], source_id=o['source_id'],
                             group=o['leakage_group'], start_s=round(o['start'] + off, 3), dur_s=round(len(c) / SR, 3)))
            n += 1
        print('moore', split, n, 'clips from', len({m['origin'] for m in meta if m['split'] == split and m['subsource'] == 'moore_broadcast'}), 'documents', flush=True)
    # document table (provenance)
    M = pd.DataFrame(meta)
    D = S.groupby('asset_sha256').agg(asset_id=('asset_id', 'first'), source_id=('source_id', 'first'),
                                      leakage_group=('leakage_group', 'first'), split=('split', 'first'),
                                      n_kwfree_segments=('start', 'size'), path=('path', 'first')).reset_index()
    D['n_clips'] = D.asset_id.map(M.origin.value_counts()).fillna(0).astype(int)
    D['source_name'] = D.source_id.map(lambda s: reg.get(s, {}).get('name', ''))
    D['original_uri'] = D.asset_sha256.map(lambda s: man[s]['aliases'][0]['original_uri'] if man[s].get('aliases') else '')
    D['local_path'] = D.path.map(lambda p: os.path.relpath(p, F))
    D = D.drop(columns='path')
    stats = dict(segments_considered=n_seen, segments_empty_asr=n_empty, segments_with_keyword=n_kw,
                 segments_keyword_free_used_pool=len(S), excluded_by_keyword=dict(excl.most_common()),
                 documents_pool={k: int(v) for k, v in S.drop_duplicates('asset_sha256').split.value_counts().items()},
                 leakage_groups_pool={k: int(v) for k, v in S.drop_duplicates('leakage_group').split.value_counts().items()},
                 documents_with_clips={k: int(v) for k, v in D[D.n_clips > 0].split.value_counts().items()},
                 hours_by_split={k: round(float(v) / 3600, 2) for k, v in S.groupby('split').dur.sum().items()})
    return clips, meta, D, stats


# ------------------------------------------------------------------------------------ GSC sources
def gsc_words(a, rng):
    G = a.gsc
    te = set(l.strip() for l in open(os.path.join(G, 'testing_list.txt')))
    va = set(l.strip() for l in open(os.path.join(G, 'validation_list.txt')))
    allw = sorted(os.path.relpath(p, G) for p in glob.glob(os.path.join(G, '*', '*.wav')) if '_background_noise_' not in p)
    allw = [p for p in allw if p.split('/')[0] not in DIGITS]
    pools = {'train': [p for p in allw if p not in te and p not in va], 'valid': sorted(va & set(allw)),
             'test': sorted(te & set(allw))}
    clips, meta = [], []
    for split, n in zip(('train', 'valid', 'test'), a.n_gsc_words):
        for p in rng.sample(pools[split], n):
            x, sr = sf.read(os.path.join(G, p), dtype='float32'); assert sr == SR
            clips.append(x)
            meta.append(dict(clip_id=f'gsc-{p}', label=UNK, cls='_unknown_', subsource='gsc_english', split=split,
                             origin=p, source_id='GSC-v2', group=p.split('/')[1].split('_')[0], start_s=0.0,
                             dur_s=round(len(x) / SR, 3)))
    return clips, meta


def gsc_noise(a, rng):
    clips, meta = [], []
    files = {os.path.basename(f)[:-4]: f for f in glob.glob(os.path.join(a.gsc, '_background_noise_', '*.wav'))}
    for split, n in (('train', a.n_gsc_noise[0]), ('test', a.n_gsc_noise[1])):
        names = sorted(k for k, v in GSC_NOISE_SPLIT.items() if v == split)
        waves = {k: sf.read(files[k], dtype='float32')[0] for k in names}
        for i in range(n):
            k = names[i % len(names)]; z = waves[k]
            L = int(rng.uniform(1.0, 2.0) * SR); st = rng.randint(0, len(z) - L - 1)
            clips.append(z[st:st + L].copy())
            meta.append(dict(clip_id=f'gscnoise-{k}-{st}', label=BG, cls='_background_', subsource='gsc_noise',
                             split=split, origin=f'_background_noise_/{k}.wav', source_id='GSC-v2', group=k,
                             start_s=round(st / SR, 3), dur_s=round(L / SR, 3)))
    return clips, meta


def field_quiet(path):
    """Longest low-energy stretch (>= 0.6 s, <= 2 s) of a raw field clip, digital zeros excluded."""
    try:
        x, sr = sf.read(path, dtype='float32')
    except Exception:
        return None
    if x.ndim > 1: x = x.mean(1)
    if len(x) < sr // 2: return None
    if sr != SR: x = resample_poly(x, SR, sr).astype(np.float32)
    e = frame_energy(x)                                      # 25 ms frames, 10 ms hop
    zero = e < 1e-10
    if (~zero).sum() < 50: return None
    floor = np.percentile(e[~zero], 10)
    act = e > max(4 * floor, e.max() / 1000)                 # > 6 dB above the floor or -30 dB re peak
    a2 = act.copy()
    for s in range(1, 11):                                    # 100 ms margin around active frames
        a2[s:] |= act[:-s]; a2[:-s] |= act[s:]
    ok = ~a2 & ~zero
    best, i, n = (0, 0), 0, len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]: j += 1
            if j - i > best[1] - best[0]: best = (i, j)
            i = j
        else:
            i += 1
    L = best[1] - best[0]
    if L < 60: return None
    st, en = best[0] * 160, min(len(x), best[0] * 160 + min(L, 200) * 160)
    return st / SR, x[st:en].copy()


def field_noise(a, rng):
    raw = os.path.join(a.raw, '02 - Agents - 2eme semaine', 'Audios')
    files = {f[:24]: f for f in os.listdir(raw) if f.endswith('.wav')}
    fr = pd.read_csv(os.path.join(ROOT, 'data', 'field_recordings.csv'))
    sp = pd.read_csv(os.path.join(ROOT, 'data', 'splits', 'splits_usable.csv'))
    gsplit = sp.groupby('group_id').speaker.first().to_dict()          # speaker-disjoint split, group level
    fr['split'] = fr.group_id.map(gsplit)
    fr = fr[fr.split.notna() & fr.oid.isin(files)]
    clips, meta = [], []
    for split, target in zip(('train', 'valid', 'test'), a.n_field_noise):
        G = fr[fr.split == split]
        per_g = {g: d.oid.sample(frac=1, random_state=rng.randint(0, 2 ** 31)).tolist() for g, d in G.groupby('group_id')}
        gs = sorted(per_g, key=lambda x: h(x, a.seed))
        order, r = [], 0
        while len(order) < 12 * target and any(r < len(v) for v in per_g.values()):
            for g in gs:
                if r < len(per_g[g]): order.append((g, per_g[g][r]))
            r += 1
        order = order[:12 * target]
        with MPool(a.workers) as p:
            out = p.map(field_quiet, [os.path.join(raw, files[o]) for _, o in order], chunksize=8)
        n = 0
        for (g, o), res in zip(order, out):
            if res is None or n >= target: continue
            st, c = res
            clips.append(c)
            meta.append(dict(clip_id=f'field-{o}', label=BG, cls='_background_', subsource='field_noise', split=split,
                             origin=o, source_id='GoomdIA-field', group=str(g), start_s=round(st, 3),
                             dur_s=round(len(c) / SR, 3)))
            n += 1
        print('field noise', split, n, 'clips from', len({m['group'] for m in meta if m['split'] == split}),
              'groups (yield', round(sum(r is not None for r in out) / max(1, len(out)), 2), ')', flush=True)
    return clips, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--foundation', required=True)
    ap.add_argument('--raw', required=True, help='Goomdia_/05-machine_learning/Data')
    ap.add_argument('--gsc', required=True)
    ap.add_argument('--out', required=True, help='output npz')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n_moore', type=int, nargs=3, default=[2000, 300, 1000])
    ap.add_argument('--n_gsc_words', type=int, nargs=3, default=[500, 100, 300])
    ap.add_argument('--n_gsc_noise', type=int, nargs=2, default=[500, 200])
    ap.add_argument('--n_field_noise', type=int, nargs=3, default=[500, 150, 300])
    a = ap.parse_args()
    rng = random.Random(a.seed)
    c1, m1, D, st = build_moore(a, rng)
    c2, m2 = gsc_words(a, rng)
    c3, m3 = gsc_noise(a, rng)
    c4, m4 = field_noise(a, rng)
    clips, meta = c1 + c2 + c3 + c4, m1 + m2 + m3 + m4
    M = pd.DataFrame(meta)
    lens = np.array([len(c) for c in clips]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, audio=np.concatenate(clips).astype(np.float32), offsets=offs, lengths=lens,
             labels=M.label.to_numpy(), split=M.split.to_numpy(), subsource=M.subsource.to_numpy(),
             clip_id=M.clip_id.to_numpy())
    M.to_csv(os.path.join(ROOT, 'data', 'open_set_clips.csv'), index=False)
    D.to_csv(os.path.join(ROOT, 'data', 'open_set_moore_documents.csv'), index=False)
    st['clips'] = {f'{c}|{s}|{sp}': int(n) for (c, s, sp), n in M.groupby(['cls', 'subsource', 'split']).size().items()}
    st['duration_s_mean'] = {f'{s}': round(float(v), 3) for s, v in M.groupby('subsource').dur_s.mean().items()}
    st['args'] = vars(a)
    json.dump(st, open(os.path.join(ROOT, 'data', 'open_set_prep.json'), 'w'), indent=1)
    print(json.dumps(st['clips'], indent=1))


if __name__ == '__main__':
    main()
