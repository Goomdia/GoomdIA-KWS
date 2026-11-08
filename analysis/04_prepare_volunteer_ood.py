"""Out-of-domain test set: the volunteer web recordings (April-July 2024, own devices, unsupervised).

Each clip is decoded, down-mixed, resampled to 16 kHz and trimmed with a simple energy VAD
(the speech span is kept; if longer than 1.5 s, the 1.5 s window with maximum energy is kept),
which approximates the diarisation-based trimming applied to the field corpus.
Clips shorter than 0.3 s or without detectable speech are discarded.
Output: npz (audio/offsets/lengths/oids/labels) using the label dictionary of the field corpus.
"""
import base64, io, json, os, sys
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SRC, POOL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]   # audios.json, pool.npz, output npz
HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000

z = np.load(POOL)
dictionary = list(z['dictionary'])
meta = pd.read_csv(os.path.join(HERE, '..', 'data', 'field_recordings.csv'), usecols=['oid', 'word_fr'])
lab = pd.DataFrame({'oid': z['oids'], 'label': [dictionary[i] for i in z['labels']]}).merge(meta, on='oid')
fr2lab = lab.groupby('word_fr').label.agg(lambda s: s.value_counts().index[0]).to_dict()
norm = lambda s: s.replace("'", '’').strip()
fr2lab = {norm(k): v for k, v in fr2lab.items()}


def vad_trim(x, win=0.025, hop=0.010, max_len=1.5, pad=0.1):
    w, h = int(win * SR), int(hop * SR)
    if len(x) < w: return None
    fr = np.lib.stride_tricks.sliding_window_view(x, w)[::h]
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    floor = np.percentile(rms, 10)
    thr = max(floor * 3.0, rms.max() * 0.1)
    act = np.where(rms > thr)[0]
    if len(act) == 0 or rms.max() < 1e-3: return None
    s = max(0, act[0] * h - int(pad * SR)); e = min(len(x), act[-1] * h + w + int(pad * SR))
    seg = x[s:e]
    L = int(max_len * SR)
    if len(seg) > L:                                     # keep the most energetic 1.5 s window
        cs = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64) ** 2)])
        en = cs[L:] - cs[:-L]                            # energy of every 1.5 s window (O(n))
        k = int(np.argmax(en))
        seg = seg[k:k + L]
    return seg.astype(np.float32)


audio, offs, lens, oids, labels, kept, drop = [], [], [], [], [], 0, {'short': 0, 'novad': 0, 'label': 0}
pos = 0
with open(SRC, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        fr = norm(d.get('French', ''))
        if fr not in fr2lab: drop['label'] += 1; continue
        b = d.get('audioData') or ''
        try:
            x, sr = sf.read(io.BytesIO(base64.b64decode(b)), dtype='float32')
        except Exception:
            drop['short'] += 1; continue
        if x.ndim > 1: x = x.mean(1)
        if len(x) < 0.3 * sr: drop['short'] += 1; continue
        if sr != SR:
            g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        seg = vad_trim(x)
        if seg is None or len(seg) < int(0.2 * SR): drop['novad'] += 1; continue
        audio.append(seg); offs.append(pos); lens.append(len(seg)); pos += len(seg)
        oids.append(d['_id']['$oid']); labels.append(dictionary.index(fr2lab[fr]))
np.savez(OUT, audio=np.concatenate(audio), offsets=np.array(offs), lengths=np.array(lens), oids=np.array(oids),
         labels=np.array(labels), dictionary=np.array(dictionary))
print('kept', len(oids), 'dropped', drop, 'hours', round(pos / SR / 3600, 2))
