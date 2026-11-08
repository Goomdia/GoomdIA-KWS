"""Stream the MongoDB JSON exports, keep metadata only (no audio, no PII), compute WAV header info."""
import json, base64, struct, sys, os, datetime
import pandas as pd
import numpy as np

def wav_info(b64):
    if not b64: return dict(sr=np.nan, ch=np.nan, bits=np.nan, dur=np.nan, nbytes=0)
    head = base64.b64decode(b64[:64] + '=' * (-len(b64[:64]) % 4))
    nbytes = len(b64) * 3 // 4 - b64[-2:].count('=')
    try:
        if head[:4] != b'RIFF': raise ValueError
        ch, sr = struct.unpack('<HI', head[22:28]); bits = struct.unpack('<H', head[34:36])[0]
        dur = (nbytes - 44) / (sr * ch * bits / 8)
    except Exception:
        ch = sr = bits = dur = np.nan
    return dict(sr=sr, ch=ch, bits=bits, dur=dur, nbytes=nbytes)

def stream(path, kind):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            oid = d['_id']['$oid']
            r = dict(kind=kind, line=i, oid=oid, ts=datetime.datetime.fromtimestamp(int(oid[:8], 16), datetime.UTC).replace(tzinfo=None),
                     French=d.get('French'), Moore=d.get('Moore', d.get('traduction')),
                     session=str(d.get('sessionCount')), gender=d.get('gender'), age=d.get('ageRange'))
            if kind == 'agent':
                r['agent'] = d.get('agentUserMail', '').split('@')[0]
                r.update(wav_info(d.get('audioData')))
            elif kind == 'citadel':
                r['agent'] = d.get('agentUserMail', '').split('@')[0]
                wi = wav_info(d.get('audioDataMoore')); r.update(wi)
                r['dur_fr'] = wav_info(d.get('audioDataFrench'))['dur']
            else:
                r['user'] = d.get('userID')
                r.update(wav_info(d.get('audioData')))
            rows.append(r)
    return pd.DataFrame(rows)

base = sys.argv[1]; out = sys.argv[2]
w1 = base + '/01 - Agents _ 1ere semaine/00 - Data from mongodb/'
dfs = {
    'volunteer': stream(w1 + 'audios.json', 'volunteer'),
    'agent_w1': stream(w1 + 'agentaudios.json', 'agent'),
    'citadel': stream(w1 + 'agentaudiocitadels.json', 'citadel'),
    'agent_w2': stream(base + '/02 - Agents - 2eme semaine/agentaudios.json', 'agent'),
}
prof = pd.read_json(w1 + 'profiles.json', lines=True)[['userID', 'gender', 'ageRange', 'sessionCount']]
prof.to_pickle(out + '/profiles.pkl')
for k, v in dfs.items():
    v.to_pickle(f'{out}/{k}.pkl'); print(k, len(v))
