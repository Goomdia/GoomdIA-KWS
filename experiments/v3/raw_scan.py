"""Stream the raw field export (JSON lines, base64 WAV) and write per-record audio diagnostics
(no audio, no personal data) to raw_scan.csv. CPU only, read-only."""
import json, base64, struct, sys, csv
import numpy as np
P = sys.argv[1]; OUT = sys.argv[2]
keys = None
with open(P, 'rb') as f, open(OUT, 'w', newline='') as o:
    w = csv.writer(o); w.writerow(['oid','session','b64_len','nbytes','riff','data_size','n_samples_hdr','sr_hdr','nch','bits','n_nonzero','max_abs','extra_keys'])
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        oid = d['_id']['$oid']; a = d.get('audioData') or ''
        b = base64.b64decode(a) if a else b''
        riff = b[:4] == b'RIFF'
        sr = nch = bits = ds = -1
        pcm = b''
        if len(b) >= 44 and riff:
            nch, sr = struct.unpack('<HI', b[22:28]); bits = struct.unpack('<H', b[34:36])[0]
            i = b.find(b'data'); ds = struct.unpack('<I', b[i+4:i+8])[0] if i >= 0 else -1
            pcm = b[i+8:] if i >= 0 else b''
        x = np.frombuffer(pcm[:len(pcm)//2*2], dtype='<i2') if bits == 16 else np.zeros(0, 'i2')
        w.writerow([oid, d.get('sessionCount'), len(a), len(b), int(riff), ds, len(x), sr, nch, bits,
                    int((x != 0).sum()), int(np.abs(x.astype(int)).max()) if len(x) else 0,
                    '|'.join(sorted(set(d) - {'_id','agentUserID','agentUserMail','sessionCount','French','traduction','Language','audioData'}))])
