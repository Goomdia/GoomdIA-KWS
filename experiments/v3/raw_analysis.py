"""Join raw_scan.csv (from raw_scan.py) with field_recordings.csv; characterise empty clips. CPU, read-only."""
import json,glob,pandas as pd,numpy as np,os
H=os.path.dirname(os.path.abspath(__file__))
d=[]
for f in glob.glob(H+'/../runs/*/summary.json'):
    S=json.load(open(f)); d.append((S['args']['name'],S['train_time_s'],S['history'][-1]['time_s']))
x=pd.DataFrame(d,columns=['n','tt','last']); print('train_time_s - last epoch time_s:',(x.tt-x['last']).describe().round(1).to_dict())
r=pd.read_csv(H+'/raw_scan.csv',keep_default_na=False,dtype={'session':str}); m=pd.read_csv(H+'/../../data/field_recordings.csv')
print(len(r), r.oid.nunique(), r.session.value_counts().to_dict(), r.extra_keys.value_counts(dropna=False).to_dict())
j=m.merge(r,on='oid',how='left'); print('unmatched',j.nbytes.isna().sum())
for k,g in j.groupby('empty'):
    print('--- empty =',k,len(g)); print(' nbytes',g.nbytes.describe().round(0).to_dict()); print(' riff',g.riff.value_counts().to_dict(),'data_size',g.data_size.describe().round(0).to_dict()); print(' all-zero samples',(g.n_nonzero==0).sum(),'max_abs==0',(g.max_abs==0).sum(), 'sr',g.sr_hdr.value_counts().to_dict(),'nch',g.nch.value_counts().to_dict(),'bits',g.bits.value_counts().to_dict())
e=j[j['empty']]; print('empty nbytes values', e.nbytes.value_counts().head(8).to_dict(), 'b64len', e.b64_len.value_counts().head(5).to_dict())
n=j[~j['empty']]; print('non-empty: all-zero',(n.n_nonzero==0).sum(),'max_abs<100',(n.max_abs<100).sum(),'min bytes',n.nbytes.min(),'min dur',n.duration_s.min(), 'bytes/dur',(n.nbytes/n.duration_s).describe().round(0).to_dict())
print(pd.crosstab(j.session,j['empty']))
print(pd.crosstab(j.agent_id,j.session))
print('header data_size != nbytes-44:', int((j.data_size!=j.nbytes-44).sum()), 'among empty', int((e.data_size!=e.nbytes-44).sum()))
