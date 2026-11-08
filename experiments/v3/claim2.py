import pandas as pd,numpy as np,os
m=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/../../data/field_recordings.csv')
ne=m[~m['empty']]; us=m[m.usable]
for nm,d in (('all',m),('non-empty',ne),('usable',us)):
    g=d.groupby('group_id').size(); print(nm,len(d),'groups',len(g),'sum ceil(n/30)',int(np.ceil(g/30).sum()),'sum round(n/30)',int(np.round(g/30).sum()),'sum floor',int((g//30).sum()),'n/30',round(len(d)/30,1))
g=m.groupby('group_id').size(); big=g[g>30]
print('groups n<=30:',(g<=30).sum(),'clips',g[g<=30].sum(),' n>30:',len(big),'clips',big.sum())
# blocks of 30 starting in groups >=? alternative: ceil per (agent,day,gender,age)
for keys in (['agent_id','day'],['agent_id','day','gender','age']):
    gg=m.groupby(keys).size(); print(keys,len(gg),int(np.ceil(gg/30).sum()))
# fragmentary groups (<10 clips): count
print('groups with <10 clips:',(g<10).sum(),'clips',g[g<10].sum(), ' ceil sum for groups>=10:', int(np.ceil(g[g>=10]/30).sum()))
print('groups with >=15 clips: ceil', int(np.ceil(g[g>=15]/30).sum()), ' round-up w/ n>=30 floor', int((g//30).sum()))
