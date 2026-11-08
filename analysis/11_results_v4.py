"""data/results_v4_summary.csv (mean +- sd per model/init/split/subset) and data/streaming_v4_summary.csv (32-class detection, centre scoring)."""
import glob, json, os
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
rows = []
for f in glob.glob(os.path.join(ROOT, 'experiments', 'runs', '*', 'summary.json')):
    s = json.load(open(f)); a = s['args']; nm = os.path.basename(os.path.dirname(f))
    if a['model'] == 'mms300m' and not a.get('train_subset') and 'mms_lr_check' in f: continue
    rows.append(dict(model=a['model'], init=a['init'], split=a['split'], subset=a.get('train_subset') or 'full', lr=a['lr'], seed=a['seed'],
                     val=s['best_val_acc'], test=s['test']['acc'], ood=s.get('ood', {}).get('acc', np.nan), params=s['n_params'],
                     time=s['train_time_s'], mem=s.get('peak_mem_mib', np.nan), failed=s['best_val_acc'] < 0.1, name=nm))
d = pd.DataFrame(rows)
keys = ['model', 'init', 'split', 'subset', 'lr']
g = d.groupby(keys)
out = g.apply(lambda x: pd.Series(dict(
    n_seeds=int((~x.failed).sum()), n_failed=int(x.failed.sum()), seeds=','.join(map(str, sorted(x.seed[~x.failed]))),
    val_mean=x.val[~x.failed].mean(), test_acc_mean=x.test[~x.failed].mean(), test_acc_sd=x.test[~x.failed].std(ddof=1) if (~x.failed).sum() > 1 else np.nan,
    ood_acc_mean=x.ood[~x.failed].mean(), ood_acc_sd=x.ood[~x.failed].std(ddof=1) if (~x.failed).sum() > 1 else np.nan,
    params=int(x.params.iloc[0]), train_time_s_mean=x.time[~x.failed].mean(), peak_mem_mib_mean=x.mem[~x.failed].mean())), include_groups=False).reset_index()
# lr selected on seed-0 validation accuracy within (model, init, split, subset=full)
sel = d[(d.seed == 0)].sort_values('val').groupby(['model', 'init', 'split', 'subset']).tail(1)[['model', 'init', 'split', 'subset', 'lr']]
sel['selected_lr'] = True
out = out.merge(sel, on=keys, how='left'); out['selected_lr'] = out.selected_lr.fillna(False)
out.sort_values(['model', 'init', 'split', 'subset', 'lr']).to_csv(os.path.join(ROOT, 'data', 'results_v4_summary.csv'), index=False)
print(out[out.selected_lr & (out.split == 'speaker')].round(4).to_string())
f = os.path.join(ROOT, 'data', 'streaming_tuned_test_centre.csv')
t = pd.read_csv(f)
t = t[(t.kind == 'tuned') & (t.label == 'tuned') & (t['mode'] == 'centre') & t.stream.isin(['moore', 'english'])]
agg = t.groupby(['group', 'budget', 'wcap', 'stream']).agg(n_runs=('run', 'nunique'), det_mean=('det', 'mean'), det_sd=('det', 'std'),
        wrong_kw_mean=('wrong_kw', 'mean'), fa_nonkw_h_mean=('fa_nonkw_h', 'mean'), fa_nonkw_h_sd=('fa_nonkw_h', 'std')).reset_index()
agg.to_csv(os.path.join(ROOT, 'data', 'streaming_v4_summary.csv'), index=False)
print(agg[(agg.budget == 5) & (agg.wcap == 99)].round(3).to_string())
