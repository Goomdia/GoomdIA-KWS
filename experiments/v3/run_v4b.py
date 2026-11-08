"""E4 addendum: 32-class mHuBERT-147 (mid-tier model, 2 seeds). Same limits and restart logic as run_v4.py."""
import run_v4 as q
for s in (0, 1):
    q.train(f'open32-mhubert-ssl-speaker-lr3e-05-s{s}', 'mhubert', 'ssl', 3e-5, s, ['--save_model'], out=q.RUNS32, open32=True, rec=True)
print('V4B DONE', flush=True)
