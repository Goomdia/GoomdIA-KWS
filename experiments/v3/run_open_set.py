"""Experiment queue of the open-set (32-class) experiments, run sequentially; every GPU job is wrapped in
`flock <lock>` (one run per flock call) because the GPU is shared. Finished runs (summary.json) are skipped.
Hyper-parameters and learning rates are those selected for the closed-set baselines (run_queue.py):
LeTR-128 GSC / scratch lr 1e-3; wav2vec 2.0 Base lr 3e-5, bs 8, 20 epochs, patience 5, wd 1e-2, AdamW8bit.

usage: run_open_set.py PY CACHE EXTRA OUT OOD LOCK STAGES   (stages: letr_gsc, w2v2, letr_scratch, w2v2more)
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY, CACHE, EXTRA, OUT, OOD, LOCK = sys.argv[1:7]
STAGES = sys.argv[7].split(',')
W2V2 = ('--bs', '8', '--epochs', '20', '--patience', '5', '--wd', '1e-2', '--optim', 'adamw8bit')


def run(model, init, lr, seed, extra=()):
    name = f'open32-{model}-{init}-speaker-lr{lr:g}-s{seed}'
    if os.path.exists(os.path.join(OUT, name, 'summary.json')):
        return
    cmd = ['flock', LOCK, PY, os.path.join(HERE, 'train_open_set.py'), '--model', model, '--init', init,
           '--split', 'speaker', '--lr', str(lr), '--seed', str(seed), '--cache', CACHE, '--extra', EXTRA,
           '--ood', OOD, '--out', OUT, '--name', name, *extra]
    print('>>', name, flush=True)
    with open(os.path.join(OUT, name + '.log'), 'w') as log:
        log.write(' '.join(cmd) + '\n'); log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)


os.makedirs(OUT, exist_ok=True)
if 'letr_gsc' in STAGES:
    for seed in (0, 1, 2):
        run('letr128t', 'gsc', 1e-3, seed, ('--save_model',))
if 'w2v2' in STAGES:
    run('w2v2', 'ssl', 3e-5, 0, W2V2 + ('--save_model',))
if 'letr_scratch' in STAGES:
    for seed in (0, 1, 2):
        run('letr128t', 'scratch', 1e-3, seed, ('--save_model',))
if 'w2v2more' in STAGES:
    run('w2v2', 'ssl', 3e-5, 1, W2V2)
print('QUEUE DONE', STAGES, flush=True)
