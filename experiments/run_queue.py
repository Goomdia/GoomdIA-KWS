"""Experiment queue for the paper. Several workers may run the same stage concurrently on one GPU:
each run is claimed with a lock file; finished runs (summary.json present) are skipped.

stage sweep   : learning-rate selection on the validation set (speaker split, seed 0)
stage main    : {speaker, random} x {LeTR-96, LeTR-128t} x {scratch, GSC} x 3 seeds
stage scaling : LeTR-128t {scratch, GSC} trained on 10/25/50 % of the training groups x 3 seeds
stage w2v2    : wav2vec 2.0 base fine-tuning, {speaker, random} x seeds
"""
import json, os, subprocess, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PY, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
STAGES = sys.argv[4].split(',')
OOD = sys.argv[5] if len(sys.argv) > 5 else ''


def run(model, init, split, lr, seed, subset='', extra=()):
    name = f'{model}-{init}-{split}{("-" + subset) if subset else ""}-lr{lr:g}-s{seed}'
    if os.path.exists(os.path.join(OUT, name, 'summary.json')):
        return name
    try:                                   # claim the run (several workers share the queue)
        os.close(os.open(os.path.join(OUT, name + '.lock'), os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return name
    cmd = [PY, os.path.join(HERE, 'train_kws.py'), '--model', model, '--init', init, '--split', split,
           '--lr', str(lr), '--seed', str(seed), '--cache', CACHE, '--out', OUT, '--name', name, *extra]
    if subset: cmd += ['--train_subset', subset]
    if OOD: cmd += ['--ood', OOD]
    print('>>', name, flush=True)
    with open(os.path.join(OUT, name + '.log'), 'w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return name


def best_lr(model, init):
    best = None
    for f in glob.glob(os.path.join(OUT, f'{model}-{init}-speaker-lr*-s0', 'summary.json')):
        s = json.load(open(f))
        if best is None or s['best_val_acc'] > best[0]:
            best = (s['best_val_acc'], s['args']['lr'])
    return best[1]


LRS = [8e-5, 3e-4, 1e-3]
os.makedirs(OUT, exist_ok=True)
if 'sweep' in STAGES:
    for init in ('gsc', 'scratch'):
        for lr in LRS:
            run('letr128t', init, 'speaker', lr, 0)
if 'main' in STAGES:
    for init in ('gsc', 'scratch'):
        lr = best_lr('letr128t', init)
        for model in ('letr128t', 'letr96'):
            for split in ('speaker', 'random'):
                for seed in (0, 1, 2):
                    run(model, init, split, lr, seed, extra=('--save_model',) if seed == 0 else ())
if 'scaling' in STAGES:
    # one extra seed-0 LeTR-128 (GSC) model saved to disk for the streaming evaluation
    run('letr128t', 'gsc', 'speaker', best_lr('letr128t', 'gsc'), 10, extra=('--save_model',))
    # controlled seen- vs unseen-speaker comparison on identical test utterances
    for init in ('gsc', 'scratch'):
        for cond in ('ctrl_unseen', 'ctrl_seen'):
            for seed in (0, 1, 2):
                run('letr128t', init, cond, best_lr('letr128t', init), seed)
    for init in ('gsc', 'scratch'):
        lr = best_lr('letr128t', init)
        for subset in ('train_frac_10', 'train_frac_25', 'train_frac_50'):
            for seed in (0, 1, 2):
                run('letr128t', init, 'speaker', lr, seed, subset)
if 'w2v2' in STAGES:
    for split, seed in (('speaker', 0), ('speaker', 1), ('random', 0)):
        if True:
            run('w2v2', 'ssl', split, 3e-5, seed, extra=('--bs', '8', '--epochs', '20', '--patience', '5', '--wd', '1e-2', '--optim', 'adamw8bit')
                + (('--save_model',) if (split, seed) == ('speaker', 0) else ()))
if 'w2v2more' in STAGES:
    for split in ('speaker', 'random'):
        run('w2v2', 'ssl', split, 3e-5, 2, extra=('--bs', '8', '--epochs', '20', '--patience', '5', '--wd', '1e-2', '--optim', 'adamw8bit'))
print('QUEUE DONE', STAGES, flush=True)
