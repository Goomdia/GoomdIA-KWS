"""Restartable queue of the v4 experiments (E2 BC-ResNet, E3 mHuBERT, E4 32-class models, E5 MMS-300M learning curve).
One heavy job at a time: every job runs under `flock gpu.lock nice -n 10`, with 4 threads, 2 DataLoader workers, 20 s pause between
jobs. Finished runs (summary.json / checkpoint present) are skipped, so the script can be relaunched after an interruption.
Priority order: P1 BC-ResNet scratch/GSC, E3, E4, MSWC (prep, pre-training, fine-tuning), BC-8 subsets, E5.
usage: run_v4.py [phase ...]   phases: p1 e3 e4 mswc subsets e5   (default: all, in this order)
"""
import json, os, subprocess, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, '..')
PY = os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')) + '/venv/bin/python'; C = os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache'))
LOCK = C + '/gpu.lock'; POOL = C + '/pool.npz'; OOD = C + '/volunteer_ood.npz'; EXTRA = C + '/v3/open_set_clips.npz'
GSC = C + '/gsc35.npz'; MSWC = C + '/mswc'
RUNS = os.path.join(EXP, 'runs'); RUNS32 = os.path.join(HERE, 'runs'); PRE = os.path.join(HERE, 'pretrained')
ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TORCH_NUM_THREADS='4')
REC = ['--bs', '8', '--epochs', '20', '--patience', '5', '--wd', '1e-2', '--optim', 'adamw8bit']   # SSL recipe
LRS = [3e-4, 1e-3, 3e-3, 1e-2]   # 1e-2 added: BC-ResNet preferred the grid edge


def job(script, args, done, log):
    if os.path.exists(done): return
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print('>>', os.path.basename(log), time.strftime('%F %T'), flush=True)
    with open(log, 'w') as f:
        subprocess.run(['flock', LOCK, 'nice', '-n', '10', PY, script, *map(str, args)],
                       env=ENV, stdout=f, stderr=subprocess.STDOUT, check=False)
    time.sleep(20)


def summ(out, name):
    p = os.path.join(out, name, 'summary.json')
    return json.load(open(p)) if os.path.exists(p) else None


def train(name, model, init, lr, seed, extra=(), out=RUNS, subset='', open32=False, rec=False):
    script = os.path.join(HERE, 'train_open_set.py') if open32 else os.path.join(EXP, 'train_kws.py')
    a = ['--model', model, '--init', init, '--split', 'speaker', '--lr', lr, '--seed', seed, '--cache', POOL, '--out', out,
         '--name', name, '--ood', OOD]
    if open32: a += ['--extra', EXTRA]
    if subset: a += ['--train_subset', subset]
    if rec: a += REC
    job(script, a + list(extra), os.path.join(out, name, 'summary.json'), os.path.join(out, name + '.log'))
    return summ(out, name)


def n(model, init, lr, seed, subset=''):
    return f'{model}-{init}-speaker{("-" + subset) if subset else ""}-lr{lr:g}-s{seed}'


def best_lr(model, init, lrs):
    c = [(summ(RUNS, n(model, init, lr, 0))['best_val_acc'], lr) for lr in lrs if summ(RUNS, n(model, init, lr, 0))]
    return max(c)[1] if c else None


def seeds3(model, init, lr, need=3, extra=(), subset='', rec=False, start=1, max_try=6):
    """seed 0 exists; run seeds until `need` runs with best val acc >= 0.1 (failed runs are kept and replaced by the next seed)."""
    ok = 1 if (summ(RUNS, n(model, init, lr, 0, subset)) or {}).get('best_val_acc', 0) >= 0.1 else 0
    s = start
    while ok < need and s < max_try:
        r = train(n(model, init, lr, s, subset), model, init, lr, s, extra, subset=subset, rec=rec)
        ok += int(bool(r) and r['best_val_acc'] >= 0.1); s += 1


def sweep(model, init, lrs=LRS):
    for lr in lrs: train(n(model, init, lr, 0), model, init, lr, 0)
    lr = best_lr(model, init, lrs)
    if lr: seeds3(model, init, lr)
    return lr


def pretrain(model, data, epochs, bs, lr, tmax, cache):
    job(os.path.join(HERE, 'pretrain.py'), ['--model', model, '--data', data, '--cache', cache, '--epochs', epochs, '--bs', bs,
        '--lr', lr, '--max_epochs_time', tmax], os.path.join(PRE, f'{model}-{data}.json'), os.path.join(PRE, f'{model}-{data}.log'))


def phase_p1():
    job(os.path.join(HERE, 'build_gsc_cache.py'), [GSC], GSC, os.path.join(PRE, 'build_gsc.log'))
    for m in ('bcresnet1', 'bcresnet8'): pretrain(m, 'gsc', 30, 128, 2e-3, 2400, GSC)
    for m in ('bcresnet1', 'bcresnet8'):
        for init in ('scratch', 'gsc'): sweep(m, init)


def phase_e3():
    ok, s = 0, 0
    while ok < 3 and s < 6:
        r = train(n('mhubert', 'ssl', 3e-5, s), 'mhubert', 'ssl', 3e-5, s, rec=True); ok += int(bool(r) and r['best_val_acc'] >= 0.1); s += 1


def best_init(model, inits):
    c = []
    for i in inits:
        lr = best_lr(model, i, LRS)
        if lr: v = [summ(RUNS, n(model, i, lr, s))['best_val_acc'] for s in range(3) if summ(RUNS, n(model, i, lr, s))]; c.append((sum(v) / max(1, len(v)), i, lr))   # validation accuracy
    return max(c)[1:] if c else None


def phase_e4():
    init, lr = best_init('bcresnet8', ['scratch', 'gsc'])    # best validation-selected init among the two available at this stage
    ok, s = 0, 0
    while ok < 3 and s < 6:
        nm = f'open32-bcresnet8-{init}-speaker-lr{lr:g}-s{s}'
        r = train(nm, 'bcresnet8', init, lr, s, ['--save_model'], out=RUNS32, open32=True); ok += int(bool(r) and r['best_val_acc'] >= 0.1); s += 1
    train('open32-w2v2-ssl-speaker-lr3e-05-s2', 'w2v2', 'ssl', 3e-5, 2, ['--save_model'], out=RUNS32, open32=True, rec=True)
    for s in (0, 1):
        train(f'open32-mms300m-ssl-speaker-lr3e-05-s{s}', 'mms300m', 'ssl', 3e-5, s, ['--save_model'], out=RUNS32, open32=True, rec=True)
    json.dump(dict(bcresnet8_init=init, lr=lr), open(os.path.join(HERE, 'e4_bc8_choice.json'), 'w'))


def phase_mswc():
    job(os.path.join(HERE, 'prep_mswc.py'), [MSWC], MSWC + '/mswc_subset.npz', MSWC + '/prep_mswc.out')
    mc = MSWC + '/mswc_subset.npz'
    pretrain('bcresnet1', 'mswc', 25, 128, 2e-3, 2400, mc); pretrain('bcresnet8', 'mswc', 25, 128, 2e-3, 3000, mc)
    pretrain('letr128t', 'mswc', 15, 64, 3e-4, 4200, mc)
    for m in ('bcresnet1', 'bcresnet8'): sweep(m, 'mswc')
    for lr in (8e-5, 3e-4, 1e-3): train(n('letr128t', 'mswc', lr, 0), 'letr128t', 'mswc', lr, 0)
    lr = best_lr('letr128t', 'mswc', (8e-5, 3e-4, 1e-3))
    if lr: seeds3('letr128t', 'mswc', lr)


def phase_subsets():
    init, lr = best_init('bcresnet8', ['scratch', 'gsc', 'mswc'])
    for sub in ('train_frac_10', 'train_frac_25', 'train_frac_50'):
        for s in (0, 1, 2): train(n('bcresnet8', init, lr, s, sub), 'bcresnet8', init, lr, s, subset=sub)


def phase_e5():
    for sub in ('train_frac_10', 'train_frac_25', 'train_frac_50'):
        for s in (0, 1): train(n('mms300m', 'ssl', 3e-5, s, sub), 'mms300m', 'ssl', 3e-5, s, subset=sub, rec=True)


if __name__ == '__main__':
    ph = sys.argv[1:] or ['p1', 'e3', 'e4', 'mswc', 'subsets', 'e5']
    os.makedirs(PRE, exist_ok=True)
    for p in ph:
        {'p1': phase_p1, 'e3': phase_e3, 'e4': phase_e4, 'mswc': phase_mswc, 'subsets': phase_subsets, 'e5': phase_e5}[p]()
        print('PHASE', p, 'DONE', time.strftime('%F %T'), flush=True)
