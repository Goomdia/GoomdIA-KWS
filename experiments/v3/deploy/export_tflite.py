"""Export a train_kws.py checkpoint to TFLite (fp32 and dynamic-range int8) and check it against PyTorch.

Run it with the deploy venv (litert-torch, ai-edge-quantizer, torch+torchaudio; see run_phone_bench.sh header):
  $GOOMDIA_CACHE/deploy_venv/bin/python export_tflite.py --ckpt <.../model.pt> [--name NAME]
      [--arch letr128t|letr96|w2v2|mms300m] [--factory module:function] [--frontend graph|none] [--precision fp32 int8]

frontend=graph : input is the raw 16 kHz waveform (1, 24000) (1.5 s window); the whole front end is in the graph
  - LeTR: per-utterance min-max scaling + MFCC (STFT as a strided conv with the windowed DFT basis, mel matrix,
    10*log10 with the 80 dB floor, DCT matrix; verified identical to torchaudio.transforms.MFCC) + network.
  - wav2vec 2.0 / MMS: zero-mean/unit-variance normalisation + CNN feature encoder + transformer + classifier.
frontend=none  : LeTR only, input = MFCC features (1, 151, 16); the front end is then excluded from the phone latency.
int8 = dynamic-range quantisation of the FULLY_CONNECTED (nn.Linear) weights to int8, activations quantised on the
  fly by the kernel (the TFLite counterpart of torch dynamic quantisation); convolutions (incl. the DFT conv of the
  MFCC and the CNN feature encoder of wav2vec 2.0) stay fp32.
Writes <out-dir>/<name>[.nofront].<prec>.tflite and <same>.json (size, params, max abs logit difference to the
PyTorch model, argmax agreement on 50 evenly spaced speaker-disjoint test clips (centre 1.5 s) + 2 synthetic signals).
"""
import argparse, json, os, sys, time
import numpy as np, torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True, help='model.pt, or random:ARCH[:K] for an architecture-only (random init) export')
ap.add_argument('--arch', default=None); ap.add_argument('--factory', default=None)
ap.add_argument('--name', default=None, help='default: name of the run directory')
ap.add_argument('--out-dir', default=os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')) + '/tflite')
ap.add_argument('--seconds', type=float, default=1.5)
ap.add_argument('--frontend', default='graph', choices=['graph', 'none'])
ap.add_argument('--precision', nargs='+', default=['fp32', 'int8'], choices=['fp32', 'int8'])
ap.add_argument('--n-real', type=int, default=50)
ap.add_argument('--requant', action='store_true', help='with --reuse: keep the fp32 file but redo the int8 quantisation')
ap.add_argument('--reuse', action='store_true', help='keep existing .tflite files (if they load) and only verify')
a = ap.parse_args()
torch.set_num_threads(4)

import litert_torch
from ai_edge_litert.interpreter import Interpreter

ck = a.ckpt if C.is_random(a.ckpt) else os.path.abspath(a.ckpt)
model, arch, k = C.load_model(ck, a.arch, a.factory)
name = a.name or ('random-' + arch if C.is_random(ck) else os.path.basename(os.path.dirname(ck)))
os.makedirs(a.out_dir, exist_ok=True)
kind = C.model_kind(model)
tag = name + ('.nofront' if a.frontend == 'none' else '')
N = int(a.seconds * C.SR)


def torch_reference(x):
    """What train_kws would compute for one utterance: data-pipeline min-max scaling, then model(x, lens)."""
    with torch.no_grad():
        if kind in ('letr', 'bc'): x = C.MinMax()(x)
        return model(x, torch.tensor([x.shape[1]]))


# test inputs: real speaker-disjoint test clips (centre 1.5 s), plus two synthetic signals
xs = [C.reference_input(model, a.seconds, s) for s in (0, 1)]
try:
    import pandas as pd
    pool = C.T.Pool(C.CACHE)
    sp = pd.read_csv(os.path.join(C.ROOT, 'data', 'splits', 'splits_usable.csv'))
    te = sp.oid[sp['speaker'] == 'test'].tolist()
    for o in [te[i] for i in np.linspace(0, len(te) - 1, a.n_real).astype(int)]:
        w = torch.from_numpy(np.asarray(pool.wave(pool.index[o]), dtype=np.float32))
        w = w[max(0, (len(w) - N) // 2):][:N]
        xs.append(torch.nn.functional.pad(w, (0, N - len(w)))[None])
except Exception as e:
    print('real clips unavailable:', e)

if a.frontend == 'graph':
    net, sample = C.FullGraph(model).eval(), xs[0]
    to_in = lambda x: x
else:
    assert kind == 'letr', '--frontend none is implemented for LeTR only'
    net = C.NetOnly(model).eval()
    to_in = lambda x: model.features(C.MinMax()(x)).detach()
    sample = to_in(xs[0])

t0 = time.time()
fp32_path = os.path.join(a.out_dir, f'{tag}.fp32.tflite')


def loads(p):
    try:
        i = Interpreter(model_path=p); i.allocate_tensors(); return True
    except Exception:
        return False


reuse = a.reuse and loads(fp32_path)
int8_path = fp32_path.replace('fp32', 'int8')
reuse_int8 = reuse and not a.requant and loads(int8_path)
if not reuse:
    with torch.no_grad():
        litert_torch.convert(net, (sample,)).export(fp32_path)
    print(f'converted {tag} in {time.time() - t0:.0f} s', flush=True)

paths = {'fp32': fp32_path, 'int8': fp32_path.replace('fp32', 'int8')}
if 'int8' in a.precision and not reuse_int8:
    from ai_edge_quantizer import quantizer, qtyping
    qt = quantizer.Quantizer(fp32_path)
    # wav2vec 2.0 Base: int8 dynamic quantisation of the feature-projection layer (input = post-LayerNorm CNN features
    # with large outlier channels) destroys the model (argmax agreement 0/22); every other FC layer is quantised.
    rx = r'^(?!.*feature_projection).*$' if kind == 'ssl' else '.*'
    qt.add_dynamic_config(regex=rx, operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=8)
    qt.quantize().export_model(paths['int8'], overwrite=True)


def run_tflite(path, x):
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i['index'], x.numpy().astype(np.float32)); it.invoke()
    return torch.from_numpy(it.get_tensor(o['index']).copy())


ref = [torch_reference(x) for x in xs]
for prec in a.precision:
    outs = [run_tflite(paths[prec], to_in(x)) for x in xs]
    d = max(float((o - r).abs().max()) for o, r in zip(outs, ref))
    agree = float(np.mean([int(o.argmax() == r.argmax()) for o, r in zip(outs, ref)]))
    meta = dict(name=name, ckpt=ck if C.is_random(ck) else os.path.relpath(ck, C.ROOT), arch=arch, n_classes=k, precision=prec,
                frontend_in_graph=a.frontend == 'graph', input_shape=list(to_in(xs[0]).shape),
                window_s=a.seconds, tflite=paths[prec], size_mb=round(C.file_mb(paths[prec]), 3),
                params=sum(p.numel() for p in model.parameters()), max_abs_logit_diff_vs_torch=d,
                max_abs_logit=float(max(r.abs().max() for r in ref)), argmax_agreement=agree, n_check=len(xs),
                litert_torch=litert_torch.__version__ if hasattr(litert_torch, '__version__') else '')
    json.dump(meta, open(paths[prec][:-len('.tflite')] + '.json', 'w'), indent=1)
    print(f"{tag} {prec}: size {meta['size_mb']:.2f} MB, max|dlogit| vs torch {d:.2e} "
          f"(max|logit| {meta['max_abs_logit']:.1f}), argmax agreement {agree:.2f} on {len(xs)} inputs", flush=True)
