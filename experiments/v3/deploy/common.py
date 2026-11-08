"""Shared helpers of the deployment-cost scripts: build a KWS model from a checkpoint and wrap it for export.

A checkpoint is the `model.pt` state dict written by experiments/train_kws.py. The architecture is read from the
`summary.json` stored next to it (args.model), or given with --arch. Models that are not defined in train_kws.py
(e.g. BC-ResNet, mHuBERT) are added with --factory module:function, where function(arch, n_classes) returns an
nn.Module with the same forward signature as train_kws models, forward(x, lens) on a (B, N) 16 kHz waveform.
"""
import importlib, json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, '..', '..'))              # experiments/
ROOT = os.path.abspath(os.path.join(EXP, '..'))
sys.path[:0] = [EXP, os.path.join(EXP, 'v3')]
import train_kws as T                                              # noqa: E402  (LeTRKWS, W2V2KWS, SSL, Pool, ...)

SR = 16000
CACHE = os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')) + '/pool.npz'
LOCK = os.environ.get('GOOMDIA_CACHE', os.path.expanduser('~/goomdia_kws_cache')) + '/gpu.lock'


class W2V2Eval(T.W2V2KWS):
    """W2V2KWS built from the config only (no pretrained weights downloaded or loaded), same forward."""
    def __init__(self, n_classes, name):
        nn.Module.__init__(self)
        from transformers import AutoConfig, Wav2Vec2ForSequenceClassification
        from transformers import HubertForSequenceClassification
        cls = HubertForSequenceClassification if 'hubert' in name.lower() else Wav2Vec2ForSequenceClassification
        self.m = cls(AutoConfig.from_pretrained(name, num_labels=n_classes))


def is_random(ckpt):
    return ckpt.startswith('random:')


def infer_arch(ckpt, arch=None):
    if is_random(ckpt): return ckpt.split(':')[1]              # 'random:ARCH[:K]' = architecture only, random init
    if arch: return arch
    s = os.path.join(os.path.dirname(ckpt), 'summary.json')
    if os.path.exists(s): return json.load(open(s))['args']['model']
    raise SystemExit(f'cannot infer the architecture of {ckpt}: pass --arch')


def n_classes_of(sd):
    for k in ('model.classifier_head.weight', 'm.classifier.weight'):
        if k in sd: return sd[k].shape[0]
    return None


def load_model(ckpt, arch=None, factory=None, n_classes=None):
    """Return (model.eval(), arch, n_classes). The state dict is loaded strictly."""
    arch = infer_arch(ckpt, arch)
    if is_random(ckpt):
        sd = None; torch.manual_seed(0)
        k = n_classes or (int(ckpt.split(':')[2]) if ckpt.count(':') > 1 else 30)
    else:
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        k = n_classes or n_classes_of(sd)
    if factory:
        mod, fn = factory.split(':')
        model = getattr(importlib.import_module(mod), fn)(arch, k)
    elif arch in T.VARIANTS:
        model = T.LeTRKWS(arch, k)
    elif arch in T.BC:
        model = T.BCResNetKWS(arch, k)
    elif arch in T.SSL:
        model = W2V2Eval(k, T.SSL[arch])
    else:
        raise SystemExit(f'unknown architecture {arch}: pass --factory module:function')
    if sd is not None: model.load_state_dict(sd)
    return model.eval(), arch, k


def model_kind(model):
    return 'letr' if isinstance(model, T.LeTRKWS) else 'bc' if isinstance(model, T.BCResNetKWS) else 'ssl' if isinstance(model, T.W2V2KWS) else 'generic'


def file_mb(path):
    return os.path.getsize(path) / 2 ** 20


class GraphMFCC(nn.Module):
    """torchaudio MFCC re-expressed with ops that exist in TFLite (STFT as one strided conv1d with the windowed DFT
    basis, power, mel matrix, 10*log10 with the 80 dB floor, DCT matrix). Same output as T.LeTRKWS.features."""
    def __init__(self, mfcc):
        super().__init__()
        ms = mfcc.MelSpectrogram; sp = ms.spectrogram
        self.n_fft, self.hop, self.top_db = sp.n_fft, sp.hop_length, mfcc.top_db
        assert sp.center and sp.pad_mode == 'reflect' and sp.win_length == sp.n_fft and sp.power == 2.0
        n = torch.arange(self.n_fft, dtype=torch.float64)
        k = torch.arange(self.n_fft // 2 + 1, dtype=torch.float64)
        ang = 2 * torch.pi * k[:, None] * n[None, :] / self.n_fft                   # (F, n_fft)
        win = sp.window.double()
        basis = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)        # (2F, n_fft)
        self.register_buffer('basis', basis.float()[:, None, :])                    # conv1d weight (2F, 1, n_fft)
        self.register_buffer('fb', ms.mel_scale.fb.clone())                         # (F, n_mels)
        self.register_buffer('dct', mfcc.dct_mat.clone())                           # (n_mels, n_mfcc)
        self.nf = self.n_fft // 2 + 1

    def forward(self, x):                                  # (B, N) -> (B, T, n_mfcc)
        x = F.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        s = F.conv1d(x, self.basis, stride=self.hop)                                # (B, 2F, T)
        p = s[:, :self.nf] ** 2 + s[:, self.nf:] ** 2                               # power spectrogram
        m = torch.matmul(p.transpose(1, 2), self.fb)                                # (B, T, n_mels)
        db = 10.0 * torch.log10(torch.clamp(m, min=1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - self.top_db)
        return torch.matmul(db, self.dct)                                           # (B, T, n_mfcc)


class GraphLogMel(nn.Module):
    """BC-ResNet front end of T.BCResNetKWS (MelSpectrogram + 1e-6, log) with the STFT as a strided conv1d."""
    def __init__(self, mel):
        super().__init__()
        sp = mel.spectrogram
        self.n_fft, self.hop = sp.n_fft, sp.hop_length
        assert sp.center and sp.pad_mode == 'reflect' and sp.win_length == sp.n_fft and sp.power == 2.0
        n = torch.arange(self.n_fft, dtype=torch.float64); k = torch.arange(self.n_fft // 2 + 1, dtype=torch.float64)
        ang = 2 * torch.pi * k[:, None] * n[None, :] / self.n_fft; win = sp.window.double()
        self.register_buffer('basis', torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0).float()[:, None, :])
        self.register_buffer('fb', mel.mel_scale.fb.clone())
        self.nf = self.n_fft // 2 + 1

    def forward(self, x):                                  # (B, N) -> (B, 1, 40, T) log-mel
        x = F.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        s = F.conv1d(x, self.basis, stride=self.hop)
        p = s[:, :self.nf] ** 2 + s[:, self.nf:] ** 2
        m = torch.matmul(p.transpose(1, 2), self.fb)       # (B, T, n_mels)
        return (m + 1e-6).log().permute(0, 2, 1).unsqueeze(1)


class MinMax(nn.Module):
    """Per-utterance min-max scaling to [-1, 1] of the data pipeline (train_kws.WaveSet)."""
    def forward(self, x):
        lo, hi = x.amin(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)
        return 2 * (x - lo) / (hi - lo + 1e-12) - 1


class FullGraph(nn.Module):
    """Raw 16 kHz waveform (B, N) -> logits, with the whole front end inside the graph.
    letr: min-max scaling + MFCC (GraphMFCC) + network. ssl: zero-mean/unit-variance (as train_kws.W2V2KWS with an
    all-ones mask; the min-max step is redundant after it) + encoder. Others: the model's own forward."""
    def __init__(self, model):
        super().__init__()
        self.kind = model_kind(model)
        self.model = model
        if self.kind == 'letr':
            self.minmax, self.front = MinMax(), GraphMFCC(model.mfcc)
        elif self.kind == 'bc':
            self.minmax, self.front = MinMax(), GraphLogMel(model.mel)

    def forward(self, x):
        if self.kind == 'letr':
            return self.model.model(self.front(self.minmax(x)))[0]
        if self.kind == 'bc':
            return self.model.model(self.front(self.minmax(x)))
        if self.kind == 'ssl':
            mu = x.mean(1, keepdim=True)
            sd = ((x - mu) ** 2).mean(1, keepdim=True).sqrt()
            return self.model.m(input_values=(x - mu) / (sd + 1e-7)).logits
        return self.model(x, torch.full((x.shape[0],), x.shape[1]))


class NetOnly(nn.Module):
    """Network on MFCC features (B, T, n_mfcc) -> logits (LeTR only), for front-end-excluded measurements."""
    def __init__(self, model):
        super().__init__(); self.net = model.model

    def forward(self, f):
        return self.net(f)[0]


def reference_input(model, seconds=1.5, seed=0):
    """A deterministic 16 kHz test waveform in [-1, 1] (random noise burst shaped like speech levels)."""
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    t = torch.arange(n) / SR
    x = 0.3 * torch.randn(1, n, generator=g) * (0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * t))[None] \
        + 0.3 * torch.sin(2 * torch.pi * 440 * t)[None]
    return x.clamp(-1, 1)
