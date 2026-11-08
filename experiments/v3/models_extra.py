"""Additional models for the v4 experiments (additive: nothing in train_kws.py changes behaviour for existing runs).

BCResNet : broadcasted residual network (Kim et al., Interspeech 2021, "Broadcasted Residual Learning for Efficient Keyword
           Spotting"), official layout: 40-bin log-mel input (30 ms window, 10 ms hop), 5x5 stride-(2,1) head conv, four
           stages n=(2,2,4,4) of BC-ResBlocks with c=(8,12,16,20)*tau/1, frequency strides (1,2,2,1), temporal dilations
           (1,2,4,8), sub-spectral normalisation (S=5), depth-wise 5x5 conv, 1x1 conv to 4*base channels, global average
           pool, 1x1 classifier. BC-ResNet-tau uses base_c = 8 * tau (tau = 1: 9.2k parameters, tau = 8: 321k).
BCResNetKWS : waveform -> log-mel -> BCResNet, same interface as LeTRKWS / W2V2KWS (forward(x, lens, aug_p)).
load_pretrained : load a pre-trained checkpoint, skipping the classification head (any key containing 'head').
"""
import os, sys, torch, torch.nn as nn, torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
BC = {'bcresnet1': 8, 'bcresnet8': 64}                       # base channel count = 8 * tau
PRETRAINED_DIR = os.path.join(HERE, 'pretrained')


def pretrained_path(model, init):
    return os.path.join(PRETRAINED_DIR, f'{model}-{init}.pt')


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super().__init__()
        self.S = S
        self.bn = nn.BatchNorm2d(C * S, eps=eps)

    def forward(self, x):                                        # x: (N, C, F, T)
        N, C, F, T = x.size()
        return self.bn(x.reshape(N, C * self.S, F // self.S, T)).reshape(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(self, planes, dilation=1, stride=1, temp_pad=(0, 1)):
        super().__init__()
        self.freq_dw_conv = nn.Conv2d(planes, planes, (3, 1), padding=(1, 0), groups=planes, stride=(stride, 1), bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, (1, 3), padding=temp_pad, groups=planes, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, 1, bias=False)

    def forward(self, x):
        out = self.ssn1(self.freq_dw_conv(x))                    # f2 (frequency-wise, 2D features)
        aux = out
        out = out.mean(2, keepdim=True)                          # frequency average pooling -> (N, C, 1, T)
        out = self.channel_drop(self.conv1x1(self.swish(self.bn(self.temp_dw_conv(out)))))   # f1 (temporal, 1D)
        return self.relu(out + x + aux)                          # broadcast over frequency, identity + auxiliary


class TransitionBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride=1, temp_pad=(0, 1)):
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.freq_dw_conv = nn.Conv2d(planes, planes, (3, 1), padding=(1, 0), groups=planes, stride=(stride, 1), bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, (1, 3), padding=temp_pad, groups=planes, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.swish = nn.SiLU()
        self.conv1x1_2 = nn.Conv2d(planes, planes, 1, bias=False)
        self.channel_drop = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out = self.ssn(self.freq_dw_conv(self.relu(self.bn1(self.conv1x1_1(x)))))
        aux = out
        out = out.mean(2, keepdim=True)
        out = self.channel_drop(self.conv1x1_2(self.swish(self.bn2(self.temp_dw_conv(out)))))
        return self.relu(aux + out)                              # channel change: no identity path


class BCResNet(nn.Module):
    def __init__(self, base_c, n_classes):
        super().__init__()
        c = [base_c * 2, base_c, int(base_c * 1.5), base_c * 2, int(base_c * 2.5), base_c * 4]
        n, strides = [2, 2, 4, 4], [1, 2, 2, 1]
        self.head = nn.Sequential(nn.Conv2d(1, c[0], 5, (2, 1), 2, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(True))
        blocks = []
        for i, reps in enumerate(n):
            d = 2 ** i
            for j in range(reps):
                if j == 0: blocks.append(TransitionBlock(c[i], c[i + 1], d, strides[i], (0, d)))
                else: blocks.append(BroadcastedBlock(c[i + 1], d, 1, (0, d)))
        self.blocks = nn.Sequential(*blocks)
        self.tail = nn.Sequential(nn.Conv2d(c[-2], c[-2], (5, 5), padding=(0, 2), groups=c[-2], bias=False),
                                  nn.Conv2d(c[-2], c[-1], 1, bias=False), nn.BatchNorm2d(c[-1]), nn.ReLU(True),
                                  nn.AdaptiveAvgPool2d(1))
        self.classifier_head = nn.Conv2d(c[-1], n_classes, 1)

    def forward(self, x):                                        # x: (N, 1, 40, T)
        return self.classifier_head(self.tail(self.blocks(self.head(x)))).flatten(1)


class BCResNetKWS(nn.Module):
    def __init__(self, variant, n_classes, n_mels=40):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=480, win_length=480, hop_length=160, f_min=20, f_max=7600,
                                                        n_mels=n_mels)
        self.model = BCResNet(BC[variant], n_classes)

    def features(self, x):                                       # (B, N) -> (B, T, 40) log-mel
        with torch.autocast(device_type='cuda', enabled=False):
            return (self.mel(x.float()) + 1e-6).log().permute(0, 2, 1)

    def forward(self, x, lens=None, aug_p=0.0):
        f = self.features(x)
        if self.training and aug_p > 0:
            from train_kws import spec_augment
            f = spec_augment(f, aug_p)
        return self.model(f.permute(0, 2, 1).unsqueeze(1))


def load_pretrained(model, path):
    """Load matching tensors of a checkpoint saved by pretrain.py; the classification head is re-initialised."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and 'head' not in k}
    missing = [k for k in own if k not in keep]
    assert all('head' in k for k in missing), missing
    model.load_state_dict(keep, strict=False)
    return len(keep)


if __name__ == '__main__':                                       # parameter counts (paper: 12 classes, 9.2k and 321k)
    for v in BC:
        m = BCResNetKWS(v, 12)
        print(v, sum(p.numel() for p in m.model.parameters()), 'params (12 classes);',
              sum(p.numel() for p in BCResNetKWS(v, 30).model.parameters()), '(30 classes)')
        print(m.train()(torch.randn(4, 16000)).shape, m.eval()(torch.randn(2, 16000 * 3)).shape)
