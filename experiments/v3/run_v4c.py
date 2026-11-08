"""LR-grid extension for BC-ResNet (1e-2; the best LR was the edge of the 3e-4..3e-3 grid), scratch and MSWC initialisations only."""
import run_v4 as q
for m in ('bcresnet1', 'bcresnet8'):
    for init in ('scratch', 'mswc'): q.sweep(m, init)
print('V4C DONE', flush=True)
