"""Installation check: load the released model and score a synthetic pileup.

    python -m rawmod.selftest            (or: rawmod-selftest)

Loads checkpoints/rawmod.pt, builds a batch of two random pileups of the shape the
model expects (channels x rows x window samples = 11 x 16 x 210), runs them through
the network and prints the two scores. It exercises PyTorch, the model code and the
checkpoint; it does not need pod5 files, Dorado or a GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / 'checkpoints' / 'rawmod.pt'
LOADER = REPO / 'scripts' / 'evaluate' / 'score_genome.py'
C, H, W = 11, 16, 210          # input channels, rows (reference + 15 reads), window samples (21 bases x 10)


def main() -> int:
    import numpy as np
    import torch
    for f in (CHECKPOINT, LOADER):
        if not f.exists():
            print(f'missing {f}', file=sys.stderr); return 1
    spec = importlib.util.spec_from_file_location('score_genome', LOADER)
    sgm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sgm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, arch = sgm.load_model(CHECKPOINT, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((2, C, H, W)).astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x)
        logit = out[0] if isinstance(out, (tuple, list)) else out
        score = torch.sigmoid(logit.reshape(-1)).cpu().numpy()
    print(f'rawmod self-test: torch {torch.__version__}, device {device.type}, model {arch}, {n_params:,} trainable parameters')
    print(f'scores of two random pileups: {score[0]:.3f} {score[1]:.3f} (any value in [0, 1] is fine)')
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
