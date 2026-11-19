#!/usr/bin/env python3
"""Load every enhancer/segmenter and run each once on a synthetic image."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from methods import METHODS, MethodBank, pick_device  # noqa: E402

dev = pick_device()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {dev}"
      + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
rng = np.random.default_rng(0)
img = (rng.random((480, 640, 3)) * 60).astype(np.uint8)
img[180:300, 250:400] = 20
bank = MethodBank(list(METHODS), dev)
for m in METHODS:
    enh, seg = bank.parts(m)
    masks = seg.boxes(enh(img), [[240, 170, 410, 310]])
    pt = seg.point(enh(img), (325, 240))
    print(f"  {m:<10} ok  box-mask px={int(masks[0].sum())}  point-mask px={int(pt.sum())}")
print("environment OK")
