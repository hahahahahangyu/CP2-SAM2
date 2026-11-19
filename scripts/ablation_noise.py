import sys, warnings, dataclasses; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import numpy as np
from methods import MethodBank
from potholemix import load_samples, read_rgb
import darken as dk
cfg=dk.load_config(); cfg0=dataclasses.replace(cfg, sigma_s=0.0, sigma_c=0.0)
bank=MethodBank(["sam","robustsam"],"cuda"); out={}
for s in load_samples():
    day=read_rgb(s.image_path); boxes=[g.box for g in s.instances]
    for tag,c in (("severe+noise",cfg),("severe,no noise",cfg0)):
        img=dk.darken_uint8(day,"severe",c,s.name)
        for m in ("sam","robustsam"):
            e,seg=bank.parts(m)
            for g,p in zip(s.instances, seg.boxes(e(img),boxes)):
                out.setdefault((m,tag),[]).append((p&g.mask).sum()/max(1,(p|g.mask).sum()))
for k,v in sorted(out.items()): print(f"{k[0]:10s} {k[1]:16s} mean IoU {np.mean(v):.3f}  (n={len(v)})")
