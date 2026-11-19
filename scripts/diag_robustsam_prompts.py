import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import numpy as np, torch
from methods import RobustSAM, UltralyticsSAM
from potholemix import load_samples, read_rgb
import darken as dk
dev="cuda"; R=RobustSAM(dev); S=UltralyticsSAM(dev); cfg=dk.load_config()
def rs(rgb, mm, **pr):
    t=torch; img=t.from_numpy(np.ascontiguousarray(rgb)).to(dev).permute(2,0,1).unsqueeze(0); size=img.shape[-2:]
    d={"image":R.tf.apply_image_torch(img.float()),"original_size":size}
    if "box" in pr: d["boxes"]=R.tf.apply_boxes_torch(t.tensor([pr["box"]],dtype=t.float32,device=dev),size).unsqueeze(0)
    else:
        P=t.tensor(pr["pts"],dtype=t.float32,device=dev); d["point_coords"]=R.tf.apply_coords_torch(P,size).unsqueeze(0); d["point_labels"]=t.ones((1,len(pr["pts"])),device=dev)
    with t.no_grad(): o=R.model.predict(R.opt,[d],multimask_output=mm)[0]
    k=int(o["iou_predictions"][0].argmax()) if mm else 0
    return o["masks"][0,k].cpu().numpy(), float(o["low_res_logits"][0,k].max())
iou=lambda m,g:(m&g).sum()/max(1,(m|g).sum())
out={}
for s in load_samples(limit=60):
    day=read_rgb(s.image_path)
    for lv in ("day","severe"):
        img=day if lv=="day" else dk.darken_uint8(day,lv,cfg,s.name)
        for g in s.instances:
            ys,xs=np.nonzero(g.mask); c=(float(np.median(xs)),float(np.median(ys)))
            for name,kw,mm in [("box",dict(box=list(map(float,g.box))),False),("box_multi",dict(box=list(map(float,g.box))),True),
                               ("pt1",dict(pts=[c]),False),("pt1_multi",dict(pts=[c]),True)]:
                m,mx=rs(img,mm,**kw); out.setdefault((lv,name),[]).append((iou(m,g.mask), m.sum()==0, mx))
            out.setdefault((lv,"sam_pt1"),[]).append((iou(S.point(img,c),g.mask),False,0))
for k,v in sorted(out.items()):
    a=np.array(v,dtype=float); print(f"{k[0]:7s} {k[1]:10s} IoU {a[:,0].mean():.3f}  empty {a[:,1].mean():.2f}  med max-logit {np.median(a[:,2]):.2f}")
