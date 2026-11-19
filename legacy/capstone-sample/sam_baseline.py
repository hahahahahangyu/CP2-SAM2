"""SAM on Pothole Mix test potholes, over methods x darkness levels.

Each pothole in the official hand-drawn mask becomes one box prompt. Every
(level, method) gets the same boxes and is scored with IoU against the daylight
mask (darkening never moves pixels). The dark image for an (image, level) is
made once and shared by all methods, so the comparison is paired.

Methods:
  sam        plain SAM
  clahe      CLAHE, then SAM
  gamma      adaptive gamma, then SAM
  zerodce    Zero-DCE, then SAM          (needs --zerodce-weights)
  robustsam  RobustSAM, no enhancement   (needs --robustsam-repo/-ckpt; CUDA)

    python scripts/sam_baseline.py --limit 10                          # daylight SAM (old baseline)
    python scripts/sam_baseline.py --levels day,mild,medium,severe \\
        --methods sam,clahe,gamma,zerodce --out runs/bench_sam
    python scripts/sam_baseline.py --levels day,mild,medium,severe \\
        --methods robustsam --out runs/bench_robustsam --resume

Results go to <out>/results.csv, one row per (image, pothole, level, method);
analyse with scripts/analyze_benchmark.py. --resume skips finished
(image, level, method) triples, so a killed HPC job can just be resubmitted.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from scripts.prepare_potholemix import DEFAULT_ROOT, load_binary_mask  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_AREA = 100  # same blob filter used when the YOLO labels were made
SIZE_BINS = [(0.002, "tiny"), (0.05, "small"), (0.2, "medium"), (np.inf, "large")]
METHODS = ("sam", "clahe", "gamma", "zerodce", "robustsam")
FIELDS = ["image", "pothole", "area_frac", "size", "level", "method", "iou",
          "pred_area_frac", "input_brightness", "seconds"]


def default_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--manifest", type=Path,
                   default=BASE / "data/manifests/potholemix_test.csv")
    p.add_argument("--levels", default="day", help="comma list from darken.LEVELS")
    p.add_argument("--methods", default="sam", help=f"comma list from {METHODS}")
    p.add_argument("--headlight", action="store_true", help="add the headlight cone")
    p.add_argument("--model", default="sam_b.pt",
                   help="ultralytics SAM weights; ViT-B matches RobustSAM-B")
    p.add_argument("--zerodce-weights", type=Path,
                   default=BASE / "Zero-DCE_code/snapshots/Epoch99.pth")
    p.add_argument("--robustsam-repo", type=Path, default=BASE / "third_party/RobustSAM")
    p.add_argument("--robustsam-ckpt", type=Path,
                   default=BASE / "weights/robustsam_checkpoint_b.pth")
    p.add_argument("--device", default=default_device())
    p.add_argument("--limit", type=int, default=0, help="0 = all pothole images")
    p.add_argument("--out", type=Path, default=BASE / "runs/sam_baseline")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overlays", type=int, default=8,
                   help="overlays saved per (level, method)")
    return p.parse_args()


def find_pair(root, name):
    """Manifest name '<source>__<stem>.jpg' -> raw (image, mask) paths."""
    source, stem = name.split("__", 1)
    stem = Path(stem).stem
    folder = root / "testing" / source
    image = next((folder / "images" / (stem + e) for e in IMAGE_EXTS
                  if (folder / "images" / (stem + e)).is_file()), None)
    mask = next((p for p in (folder / "masks").glob(stem + ".*")), None)
    return image, mask


def potholes(mask):
    """Split the binary mask into one boolean mask + box per pothole."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= MIN_AREA:
            out.append((labels == i, [int(x), int(y), int(x + w - 1), int(y + h - 1)]))
    return out


def size_bin(area_frac):
    return next(name for limit, name in SIZE_BINS if area_frac < limit)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def overlay(image, gt, pred, box):
    vis = image.copy()
    vis[gt & ~pred] = (0.5 * vis[gt & ~pred] + [0, 0, 127]).astype(np.uint8)    # missed: red
    vis[pred & ~gt] = (0.5 * vis[pred & ~gt] + [127, 0, 0]).astype(np.uint8)    # extra: blue
    vis[gt & pred] = (0.5 * vis[gt & pred] + [0, 127, 0]).astype(np.uint8)      # agree: green
    cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
    return vis


# --- segmenters: (BGR uint8 image, list of xyxy boxes) -> (n, H, W) bool -----

class UltralyticsSAM:
    def __init__(self, weights, device):
        from ultralytics import SAM
        self.model, self.device = SAM(weights), device

    def __call__(self, image, boxes):
        r = self.model(image, bboxes=boxes, device=self.device, verbose=False)[0]
        return r.masks.data.cpu().numpy() > 0.5


class RobustSAM:
    """Official RobustSAM repo (github.com/robustsam/RobustSAM), box prompts.

    Mirrors the repo's eval.py. ** The repo's API is not vendored here: run
    `--methods robustsam --limit 2` on the HPC first and check the IoU is sane. **
    """

    def __init__(self, repo, ckpt, device, model_size="b"):
        import torch
        sys.path.insert(0, str(repo))
        from robust_segment_anything import sam_model_registry
        from robust_segment_anything.utils.transforms import ResizeLongestSide
        self.torch, self.device = torch, device
        self.model = sam_model_registry[f"vit_{model_size}"](opt=None, checkpoint=str(ckpt))
        self.model = self.model.to(device).eval()
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def __call__(self, image, boxes):
        t = self.torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = t.as_tensor(rgb, device=self.device).permute(2, 0, 1)[None].float()
        box = t.as_tensor(boxes, dtype=t.float, device=self.device)
        data = {"image": self.transform.apply_image_torch(img),
                "boxes": self.transform.apply_boxes_torch(box, rgb.shape[:2]),
                "original_size": rgb.shape[:2]}
        with t.no_grad():
            out = self.model.predict(None, [data], multimask_output=False,
                                     return_logits=False)[0]["masks"]
        return out[:, 0].cpu().numpy() > 0


def build(methods, args):
    from scripts import enhance
    needs_sam = any(m != "robustsam" for m in methods)
    sam = UltralyticsSAM(args.model, args.device) if needs_sam else None
    pre = {"sam": None, "clahe": enhance.clahe, "gamma": enhance.gamma}
    if "zerodce" in methods:
        pre["zerodce"] = enhance.ZeroDCE(args.zerodce_weights, args.device)
    out = {m: (pre[m], sam) for m in methods if m != "robustsam"}
    if "robustsam" in methods:
        out["robustsam"] = (None, RobustSAM(args.robustsam_repo, args.robustsam_ckpt,
                                            args.device))
    return out


def load_done(path):
    if not path.is_file():
        return set()
    with open(path) as f:
        return {(r["image"], r["level"], r["method"]) for r in csv.DictReader(f)}


def main():
    args = parse_args()
    from scripts.darken import LEVELS, brightness, darken

    levels = args.levels.split(",")
    methods = args.methods.split(",")
    bad = [x for x in levels if x not in LEVELS] + [m for m in methods if m not in METHODS]
    if bad:
        sys.exit(f"Unknown level/method: {bad}")

    with open(args.manifest) as f:
        names = [r["image"] for r in csv.DictReader(f) if int(r["n_potholes"]) > 0]
    if args.limit:
        names = names[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"
    done = load_done(csv_path) if args.resume else set()
    fresh = not (args.resume and csv_path.is_file())
    out_f = open(csv_path, "w" if fresh else "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()

    pipes = build(methods, args)
    saved = {}
    tag = "+headlight" if args.headlight else ""

    for i, name in enumerate(names, 1):
        todo = [(lv, m) for lv in levels for m in methods if (name, lv + tag, m) not in done]
        if not todo:
            continue
        image_path, mask_path = find_pair(args.root, name)
        if image_path is None or mask_path is None:
            print(f"Missing raw file for {name}; skipping.")
            continue
        image = cv2.imread(str(image_path))
        gt_all = load_binary_mask(mask_path, "255,0,0")
        if gt_all.shape != image.shape[:2]:
            gt_all = cv2.resize(gt_all, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        items = potholes(gt_all)
        if not items:
            continue
        boxes = [b for _, b in items]

        dark = {}
        for lv, m in todo:
            if lv not in dark:
                dark[lv] = darken(image, lv, name, args.headlight)
            enhancer, segment = pipes[m]
            start = time.perf_counter()
            inp = enhancer(dark[lv]) if enhancer else dark[lv]
            preds = segment(inp, boxes)
            secs = time.perf_counter() - start
            for k, (gt, box) in enumerate(items):
                pred = preds[k]
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
                af = gt.sum() / gt.size
                writer.writerow({"image": name, "pothole": k, "area_frac": round(af, 5),
                                 "size": size_bin(af), "level": lv + tag, "method": m,
                                 "iou": round(iou(gt, pred), 4),
                                 "pred_area_frac": round(pred.sum() / pred.size, 5),
                                 "input_brightness": round(brightness(inp), 1),
                                 "seconds": round(secs, 3)})
                key = (lv, m)
                if saved.get(key, 0) < args.overlays:
                    d = args.out / "overlays" / f"{lv}{tag}_{m}"
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{Path(name).stem}_{k}.jpg"),
                                overlay(inp, gt, pred, box))
                    saved[key] = saved.get(key, 0) + 1
        out_f.flush()
        if i % 10 == 0 or i == len(names):
            print(f"[{i}/{len(names)}] {name}", flush=True)
    out_f.close()
    summarise(csv_path)


def summarise(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No potholes scored.")
        return
    conds = sorted({(r["level"], r["method"]) for r in rows})
    print(f"\n{'level':16s}{'method':11s}{'n':>5s}{'meanIoU':>9s}{'median':>8s}{'s/img':>7s}")
    for lv, m in conds:
        sub = [r for r in rows if r["level"] == lv and r["method"] == m]
        ious = np.array([float(r["iou"]) for r in sub])
        secs = list({r["image"]: float(r["seconds"]) for r in sub}.values())
        print(f"{lv:16s}{m:11s}{len(ious):5d}{ious.mean():9.3f}{np.median(ious):8.3f}"
              f"{np.mean(secs[1:] or secs):7.2f}")
    print(f"\nPer-pothole rows in {csv_path}; CIs and paired tests: "
          f"python scripts/analyze_benchmark.py {csv_path}")


if __name__ == "__main__":
    main()
