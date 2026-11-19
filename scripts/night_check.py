#!/usr/bin/env python3
"""Real-night check on NPD (box annotations only, so box-level scoring).

NPD has no masks, so mask IoU is impossible. For every GT box on a dark test
image (mean luma < --max-luma) and every method:

  * prompt = one positive POINT at the box centre (primary). A box prompt would
    be circular here: SAM's mask is constrained to the prompt box, so its bbox
    would match the GT box almost by construction.
  * box_iou  = IoU(bbox of predicted mask, GT box)
  * fill     = predicted-mask pixels inside the GT box / GT box area
  * spill    = predicted-mask pixels outside the GT box / predicted-mask pixels

With --box-prompt the GT box is used as the prompt instead (secondary analysis;
then only `fill` is informative).

Images are 640x640 Roboflow "stretch" resizes; that is fine for relative
comparisons between methods, but say so in the report.

    python scripts/night_check.py --npd /path/to/NPD --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import METHODS, MethodBank, mean_luma, pick_device, sync  # noqa: E402
from potholemix import read_rgb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["image", "ann_id", "method", "prompt", "box_iou", "fill", "spill", "gt_box_area",
          "pred_area", "luma_in", "sec_enhance", "sec_segment", "device"]


def mask_bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def box_iou(a, b) -> float:
    if a is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def find_split(npd: Path, split: str):
    ann = npd / "annotations" / f"instances_{split}.json"
    if not ann.exists():
        hits = list(npd.rglob(f"*{split}*.json"))
        if not hits:
            raise FileNotFoundError(f"no COCO json for split {split} under {npd}")
        ann = hits[0]
    coco = json.loads(ann.read_text())
    img_dirs = [d for d in npd.rglob("*") if d.is_dir() and d.name == split] or [npd]
    return coco, img_dirs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npd", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-luma", type=float, default=0.35, help="keep only dark images")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--box-prompt", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "night_check.csv")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    coco, img_dirs = find_split(args.npd, args.split)
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images = [im for im in coco["images"] if im["id"] in by_img]
    if args.limit:
        images = images[: args.limit]

    def locate(fn: str) -> Path:
        for d in img_dirs:
            if (d / fn).exists():
                return d / fn
        raise FileNotFoundError(fn)

    device = pick_device(args.device)
    bank = MethodBank(args.methods, device)
    prompt = "box" if args.box_prompt else "center_point"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, im in enumerate(images):
            rgb = read_rgb(locate(im["file_name"]))
            L = mean_luma(rgb)
            if L >= args.max_luma:
                skipped += 1
                continue
            kept += 1
            for m in args.methods:
                enh, seg = bank.parts(m)
                sync(device); t0 = time.perf_counter()
                x = enh(rgb)
                sync(device); t1 = time.perf_counter()
                gts = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in by_img[im["id"]]]
                if args.box_prompt:
                    preds = seg.boxes(x, gts)
                else:
                    preds = [seg.point(x, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for b in gts]
                sync(device); t2 = time.perf_counter()
                for a, b, p in zip(by_img[im["id"]], gts, preds):
                    x1, y1, x2, y2 = (int(round(v)) for v in b)
                    area = max(1, (x2 - x1) * (y2 - y1))
                    inside = int(p[y1:y2, x1:x2].sum())
                    w.writerow({
                        "image": im["file_name"], "ann_id": a["id"], "method": m, "prompt": prompt,
                        "box_iou": f"{box_iou(mask_bbox(p), b):.5f}", "fill": f"{inside / area:.5f}",
                        "spill": f"{(p.sum() - inside) / max(1, p.sum()):.5f}", "gt_box_area": area,
                        "pred_area": int(p.sum()), "luma_in": f"{L:.5f}",
                        "sec_enhance": f"{t1 - t0:.4f}", "sec_segment": f"{t2 - t1:.4f}", "device": device,
                    })
            if (k + 1) % 20 == 0:
                print(f"[{k + 1}/{len(images)}] kept {kept}, skipped (bright) {skipped}")
    print(f"done: {kept} dark images scored, {skipped} bright skipped -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
