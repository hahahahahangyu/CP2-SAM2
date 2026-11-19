#!/usr/bin/env python3
"""Methods x darkness benchmark on the Pothole Mix test split.

For every image, level (day + configs/levels.json) and method, the image is
darkened (deterministically), enhanced, and segmented with one box prompt per
GT pothole. Each pothole is scored by IoU/Dice against its daylight mask.

Output: one CSV, one row per (image, pothole, level, method):
    results/benchmark.csv

The run is resumable: rows already in the CSV are skipped, so a cloud session
that stops half-way can be restarted with the same command.

Examples
--------
    python scripts/benchmark.py                          # everything
    python scripts/benchmark.py --methods sam clahe gamma --levels day medium
    python scripts/benchmark.py --limit 5 --out results/smoke.csv   # smoke test

Sanity check: method=sam, level=day must reproduce the baseline (mean IoU 0.653).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darken as dk  # noqa: E402
from methods import METHODS, MethodBank, mean_luma, pick_device, sync  # noqa: E402
from potholemix import MANIFEST, load_samples, read_rgb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["image", "source", "pothole_id", "level", "method",
          "iou", "dice", "gt_area", "pred_area", "img_area",
          "box_x1", "box_y1", "box_x2", "box_y2",
          "luma_in", "luma_enh", "sec_enhance", "sec_segment", "n_in_image", "device"]


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return (inter / union if union else 0.0), (2 * inter / s if s else 0.0)


def overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, box) -> np.ndarray:
    out = rgb.copy()
    out[pred] = (0.5 * out[pred] + 0.5 * np.array([0, 120, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 0, 0), 2)
    cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), (255, 255, 0), 2)
    return out


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["image"], r["level"], r["method"]) for r in csv.DictReader(f)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--levels", nargs="+", help="default: day + all levels in the config")
    ap.add_argument("--config", type=Path, default=dk.DEFAULT_CONFIG)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--root", type=Path, help="Pothole Mix root (or env POTHOLEMIX_ROOT)")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "benchmark.csv")
    ap.add_argument("--overlays", type=int, default=0, help="save overlays for the first N images")
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = dk.load_config(args.config)
    levels = args.levels or ["day", *cfg.levels]
    for lv in levels:
        if lv != "day" and lv not in cfg.levels:
            ap.error(f"unknown level {lv}")
    device = pick_device(args.device)
    samples = load_samples(args.manifest, args.root, args.limit)
    print(f"{len(samples)} images, {sum(len(s.instances) for s in samples)} potholes, "
          f"levels {levels}, methods {args.methods}, device {device}")

    bank = MethodBank(args.methods, device)
    done = done_keys(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    fh = open(args.out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    t0 = time.time()
    for i, s in enumerate(samples):
        day = read_rgb(s.image_path)
        H, W = day.shape[:2]
        gts = s.instances
        for inst in gts:
            if inst.mask.shape != (H, W):
                raise ValueError(f"{s.name}: mask {inst.mask.shape} != image {(H, W)}")
        boxes = [g.box for g in gts]
        for lv in levels:
            img = day if lv == "day" else dk.darken_uint8(day, lv, cfg, s.name)
            luma_in = mean_luma(img)
            for m in args.methods:
                if (s.name, lv, m) in done or not gts:
                    continue
                enh, seg = bank.parts(m)
                sync(device); a = time.perf_counter()
                x = enh(img)
                sync(device); b = time.perf_counter()
                preds = seg.boxes(x, boxes)
                sync(device); c = time.perf_counter()
                for g, p in zip(gts, preds):
                    if p.shape != (H, W):
                        p = cv2.resize(p.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    iou, dice = iou_dice(p, g.mask)
                    writer.writerow({
                        "image": s.name, "source": s.source, "pothole_id": g.pothole_id,
                        "level": lv, "method": m, "iou": f"{iou:.5f}", "dice": f"{dice:.5f}",
                        "gt_area": g.area, "pred_area": int(p.sum()), "img_area": H * W,
                        "box_x1": g.box[0], "box_y1": g.box[1], "box_x2": g.box[2], "box_y2": g.box[3],
                        "luma_in": f"{luma_in:.5f}", "luma_enh": f"{mean_luma(x):.5f}",
                        "sec_enhance": f"{b - a:.4f}", "sec_segment": f"{c - b:.4f}",
                        "n_in_image": len(gts), "device": device,
                    })
                    if i < args.overlays:
                        d = args.out.parent / "overlays" / lv / m
                        d.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(d / f"{Path(s.name).stem}_p{g.pothole_id}.jpg"),
                                    cv2.cvtColor(overlay(x, g.mask, p, g.box), cv2.COLOR_RGB2BGR))
                fh.flush()
        el = time.time() - t0
        print(f"[{i + 1}/{len(samples)}] {s.name}  {el / 60:.1f} min, eta {el / (i + 1) * (len(samples) - i - 1) / 60:.1f} min")
    fh.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
