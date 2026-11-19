"""Pothole Mix (SHREC 2022) test split: images, GT masks and box prompts.

Rules (same as the daylight baseline, scripts/sam_baseline.py):
  * pothole pixels are red (255, 0, 0) in the mask; cracks (green) are ignored
  * every connected pothole region of >= MIN_AREA px is one instance and one box prompt
  * the test list is data/manifests/potholemix_test.csv (image, source, n_potholes),
    with names like <source>__<stem>.jpg -> <root>/testing/<source>/{images,masks}/<stem>.*

`load_binary_mask`, `DEFAULT_ROOT`, `find_pair` and the blob/box rule are copied
from capstone-sample (scripts/prepare_potholemix.py, scripts/sam_baseline.py), so
the ground truth and box prompts are exactly the daylight baseline's.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifests" / "potholemix_test.csv"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_AREA = 100            # same blob filter as the daylight baseline
EXPECTED_POTHOLES = 206   # baseline: 186 images, 206 potholes

# ---- copied verbatim from capstone-sample/scripts/prepare_potholemix.py ----
DEFAULT_ROOT = (Path.home() / "Documents" / "fyp datasets" / "SHREC2022"
                / "Pothole-Mix" / "pothole-mix")


def load_binary_mask(path, mask_color):
    mask = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if mask is None:
        return None
    r, g, b = (int(v) for v in mask_color.split(","))
    # Small tolerance in case a mask was saved with lossy compression.
    target = np.array([b, g, r], dtype=np.int16)
    close = np.abs(mask.astype(np.int16) - target).max(axis=2) <= 40
    return close.astype(np.uint8) * 255
# ----------------------------------------------------------------------------


@dataclass
class Instance:
    pothole_id: int
    box: list[int]          # x1, y1, x2, y2 (inclusive-exclusive)
    mask: np.ndarray        # bool HxW
    area: int


@dataclass
class Sample:
    name: str               # manifest name, e.g. <source>__<stem>.jpg
    source: str
    image_path: Path
    mask_path: Path
    instances: list[Instance]


def root_dir() -> Path:
    return Path(os.environ.get("POTHOLEMIX_ROOT", DEFAULT_ROOT)).expanduser()


def find_pair(root: Path, name: str):
    """Manifest name '<source>__<stem>.jpg' -> raw (image, mask) paths (as in sam_baseline.py)."""
    source, stem = name.split("__", 1)
    stem = Path(stem).stem
    folder = root / "testing" / source
    image = next((folder / "images" / (stem + e) for e in IMAGE_EXTS
                  if (folder / "images" / (stem + e)).is_file()), None)
    mask = next((p for p in (folder / "masks").glob(stem + ".*")), None)
    return image, mask


def instances_from_mask(mask: np.ndarray) -> list[Instance]:
    """One instance per connected blob >= MIN_AREA px, as in sam_baseline.potholes()."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= MIN_AREA:
            # inclusive x2/y2, exactly as the baseline built its box prompts
            out.append(Instance(len(out), [int(x), int(y), int(x + w - 1), int(y + h - 1)], lab == k, int(area)))
    return out


def load_samples(manifest: Path = MANIFEST, root: Path | None = None, limit: int | None = None) -> list[Sample]:
    root = Path(root or root_dir()).expanduser()
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["n_potholes"]) > 0]
    if limit:
        rows = rows[:limit]
    samples = []
    for r in rows:
        img_p, mask_p = find_pair(root, r["image"])
        if img_p is None or mask_p is None:
            print(f"Missing raw file for {r['image']}; skipping.", file=sys.stderr)
            continue
        mask = load_binary_mask(mask_p, "255,0,0")
        full = cv2.imread(str(img_p)).shape[:2]
        if mask.shape != full:
            mask = cv2.resize(mask, (full[1], full[0]), interpolation=cv2.INTER_NEAREST)
        inst = instances_from_mask(mask)
        if inst:
            samples.append(Sample(r["image"], r["source"], img_p, mask_p, inst))
    n = sum(len(s.instances) for s in samples)
    if not limit and (len(samples), n) != (186, EXPECTED_POTHOLES):
        print(f"WARNING: {len(samples)} images / {n} potholes; baseline had 186 / {EXPECTED_POTHOLES}.",
              file=sys.stderr)
    return samples


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
