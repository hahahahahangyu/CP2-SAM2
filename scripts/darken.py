#!/usr/bin/env python3
"""Synthesise low-light versions of daylight pothole images.

Darkening: Lv, Li & Lu (arXiv 1908.00682), Eq. 1, per RGB channel, I in [0, 1]:

    I_dark = beta * (alpha * I) ** gamma

Noise: Lv et al. Eq. 2, which uses the CBDNet setup (Guo et al., CVPR 2019,
arXiv 1807.04686). Noise is added in linear irradiance, then the camera
response is re-applied:

    L      = f^-1(I_dark)                      # undo the camera response
    L_n    = L + N(0, L * sigma_s^2 + sigma_c^2)  # heteroscedastic Gaussian
                                                 #   = Poisson-Gaussian approx.
    I_out  = f(L_n)

  * CBDNet samples sigma_s ~ U[0, 0.16] and sigma_c ~ U[0, 0.06]. We FIX one
    pair for all levels (one simulated camera; darker scenes then get noisier
    relative to signal, as on a real sensor).
  * Simplifications vs. Eq. 2 (state them in the report): f is the sRGB
    transfer curve rather than a curve sampled from the DoRF database, and
    there is no Bayer mosaic/demosaic (M) or JPEG step.

Levels are read from configs/levels.json (fixed values, no random sampling), and
noise is seeded per (image, level), so outputs are bit-identical on re-runs.
Masks are untouched: darkening is photometric, so daylight masks remain valid.

Usage
-----
  # write darkened copies (for figures / inspection; benchmark.py darkens on the fly)
  python scripts/darken.py write --in <images> --out data/dark

  # luminance stats of any folder (e.g. NPD night images)
  python scripts/darken.py luma --in <images>

  # fit each level's gamma so mean luma on Pothole Mix matches NPD's dark subset
  python scripts/darken.py calibrate --in <potholemix test images> \
      --targets mild=0.21 medium=0.146 severe=0.081 --save
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "levels.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LV_RANGES = {"alpha": (0.9, 1.0), "beta": (0.5, 1.0), "gamma": (1.5, 5.0)}


@dataclass(frozen=True)
class Level:
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class Config:
    levels: dict[str, Level]
    sigma_s: float
    sigma_c: float
    seed: int


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    raw = json.loads(Path(path).read_text())
    return Config(
        levels={k: Level(**v) for k, v in raw["levels"].items()},
        sigma_s=raw["noise"]["sigma_s"],
        sigma_c=raw["noise"]["sigma_c"],
        seed=raw.get("seed", 0),
    )


def save_config(cfg: Config, path: Path = DEFAULT_CONFIG) -> None:
    raw = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    raw["levels"] = {k: asdict(v) for k, v in cfg.levels.items()}
    raw["noise"] = {"sigma_s": cfg.sigma_s, "sigma_c": cfg.sigma_c}
    raw["seed"] = cfg.seed
    Path(path).write_text(json.dumps(raw, indent=2) + "\n")


# ---------------------------------------------------------------- core maths
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def luma(img01: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an HxWx3 image in [0, 1] (same measure for NPD and synthetic)."""
    return img01 @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def darken_clean(img01: np.ndarray, lv: Level) -> np.ndarray:
    """Eq. 1 only (no noise)."""
    return (lv.beta * np.power(lv.alpha * img01, lv.gamma)).astype(np.float32)


def add_noise(img01: np.ndarray, sigma_s: float, sigma_c: float, rng: np.random.Generator) -> np.ndarray:
    lin = srgb_to_linear(img01)
    std = np.sqrt(lin * sigma_s**2 + sigma_c**2)
    lin = lin + rng.standard_normal(lin.shape, dtype=np.float32) * std
    return linear_to_srgb(lin).astype(np.float32)


def seed_for(key: str, level: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{level}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def darken(img01: np.ndarray, level: str, cfg: Config, key: str) -> np.ndarray:
    """Full pipeline: Eq. 1 + CBDNet noise, deterministic for (key, level)."""
    rng = np.random.default_rng(seed_for(key, level, cfg.seed))
    out = darken_clean(img01, cfg.levels[level])
    if cfg.sigma_s > 0 or cfg.sigma_c > 0:
        out = add_noise(out, cfg.sigma_s, cfg.sigma_c, rng)
    return np.clip(out, 0.0, 1.0)


def darken_uint8(rgb: np.ndarray, level: str, cfg: Config, key: str) -> np.ndarray:
    out = darken(rgb.astype(np.float32) / 255.0, level, cfg, key)
    return np.round(out * 255.0).astype(np.uint8)


# ---------------------------------------------------------------- io helpers
def load_rgb01(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def luma_stats(values: np.ndarray) -> str:
    return (f"n={len(values)}  mean {values.mean():.4f}  median {np.median(values):.4f}  "
            f"p10 {np.percentile(values, 10):.4f}  p90 {np.percentile(values, 90):.4f}")


# ---------------------------------------------------------------- commands
def cmd_write(args, cfg: Config) -> int:
    images = list_images(args.inp)
    levels = args.levels or list(cfg.levels)
    rows = []
    for i, path in enumerate(images, 1):
        rel = path.relative_to(args.inp)
        img = load_rgb01(path)
        for lv in levels:
            out = darken(img, lv, cfg, rel.as_posix())
            dst = (args.out / lv / rel).with_suffix(".png")  # PNG: keep the noise exact
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(out * 255).astype(np.uint8)).save(dst)
            rows.append({"image": rel.as_posix(), "level": lv,
                         "luma_day": round(float(luma(img).mean()), 5),
                         "luma_dark": round(float(luma(out).mean()), 5)})
        if i % 25 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {rel}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "darken_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for lv in levels:
        print(f"{lv:>7}: " + luma_stats(np.array([r["luma_dark"] for r in rows if r["level"] == lv])))
    return 0


def cmd_luma(args, cfg: Config) -> int:
    images = list_images(args.inp)
    vals = np.array([luma(load_rgb01(p)).mean() for p in images])
    print(luma_stats(vals))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "mean_luma"])
            for p, v in zip(images, vals):
                w.writerow([p.relative_to(args.inp).as_posix(), f"{v:.5f}"])
    return 0


def cmd_calibrate(args, cfg: Config) -> int:
    """Fit gamma per level (bisection; larger gamma -> darker), noise included.

    Targets come as level=luma pairs, e.g. mild=0.21 medium=0.146 severe=0.08
    (NPD dark-subset p75 / median / p25). alpha and beta stay as configured.
    """
    targets = dict(t.split("=") for t in args.targets)
    images = list_images(args.inp)
    if args.sample and len(images) > args.sample:
        images = list(np.random.default_rng(0).choice(images, args.sample, replace=False))
    imgs = [load_rgb01(p) for p in images]
    print(f"Pothole Mix sample: {len(imgs)} images, daylight " + luma_stats(np.array([luma(i).mean() for i in imgs])))

    levels = dict(cfg.levels)
    ok = True
    for name, tgt in targets.items():
        tgt = float(tgt)
        base = cfg.levels[name]

        def mean_luma(gamma: float) -> float:
            c = Config({name: Level(base.alpha, base.beta, gamma)}, cfg.sigma_s, cfg.sigma_c, cfg.seed)
            return float(np.mean([luma(darken(im, name, c, str(i))).mean() for i, im in enumerate(imgs)]))

        lo, hi = 1.0, 8.0
        if not (mean_luma(hi) <= tgt <= mean_luma(lo)):
            print(f"{name}: target {tgt} not reachable with gamma in [{lo}, {hi}] (beta={base.beta}); "
                  f"lower beta or the noise floor", file=sys.stderr)
            ok = False
            continue
        for _ in range(25):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_luma(mid) > tgt else (lo, mid)
        gamma = round((lo + hi) / 2, 3)
        print(f"{name:>7}: gamma {base.gamma} -> {gamma}   mean luma {mean_luma(gamma):.4f} (target {tgt})")
        g_lo, g_hi = LV_RANGES["gamma"]
        if not g_lo <= gamma <= g_hi:
            print(f"  WARNING: gamma outside the Lv et al. range [{g_lo}, {g_hi}]", file=sys.stderr)
        levels[name] = Level(base.alpha, base.beta, gamma)
    if args.save and ok:
        save_config(Config(levels, cfg.sigma_s, cfg.sigma_c, cfg.seed), args.config)
        print(f"saved to {args.config}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="write darkened copies")
    w.add_argument("--in", dest="inp", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--levels", nargs="+")

    lu = sub.add_parser("luma", help="luminance stats of a folder")
    lu.add_argument("--in", dest="inp", type=Path, required=True)
    lu.add_argument("--csv", type=Path, help="also write per-image luma")

    c = sub.add_parser("calibrate", help="fit each level's gamma to a target mean luma")
    c.add_argument("--in", dest="inp", type=Path, required=True)
    c.add_argument("--targets", nargs="+", required=True, metavar="LEVEL=LUMA",
                   help="e.g. mild=0.21 medium=0.146 severe=0.08 (NPD dark-subset p75/median/p25)")
    c.add_argument("--sample", type=int, default=60, help="images used for the fit (0 = all)")
    c.add_argument("--save", action="store_true", help="write the fitted gamma into the config")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return {"write": cmd_write, "luma": cmd_luma, "calibrate": cmd_calibrate}[args.cmd](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
