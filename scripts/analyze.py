#!/usr/bin/env python3
"""Statistics for results/benchmark.csv (and optionally night_check.csv).

Tables written to results/tables/:
  summary.csv      per method x level: n, mean IoU [95% CI], median, %>=0.75, %<0.5, s/image
  drop.csv         per method x level: mean(IoU_level - IoU_day) [paired 95% CI], Wilcoxon p
  vs_sam.csv       per level x method: mean(IoU_method - IoU_sam) [paired CI], Wilcoxon p, Holm p
  by_size.csv      per method x level x size bin: n, mean IoU [CI]
  night.csv        per method: box IoU / fill / spill [CI]   (if --night given)
  summary.md       the main tables as markdown, for the report

CIs: percentile bootstrap, B=2000, resampling IMAGES (all potholes of a resampled
image come along). Potholes in one image share lighting and are not independent,
so this is the defensible unit; with ~1.1 potholes per image it is close to the
per-pothole bootstrap used for the daylight baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
B, SEED = 2000, 0
LEVEL_ORDER = ["day", "mild", "medium", "severe"]
METHOD_ORDER = ["sam", "clahe", "gamma", "zerodce", "robustsam"]

# Same rule as the daylight baseline (sam_baseline.SIZE_BINS): GT area / image area.
SIZE_EDGES = [0.0, 0.002, 0.05, 0.2, np.inf]
SIZE_LABELS = ["tiny", "small", "medium", "large"]


def cluster_boot(values: np.ndarray, clusters: np.ndarray, rng, b: int = B) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling clusters with replacement."""
    codes, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(codes))
    cnts = np.bincount(inv, minlength=len(codes)).astype(float)
    k = len(codes)
    draws = rng.multinomial(k, np.full(k, 1 / k), size=b)          # b x k resample counts
    means = (draws @ sums) / (draws @ cnts)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcox_p(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if len(d) < 10 or np.allclose(d, 0):
        return float("nan")
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def holm(p: pd.Series) -> pd.Series:
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * ok.values))
    out = pd.Series(np.nan, index=p.index)
    out[ok.index] = adj
    return out


def fmt_ci(m, lo, hi) -> str:
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def order(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("level", "method") if c in df]
    for c, o in (("level", LEVEL_ORDER), ("method", METHOD_ORDER)):
        if c in df:
            df[c] = pd.Categorical(df[c], [x for x in o if x in set(df[c])] + sorted(set(df[c]) - set(o)))
    return df.sort_values(keys).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=REPO / "results" / "benchmark.csv")
    ap.add_argument("--night", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(args.bench)
    df["key"] = df["image"] + "#" + df["pothole_id"].astype(str)
    df["size"] = pd.cut(df["gt_area"] / df["img_area"], SIZE_EDGES, labels=SIZE_LABELS, right=False)

    # per-image timing (rows repeat it per pothole)
    per_img = df.drop_duplicates(["image", "level", "method"])
    per_img = per_img.assign(sec=per_img.sec_enhance + per_img.sec_segment)
    sec = per_img.groupby(["method", "level"]).sec.mean()

    # 1. summary
    rows = []
    for (m, lv), g in df.groupby(["method", "level"]):
        lo, hi = cluster_boot(g.iou.values, g.image.values, rng)
        rows.append({"method": m, "level": lv, "n": len(g), "mean_iou": g.iou.mean(), "ci_lo": lo, "ci_hi": hi,
                     "median_iou": g.iou.median(), "pct_ge_075": 100 * (g.iou >= 0.75).mean(),
                     "pct_lt_05": 100 * (g.iou < 0.5).mean(), "sec_per_image": sec.get((m, lv), np.nan)})
    summary = order(pd.DataFrame(rows))
    summary.to_csv(args.out / "summary.csv", index=False)

    wide = df.pivot_table(index="key", columns=["method", "level"], values="iou")
    img_of = df.drop_duplicates("key").set_index("key")["image"]

    def paired(a: tuple, b: tuple) -> dict:
        sub = wide[[a, b]].dropna()
        d = (sub[a] - sub[b]).values
        lo, hi = cluster_boot(d, img_of.loc[sub.index].values, rng)
        return {"n": len(d), "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": wilcox_p(d)}

    # 2. drop from daylight, per method
    rows = []
    for m in df.method.unique():
        for lv in df.level.unique():
            if lv == "day" or (m, "day") not in wide.columns or (m, lv) not in wide.columns:
                continue
            rows.append({"method": m, "level": lv, **paired((m, lv), (m, "day"))})
    drop = order(pd.DataFrame(rows))
    if len(drop):
        drop["holm_p"] = holm(drop.wilcoxon_p)
    drop.to_csv(args.out / "drop.csv", index=False)

    # 3. each method vs plain SAM at the same level
    rows = []
    for lv in df.level.unique():
        for m in df.method.unique():
            if m == "sam" or (m, lv) not in wide.columns or ("sam", lv) not in wide.columns:
                continue
            rows.append({"level": lv, "method": m, **paired((m, lv), ("sam", lv))})
    vs = order(pd.DataFrame(rows))
    if len(vs):
        vs["holm_p"] = holm(vs.wilcoxon_p)
    vs.to_csv(args.out / "vs_sam.csv", index=False)

    # 4. by size
    rows = []
    for (m, lv, sz), g in df.groupby(["method", "level", "size"], observed=True):
        lo, hi = cluster_boot(g.iou.values, g.image.values, rng) if len(g) > 1 else (np.nan, np.nan)
        rows.append({"method": m, "level": lv, "size": sz, "n": len(g), "mean_iou": g.iou.mean(),
                     "ci_lo": lo, "ci_hi": hi, "small_n": len(g) < 30})
    by_size = order(pd.DataFrame(rows))
    by_size.to_csv(args.out / "by_size.csv", index=False)

    md = ["# Benchmark summary", "",
          f"Bootstrap B={B}, resampling images. Size bins: {dict(zip(SIZE_LABELS, SIZE_EDGES[1:]))} "
          "(upper edges, fraction of image area; same bins as the daylight baseline).", "",
          "## Mean IoU [95% CI]", ""]
    piv = summary.assign(cell=[fmt_ci(r.mean_iou, r.ci_lo, r.ci_hi) for r in summary.itertuples()]) \
        .pivot(index="method", columns="level", values="cell")
    md += [piv.to_markdown(), "", "## Drop from daylight (paired)", ""]
    if len(drop):
        md += [drop.assign(diff=[fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi) for r in drop.itertuples()])
               [["method", "level", "n", "diff", "wilcoxon_p", "holm_p"]].to_markdown(index=False, floatfmt=".2g"), ""]
    md += ["## Method minus plain SAM, same level (paired)", ""]
    if len(vs):
        md += [vs.assign(diff=[fmt_ci(r.mean_diff, r.ci_lo, r.ci_hi) for r in vs.itertuples()])
               [["level", "method", "n", "diff", "wilcoxon_p", "holm_p"]].to_markdown(index=False, floatfmt=".2g"), ""]

    # 5. night (box level)
    if args.night and args.night.exists():
        nd = pd.read_csv(args.night)
        rows = []
        for (m, pr), g in nd.groupby(["method", "prompt"]):
            r = {"method": m, "prompt": pr, "n_boxes": len(g)}
            for col in ("box_iou", "fill", "spill"):
                lo, hi = cluster_boot(g[col].values, g.image.values, rng)
                r.update({col: g[col].mean(), f"{col}_lo": lo, f"{col}_hi": hi})
            r["pct_box_iou_ge_05"] = 100 * (g.box_iou >= 0.5).mean()
            rows.append(r)
        night = order(pd.DataFrame(rows))
        night.to_csv(args.out / "night.csv", index=False)
        nw = nd.pivot_table(index=["image", "ann_id"], columns="method", values="box_iou").reset_index()
        nrows = []
        for m in [c for c in nw.columns if c in METHOD_ORDER and c != "sam"]:
            if "sam" in nw:
                sub = nw[["image", m, "sam"]].dropna()
                d = (sub[m] - sub["sam"]).values
                lo, hi = cluster_boot(d, sub.image.values, rng)
                nrows.append({"method": m, "n": len(d), "diff_box_iou": fmt_ci(d.mean(), lo, hi), "wilcoxon_p": wilcox_p(d)})
        md += ["## Real night (NPD, box level)", "", night.to_markdown(index=False, floatfmt=".3f"), ""]
        if nrows:
            nv = pd.DataFrame(nrows)
            nv["holm_p"] = holm(nv.wilcoxon_p)
            md += ["Box IoU, method minus plain SAM:", "", nv.to_markdown(index=False, floatfmt=".2g"), ""]

    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
