#!/usr/bin/env python3
"""Report figures from results/tables/*.csv (+ overlays). CPU only.

    python scripts/figures.py            # -> results/figures/*.png and *.pdf

fig1_iou_vs_darkness   mean IoU [95% CI] per method across day -> severe
fig2_vs_sam            paired difference vs plain SAM at each level [95% CI]
fig3_by_size           mean IoU by pothole size bin (small multiples)
fig4_night             NPD real-night box-level IoU per method [95% CI]
fig5_examples          one pothole: methods x levels overlay grid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEVELS = ["day", "mild", "medium", "severe"]
LEVEL_LABEL = {"day": "Day", "mild": "Mild", "medium": "Medium", "severe": "Severe"}
METHODS = ["sam", "clahe", "gamma", "zerodce", "robustsam"]
NAME = {"sam": "SAM", "clahe": "CLAHE + SAM", "gamma": "Gamma + SAM",
        "zerodce": "Zero-DCE + SAM", "robustsam": "RobustSAM"}
# categorical slots 1-5 of the reference palette, fixed order; marker = secondary encoding
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig_iou_vs_dark(summary: pd.DataFrame, out: Path, luma: dict):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = range(len(LEVELS))
    off = {m: (i - 2) * 0.06 for i, m in enumerate(METHODS)}
    for m in METHODS:
        g = summary[summary.method == m].set_index("level").reindex(LEVELS)
        if g.mean_iou.isna().all():
            continue
        xs = [i + off[m] for i in x]
        ax.errorbar(xs, g.mean_iou, yerr=[g.mean_iou - g.ci_lo, g.ci_hi - g.mean_iou],
                    color=COLOR[m], marker=MARKER[m], ms=6, lw=2, capsize=0, elinewidth=1.2,
                    mec="white", mew=1, label=NAME[m])
    ax.set_xticks(list(x), [f"{LEVEL_LABEL[l]}\n(luma {luma[l]:.3f})" if l in luma else LEVEL_LABEL[l]
                            for l in LEVELS])
    ax.set_ylabel("Mean IoU vs daylight mask")
    ax.set_xlim(-0.3, len(LEVELS) - 0.7)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("SAM pothole mask IoU as the image darkens (95% bootstrap CI)", loc="left", fontsize=10)
    save(fig, out, "fig1_iou_vs_darkness")


def fig_vs_sam(vs: pd.DataFrame, out: Path):
    others = [m for m in METHODS if m != "sam" and m in set(vs.method)]
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(8, 2.8), sharey=True, sharex=True)
    for ax, lv in zip(axes, LEVELS):
        g = vs[vs.level == lv].set_index("method").reindex(others)
        for i, m in enumerate(others):
            r = g.loc[m]
            if pd.isna(r.mean_diff):
                continue
            ax.errorbar(r.mean_diff, i, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]],
                        color=COLOR[m], marker=MARKER[m], ms=6, elinewidth=1.4, capsize=0, mec="white")
            if r.holm_p < 0.05:
                ax.annotate("*", (r.ci_hi, i), xytext=(3, -1), textcoords="offset points", color=INK, va="center")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(LEVEL_LABEL[lv], fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(range(len(others)), [NAME[m] for m in others])
        ax.invert_yaxis() if ax is axes[0] else None
    fig.supxlabel("IoU difference vs plain SAM (paired, 95% CI; * Holm p < 0.05)", fontsize=9, y=-0.06)
    save(fig, out, "fig2_vs_sam")


def fig_by_size(by_size: pd.DataFrame, out: Path):
    sizes = [s for s in ["tiny", "small", "medium", "large"] if s in set(by_size["size"])]
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.2 * len(sizes), 2.9), sharey=True)
    for ax, sz in zip(axes, sizes):
        for m in METHODS:
            g = by_size[(by_size.method == m) & (by_size["size"] == sz)].set_index("level").reindex(LEVELS)
            if g.mean_iou.isna().all():
                continue
            ax.plot(range(len(LEVELS)), g.mean_iou, color=COLOR[m], marker=MARKER[m], ms=5, lw=1.6,
                    mec="white", label=NAME[m])
        n = int(by_size[(by_size["size"] == sz)].n.iloc[0])
        ax.set_title(f"{sz} (n={n}{', few' if n < 30 else ''})", fontsize=9)
        ax.set_xticks(range(len(LEVELS)), [LEVEL_LABEL[l][:3] for l in LEVELS])
    axes[0].set_ylabel("Mean IoU")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Mean IoU by pothole size (GT area / image area)", x=0.01, y=1.06, ha="left", fontsize=10)
    save(fig, out, "fig3_by_size")


def _dots(ax, g, col, label_fmt="{:.3f}"):
    for i, (m, r) in enumerate(g.iterrows()):
        if pd.isna(r[col]):
            continue
        ax.errorbar(r[col], i, xerr=[[r[col] - r[f"{col}_lo"]], [r[f"{col}_hi"] - r[col]]],
                    color=COLOR[m], marker=MARKER[m], ms=7, elinewidth=1.5, capsize=0, mec="white")
        ax.annotate(label_fmt.format(r[col]), (r[f"{col}_hi"], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(g)), [NAME[m] for m in g.index])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)


def fig_night(night: pd.DataFrame, out: Path, night_box: pd.DataFrame | None = None):
    """Panel A: centre-point prompt, box IoU. RobustSAM is shown but flagged: it returns
    empty masks for single-point prompts even in daylight (scripts/diag_robustsam_prompts.py),
    so its point-prompt score measures prompt support, not night robustness.
    Panel B (if given): GT-box prompt, fill = share of the labelled box covered by the mask."""
    order_ = [m for m in METHODS if m in set(night.method)]
    panels = 2 if night_box is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 2.8), sharey=True, squeeze=False)
    a = axes[0, 0]
    g = night.set_index("method").reindex(order_)
    _dots(a, g, "box_iou")
    if "robustsam" in g.index:
        a.annotate("no single-point support\n(empty even in daylight)", (g.loc["robustsam", "box_iou_hi"], order_.index("robustsam")),
                   xytext=(30, 0), textcoords="offset points", va="center", fontsize=7, color=INK2)
    a.set_xlabel("Box IoU, centre-point prompt")
    a.set_title(f"A  Point prompt (n={int(g.n_boxes.iloc[0])} boxes)", loc="left", fontsize=9)
    if night_box is not None:
        b = axes[0, 1]
        _dots(b, night_box.set_index("method").reindex(order_), "fill")
        b.set_xlabel("Fill (mask share of labelled box), box prompt")
        b.set_title("B  Box prompt", loc="left", fontsize=9)
    fig.suptitle("Real night images, NPD test (280 dark images, 95% CI)", x=0.01, y=1.06, ha="left", fontsize=10)
    save(fig, out, "fig4_night")


def fig_examples(overlays: Path, out: Path, stem: str | None):
    if not overlays.exists():
        return
    files = sorted((overlays / "day" / "sam").glob("*.jpg"))
    if not files:
        return
    pick = next((f for f in files if stem and f.stem.startswith(stem)), files[0]).name
    cols = [m for m in METHODS if (overlays / "day" / m).exists()]
    fig, axes = plt.subplots(len(LEVELS), len(cols), figsize=(1.9 * len(cols), 1.5 * len(LEVELS)))
    for i, lv in enumerate(LEVELS):
        for j, m in enumerate(cols):
            ax = axes[i, j]
            p = overlays / lv / m / pick
            if p.exists():
                ax.imshow(plt.imread(p))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(NAME[m], fontsize=8)
            if j == 0:
                ax.set_ylabel(LEVEL_LABEL[lv], fontsize=8)
    fig.suptitle(f"{pick}: red = GT outline, blue = predicted mask, yellow = box prompt "
                 "(rows 2-4 shown after enhancement)", fontsize=8, x=0.01, ha="left")
    save(fig, out, "fig5_examples")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    ap.add_argument("--bench", type=Path, default=REPO / "results" / "benchmark.csv")
    ap.add_argument("--overlays", type=Path, default=REPO / "results" / "overlays")
    ap.add_argument("--example", help="overlay file stem to show in fig5")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "figures")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    bench = pd.read_csv(a.bench)
    luma = bench.drop_duplicates(["image", "level"]).groupby("level").luma_in.mean().to_dict()
    fig_iou_vs_dark(pd.read_csv(a.tables / "summary.csv"), a.out, luma)
    fig_vs_sam(pd.read_csv(a.tables / "vs_sam.csv"), a.out)
    fig_by_size(pd.read_csv(a.tables / "by_size.csv"), a.out)
    if (a.tables / "night.csv").exists():
        nb = REPO / "results" / "tables_nightbox" / "night.csv"
        fig_night(pd.read_csv(a.tables / "night.csv"), a.out, pd.read_csv(nb) if nb.exists() else None)
    fig_examples(a.overlays, a.out, a.example)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
