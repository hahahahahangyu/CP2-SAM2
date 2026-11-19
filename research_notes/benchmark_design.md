# Methods × darkness benchmark: design (draft, updated 2026-11-19)

## Grid
- **Images:** Pothole Mix test split, the same 206 potholes as the daylight baseline.
- **Conditions (4):** day, mild, medium and severe, from `scripts/darken.py`.
  - Parameters are fixed, and the noise is seeded so it is identical on every run.
  - Calibrated on the 186 test images (daylight mean luma 0.561) against the NPD dark subset (luma < 0.35):
    - mild: γ 2.41, which gives 0.210 (NPD p75).
    - medium: γ 2.93, which gives 0.146 (NPD median). Your proposed 3.0 gives 0.143.
    - severe: γ 5.0, the top of the Lv range, which gives 0.087 (about NPD p27). The noise floor stops it reaching p25 inside the paper's range.
  - Noise follows CBDNet: σ² = L·σs² + σc² in linear space, with σs = 0.08 and σc = 0.03 (the midpoints of CBDNet's ranges).
- **Methods (5):** SAM ViT-B · CLAHE → SAM · gamma → SAM · Zero-DCE → SAM · RobustSAM-B.
- 5 × 4 = 20 cells, 206 instances each.

## Controls
- **Same prompt everywhere:** one box per pothole, from the *daylight* GT mask: a tight, inclusive bbox of each blob ≥ 100 px, with no jitter. This is exactly the baseline's rule.
- **Same GT:** daylight mask; darkening is photometric only.
- **Same SAM checkpoint and single-mask output** (`multimask_output=False`) for all methods. Best-of-3 is a possible follow-up.
- **Pre-processing parameters fixed before looking at results:**
  - CLAHE on the L channel of LAB, clipLimit 2.0, tile 8×8.
  - Gamma: adaptive, γ = log(0.5)/log(mean luma), clamped to [0.2, 3]. It has no free parameter, and it is fair across levels where a single fixed γ would not be.
  - Zero-DCE: the official pretrained weights, no fine-tuning.
- Enhancement also runs on Day (L0), to check that it doesn't hurt daylight images.

## Outputs
`results/benchmark.csv`, with one row per (image, pothole_id, level, method). Columns: IoU, Dice, GT/pred area, box, input luma, seconds. `night_check.csv` holds the NPD results.

## Statistics
- **Mean IoU per cell with a 95% CI:** percentile bootstrap, B = 2,000. Resample *images*, not potholes, because potholes in the same image are correlated.
- **Paired tests at each level:** each method vs plain SAM on the same instances.
  - Wilcoxon signed-rank test.
  - Paired bootstrap CI of the mean ΔIoU, which is the effect size to report.
  - Holm correction over the 4 × 4 comparisons.
- **Degradation:** ΔIoU (Lk − Day) per method, using the same paired bootstrap.
- **Size breakdown:**
  - Bins follow the baseline, by GT area as a fraction of the image: tiny < 0.002, small < 0.05, medium < 0.2, and large otherwise. The counts are 8 / 96 / 92 / 10.
  - Report the cell means with CIs inside each bin.
  - Report n per bin; flag any bin with fewer than about 30 instances.
- **Optional:** a mixed model, IoU ~ method × level × size + (1 | image), as a single summary test.

## Real-night check (NPD)
- **NPD has boxes only,** so there is no mask IoU. See `npd_inspection.md`.
- **Images:** the test split, filtered to luma < 0.35, leaves 280 images and about 520 boxes.
- **Prompt:** one positive point at the centre of the GT box. A box prompt would be circular, because SAM's mask is constrained to the box it is given.
- **Scores:**
  - IoU between the predicted mask's bbox and the GT box. This is the main score.
  - fill: the share of the GT box that the mask covers.
  - spill: the share of the mask that falls outside the box.
- **Reporting:** keep these results separate from the synthetic table. They answer "do the method rankings hold on real night images?", not "what is the exact IoU?".
- **Caveats:**
  - The Roboflow images are stretched to 640×640.
  - This copy of NPD is a Roboflow re-upload, so cite it carefully.

## Open
- Pin ultralytics to the version the baseline used on the Mac (`pip show ultralytics`).
- **Optional:** a headlight-cone/glare variant. This is extra, so only do it if time allows.
