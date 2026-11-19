# FYP handoff: SAM on low-light potholes

Written 2026-11-19 to carry the project into a new chat. Deadline: **31/12/2026**.

---

## 1. Why the project changed

The original FYP measured potholes from photos. An ArUco marker on the road set the scale, and YOLO-seg outlined the pothole (Milestones 1–3 in this repo, branch `milestone-3-yolo-seg`).

The supervisor's feedback made that hard to defend:
- A marker on the road isn't practical in real life.
- She asked for about 1,000 own images, which isn't realistic in the time left.

The user decided:
- **No marker-based images.**
- **No more photo collection.** Only public datasets.
- Keep the work time-efficient and defendable, with a real research gap.

## 2. The new project in one paragraph

**Does SAM still outline potholes correctly when the road is dark, and what fixes it best?**
I benchmark the Segment Anything Model (SAM) on potholes at increasing darkness. Five methods are compared:
1. plain SAM
2. SAM on images brightened with CLAHE
3. SAM on images brightened with gamma correction
4. SAM on images brightened with Zero-DCE
5. RobustSAM, a SAM variant trained for degraded images

The darkness is simulated on daylight images from Pothole Mix, which have hand-drawn masks. The result is then checked on real night images from NPD.

**Research gap:** SAM has been tested on potholes in daylight, and RobustSAM and low-light enhancers have been tested on general scenes. No study measures how SAM's pothole masks degrade with darkness, or which low-light fix works best for road damage.

**Practical value:** road inspection vans and dashcams often drive at dusk or night. The project shows whether SAM-based labelling or inspection can be trusted after dark, and which cheap preprocessing step to add.

## 3. How it works (pipeline)

1. **Daylight ground truth.** Use the Pothole Mix test split.
   - Each pothole in the hand-drawn mask is a connected component (area ≥ 100 px) and gets its own box prompt.
2. **Darken.** Apply Lv, Li & Lu (arXiv 1908.00682), Eq. 1: `I_out = β · (α · I_in)^γ`.
   - The paper's random ranges are α ∈ U(0.9, 1), β ∈ U(0.5, 1), γ ∈ U(1.5, 5).
   - Fixed levels proposed:

     | Level | α | β | γ |
     |---|---|---|---|
     | Mild | 0.95 | 0.90 | 1.5 |
     | Medium | 0.95 | 0.75 | 3.0 |
     | Severe | 0.95 | 0.50 | 5.0 |

   - Calibrate **Medium** so its mean brightness matches the real night images. Bučko Night has a median brightness of 26 on a 0–255 scale. Also check NPD's brightness.
   - Eq. 2 adds Poisson and Gaussian noise. The paper defers the noise values to CBDNet, and I haven't looked them up yet.
   - Planned extras: a headlight cone (bright near the bottom centre, dark at the edges) and glare.
3. **Enhance (optional).** Brighten with CLAHE, gamma correction or Zero-DCE.
   - Zero-DCE weights: `Zero-DCE_code/snapshots/Epoch99.pth` (320 KB, CC BY-NC 4.0).
4. **Segment.** Give SAM or RobustSAM the same box prompt for every condition.
5. **Score.** Compute IoU against the original daylight mask. Darkening doesn't move pixels, so the daylight mask stays valid.
6. **Real-night check.** Use NPD (see §5). Bučko Night is not a test set: it is used only to calibrate darkness and for a few example figures.

**Key comparison:** the drop in IoU from daylight to each darkness level, per method. This answers whether plain SAM is enough or whether RobustSAM or enhancement is needed. If even RobustSAM fails at Severe, that is also a finding: it gives the darkness limit for night inspection.

## 4. Metrics

- IoU per pothole.
- Mean and median IoU, with a bootstrap 95% CI (2,000 resamples).
- Share of potholes with IoU ≥ 0.75, and share with IoU < 0.5.
- Breakdown by pothole size (tiny, small, medium, large, by area fraction).
- Seconds per image.
- A paired comparison between methods: the same potholes under every condition, using a paired bootstrap or Wilcoxon test.

## 5. Datasets

| Dataset | Use | Location | Notes |
|---|---|---|---|
| **SHREC 2022 Pothole Mix** | Daylight ground truth, simulated darkness | `~/Documents/fyp datasets/SHREC2022/Pothole-Mix/pothole-mix/` (raw, full resolution) | CC BY-NC 3.0. Masks: **red (255,0,0) = pothole**, green = crack. Test split: **186 images, 206 potholes** (196 of them from pothole600). Manifest: `data/manifests/potholemix_test.csv`. Every image is daylight (brightness 84–214, median about 130–140). |
| **NPD (Nighttime Pothole Dataset)** | Real-night evaluation | `/Users/hangyu/Downloads/NPD` (Roboflow COCO export; **boxes only**, mixed day/night; see `NPD_inspection.md`) | Ling, Shi et al., *Electronics* 2024, DOI 10.3390/electronics13193790; repo github.com/hhaozhang/NPD. 3,831 images. Research and non-commercial use only. **Check first: masks or only boxes?** Masks allow a mask-level IoU. Boxes allow only the box-level check below. |
| **Bučko Night** | **Not a test set.** Used only for brightness calibration and a few example figures. | `/Users/hangyu/Downloads/Potholes_dataset/Night` | Mostly unusable: boxes are tiny (median 85×32 px), with heavy glare and blur. Only 3–5 potholes are clear (e.g. frames 101–107), too few to score. Its night brightness (16–33, median 26) sets the Medium darkness level, and the clear frames can illustrate the report. 310 frames, 220 pothole boxes, from 2 videos. |
| RDD2022 | Not used in the new plan | `~/Documents/fyp datasets/RDD2022_fullDataset/` | Boxes only. |

**Box-level real-night check** (if a dataset has only boxes):
- Prompt SAM with a point at the centre of each labelled box, or with a jittered or loose box.
- Take the bounding box of SAM's mask.
- Compute IoU with the labelled box.

This checks that SAM finds the right object at night without needing hand-drawn night masks. Drawing night masks by hand was dropped, to save time.

## 6. Models and tools

| Item | Detail |
|---|---|
| SAM ViT-B | `sam_b.pt` (375 MB, in the repo root, gitignored), run through `ultralytics.SAM`. The same backbone as RobustSAM-B, so the comparison is fair. |
| SAM ViT-L / ViT-H | Optional on the HPC, to check whether a bigger model is more robust. |
| RobustSAM | CVPR 2024. Weights on Hugging Face at `robustsam/robustsam`: ViT-B 0.61 GB, L 1.65 GB, H 3.18 GB. MIT licence. Its official setup assumes **CUDA and Python 3.10**, so **run it on the HPC**. |
| Zero-DCE | Low-light enhancer. Weights `Epoch99.pth`. |
| CLAHE, gamma | OpenCV. No training needed. |
| Compute | Mac (MPS) for small runs. **HPC (CUDA)** for RobustSAM and the full grid of methods × darkness levels. Use `--device cuda` (or `0`) on the HPC. |
| Dropped | SALS (ICRA 2025, no licence). Lighted-SAM (no code found). |

No model training is required. This is a zero-shot benchmark, which keeps it feasible before the deadline.

## 7. Results so far

Plain SAM ViT-B, daylight, Pothole Mix test split (`scripts/sam_baseline.py`):
- 186 images, 206 potholes.
- **Mean IoU 0.653 (95% CI 0.623–0.680), median 0.719.**
- 42% of potholes have IoU ≥ 0.75, and 19% have IoU < 0.5.
- About 1.06 s per image on MPS.
- By size:

  | Size | Mean IoU | n |
  |---|---|---|
  | tiny | 0.389 | 8 |
  | small | 0.717 | 96 |
  | medium | 0.613 | 92 |
  | large | 0.631 | 10 |

- Common failure: SAM misses the pothole's rim. Not tested yet: `multimask_output`, picking the best of SAM's 3 candidate masks.

## 8. Repo state

- Branch `milestone-3-yolo-seg`. The new work is **uncommitted**: `scripts/sam_baseline.py`, `research_notes/`, `reports/`.
- `scripts/sam_baseline.py`:
  - Reads the raw Pothole Mix test images and masks.
  - Prompts each pothole with its own box.
  - Writes `runs/sam_baseline/results.csv` (image, pothole, area_frac, iou) and colour overlays (green = agree, red = missed, blue = extra).
  - Arguments: `--limit`, `--overlays`, `--model`, `--device`, `--out`.
- The old milestone code (`pothole/`, `measure_*.py`, YOLO training) stays as history.
- Proposal draft (Claude Doc): https://claude.ai/artifact/DcvwymqgjbTvvTwryLRja3. It has 5 pending suggestions:
  - the darkening formula
  - the noise note
  - the Bučko paragraph
  - the dataset table
  - the self-check
  It still mentions hand-drawn night masks and needs updating to the NPD and box-level plan.

## 9. Next steps

1. Inspect NPD: annotation format (masks, boxes or polygons), image count, resolution and brightness.
2. Set up the HPC environment: Python 3.10, CUDA PyTorch, ultralytics, RobustSAM repo and weights, Zero-DCE.
3. Write `scripts/darken.py` with the 3 levels, noise, and an optional headlight cone. Calibrate Medium to the real night brightness.
4. Extend `sam_baseline.py` into a benchmark over methods × levels, with a single results CSV.
5. Run plain SAM on the Mac or HPC, and RobustSAM on the HPC.
6. Run the real-night check on NPD.
7. Analyse results: IoU-drop curves, paired tests and the size breakdown. Make figures.
8. Update the proposal doc, then write the report.

## 10. Constraints to respect

- No marker-based images and no new photo collection.
- Marker measurement is out of scope. The output is mask quality (IoU), not size in cm.
- Every dataset licence is non-commercial (CC BY-NC, NPD research-only), which is fine for an FYP. Cite each dataset.
