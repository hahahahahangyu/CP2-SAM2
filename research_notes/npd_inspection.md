# NPD inspection (2026-11-19, read-only)

Folder: `/Users/hangyu/Downloads/NPD` (nothing was modified).

## What it is
- A Roboflow export: "坑洼检测 (pothole detection) v3, 2024-06-09", CC BY 4.0, uploaded by a Roboflow user
  (universe.roboflow.com/project-5iscr/-j2sbh). The folder itself does not say it is the official Ling, Shi et al. 2024 release, so cite it with care.
- One class: `pothole`.
- Format: COCO JSON (`annotations/instances_{train,valid,test}.json`) plus flat image folders.

## Annotation type: boxes only
- Every annotation has `segmentation: []` (17,055 of 17,055 annotations). Only `bbox` is filled in, and `area` = bbox w×h.
- **So a mask-level night evaluation is not possible with NPD.** Use the box-level fallback (see benchmark_design.md).

## Counts and resolution
| split | images | boxes | max boxes/img | notes |
|---|---|---|---|---|
| train | 7,801 | 14,944 | 24 | 2,625 source images × ~3 Roboflow versions (random ±20% brightness aug) |
| valid | 750 | 1,404 | 21 | 1 version each |
| test | 379 | 707 | 7 | 1 version each |

- Every image is **640×640**, resized from the original size with a "Stretch" resize, so the aspect ratio is distorted. Box coordinates are in the 640×640 space.
- Median box area is about 11.7k px², which is roughly 3% of the image.
- Filename groups: numeric IDs (~90%), `Vid_{Ca,Pr}_{evening,night}_*` video frames, `ddpi*`, `img*`, `pothole_image_*`.

## Brightness (Rec. 601 luma in [0,1], the same definition as darken.py)
| set | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| test | 379 | 0.253 | 0.175 | 0.057 | 0.586 |
| valid | 750 | 0.250 | 0.176 | 0.068 | 0.587 |
| train (1 per source) | 2,625 | 0.267 | 0.203 | 0.039 | 0.601 |
| all | 3,754 | 0.262 | 0.194 | | |

- The distribution is **bimodal**. 37% of images are below 0.15 and 61% are below 0.25, but 28% are above 0.35 and look like dusk or lit scenes. The test split has ~100 such bright images.
- The dark subset (luma < 0.35, n=2,716) has mean 0.149, median 0.146, p25 0.081, p75 0.210.
- By group: `Vid_night` has median 0.100, while `Vid_evening` has median 0.485.
- Inside the boxes, luma is slightly above the whole-image value (test 0.270).

## Recommendations
1. **Calibrating Medium:** use the NPD dark subset (luma < 0.35). Set Medium ≈ 0.15 (its median), and let Mild/Severe bracket it at about p75 0.21 and p25 0.08. The all-image mean (0.26) is inflated by the dusk images.
2. **Real-night check:** use test (+ valid if more n is needed), filtered to luma < 0.35 (test has 280 such images). Prompt SAM with the GT boxes. Score each prediction by the IoU of its mask's bbox against the GT box, plus the fraction of the box the mask fills. Report this separately from the mask-IoU table.
3. Avoid the train split for brightness stats. Its Roboflow ±20% brightness augmentation changes luma.
