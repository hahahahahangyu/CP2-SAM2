# CP2-SAM2: Does SAM still outline potholes in the dark?

A zero-shot benchmark for a final-year project. It asks how well SAM's pothole masks hold up as images get darker, and which low-light fix works best for road damage. There is no training and no new photos: only public datasets and released weights are used.

This project replaces an earlier version that measured pothole size with an ArUco marker and YOLO-seg ([capstone-sample](https://github.com/hahahahahangyu/capstone-sample), branch `milestone-3-yolo-seg`). That direction was dropped because a marker on the road isn't practical.

## Design in one paragraph

1. Take the Pothole Mix (SHREC 2022) test split, which has hand-drawn masks: 186 images and 206 potholes.
2. Darken each image to 3 levels with Lv, Li & Lu's model (Eq. 1, `I = β(αI)^γ`) plus CBDNet sensor noise. The levels are calibrated to the brightness of real night images in NPD.
3. Segment every pothole with the same box prompt, using 5 methods:
   - plain SAM ViT-B
   - CLAHE → SAM
   - adaptive gamma → SAM
   - Zero-DCE → SAM
   - RobustSAM-B
4. Score each mask by IoU against the daylight mask.
5. Repeat the check on real night images from NPD. NPD has boxes only, so this check is scored at box level.

The full plan and statistics are in [`research_notes/benchmark_design.md`](research_notes/benchmark_design.md).

## Layout

```
configs/levels.json          darkness levels (α, β, γ) + noise; fitted by darken.py calibrate
scripts/
  darken.py                  Eq.1 + CBDNet noise; `write`, `luma`, `calibrate` commands
  potholemix.py              test list, GT masks, one box per pothole (≥100 px)
  methods.py                 enhancers (none/CLAHE/gamma/Zero-DCE) + segmenters (SAM, RobustSAM)
  benchmark.py               methods × levels → results/benchmark.csv (resumable)
  night_check.py             NPD box-level check → results/night_check.csv
  analyze.py                 CIs, paired tests, size breakdown → results/tables/
  check_env.py               loads every model once (run after setup)
data/manifests/potholemix_test.csv   test list (504 rows; the 186 with n_potholes > 0 are used)
docs/RUN_ON_CLOUD.md         step-by-step guide for the cloud GPU run
legacy/capstone-sample/      original sam_baseline.py + prepare_potholemix.py (reference only)
cloud/setup_gpu.sh           one-time GPU machine setup (Python 3.10, CUDA torch, weights)
cloud/run_all.sh             smoke test → benchmark → night check → stats
research_notes/              design, NPD inspection, handoff notes
```

Datasets, weights, third-party code and overlay images are git-ignored.

## Running

**Mac (MPS):** use this for small runs. RobustSAM is optional here.

```bash
pip install torch torchvision && pip install -r requirements.txt
export POTHOLEMIX_ROOT="$HOME/Documents/fyp datasets/SHREC2022/Pothole-Mix/pothole-mix"
python scripts/benchmark.py --methods sam clahe gamma --limit 5 --out results/smoke.csv
```

**Cloud GPU:** follow [`docs/RUN_ON_CLOUD.md`](docs/RUN_ON_CLOUD.md). In short:

```bash
bash cloud/setup_gpu.sh
mkdir -p results
POTHOLEMIX_ROOT=data/raw/pothole-mix NPD_ROOT=data/raw/NPD \
  nohup bash cloud/run_all.sh > results/run.log 2>&1 &
```

**Sanity check:** the `sam` / `day` cell must reproduce the daylight baseline, a mean IoU of 0.653 (95% CI 0.623–0.680). The ground-truth loader already reproduces the baseline's 186 images, 206 potholes and size counts (8 / 96 / 92 / 10).

**Darkness levels:** these are already calibrated, in `configs/levels.json`. Mean luma is mild 0.210, medium 0.146 and severe 0.087, against the NPD dark-subset p75, median and p25.

## Data and licences

- **Pothole Mix (SHREC 2022):** CC BY-NC 3.0.
- **NPD:** Ling, Shi et al., *Electronics* 2024, doi:10.3390/electronics13193790. For research and non-commercial use. The local copy is a Roboflow export with images stretched to 640×640.
- **Code and weights:**
  - RobustSAM: Chen et al., CVPR 2024. MIT licence.
  - Zero-DCE: Guo et al., CVPR 2020. The weights are CC BY-NC 4.0.
  - SAM: Kirillov et al., 2023, run through ultralytics (AGPL-3.0).
- **Darkening model:** Lv, Li & Lu, arXiv:1908.00682.
- **Noise model:** Guo et al. (CBDNet), CVPR 2019, arXiv:1807.04686.
