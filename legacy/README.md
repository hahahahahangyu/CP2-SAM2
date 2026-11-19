# Legacy code from capstone-sample

Copied unchanged from `capstone-sample` (branch `milestone-3-yolo-seg`, uncommitted work as of 2026-11-19), for provenance:

- `sam_baseline.py`: the script that produced the daylight baseline (mean IoU 0.653, 186 images, 206 potholes). It was later extended with methods and darkness levels.
- `prepare_potholemix.py`: Pothole Mix → YOLO-seg conversion. It also wrote `data/manifests/potholemix_test.csv`.

These files are **not run** in this repo: they import modules (`pothole.measure`, `scripts.darken`, `scripts.enhance`) that only exist in the old repo. The parts the benchmark needs were copied into `scripts/potholemix.py`:
- `load_binary_mask`
- `DEFAULT_ROOT`
- `find_pair`
- the ≥100 px blob rule
- the inclusive box rule
- the size bins

On 2026-11-19 these copies reproduced the baseline's 186 images and 206 potholes, with size bins tiny 8, small 96, medium 92 and large 10.
