#!/usr/bin/env bash
# One-time setup on the cloud GPU machine (Linux + NVIDIA driver).
# Creates .venv (Python 3.10), installs CUDA PyTorch, ultralytics, RobustSAM and
# Zero-DCE code, downloads the weights, then runs a quick self-check.
#
#   bash cloud/setup_gpu.sh                 # default CUDA 12.1 wheels
#   TORCH_CUDA=cu118 bash cloud/setup_gpu.sh  # if nvidia-smi shows CUDA < 12.1
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_CUDA=${TORCH_CUDA:-cu121}

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "nvidia-smi failed: no GPU on this machine?"; exit 1; }

echo "== Python 3.10 venv (via uv)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== PyTorch ($TORCH_CUDA)"
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
uv pip install -r requirements.txt

echo "== third-party code"
mkdir -p third_party weights
[ -d third_party/RobustSAM ] || git clone --depth 1 https://github.com/robustsam/RobustSAM third_party/RobustSAM
[ -d third_party/Zero-DCE ]  || git clone --depth 1 https://github.com/Li-Chongyi/Zero-DCE third_party/Zero-DCE

echo "== weights"
[ -f weights/robustsam_checkpoint_b.pth ] || curl -L -o weights/robustsam_checkpoint_b.pth \
  https://huggingface.co/robustsam/robustsam/resolve/main/model_checkpoint/robustsam_checkpoint_b.pth
cp -n third_party/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth weights/ 2>/dev/null || true
[ -f weights/sam_b.pt ] || (cd weights && python -c "from ultralytics import SAM; SAM('sam_b.pt')")

echo "== self-check"
python scripts/check_env.py
echo "Setup done. Activate later with: source .venv/bin/activate"
