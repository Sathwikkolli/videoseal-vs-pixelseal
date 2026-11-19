#!/bin/bash
# One-time setup on a Great Lakes LOGIN node (needs internet; takes ~10-15 min):
#   bash ~/videoseal-vs-pixelseal/slurm/setup_env.sh
# Creates the conda env on scratch (not in the home quota), installs Video Seal (pinned commit) and pre-downloads the
# Video Seal 1.0 and Pixel Seal checkpoints into $VSPS_CKPT so GPU jobs never download.
set -euo pipefail
source "$(dirname "$0")/config.sh"

VIDEOSEAL_COMMIT=f12875a9b5e711e0c6e2dc3a22aca18231ce4a1a

if ! command -v conda >/dev/null 2>&1; then
    module load python3.10-anaconda 2>/dev/null || module load python/3.10 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"

mkdir -p "$(dirname "$VSPS_ENV")" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_HOME"
if [ ! -x "$VSPS_ENV/bin/python" ]; then
    # ffmpeg from conda-forge is built with libx264 and libvmaf
    conda create -y -p "$VSPS_ENV" -c conda-forge python=3.10 "ffmpeg>=6" pip
fi
conda activate "$VSPS_ENV"

pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

mkdir -p "$VSPS_REPO/third_party"
if [ ! -d "$VSPS_REPO/third_party/videoseal/.git" ]; then
    git clone https://github.com/facebookresearch/videoseal.git "$VSPS_REPO/third_party/videoseal"
fi
git -C "$VSPS_REPO/third_party/videoseal" fetch -q origin
git -C "$VSPS_REPO/third_party/videoseal" checkout -q "$VIDEOSEAL_COMMIT"
pip install -e "$VSPS_REPO/third_party/videoseal"
pip install -r "$VSPS_REPO/requirements.txt"
pip install -e "$VSPS_REPO"

mkdir -p "$VSPS_DATA" "$VSPS_CKPT" "$VSPS_OUT"
python - <<EOF
import sys
sys.path.insert(0, "$VSPS_REPO/src")
from vsbench.models import load_model, parse_spec, nbits_of
for name in ["videoseal", "pixelseal"]:
    m = load_model(parse_spec(name, 4), "cpu", "$VSPS_CKPT")
    print(name, "ok:", nbits_of(m), "bits")
import lpips
lpips.LPIPS(net="alex", verbose=False)  # caches AlexNet weights in $TORCH_HOME (scratch)
print("lpips ok")
EOF

echo
echo "ffmpeg: $(ffmpeg -version | head -1)"
ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf && echo "libvmaf: yes" || echo "libvmaf: NO (VMAF will be skipped)"
echo "env '$VSPS_ENV' ready; data dir: $VSPS_DATA"
