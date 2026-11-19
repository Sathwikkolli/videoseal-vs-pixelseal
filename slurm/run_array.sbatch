#!/bin/bash
#SBATCH --job-name=vsps_run
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --account=hafiz1
#SBATCH --array=0-29
#SBATCH --output=logs/%x_%A_%a.log

# One array task per manifest row (30 clips = 3 tiers x 10).
# Every task runs on the same partition so embedding times are comparable; use
# --partition=spgpu (A40) instead of gpu (V100) if you prefer, but not a mix.
# Submit from the repo:  cd ~/videoseal-vs-pixelseal && mkdir -p logs && sbatch slurm/run_array.sbatch
set -euo pipefail
source "${VSPS_REPO:-$HOME/videoseal-vs-pixelseal}/slurm/config.sh"
vsps_activate
cd "$VSPS_REPO"
export TMPDIR=${TMPDIR:-/tmp}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/run_one.py \
    --manifest "$VSPS_DATA/manifest.csv" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --out "$VSPS_OUT" \
    --ckpt-dir "$VSPS_CKPT" \
    --models ${VSPS_MODELS:-videoseal pixelseal} \
    --step-size "${VSPS_STEP:-4}"
