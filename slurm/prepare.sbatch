#!/bin/bash
#SBATCH --job-name=vsps_prepare
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --account=hafiz1
#SBATCH --output=%x_%j.log

# Normalize raw clips to 1080p30 10 s, measure SI/TI, pick 10 clips per motion tier.
# Run after scripts/fetch_videos.py. Submit:  sbatch ~/videoseal-vs-pixelseal/slurm/prepare.sbatch
set -euo pipefail
source "${VSPS_REPO:-$HOME/videoseal-vs-pixelseal}/slurm/config.sh"
vsps_activate
cd "$VSPS_REPO"

python scripts/prepare.py --data "$VSPS_DATA" --workers "$SLURM_CPUS_PER_TASK"
python scripts/select_videos.py --data "$VSPS_DATA" --per-tier "${VSPS_PER_TIER:-10}"
