#!/bin/bash
#SBATCH --job-name=vsps_aggregate
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:20:00
#SBATCH --account=hafiz1
#SBATCH --output=%x_%j.log

# Tables, statistics, plots and the HTML video viewer. Submit after the array finishes:
#   sbatch --dependency=afterany:<array_job_id> slurm/aggregate.sbatch
set -euo pipefail
source "${VSPS_REPO:-$HOME/videoseal-vs-pixelseal}/slurm/config.sh"
vsps_activate
cd "$VSPS_REPO"

python scripts/aggregate.py --out "$VSPS_OUT"
python scripts/gallery.py --out "$VSPS_OUT"
echo "viewer: $VSPS_OUT/index.html"
