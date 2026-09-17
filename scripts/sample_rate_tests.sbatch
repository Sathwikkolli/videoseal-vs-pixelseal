#!/bin/bash
#SBATCH --job-name=aware_sr_tests
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:30:00
#SBATCH --account=hafiz1
#SBATCH --output=aware_sr_tests_%j.log

# --- Multi sample rate AWARE: correctness checks (not a performance evaluation) ---
#
#   1. unit tests    tests/          rate params, detector equivalence, VAD, service validation,
#                                    embed/detect on synthetic audio, 16 kHz output vs. original AWARE
#   2. Emilia smoke  scripts/emilia_smoke.py   4 real clips x 8 rates through the public API
#
# Submit from anywhere:   sbatch $HOME/aware-multi-sr/scripts/sample_rate_tests.sbatch
# Override paths:         AWARE_REPO=/path/to/aware-multi-sr EMILIA_CSV=/path/manifest.csv sbatch ...

export AWARE_REPO=${AWARE_REPO:-$HOME/aware-multi-sr}
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare

cd "$AWARE_REPO" || exit 1
echo "repo:   $AWARE_REPO ($(git rev-parse --short HEAD))"
echo "python: $(which python)"

echo; echo "===== 1/2 unit tests ====="
python -m unittest discover -s tests -v
unit_status=$?

echo; echo "===== 2/2 Emilia smoke test ====="
python scripts/emilia_smoke.py --csv "$EMILIA_CSV" --out "$AWARE_REPO/results/emilia_smoke_${SLURM_JOB_ID:-local}"
smoke_status=$?

echo; echo "===== summary ====="
echo "unit tests:   $([ $unit_status -eq 0 ] && echo PASS || echo FAIL)"
echo "emilia smoke: $([ $smoke_status -eq 0 ] && echo PASS || echo FAIL)"
[ $unit_status -eq 0 ] && [ $smoke_status -eq 0 ]
