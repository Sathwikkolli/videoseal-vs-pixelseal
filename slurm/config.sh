# Shared settings, sourced by every job and by setup_env.sh. Override any of these by
# exporting them before `sbatch` (sbatch passes your environment through by default).

export VSPS_REPO=${VSPS_REPO:-$HOME/videoseal-vs-pixelseal}
export VSPS_ENV=${VSPS_ENV:-vsps}
export VSPS_ACCOUNT=${VSPS_ACCOUNT:-hafiz1}
# scratch is fast and large but purged after ~60 days of no access; copy results you keep
export VSPS_DATA=${VSPS_DATA:-/scratch/${VSPS_ACCOUNT}_root/${VSPS_ACCOUNT}/$USER/vsps}
export VSPS_CKPT=${VSPS_CKPT:-$VSPS_DATA/ckpts}
export VSPS_OUT=${VSPS_OUT:-$VSPS_DATA/out}

vsps_activate() {
    if ! command -v conda >/dev/null 2>&1; then
        module load python3.10-anaconda 2>/dev/null || module load python/3.10 2>/dev/null || true
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$VSPS_ENV"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    export TORCHDYNAMO_DISABLE=1
}
