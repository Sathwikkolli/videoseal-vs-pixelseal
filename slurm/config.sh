# Shared settings, sourced by every job and by setup_env.sh. Override any of these by
# exporting them before `sbatch` (sbatch passes your environment through by default).

export VSPS_REPO=${VSPS_REPO:-$HOME/videoseal-vs-pixelseal}
export VSPS_ACCOUNT=${VSPS_ACCOUNT:-hafiz1}
# scratch is fast and large but purged after ~60 days of no access; copy results you keep
export VSPS_SCRATCH=${VSPS_SCRATCH:-/scratch/${VSPS_ACCOUNT}_root/${VSPS_ACCOUNT}/$USER}
export VSPS_DATA=${VSPS_DATA:-$VSPS_SCRATCH/vsps}
# the conda env (~6 GB with torch) and every download cache live on scratch, not in the
# 80 GB home quota; VSPS_ENV is a path, so `conda activate` works without touching ~/.conda
export VSPS_ENV=${VSPS_ENV:-$VSPS_SCRATCH/envs/vsps}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$VSPS_SCRATCH/.cache/conda_pkgs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$VSPS_SCRATCH/.cache/pip}
export TORCH_HOME=${TORCH_HOME:-$VSPS_SCRATCH/.cache/torch}
export HF_HOME=${HF_HOME:-$VSPS_SCRATCH/.cache/huggingface}
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
