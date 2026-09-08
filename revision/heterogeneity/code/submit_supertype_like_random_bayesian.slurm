#!/bin/bash
#SBATCH --job-name=het_stlr_bayes
#SBATCH --partition=gpu
#SBATCH --account=brlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/heterogeneity/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/heterogeneity/results/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python
OUTDIR="$REPO/revision/heterogeneity/results/supertype_like_random/bayesian"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$REPO/revision/heterogeneity/results/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"
"$PYTHON" revision/heterogeneity/code/run_bayes_split.py \
  --grouping supertype_like_random \
  --outdir "$OUTDIR" \
  --level subclass \
  --channel joint \
  --infection-model copy_number_dropout \
  --dropout-prior-label default_beta_1_9 \
  --activity-model direct \
  --negative-control-mode ordinary \
  --method svi \
  --steps 30000 \
  --num-posterior 1000 \
  "$@"
echo "end=$(date --iso-8601=seconds)"
