#!/bin/bash
#SBATCH --job-name=sfv6_multibc_bayes
#SBATCH --partition=gpu
#SBATCH --account=brlab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/Bayes_multiBC/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/Bayes_multiBC/logs/%x_%j.err

# Extra arguments are passed through to run_bayes_multibc.py, e.g. --overwrite.
set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
HERE="$REPO/revision/Bayes_multiBC"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python
H5AD="$REPO/revision/Data/scdata_260821_SFv6_multiBC_FINAL.h5ad"
DNA_CSV="$REPO/revision/Data/STARR_seq_Counts_wReps_DNA.csv"
OUTDIR="$HERE/bayesian"

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
  "$OUTDIR" \
  "$HERE/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices())"

"$PYTHON" "$HERE/run_bayes_multibc.py" \
  --h5ad "$H5AD" \
  --dna-csv "$DNA_CSV" \
  --outdir "$OUTDIR" \
  --kmax 60 \
  --steps 30000 \
  --lr 0.005 \
  --num-posterior 1000 \
  --posterior-sites log_gamma log_rho log_a \
  --seed 0 \
  "$@"
echo "end=$(date --iso-8601=seconds)"
