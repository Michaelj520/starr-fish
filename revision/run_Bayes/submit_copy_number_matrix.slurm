#!/bin/bash
#SBATCH --job-name=sfv8_copy_number
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/copy_number_%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/run_Bayes/logs/copy_number_%x_%j.err
#
# The <dataset>/copy_number/ folder, from a fit that saved log_rho and log_a.
#
#   sbatch --job-name=cn_old submit_copy_number_matrix.slurm old
#   sbatch --job-name=cn_new submit_copy_number_matrix.slurm new
#
# Writes into <outdir>:
#   copy_number.npz                       E[k_ij | t7_ij, cre_ij], mean + sd, float32
#   copy_number_copies.csv.gz             the mean, cells x cCREs
#   activity_normalized.npz               copies, posterior activity, and that
#                                         activity / the cell type's control
#                                         reference (NaN below pooled control T7 50)
#   activity_posterior.npz                the posterior activity alone
#   negative_control_reference_check.csv  each per-cell estimator on the 7 controls
#                                         against its parameter-level target
#
# A file already on disk is skipped, never overwritten. Each step writes to a
# .partial name and renames on success, so a job killed mid-write leaves nothing
# that a rerun would mistake for a finished file; resubmitting resumes.
#
# CPU only, on purpose: this streams ~160M pairs in chunks and never touches a
# GPU. Cost scales with the posterior, not the data -- the all-zero baseline
# alone is n_cell_types x n_cre patterns -- so the posterior is thinned to 200
# evenly-spaced draws, which is within Monte Carlo error of the full 1,000 for a
# posterior *mean*. Measured at MAX_DRAWS=200 on the old dataset, step 1:
# 18m12s, 6.8 GB peak RSS, of which roughly a third is writing the 159M-value
# CSV. Step 2 is a second full pass of about the same compute; steps 3-4 take
# minutes (step 4 reconstructs only the 7 control columns).

set -euo pipefail

DATASET="${1:?usage: submit_copy_number_matrix.slurm <old|new>}"

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
LOGS="$REPO/revision/run_Bayes/logs"
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
OLD_RESULTS="$REPO/revision/Bayes_OldData"
NEW_RESULTS="$REPO/revision/Bayes_NewData"
CHECK="$REPO/revision/run_Bayes/check_negative_control_reference.py"
EXTRACT="$REPO/revision/run_Bayes/extract_activity_posterior.py"
MAX_DRAWS=${MAX_DRAWS:-200}

case "$DATASET" in
  old|origin)
    FIT="$OLD_RESULTS/bayesian"
    H5AD="$REPO/revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
    OUTDIR="$OLD_RESULTS/copy_number"
    ;;
  new)
    FIT="$NEW_RESULTS/bayesian"
    H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
    OUTDIR="$NEW_RESULTS/copy_number"
    ;;
  *)
    echo "unknown dataset '$DATASET' (expected 'old' or 'new')" >&2
    exit 2
    ;;
esac

export MPLBACKEND=Agg
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p "$XDG_CACHE_HOME" "$OUTDIR" "$LOGS"

# Leftovers of a killed run; finished files never carry this suffix.
rm -f "$OUTDIR"/*.partial*

# have FILE... -> true when every FILE exists.
have() {
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || return 1
  done
}

cd "$REPO"
echo "dataset=$DATASET"
echo "fit=$FIT"
echo "h5ad=$H5AD"
echo "outdir=$OUTDIR"
echo "max_draws=$MAX_DRAWS"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"

echo "--- step 1/4: copy number (mean + sd, mean as CSV) ---"
if have "$OUTDIR/copy_number.npz" "$OUTDIR/copy_number_copies.csv.gz"; then
  echo "exists, skipping"
else
  "$PYTHON" -m baystarrfish copy-number \
    --fit-dir "$FIT" \
    --h5ad "$H5AD" \
    --out "$OUTDIR/copy_number.partial.npz" \
    --csv "$OUTDIR/copy_number.partial" \
    --csv-which copies \
    --csv-decimals 5 \
    --with-sd \
    --max-draws "$MAX_DRAWS"
  mv "$OUTDIR/copy_number.partial_copies.csv.gz" "$OUTDIR/copy_number_copies.csv.gz"
  mv "$OUTDIR/copy_number.partial.npz" "$OUTDIR/copy_number.npz"
fi

echo "--- step 2/4: posterior activity, raw and control-normalised ---"
if have "$OUTDIR/activity_normalized.npz"; then
  echo "exists, skipping"
else
  "$PYTHON" -m baystarrfish copy-number \
    --fit-dir "$FIT" \
    --h5ad "$H5AD" \
    --out "$OUTDIR/activity_normalized.partial.npz" \
    --with-activity \
    --with-activity-normalized \
    --max-draws "$MAX_DRAWS"
  mv "$OUTDIR/activity_normalized.partial.npz" "$OUTDIR/activity_normalized.npz"
fi

echo "--- step 3/4: posterior activity alone ---"
if have "$OUTDIR/activity_posterior.npz"; then
  echo "exists, skipping"
else
  "$PYTHON" "$EXTRACT" \
    --source "$OUTDIR/activity_normalized.npz" \
    --out "$OUTDIR/activity_posterior.partial.npz"
  mv "$OUTDIR/activity_posterior.partial.npz" "$OUTDIR/activity_posterior.npz"
fi

echo "--- step 4/4: negative-control calibration check ---"
if have "$OUTDIR/negative_control_reference_check.csv"; then
  echo "exists, skipping"
else
  "$PYTHON" "$CHECK" \
    --fit-dir "$FIT" \
    --h5ad "$H5AD" \
    --out "$OUTDIR/negative_control_reference_check.partial.csv" \
    --max-draws "$MAX_DRAWS"
  mv "$OUTDIR/negative_control_reference_check.partial.csv" \
    "$OUTDIR/negative_control_reference_check.csv"
fi

ls -la "$OUTDIR"
echo "end=$(date --iso-8601=seconds)"
