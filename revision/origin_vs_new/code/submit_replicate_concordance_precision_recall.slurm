#!/bin/bash
#SBATCH --job-name=sfv8_concordant_pr
#SBATCH --partition=cpu
#SBATCH --account=brlab
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/origin_vs_new/results/logs/%x_%j.out
#SBATCH --error=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish/revision/origin_vs_new/results/logs/%x_%j.err
#
# Replicate-concordant precision/recall across T7 thresholds.
#
# The original-experiment threshold series already exist under
# revision/bayesian_vs_fold_change/results/tables/ (they are what the published
# method_activity_t7_filter figure is built from), so only the new low-dose
# series are computed here, both with --skip-filtered because the figure uses
# the all-seven-control reference only. The bootstrap series then costs a single
# pass over the 10 GB replicate array instead of one pass per threshold.
#
#   sbatch revision/origin_vs_new/code/submit_replicate_concordance_precision_recall.slurm

set -euo pipefail

REPO=/gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
PYTHON=/gpfs/commons/home/guojiezhong/miniconda3/envs/scvi/bin/python
ANALYSIS="$REPO/revision/origin_vs_new"
BVF_CODE="$REPO/revision/bayesian_vs_fold_change/code"
NEW_RESULTS="$REPO/revision/Bayes_NewData"
NEW_BAYES="$NEW_RESULTS/bayesian"
NEW_BOOTSTRAP="$REPO/revision/Bootstrap_NewData"
NEW_H5AD="$REPO/revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
NEW_TABLES="$NEW_RESULTS/tables"
NEW_FIGURES="$NEW_RESULTS/figures"
THRESHOLDS="5 10 20 50 100"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-local}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$NEW_TABLES" \
  "$NEW_FIGURES" \
  "$ANALYSIS/results/comparison/tables" \
  "$ANALYSIS/results/comparison/figures" \
  "$ANALYSIS/results/logs"

cd "$REPO"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"

# 1. New low-dose Bayesian mean-control tests at every threshold.
"$PYTHON" "$BVF_CODE/figure_work/test_mean_negative_control_activity_threshold_series.py" \
  --bayes-dir "$NEW_BAYES" \
  --h5ad "$NEW_H5AD" \
  --t7-thresholds $THRESHOLDS \
  --q-cutoff 0.05 \
  --skip-filtered \
  --tables-dir "$NEW_TABLES" \
  --figures-dir "$NEW_FIGURES" \
  --unfiltered-stem new_mean_negative_control_tests_t7_series

# 2. New low-dose bootstrap mean-control tests at every threshold.
"$PYTHON" "$BVF_CODE/figure_work/test_bootstrap_mean_negative_control_activity_threshold_series.py" \
  --bootstrap-dir "$NEW_BOOTSTRAP" \
  --h5ad "$NEW_H5AD" \
  --t7-thresholds $THRESHOLDS \
  --q-cutoff 0.05 \
  --skip-filtered \
  --tables-dir "$NEW_TABLES" \
  --figures-dir "$NEW_FIGURES" \
  --unfiltered-stem new_bootstrap_mean_negative_control_tests_t7_series

# 3. Concordance filter, precision/recall, figure.
"$PYTHON" "$ANALYSIS/code/plot_replicate_concordance_precision_recall.py" \
  --t7-thresholds $THRESHOLDS \
  --q-cutoff 0.05 \
  --common-pairs

# 4. Reproduction check: without the cross-method intersection the Bayesian arm
#    must reproduce the saved T7 >= 50 concordance table bit for bit.
"$PYTHON" "$ANALYSIS/code/plot_replicate_concordance_precision_recall.py" \
  --t7-thresholds $THRESHOLDS \
  --q-cutoff 0.05 \
  --no-common-pairs \
  --validate-calls "$ANALYSIS/results/comparison/tables/overlap_t7_ge50_significant_call_concordance_calls.csv.gz" \
  --validate-threshold 50 \
  --stem replicate_concordant_bh_call_precision_recall_per_method_pairs

echo "end=$(date --iso-8601=seconds)"
