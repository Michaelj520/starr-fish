# Original versus new low-dose Bayesian activity

This analysis compares the existing production Bayesian fit of

`revision/Data/scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad`

with a new, otherwise identical fit of

`revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad`.

## Fixed analysis choices

- Model: joint cCRE/T7 copy-number model with zero inflation
  (`copy_number_dropout`), direct subclass activity, ordinary negative
  controls, AutoNormal SVI, 30,000 steps, learning rate 0.005, `Kmax=60`,
  seed 0, and 1,000 posterior draws.
- Blacklist: the manuscript base blacklist plus cCREs with barcode mismatch
  above 20%. The fit is required to reproduce the exact blacklist saved by
  the original run.
- Negative controls: the same seven ordinary negative controls used in the
  original production fit.
- Statistical test: within every posterior draw, target `log_gamma` minus
  the mean `log_gamma` of the seven ordinary negative controls. A pair is
  tested only when target T7 is at least 50 and total T7 across the seven
  controls is at least 50 in that subclass. Right-tail posterior p-values
  are adjusted with BH across all eligible pairs, with significance at
  `q <= 0.05`.

## Comparison scopes

The output deliberately separates two questions:

1. `all_common_activity`: posterior-mean activity for every shared
   subclass-target pair, independent of T7 eligibility. Both raw
   `log_gamma` and the comparable negative-control-centered activity are
   retained.
2. `overlap_t7_ge50_pairs`: pairs that have target T7 >= 50 and combined
   seven-control T7 >= 50 in both datasets. This identical overlap is the
   primary universe for activity correlation, BH correction, and
   significant-call concordance. Each fit's BH q-values are recomputed using
   only these same pairs.

Dataset-specific eligible and significant pairs are retained only as
secondary provenance so gains or losses caused by changed T7 coverage are
not hidden.

## Layout

- `../run_Bayes/submit_new_bayesian.slurm`: production GPU fit.
- `code/submit_statistics_and_compare.slurm`: dependent CPU statistics and
  comparison job.
- `code/compare_origin_vs_new.py`: validated table/figure generation.
- `../run_Bootstrap/submit_new_bootstrap.slurm`: exact 10,000-replicate bootstrap on the
  new data. It uses 12 parallel workers on 14 CPUs with 256 GB; the completed
  run peaked at approximately 141 GiB. Worker count is execution-only:
  replicate `i` is seeded with `random_state=i`, so it does not alter the
  bootstrap samples.
- `code/submit_bootstrap_statistics_and_compare.slurm`: dependent bootstrap
  T7-filtered statistics and comparison.
- `code/compare_bootstrap_origin_vs_new.py`: old-versus-new bootstrap
  overlap comparison.
- `code/submit_replicate_concordance_precision_recall.slurm`: the
  replicate-concordant precision-recall job. It computes the two missing new
  low-dose threshold series - Bayesian and bootstrap mean-control tests at T7
  5, 10, 20, 50 and 100, both with `--skip-filtered` because the figure uses the
  all-seven-control reference only - into `../Bayes_NewData/tables/`, then draws
  the figure and finally re-runs the analysis with `--no-common-pairs
  --validate-calls` as a reproduction check against the saved T7 >= 50
  concordance table. The original-experiment series already exist under
  `../bayesian_vs_fold_change/results/tables/`.
- `code/plot_activity_ccc_vs_t7_cutoff.py`: combined Bayesian/bootstrap
  Lin's-CCC sensitivity curve at T7 cutoffs 0, 1, 2, 5, 10, 20, 50, 100,
  200, and 500. It also reports mean within-cell-type CCC across cCREs and
  mean within-cCRE CCC across cell types, excluding units supported by fewer
  than 10 pairs. Unit-averaged curves show mean +/- one sample standard
  deviation. Each point uses the exact old/new overlap at its cutoff and
  reports method-specific pair or unit counts on the x-axis.
- `code/plot_unit_activity_correlation_scatter.py`: per-unit old-versus-new
  activity concordance scatter. Each overlap-filtered cCRE-cell-type pair
  contributes to one point per panel: left panel one point per cCRE
  (correlation across its cell types), right panel one point per cell type
  (correlation across its cCREs). y is the within-unit Pearson r of
  negative-control-centered activity, x is the unit's supported pair count
  (log scale), and colour is the median per-pair `min(original, new)` target
  T7. Units with fewer than 10 pairs are dropped; the pooled pair-level r and
  the mean within-unit r are drawn as reference lines, and the three most and
  least concordant units per panel are labelled.
- `code/plot_raw_count_concordance_scatter.py`: raw-count concordance scatter.
  Each dot is one cCRE-cell-type pair. Both panels show raw transcript counts
  summed over the cells of the subclass in each experiment - the T7 transcript
  species on the left and the cCRE transcript species on the right - read
  straight from
  `obsm/T7CRE` and `obsm/CRE` one cCRE at a time so the 500-gene expression
  matrix is never loaded. Both axes are `count + 1` on a log scale with the
  `y = x` identity line; each panel reports Pearson r on `log1p` counts,
  Spearman rho on raw counts, and the new/original total-count ratio. Two
  scopes are drawn: all shared non-blacklisted pairs (324 subclasses x 389
  cCREs) and the T7 >= 50 overlap universe plus the negative-control pairs of
  those subclasses. The seven negative controls are highlighted in every
  panel. The streamed T7 totals are required to reproduce the published
  overlap test-table totals exactly, which validates the loader.
- `code/plot_call_concordance.py`: significant-call concordance emitted as a
  matched pair of figures from one control reference. Both bases start from the
  same right-tail posterior p-value of the saved pair tables: `raw_p` thresholds
  the uncorrected p-value, so the counts do not depend on how many pairs entered
  the correction, and `bh_q` applies BH across the pairs of that universe. It
  reuses the saved tables and the `call_metrics` helper from
  `compare_origin_vs_new.py`; no refit is needed. `--source overlap` (default)
  uses the primary mean negative-control reference, and
  `--source mean_plus_1sd` the draw-wise mean+1 SD reference. The recomputed BH
  calls are required to reproduce the `*_significant_common_q` columns of the
  source table exactly.
- `code/plot_replicate_concordance_precision_recall.py`: the published
  T7-threshold precision-recall figure
  (`revision/bayesian_vs_fold_change/results/figures/final/`
  `method_activity_t7_filter_precision_recall.pdf`) restricted, for both
  methods, to the cCRE-cell-type pairs whose call agrees between the two
  experiments. At every T7 threshold (5, 10, 20, 50, 100) and for each method -
  Bayesian `Joint+dropout mean controls` and `Bootstrap mean controls` - the
  original and new low-dose threshold-series test tables are intersected on
  their eligible pairs, BH is recomputed inside that shared universe separately
  per experiment (the definition behind
  `overlap_t7_ge50_significant_call_concordance_bh_q`), and only the concordant
  pairs - `both_significant` or `neither_significant` at `q <= 0.05` - are
  evaluated. The tested universe of the figure is therefore the concordant set,
  so the recall denominator and the naive-precision baseline are restricted to
  it as well; the plotted `q_right` is the per-pair `max(origin_q, new_q)`,
  which reproduces the shared call on concordant pairs. `--common-pairs` (the
  default, matching the published figure) first intersects the eligible pairs of
  all four test tables at each threshold, so both methods start from one
  identical universe before their own concordance filter. `--no-common-pairs`
  lets each method use its own origin/new intersection; at T7 >= 50 that
  reproduces the saved Bayesian concordance table exactly, which
  `--validate-calls` requires. It imports `ASSAYS`,
  `assay_positive_for_tests`, `benchmark_assay`, and `plot_precision_recall`
  from `revision/bayesian_vs_fold_change/code/plot_t7_filter_precision_recall.py`,
  so the metric definitions, the panel layout, the method colours, and the
  dashed naive-precision baseline are the published ones.
- `code/plot_origin_vs_new_heatmap.py`: matched original/new activity heatmap
  with test diagnostics. It plots the exported per-dataset matrices and derives
  nothing but the shared-universe BH: the colours are read from
  `../Bayes_OldData/tables/subclass_cre_activity_matrix.csv.gz` and
  `../Bayes_NewData/tables/subclass_cre_activity_matrix.csv.gz`, the displayed
  universe from the matching `_target_t7_matrix.csv.gz` at `--t7-threshold`
  (default 50, applied to both datasets), and `p_right` from
  `_significance.csv.gz`. Cells carry a star for shared-universe BH `q <= 0.05`,
  recomputed here from `p_right` because the family is the intersection of the
  two datasets rather than either one alone, and a black box for an ATAC peak;
  pairs that reach nominal `p <= 0.05` without BH significance are counted in
  the panel titles but not marked. `--control-reference` selects which exported
  significance table is read: `mean` takes the mean-control null and draws the
  control-spread strip as the SD across the seven control posterior-mean
  activities in `_negative_control_activity_matrix.csv.gz`, while `mean_plus_1sd`
  (the default) takes the `subclass_cre_mean_plus_1sd_significance.csv.gz`
  export. Regenerate the matrices with
  `revision/run_Bayes/submit_activity_matrices.slurm`. `--restrict-calls`,
  `--restrict-status-column`, and `--restrict-status` subset the displayed pairs
  by a call-status column, which is how the replicate-concordant heatmap is
  produced:

  ```
  python plot_origin_vs_new_heatmap.py \
    --control-reference mean \
    --restrict-calls ../results/comparison/tables/overlap_t7_ge50_significant_call_concordance_calls.csv.gz \
    --stem origin_vs_new_replicate_concordant_activity_heatmap_t7_ge50
  ```
- `../Bayes_NewData/bayesian/`: new posterior and model diagnostics. Stores
  `log_gamma`, `log_rho` and `log_a`, so copy number is recoverable from it.
- `../Bootstrap_NewData/`: new bootstrap arrays, activity, and q-values.
- `../Bayes_NewData/tables/`: new T7-filtered statistical tests.
- `../Bayes_OldData/bayesian/`: original-dataset posterior, same three sites.
- `../Bayes_OldData/copy_number/`: reconstructed AAV copy-number matrix.
- `results/comparison/`: pair-level tables, summaries, and figures.
- `results/comparison/figures/activity_concordance_ccc_vs_t7_cutoff.pdf`:
  combined cutoff-sensitivity figure.
- `results/comparison/tables/activity_concordance_ccc_vs_t7_cutoff.csv`:
  plotted CCC, Pearson, Spearman, pair counts, and mean activities.
- `results/comparison/figures/mean_celltype_ccc_vs_t7_cutoff.pdf` and
  `mean_ccre_ccc_vs_t7_cutoff.pdf`: unit-averaged cutoff-sensitivity figures.
- `results/comparison/tables/mean_celltype_ccc_vs_t7_cutoff.csv` and
  `mean_ccre_ccc_vs_t7_cutoff.csv`: plotted unit-averaged CCC summaries;
  corresponding `*_ccc_by_t7_cutoff.csv.gz` files contain every retained
  unit-level CCC and support count.
- `results/comparison/figures/unit_activity_correlation_scatter_t7_ge50.pdf`:
  per-cCRE and per-cell-type concordance scatter; the matching
  `tables/unit_activity_correlation_scatter_t7_ge50_values.csv` holds every
  retained unit's Pearson r, Spearman rho, Lin's CCC, mean activities, mean
  change, and support counts, and `*_manifest.json` records the pooled and
  panel-level summaries.
- `results/comparison/figures/raw_count_concordance_scatter_all_shared_pairs.pdf`
  and `raw_count_concordance_scatter_overlap_t7_ge50.pdf`: raw T7 and cCRE
  transcript-count scatters. `tables/raw_count_concordance_pair_counts.csv.gz` holds every
  pair's four raw totals with negative-control and overlap flags, and
  `tables/raw_count_concordance_manifest.json` records the per-panel
  correlations, totals, ratios, and zero-pair counts.
- `results/comparison/figures/overlap_t7_ge50_significant_call_concordance_raw_p.pdf`
  and `..._bh_q.pdf`: matched raw-p and BH call concordance on the mean
  negative-control reference; `shared_pair_significant_call_concordance_raw_p.pdf`
  and `..._bh_q.pdf` are the same pair on the mean+1 SD reference. Each prefix
  also gets `tables/*_significant_call_concordance_calls.csv.gz` (per-pair
  p-values, BH q-values, and both call statuses) and
  `tables/*_significant_call_concordance_summary.json` (metrics for both bases).
- `results/comparison/figures/replicate_concordant_bh_call_precision_recall.pdf`
  (and `.png`): precision/recall of the replicate-concordant calls of both
  methods against both assays, at T7 >= 5, 10, 20, 50 and 100.
  `tables/replicate_concordant_bh_call_precision_recall.csv` holds every
  TP/significant/assay-positive/tested count with the Fisher statistics,
  `..._calls.csv.gz` every pair's origin/new p-values, shared-universe BH
  q-values and call status per method and threshold, and `..._manifest.json` the
  common-pair counts, the four call-status counts, and the per-assay coverage of
  each method's concordant universe. `..._per_method_pairs*` is the same
  analysis without the cross-method intersection, kept because it is the arm
  that reproduces the saved T7 >= 50 concordance table.
- `results/comparison/figures/origin_vs_new_replicate_concordant_activity_heatmap_t7_ge50.pdf`:
  the same heatmap style restricted to the 925 replicate-concordant pairs
  (`bh_q_call_status` of `both_significant` or `neither_significant`) on the mean
  negative-control reference, with the usual `*_values.csv.gz` and
  `*_manifest.json` beside it.
- `results/comparison/bootstrap/`: overlap-filtered old-versus-new bootstrap
  tables, report, and figures.
- `results/logs/`: Slurm logs.

The submitted job IDs and final state are recorded in
`results/submitted_jobs.json`.
