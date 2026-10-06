# run_Bayes

Fits the posterior for each dataset and exports the canonical tables every
activity figure reads.

## Fitting

- `run_bayes.py`: the SVI fit. `submit_new_bayesian.slurm` runs it for the new
  low-dose dataset into `../Bayes_NewData/bayesian/`; the original run's
  posterior lives in `../Bayes_OldData/bayesian/`.

## Activity matrices (the figure inputs)

- `export_activity_matrix.py`: reduces one posterior to the tables below.
- `activity_matrix_io.py`: owns the file-name convention and the readers, so the
  writer and the plotting scripts cannot drift apart.
- `submit_activity_matrices.slurm`: runs the export for both datasets, for the
  mean-control null and the mean+1SD null.
- `test_activity_matrix_export.py`: the contract tests. The fast half runs on a
  synthetic fit and always runs; the rest checks the real exports and skips until
  they exist. Run after any change here:
  `pytest revision/run_Bayes/test_activity_matrix_export.py -q`

Each dataset gets, in its own `tables/` directory:

| file | content |
|---|---|
| `subclass_cre_activity_matrix.csv.gz` | subclass x target cCRE, **the value the heatmaps colour**: posterior mean target `log_gamma` minus the posterior mean of the seven ordinary negative controls |
| `subclass_cre_beta_t7_activity_matrix.csv.gz` | subclass x **every ordinary** cCRE, controls included: posterior mean `log_gamma` minus `mean(log(beta_t7))` — the scale the method-comparison figures plot |
| `subclass_cre_p_value_matrix.csv.gz` | subclass x target cCRE, `p_right`, unfiltered |
| `subclass_cre_q_value_matrix_t7_ge50.csv.gz` | subclass x target cCRE, BH `q` over this dataset's T7 >= 50 pairs, `NaN` outside |
| `subclass_cre_target_cre_matrix.csv.gz` | subclass x target cCRE raw `obsm["CRE"]` totals |
| `subclass_cre_target_t7_matrix.csv.gz` | subclass x target cCRE raw `obsm["T7CRE"]` totals — the source of every T7 mask |
| `subclass_cre_negative_control_activity_matrix.csv.gz` | subclass x 7 controls, posterior-mean `log_gamma`, for the control-spread strip |
| `subclass_cre_significance.csv.gz` | one row per pair: `p_right`, `q_right_t7_ge50`, the effect interval, and the CRE, T7 and cell counts |
| `subclass_cre_matrix_manifest.json` | provenance, shapes, `mean_log_beta_t7`, and the cross-check against the previously shipped test table |

`subclass_cre_mean_plus_1sd_*` are the same products under the stricter
mean+1SD control reference. Read them through `activity_matrix_io.load_dataset`
rather than by name; `matrix_paths` is the only place a filename is spelled out.

### Two activity scales, on purpose

Both come from the same `log_gamma` draws and differ only in what is subtracted.

- **Control-centered** (`_activity_matrix`) subtracts the control mean *within each
  subclass*. That is the biological effect against a null measured in the same
  cells, so it is what the heatmaps colour and what every significance call tests.
- **beta_t7-referenced** (`_beta_t7_activity_matrix`) subtracts one global scalar,
  `mean(log(beta_t7))`. Each method keeps its own per-subclass offset, which is
  exactly what makes different models comparable on a single axis — centering each
  one against its own controls would absorb the difference being measured.

The beta_t7 matrix is the one output that keeps the negative-control columns: it
carries no test, and the method-comparison scatters plot the controls alongside
the targets. Everything else is target-only.

### Axes

The matrices are **unfiltered**: every fitted target pair is exported, `NaN`
only where the model has no pair. Figures apply their own T7 mask from the T7
matrix, which is why one export serves every threshold.

The posterior already excludes the blacklisted cCREs, so for the original dataset
the axes are 328 subclasses x 382 targets, and 328 x 389 for the beta_t7 matrix
(382 targets + 7 controls).

### The q universe

`p_right` is universe-free and is stored as computed. BH `q` is not — it depends
on which pairs are in the family — so `q_right_t7_ge50` is BH over *this*
dataset's T7 >= 50 pairs and is `NaN` outside it. A figure plotting a different
family (the shared original/new universe, for instance) must re-run BH over that
family from `p_right`; `plot_origin_vs_new_heatmap.py` does exactly that and is
the only place where `q` is derived rather than read.

`activity_matrix_io.q_column_for` / `call_column_for` / `t7_token` build the
column and filename tokens from the threshold, so the writer and every reader
agree on the spelling.

## Copy-number folder (per-cell quantities)

`submit_copy_number_matrix.slurm <old|new>` writes `<dataset>/copy_number/`:

| file | content |
|---|---|
| `copy_number.npz` | `E[k \| t7, cre]` mean and sd per (cell, cCRE), float32 |
| `copy_number_copies.csv.gz` | the mean, cells x cCREs |
| `activity_normalized.npz` | copies, Gamma-conjugate posterior activity, and that activity divided per draw by the cell type's control reference (`NaN` below pooled control T7 50) |
| `activity_posterior.npz` | the posterior activity alone (`extract_activity_posterior.py`) |
| `negative_control_reference_check.csv` | each per-cell estimator on the 7 controls against its parameter-level target (`check_negative_control_reference.py`) |

All four products use the same 200 evenly-spaced draws (`MAX_DRAWS`). Files
already present are skipped, so the old dataset is a no-op and an interrupted
job can be resubmitted. Rerunning the two scripts on the old fit reproduces its
shipped files: `activity_posterior.npz` bit for bit, the check table to 6e-8
relative (float32 rounding).

## Raw count aggregation

The subclass x cCRE count totals come from `baystarrfish.data.read_grouped_counts`,
which streams `obsm/CRE` and `obsm/T7CRE` one cCRE at a time and sums over the cells
of each subclass (excluding cells with an unassigned `subclass_name` or `class_name`).
It is the single definition of that aggregation;
`test_individual_negative_control_loo_empirical_fdr.load_grouped_t7` is now the
T7-only view onto it.

*Follow-up:* `origin_vs_new/code/plot_raw_count_concordance_scatter.grouped_pair_counts`
still carries its own copy of the same loop and should be collapsed onto
`read_grouped_counts` when that folder is next touched.

## Consumers

- `../origin_vs_new/code/plot_origin_vs_new_heatmap.py` — both panels, both
  datasets, shared universe.
- `../bayesian_vs_fold_change/code/plot_method_activity_heatmap.py` via
  `--activity-matrix-dir` — the published `joint_dropout_activity_heatmap`
  figures.
- `../bayesian_vs_fold_change/code/plot_method_activity_correlation.py` — reads
  `_beta_t7_activity_matrix` for each of the four arms in `../Bayesian_ablation/`,
  instead of reopening their posteriors. One of those arms, `bayesian_joint_dropout`,
  is a symlink to `../Bayes_OldData`, so the production fit appears in the matrix as
  `Joint+dropout` rather than as a separate method.
  `figure_work/plot_stripe_count_diagnostics.py` shares that loader.
