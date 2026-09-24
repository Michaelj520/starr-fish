# SFv6 multi-barcode activities

## GLM activity (current plot)

`compute_glm_activities.py` reproduces the estimator in
`STARRFISH_in_vitro/glm.py::glm_fit_total`, with `norm_by_vector=False`,
`norm_by_nanopore=True`, and `use_fov_covariate=False`:

```text
RNA[cell, barcode] = intercept[barcode] + slope[barcode] * total_RNA[cell] + error
activity[barcode] = slope[barcode] / ln(DNA_count[barcode])
log_activity[barcode] = ln(activity[barcode])
```

Despite its name, the original function calls `smf.ols`, so this is unweighted
Gaussian regression with an identity link, rather than a Poisson or negative
binomial count GLM. Total RNA includes all 300 reporter barcodes, including
the target itself. An intercept is fitted independently for each barcode.
There is no FOV adjustment, T7 normalization, CPM normalization, or control
centering. The result is an empirical regression score, not an RNA/DNA fold
change; it depends on the raw DNA count scale through `ln(DNA_count)`.

The implementation solves the same OLS equations at full precision rather
than parsing rounded HTML regression summaries. Slope standard errors are
classical OLS estimates; activity standard errors divide them by the same
`ln(DNA_count)` factor (treating DNA as fixed). This corrects the original
function's unscaled standard-error output. Nonpositive slopes have no log
activity. DNA <= 1, constant total RNA, and fewer than three cells are flagged.
These rows and their available raw estimates remain in the tables.

Run with the `bayes-jax` Python environment used below:

```bash
python revision/barcode_shuffling/compute_glm_activities.py
python revision/barcode_shuffling/plot_activities.py --method glm
```

GLM tables and provenance are in `results/glm/`. Both global and within-Leiden
fits are exported at barcode and pooled CRE levels. For a pooled CRE, RNA
and DNA counts are summed over its barcodes, and its RNA is regressed on the
original total RNA per cell. The separate `mean_barcode_log_activity` column
averages individual barcode log activities and propagates invalid values.

The current plot defaults to GLM activity and is saved as
`figures/glm/global_ccre_barcode_glm_activity_boxplot.{png,pdf}`. It shows
one point per eligible barcode, ordered by increasing mean plotted log
activity per CRE, with controls last. Following the original plotting code,
DNA counts must be > 10; `--min-dna` changes this threshold. The supplied data
exclude `CRE234_bar115` (DNA count 7), leaving 299 points, four for CRE234 and
five for every other construct. Nonfinite log activities are also excluded.
The plot CSV retains every barcode, its inclusion flag, exclusion reason,
and jitter position. There is no zero reference line for this regression score.

## RNA/DNA fold-change activity

`compute_activities.py` computes log-fold-change activities using RNA from
`revision/Data/scdata_260821_SFv6_multiBC.h5ad` and nanopore DNA from
`revision/Data/STARR_seq_Counts_wReps_DNA.csv`. The RNA input has 27,889 cells,
300 CRE–barcode features (59 CREs plus the barcode-only control group, each
with five barcodes), and 44 Leiden clusters. Counts come from
`obsm['X_raw']`, whose row sums match `obs['total_counts']`; `X` is transformed
and is not used. There are no T7 measurements or annotated cell types in this
file. Leiden labels are clusters derived from these reporter features.

## Definition

For group g and barcode b, sum RNA counts across its cells to obtain R[g,b].
Let D[b] be the corresponding `DNA_count` in the CSV. Normalize each library
over the same 300 features, including the five barcode-only controls:

```text
RNA_CPM[g,b] = 1e6 * R[g,b] / sum_b R[g,b]
DNA_CPM[b]   = 1e6 * D[b]   / sum_b D[b]
barcode activity[g,b] = ln(RNA_CPM[g,b] / DNA_CPM[b])
```

The CSV has 300 unique barcodes and 2,325,338 DNA reads. Matching uses full
element identities, with the explicit alias `barcode_only_barNNN` in the DNA
table to `barcode only_barNNN` in the RNA data. Missing/extra/duplicate
elements cause an error. DNA CPM is recomputed from `DNA_count`; it agrees
with the supplied `CPM_D` column. The CSV's `count1`, `count2`, and precomputed
activity columns are not used: the numerator is the H5AD RNA data.

For the CRE-level tables, `log_fold_change` pools RNA and DNA across a CRE's
barcodes before computing their CPM ratio. The separate
`mean_barcode_log_fold_change` column is the arithmetic mean of individual
barcode log activities, matching the plot's ordering. These summaries differ
because log and averaging do not commute and pooled ratios weight barcodes
by DNA abundance. All cells contribute; global RNA counts are pooled directly.
For grouped results, RNA CPM is normalized within each Leiden cluster using
the same input DNA library for every cluster.

**This RNA/DNA definition replaces the earlier RNA-only control ratio.**
Barcode-only controls are retained and marked `is_control` for display;
they are not used to center or normalize the activity.

Natural logs and no pseudocount are the defaults. `--log-base 2` requests log2.
With no pseudocount, zero RNA gives `-inf`. Zero DNA or an empty RNA library
gives `NaN`, even with a pseudocount. `--pseudocount P` adds P to both CPM
values after library normalization. There is no T7 or negative-control
normalization. DNA corrects for input-library abundance, not cell-specific
construct delivery. No barcode permutation, bootstrap, significance testing,
or cell-type annotation is performed.

## Run

From the repository root, in an environment with anndata, numpy, pandas, and scipy:

```bash
python revision/barcode_shuffling/compute_activities.py
```

The environment used for verification is:

```bash
/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python revision/barcode_shuffling/compute_activities.py
```

Options include `--h5ad`, `--dna-csv`, `--outdir`, `--groupby` (default `leiden`),
`--log-base`, and `--pseudocount` (in CPM units).
Re-running replaces the generated tables in the selected output directory.

## Outputs

All outputs are written to `results/` by default:

- `{global,grouped}_{barcode,cre}_activities.csv`: long tables containing
  activity, raw RNA/DNA counts, library totals, RNA/DNA CPM, cell and barcode
  counts, positive-cell counts, and control flags. CRE tables also include
  the mean barcode log activity.
- `{global,grouped}_{barcode,cre}_log_fold_change.csv`: activity matrices
  with groups as rows and barcodes or CRE labels as columns. Global rows are
  labeled `all`; grouped rows use the selected observation column.
- `feature_metadata.csv`: feature-to-CRE/barcode mapping, matched DNA counts,
  DNA CPM, and control flags.
- `run_manifest.json`: RNA/DNA source files, dimensions, grouping,
  formula, log base, pseudocount, and zero handling.

Tests check hand-calculated RNA/DNA ratios, library-depth invariance, DNA
alignment and aliases, invalid inputs, zero RNA/DNA, and sparse/dense equivalence:

```bash
python -m unittest discover -s revision/barcode_shuffling -p 'test_*.py'
```

## Barcode boxplot

After computing activities, run (also requires matplotlib):

```bash
python revision/barcode_shuffling/plot_activities.py --method rna-dna
```

`figures/global_ccre_barcode_activity_boxplot.{png,pdf}` shows cCREs on the
x-axis and global log-fold-change activity on the y-axis. Each point is one
barcode's activity across all cells; each box summarizes the five barcodes
for that cCRE (median, quartiles, and whiskers within 1.5 IQR). All points,
including outliers, appear once in the jitter layer. The barcode-only control
group is shown at the right in orange. The dashed zero line means RNA CPM
equals DNA CPM; controls are not forced to zero. The adjacent CSV records
the plotted values and jitter
coordinates and category order. The default (`--order mean`) orders cCREs
by the arithmetic mean of their five barcode log activities, lowest first,
with controls last. `--order median` uses median barcode activity and
`--order name` uses cCRE name order.
