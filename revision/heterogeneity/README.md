# Subclass heterogeneity via random and annotated subgroup splitting

This analysis probes activity-estimate stability by splitting the highest-T7
subclasses into random subgroups and asking how far the per-subgroup estimates
recover the intact-subclass estimate under the Bayesian joint+dropout model.
The parallel annotated-supertype analysis asks the same question using the
biological `supertype_name` labels already present in the input h5ad.

The original heterogeneity code and results were moved to
`revision/archive/heterogeneity_old/`.

## Design

1. **Top subclasses** — the 10 subclasses with the largest `total_t7` in
   `revision/Data/subclass_total_t7_counts.csv` (names standardized to match the
   model's `subclass` labels: Allen numeric prefix dropped, `/`→`-`).
2. **Split** — within each top subclass, the combined sec1+sec2 cells are
   randomly partitioned into 5 subgroups (`<subclass>_group_<i>`) with a fixed
   seed (`relabel.SPLIT_SEED`, independent RNG stream per subclass). Only the
   `subclass` label changes, so the five subgroups stay nested under the same
   parent `class`. All other subclasses are kept intact and merged back in.
3. **Fit** — `run_bayes_split.py` fits the split labels with the production
   joint CRE/T7 copy-number+dropout model: direct activity, ordinary annotated
   negative controls, default Beta(1, 9) dropout priors, SVI 30k steps, and
   1000 posterior draws.
4. **Agreement metric** — the **intact-subclass** Bayesian estimate is reused
   from `revision/bayesian_vs_fold_change/results/bayesian` (no recompute).
   For every cCRE and each of the 10 selected cell types,
   `make_heterogeneity_plots.py` compares:
   - x = negative-control-centred activity in the intact cell type;
   - y = the unweighted mean activity across its 5 random cell subsets.
   Each panel is one cell type and contains all fitted cCREs as points; vertical
   bars show the SD across the 5 subset estimates. Panels also include the y=x
   line, Lin's concordance correlation coefficient (CCC), and mean absolute
   error (MAE), and are ordered from largest to smallest original cell count.
   Bootstrap is not used.

## Annotated-supertype design

The parallel workflow uses the same input h5ad, top-ten subclasses, intact
Bayesian reference, model configuration, calibration, and cCRE set. For cells
in each target subclass, the Allen numeric prefix is removed from
`obs["supertype_name"]` and that standardized label becomes the model's
subclass label. Cells outside the ten targets retain their original subclass.

The ten targets contain 5, 1, 6, 14, 5, 8, 5, 9, 4, and 6 annotated
supertypes, respectively (63 total). Every annotated supertype is retained,
including the smallest five-cell group. The plotted y-value is the
**cell-count-weighted mean** of the supertype activity estimates. This makes the
composition of the comparison match the intact cell type and prevents a
five-cell supertype from contributing as much to the agreement point as a
large supertype.

The vertical bars deliberately remain the **unweighted sample SD across
supertypes**. They describe between-supertype biological heterogeneity, not
posterior uncertainty or uncertainty in the weighted mean. Each panel reports
the median of these supertype SDs as a compact heterogeneity summary. The
unweighted mean is also retained in the output table, which makes it possible
to contrast a typical annotated supertype with the abundance-weighted intact
population. `Endo NN` has one annotated supertype, so its points and agreement
metrics are shown without SD bars and its supertype SD is unavailable.

To measure heterogeneity directly, every unordered pair of annotated
supertypes is compared **within its parent cell type**. Lin's CCC is calculated
between the two inferred activity vectors across all 389 fitted,
non-blacklisted cCREs. Each pair contributes once to its parent distribution;
lower CCC indicates more divergent inferred activity profiles. The pair table
retains both supertype cell counts because low CCC involving a very small group
may reflect greater estimation noise as well as biological heterogeneity.
`Endo NN` has no pairwise value because one supertype cannot form a pair.

## Size-matched random null (`supertype_like_random`)

The annotated-supertype pairwise CCC mixes two effects: real between-supertype
biology and estimation noise, which grows as a group shrinks. The null run
removes the first and keeps the second. Within each of the ten target
subclasses the cells are shuffled once and cut, **without replacement**, into
blocks whose sizes are exactly the annotated supertype sizes of that subclass
(63 groups in total, identical size profile per parent, seeded by
`relabel.SPLIT_SEED`). Group `i` of a parent carries the cell count of that
parent's `i`-th annotated supertype and is labelled
`<subclass>_random_<i>`; the matched supertype is recorded in
`cell_group_assignment.csv` for traceability.

The fit, calibration, cCRE set, intact reference, and every plotting step are
otherwise identical to the annotated-supertype workflow, so
`results/supertype_like_random/figures/bayesian_supertype_pairwise_ccc.pdf`
is directly comparable panel-for-panel with the annotated version under
`results/supertype/`. Pairwise CCC that is high in the null but low in the
annotated run is biological heterogeneity; pairwise CCC that is low in both is
estimation noise at that cell support.

```bash
bash revision/heterogeneity/code/submit_supertype_like_random_all.sh
```

### Paired test against the annotated supertypes

Once both fits exist, the annotated pairwise-CCC figure is redrawn with the
null beside it:

```bash
python revision/heterogeneity/code/make_heterogeneity_plots.py \
  --split-bayes-dir revision/heterogeneity/results/supertype/bayesian \
  --null-split-bayes-dir revision/heterogeneity/results/supertype_like_random/bayesian \
  --outdir revision/heterogeneity/results/supertype
```

`figures/bayesian_supertype_pairwise_ccc.pdf` then shows two boxes per cell
type — annotated supertype pairs (red) and their size-matched random
counterparts (blue). Annotated pair `(i, j)` is matched to the null pair built
from the random groups carrying the same two cell counts, so the comparison is
paired and a paired t-test on the CCC difference is well defined. The join is
keyed on membership ordinals and the two cell counts are re-checked afterwards,
so a mismatched null raises rather than silently comparing different-sized
groups.

CCC is bounded and its marginal distribution is skewed (Shapiro-Wilk
p ~ 0.003 for both the annotated and the null CCCs), so the **Wilcoxon
signed-rank test is the primary statistic**, with the Hodges-Lehmann shift and
the rank-biserial correlation as effect sizes. Three further results are
reported per cell type: an exact sign test (assumes only that a difference is
equally likely to fall either way), a sign-flip randomization test (assumes
only that the annotated and null member of a pair are exchangeable, enumerated
exactly at n <= 20 and sampled with 20,000 draws above that), and the paired
t-test with a Shapiro-Wilk check on the differences it actually assumes. The
differences are in fact close to normal (W = 0.995, p = 0.66 pooled), so the
four tests agree; the rank-based ones are reported because that agreement is
an empirical finding, not something the design guarantees.

**The pairs are not independent** — each supertype appears in many pairs — so
every pair-level p-value is anticonservative. The conservative companion
collapses each cell type to its median difference and tests the nine resulting
independent values with an exactly enumerated sign-flip randomization
(`bayesian_supertype_pairwise_ccc_cell_type_test.csv`). That is the number to
quote when the claim is about cell types in general rather than about these
particular supertype pairs.

New tables under `results/supertype/tables/`:

- `bayesian_supertype_pairwise_ccc_null.csv` — the null pairwise CCCs recomputed
  under this run's calibration and cCRE set.
- `bayesian_supertype_pairwise_ccc_vs_null.csv` — one row per matched pair with
  both CCCs and their difference.
- `bayesian_supertype_pairwise_ccc_null_tests.csv` — per-cell-type and pooled
  Wilcoxon, sign, randomization, and t-test p-values with Hodges-Lehmann shift,
  rank-biserial correlation, Cohen's dz, and the Shapiro-Wilk check.
- `bayesian_supertype_pairwise_ccc_cell_type_test.csv` — the cell-type-level
  test that treats each cell type as one independent unit.

Outputs mirror `results/supertype/` file-for-file (same table, summary, and
figure names) under `results/supertype_like_random/`.

## Run

```bash
bash revision/heterogeneity/code/submit_all.sh
```

Submits the Bayesian fit (1 GPU, 96 GB) and its dependent plotting job
(`afterok`). To run the plot alone once the fit exists:

```bash
sbatch revision/heterogeneity/code/submit_plots.slurm
```

Local smoke tests: the fit script accepts `--max-cells` / `--max-cres`.

`make_heterogeneity_plots.py` accepts:

- `--calibration negctrl_only` (default): negative-control centering only,
  retaining a common per-cCRE scale across the two independent fits.
- `--ncols 5`: control the cell-type panel layout.

To submit the independent annotated-supertype fit and its dependent plot:

```bash
bash revision/heterogeneity/code/submit_supertype_all.sh
```

The underlying fit command uses `run_bayes_split.py --grouping supertype`.
The default remains `--grouping random`, so existing commands and output paths
retain their original behavior.

To run the same annotated-supertype comparison with the manuscript bootstrap
estimator instead of the Bayesian model:

```bash
bash revision/heterogeneity/code/submit_supertype_bootstrap_all.sh
```

This runs 10,000 bootstrap iterations with 62 workers, writes the fit under
`results/supertype/bootstrap/`, and then reuses the intact-subclass bootstrap
from `revision/Bootstrap_OldData`. The plot applies negative-control-only
centering to both runs, matching the Bayesian figure's cross-fit calibration.

## Outputs (`results/`)

- `split/bayesian/` — full Bayesian fit results on the split labels.
- `tables/bayesian_subset_vs_whole.csv` — intact activity, mean subset
  activity, subset SD, and differences for every cell-type/cCRE pair.
- `tables/bayesian_subset_vs_whole_summary.csv` — per-cell-type and overall CCC,
  Pearson correlation, mean error, MAE, and RMSE.
- `raw/combined_activity_bayesian.csv` and
  `raw/split_activity_bayesian.csv` — activity matrices actually compared.
- `figures/bayesian_subset_mean_vs_whole.{pdf,png}` — one panel per cell type,
  with all fitted cCREs as points, subset-SD bars, and a diagonal y=x line.
- `logs/` — Slurm stdout/stderr.

## Annotated-supertype outputs (`results/supertype/`)

- `bayesian/` — Bayesian fit using the 63 annotated supertype labels, including
  `cell_group_assignment.csv`, `subgroup_cell_counts.csv`, and a manifest with
  the complete parent-to-supertype mapping.
- `tables/bayesian_supertype_vs_whole.csv` — intact activity, cell-count-weighted
  and unweighted mean supertype activity, unweighted supertype SD,
  contributing-supertype count, and both weighted and unweighted differences
  for every cell-type/cCRE pair.
- `tables/bayesian_supertype_vs_whole_summary.csv` — per-cell-type and overall
  weighted-agreement metrics plus median/mean supertype SD.
- `tables/bayesian_supertype_pairwise_ccc.csv` — one row per within-parent
  supertype pair, with CCC, Pearson correlation, MAE, RMSE, and both supertype
  cell counts.
- `tables/bayesian_supertype_pairwise_ccc_summary.csv` — pairwise-CCC
  distribution summaries for each parent cell type.
- `raw/combined_activity_bayesian.csv` and
  `raw/supertype_activity_bayesian.csv` — activity matrices actually compared.
- `figures/bayesian_supertype_mean_vs_whole.{pdf,png}` — one panel per target
  cell type, with cell-count-weighted cCRE means, unweighted supertype-SD bars
  where defined, and y=x.
- `figures/bayesian_supertype_pairwise_ccc.{pdf,png}` — within-parent pairwise
  CCC distributions; each red point is an annotated-supertype pair. The blue
  diamond for each parent is the CCC of its cell-count-weighted supertype mean
  versus the intact whole-cell-type activity, providing the aggregate-agreement
  baseline from the mean-vs-whole figure.
- `figures/bayesian_supertype_pairwise_ccc_vs_min_cells.{pdf,png}` — pairwise
  CCC against the smaller supertype's cell count on a log x-axis, colored by
  parent cell type and annotated with the overall Spearman association.
- `bootstrap/` — annotated-supertype manuscript bootstrap fit, including the
  10,000-iteration activity array and the same subgroup mapping metadata.
- `tables/bootstrap_supertype_vs_whole.csv` and its `_summary.csv` companion —
  bootstrap weighted-agreement data and supertype-heterogeneity metrics.
- `figures/bootstrap_supertype_mean_vs_whole.{pdf,png}` — the bootstrap version
  of the annotated-supertype agreement figure.
- `figures/bootstrap_supertype_pairwise_ccc.{pdf,png}` — the corresponding
  bootstrap within-parent pairwise-CCC distributions.
- `figures/bootstrap_supertype_pairwise_ccc_vs_min_cells.{pdf,png}` — bootstrap
  pairwise CCC against minimum pair cell support.
