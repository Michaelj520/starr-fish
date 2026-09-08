# Silencer annotation: cCRE x cell-type ChromHMM state coverage

Annotates the 390 STARR-FISH cCREs (`STARRFISH_in_vivo/Data/CRE.bed`) with the
per-cell-type ChromHMM 8-state segmentations in `revision/Data/ChromStates/`
(151 `*_8states_dense.bed` files, one per Allen subclass), and reports the
**repressive fraction** = fraction of each cCRE's bp covered by `Chr-R` or
`Hc-P` — the two states used here as the silencer/repressed signature.

## Input format

- `CRE.bed`: 5 columns — `chrom, start, end, category, name`. `category` is one
  of `subclass` (360), `region` (18), `Positive control` (12); `name` is
  `CRE###` and is unique. Most cCREs are 499 bp; positive controls vary
  (162-1580 bp), so coverage is always normalised by the cCRE's own length.
- ChromStates BEDs: dense (non-overlapping, gap-free) 9-column segmentations
  with a leading `track` line; state in column 4, one of
  `Chr-A, Chr-Pr, Chr-Po, Chr-O, Chr-R, Hc-P, Hc-H, ND`.

## Method

`code/annotate_chromstates.py` streams every segmentation once (one process per
file, `ProcessPoolExecutor`), and for each line intersects it against the cCREs
of that chromosome via a bisect lookup on sorted starts, accumulating overlap bp
into a `[n_cre, n_state]` int64 matrix. Fractions are `bp / cCRE length`.
Because the segmentations are non-overlapping, the repressive fraction is the
sum of the `Chr-R` and `Hc-P` fractions. Verified bp-exact against
`bedtools intersect -wao` on `027_L6b_EPd_Glut`.

**chrX is not segmented** in any ChromStates file, so the 14 chrX cCREs get
`NaN` (unmeasured), never 0. Summaries use NaN-aware reductions and report
`n_celltype_measured` / `n_cre_measured`.

**Caveat:** `ND` (no data) covers ~96-99% of the panel's cCRE bp in the two cell
types spot-checked, so most cCRE x cell-type entries are unmeasured *within* a
covered chromosome as well. Read the repressive fraction next to
`mean_ND_fraction` before calling anything non-repressive.

## Run

```bash
conda activate scvi
cd /gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
python revision/silencer/code/annotate_chromstates.py --workers 16
```

Defaults resolve to the paths above; override with `--cre-bed`,
`--chromstate-dir`, `--out-dir`. Runtime is I/O bound (~6 GB of BED text),
a few minutes with 16 workers.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `cre_by_celltype_repressive_fraction.csv` | **headline** 390 x 151 matrix, fraction of cCRE bp in `Chr-R` + `Hc-P` |
| `cre_by_celltype_<state>_fraction.csv` | same matrix for each of the 8 states individually (`Chr_R`, `Hc_P`, `ND`, ...) |
| `cre_by_celltype_state_fraction_long.csv.gz` | long form: `cCRE, cell_type, state, fraction` (390 x 151 x 8 rows) |
| `cre_repressive_summary.csv` | per cCRE: coordinates, category, mean/median/max repressive fraction across cell types, `n_celltype_repressive_gt50pct`, `mean_ND_fraction` |
| `celltype_repressive_summary.csv` | per cell type: mean repressive fraction, `n_cre_repressive_gt50pct`, `mean_ND_fraction` |

---

# Left-tail silencer test and precision/recall against the annotation

## Model change (shared package)

`baystarrfish.stats.negative_control_test` gained an `alternative` argument
(`"greater"` default = unchanged behaviour and unchanged `p_right`/`q_right`
columns; `"less"` = the silencer direction, emitting `p_left`/`q_left` and
`posterior_probability_below_control_reference`). The contrast itself is the
same draw-wise quantity; only the threshold sign and the tail side differ:

```
contrast_d = log_gamma[d, s, j] - mean_{controls} log_gamma[d, s, j'] + effect_threshold
p_left     = P(contrast >= 0)
q_left     = BH(p_left)
```

Verified against the shipped right-tail run
(`joint_dropout_direct_activity_mean_negative_control_tests_t7_ge50.csv.gz`):
identical contrasts to 4.8e-10 on all 3401 shared pairs, and
`p_left + p_right == 1` exactly.

## Scripts

| Script | Role |
|---|---|
| `code/test_silencer_left_tail.py` | left-tail test on the `revision/Bayes_OldData/bayesian` posterior (production joint+dropout fit, 1000 draws, 328 subclasses, 389 cCREs); BH within each T7 threshold |
| `code/silencer_truth.py` | loads the annotation as cell type x cCRE, mapping ChromStates filenames to Allen subclass names through the 3-digit subclass number and `abc_atlas/cluster_annotation_term.csv` (the filename's underscores are lossy, so the number is the only safe key) |
| `code/plot_silencer_precision_recall.py` | q-value figures + precision/recall vs the annotation at one T7 threshold, sweeping the truth cutoff (`--silencer-cutoffs`) |

```bash
conda activate scvi
cd /gpfs/commons/groups/ren_lab/guojiezhong/starr-fish
python revision/silencer/code/test_silencer_left_tail.py --t7-thresholds 50
PYTHONPATH=revision/silencer/code python \
    revision/silencer/code/plot_silencer_precision_recall.py \
    --t7-threshold 50 --silencer-cutoffs 0 0.1 0.2 0.5 0.8
```

## Definitions

- **Call**: `q_left <= 0.05` (BH over all 3401 eligible pairs at T7 >= 50). The
  call set is fixed across the sweep -- 760 of the 2290 annotated pairs.
- **Annotated silencer (truth)**: repressive fraction (`Chr-R` + `Hc-P`)
  **strictly greater** than the cutoff, swept over 0, 0.1, 0.2, 0.5, 0.8.
  `> 0` means "any repressive coverage at all"; 2011 of the 2290 pairs sit
  exactly at 0, so that is the only cutoff with ties to break.
- **precision** = TP / #significant, **recall** = TP / #annotated silencers, on
  the **all annotated** set: every tested pair whose cCRE lies on a chromosome
  present in that cell type's segmentation (150 of 328 subclasses, 382 of 389
  cCREs; chrX is `NaN` and excluded). Same convention as
  `revision/bayesian_vs_fold_change/.../method_activity_t7_filter_precision_recall.pdf`.
- **Baseline**: annotation prevalence among tested pairs (dashed) = the precision
  of calling everything. Enrichment is `precision / prevalence`, with a one-sided
  Fisher exact test. Ranking metrics use the depletion
  (`-effect_vs_control_reference_mean`) as the score.

## Result

The verdict does not depend on how permissive the silencer label is. At
T7 >= 50, on 2290 annotated pairs with 760 calls:

| cutoff | positives | prevalence | precision | recall | precision/prevalence | Fisher p | AP | AP/chance |
|---|---|---|---|---|---|---|---|---|
| > 0   | 279 | 0.122 | 0.128 | 0.348 | 1.05 | 0.30 | 0.137 | 1.13 |
| > 0.1 | 268 | 0.117 | 0.122 | 0.347 | 1.05 | 0.31 | 0.132 | 1.13 |
| > 0.2 | 265 | 0.116 | 0.122 | 0.351 | 1.06 | 0.26 | 0.131 | 1.13 |
| > 0.5 | 235 | 0.103 | 0.097 | 0.315 | 0.95 | 0.74 | 0.114 | 1.11 |
| > 0.8 | 194 | 0.085 | 0.086 | 0.335 | 1.01 | 0.49 | 0.100 | 1.18 |

Precision falls from 0.128 to 0.086 as the cutoff tightens, but only because the
prevalence falls with it: enrichment stays within 5% of chance everywhere and no
cutoff is significant (Fisher p >= 0.26). Recall is flat at ~0.35 -- the calls
cover about a third of annotated silencers at every cutoff, which is what
labelling a third of the panel at random would do. Ranking by depletion is
likewise flat (AP 1.11-1.18x chance), and the threshold-free version says the
same: Spearman rho = 0.028 (p = 0.18, n = 2290) between depletion and repressive
fraction, 0.019 (p = 0.36) using posterior evidence `1 - p_left`.

The relaxed cutoffs are the more sensitive test -- more positives, higher
prevalence -- and they also come out flat, so the null result is not a
small-positive-set artefact. Left-tail activity and ChromHMM repression are
essentially unrelated on this panel.

For reference, the earlier `ND < 0.5` restriction (dropped from the script) only
removed unmeasured negatives: it raised both precision and prevalence to ~0.61
and left enrichment at 1.03, i.e. the same conclusion.

## Outputs

| File | Contents |
|---|---|
| `results/silencer_left_tail_tests.csv.gz` | one row per eligible (cell type, cCRE, T7 threshold): effect, 90% interval, `p_left`, `q_left` |
| `results/silencer_left_tail_tests_annotated.csv.gz` | the T7 >= 50 rows that have an annotation, with `repressive_fraction` and `nd_fraction` |
| `results/silencer_left_tail_precision_recall.csv` | TP/FP/FN/TN, precision, recall, F1, prevalence, enrichment, Fisher p, tie count per truth cutoff |
| `results/silencer_left_tail_pr_curves.csv.gz`, `_pr_metrics.csv` | ranking-based PR curves and average precision per cutoff |
| `results/silencer_left_tail_benchmark_manifest.json` | provenance, coverage counts, rank association, all metrics |
| `figures/silencer_left_tail_precision_recall.pdf` | precision (vs prevalence), recall, and enrichment across the five cutoffs |
| `figures/silencer_left_tail_qvalues.pdf` | q-value histogram by class, q-value ECDF per cutoff, volcano coloured by repressive fraction |
| `figures/silencer_left_tail_pr_curves.pdf` | PR curves, one per cutoff, with per-cutoff chance lines |

---

# Two-experiment consistency filter, then precision/recall

Adds the second dataset and only scores the pairs both experiments agree on.
Truth cutoff is fixed at repressive fraction **> 0.1**, evaluation set is
**all annotated** pairs, eligibility is **T7 >= 50** in both arms.

| Arm | Posterior | Data |
|---|---|---|
| Bayesian, original data | `revision/Bayes_OldData/bayesian` | `scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad`, 328 subclasses, 3401 eligible pairs |
| Bayesian, new low-dose data | `revision/Bayes_NewData/bayesian` | `scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad`, 331 subclasses, 353k cells, 1366 eligible pairs |

Same model (joint copy-number + dropout, direct activity, ordinary negative
controls), same left-tail test, same T7 filter -- the arms differ only in the
experiment behind them. Both come from `code/test_silencer_left_tail.py`:

```bash
python revision/silencer/code/test_silencer_left_tail.py --t7-thresholds 50
python revision/silencer/code/test_silencer_left_tail.py \
    --bayes-dir revision/Bayes_NewData/bayesian \
    --h5ad revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad \
    --t7-thresholds 50 --stem silencer_left_tail_tests_newdata
PYTHONPATH=revision/silencer/code python \
    revision/silencer/code/plot_consistent_precision_recall.py \
    --silencer-cutoff 0.1 --t7-threshold 50 --q-cutoff 0.05
```

**Consistency filter**: keep pairs eligible in both arms whose call agrees --
`q_left <= 0.05` in both or in neither. On that subset "significant in both" and
"significant in either" coincide, so the call rule needs no further choice.

## Coverage

Annotated eligible pairs per arm: 2290 for the old data (28 cell types, 185
cCREs, 268 positives at > 0.1) and 1016 for the new data (142 positives). 1170
pairs are eligible in **both** arms (22 cell types, 117 cCREs), 876 of them
annotated (16 cell types, 114 cCREs, 134 positives); the consistency filter keeps
639 of those 876. Concordance is measured on the overlap, precision/recall on the
per-arm sets described under Result.

## The two experiments do agree

On the 876 annotated common pairs:

```
             new sig   new ns
original sig     158      187
original ns       50      481

call agreement 0.73 vs 0.57 expected by chance -> Cohen kappa = 0.39 (Fisher p = 5e-35)
effect Spearman rho = 0.63 (p = 1e-126)
right-tail positive control on the 1170 common pairs: OR 7.5, p = 2e-19
```

The original-data arm calls about twice as many pairs (447 vs 255 of 1170), so
most disagreement is one arm being more sensitive rather than the two pointing in
opposite directions.

## Result

Each single arm is scored on **its own** annotated eligible pairs; only the
consistent arm is restricted to the overlap. Prevalence therefore differs per
bar, so each bar is read against its own dashed baseline and `precision` is not
comparable across bars -- `prec/prev` and `AP/chance` are.

| arm | tested | positives | calls | precision | prevalence | prec/prev | Fisher p | recall | AP | AP/chance |
|---|---|---|---|---|---|---|---|---|---|---|
| Bayesian, original data | 2290 | 268 | 760 | 0.122 | 0.117 | 1.05 | 0.31 | 0.347 | 0.132 | 1.13 |
| Bayesian, new low-dose data | 1016 | 142 | 246 | 0.150 | 0.140 | 1.08 | 0.32 | 0.261 | 0.145 | 1.03 |
| Consistent (both arms agree) | 639 | 100 | 158 | 0.171 | 0.156 | 1.09 | 0.32 | 0.270 | 0.178 | 1.14 |

Requiring consistency does not change the verdict. Enrichment creeps from 1.05x
to 1.09x chance across the three arms and no arm is significant (Fisher
p = 0.31-0.32); ranking by depletion is 1.03-1.14x chance. The precision *level*
rises left to right (0.122 -> 0.150 -> 0.171) only because the sets get more
enriched in annotated silencers (prevalence 0.117 -> 0.140 -> 0.156): the calls
track the base rate, not the annotation.

The old-data row is identical to the `> 0.1` row of the cutoff sweep above (2290
pairs, 760 calls, 268 positives) -- same test, same evaluation set -- so the two
tables line up.

So the earlier null result is not explained by a noisy single experiment: two
datasets that agree with each other (kappa 0.39, rho 0.63 on their 876 shared
annotated pairs) still carry no usable signal about which cCREs are
ChromHMM-repressed.

## Outputs

| File | Contents |
|---|---|
| `results/silencer_left_tail_tests_newdata.csv.gz` | new-data left-tail tests (1366 eligible pairs at T7 >= 50, 303 significant) |
| `results/silencer_two_experiment_tests_annotated.csv.gz` | the 876 annotated common pairs with both arms' q-values, effects and the annotation |
| `results/silencer_two_experiment_precision_recall.csv` | the three-arm table above |
| `results/silencer_two_experiment_pr_curves.csv.gz`, `_pr_metrics.csv` | ranking curves and average precision per arm |
| `results/silencer_two_experiment_manifest.json` | provenance, coverage, concordance on all common and on annotated pairs, right-tail control, all metrics |
| `figures/silencer_two_experiment_concordance.pdf` | effect scatter coloured by call agreement, 2x2 call table with kappa, q-value histograms |
| `figures/silencer_two_experiment_precision_recall.pdf` | precision, recall, enrichment and AP/chance for the three arms |
| `figures/silencer_two_experiment_pr_curves.pdf` | PR curves per arm |
