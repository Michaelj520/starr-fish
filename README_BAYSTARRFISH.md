# BAYSTARRFISH

A STARR-FISH in a bay. `BAY` carries the Bayes.

Bayesian hierarchical inference of cis-regulatory element (cCRE) activity from
single-cell [STARR-FISH](README.md) data — an imaging-based reporter assay that
measures the transcriptional capacity of enhancer reporter constructs in a
multiplexed, spatially resolved manner at single-molecule resolution.

---

## The problem

Each AAV carries a candidate cCRE driving a reporter plus a barcode. Per cell and
per barcode you observe two transcript counts — **T7** (constitutive, calibrates
infection) and **cCRE** (enhancer-driven). Neither is a DNA measurement.

The number of AAV genomes `k` that infected a cell for a given barcode is
**unobserved**, and infection is very rare: 99.2% of T7 entries and 99.85% of
cCRE entries are zero. A fold change over that background is dominated by which
cells happened to be infected, not by enhancer strength.

## The model

Treat `k` as a latent variable and integrate it out.

```
latent copies   k_ij      ~ Poisson(lambda_sj),  lambda_sj = rho_s * a_j

T7 channel      t7_ij  | k ~  delta_0                                                  if k = 0
                              ZINB(drop = d_t7,  mean = k * beta_t7,   disp = phi_t7)  if k > 0

cCRE channel    cre_ij | k ~  delta_0                                                  if k = 0
                              ZINB(drop = d_cre, mean = k * gamma_sj,  disp = phi_cre) if k > 0

where  ZINB(drop = d, mean = m, disp = phi)  =  d * delta_0 + (1 - d) * NB2(mean = m, disp = phi)
```

`delta_0` is the point mass at zero: an uninfected cell gives probability exactly
1 to a zero and 0 to any positive count, never "an NB with mean 0". So an
observed zero is one of three things — an uninfected cell (`k = 0`, probability
`exp(-lambda_sj)`, carrying no information about `gamma`), an infected cell whose
transcript was missed by the measurement (probability `d_c`), or an infected cell
that genuinely sampled a zero from the NB. Keeping those apart is what stops rare
infection from masquerading as low activity.

The zero inflation sits inside the emission distribution and only at `k > 0`: it
describes the *measurement*, while `exp(-lambda_sj)` describes the *biology*.
Inflating the `k = 0` row as well would double-count the same zero and let `d_c`
absorb the infection rate.

Both channels share the same `k`, so the zero cases are coupled: `k = 0` forces
`t7_ij = 0` **and** `cre_ij = 0` simultaneously, while `(t7 > 0, cre = 0)` is
only reachable through the `k > 0` zero branch. That coupling is what identifies
`d_cre` separately from `gamma`.

With `infection_model="copy_number"`, `d_t7 = d_cre = 0` and each ZINB collapses
to the plain NB2. With `infection_model="copy_number_dropout"` (the production
fit) each channel carries one global `d_c ~ Beta(1, 9)`.
`infection_model="binary"` replaces the copy count with a shared Bernoulli gate,
`p_infected = 1 - exp(-rho_s a_j)`: the `k = 0` row becomes the "not infected"
branch and both channels share one gate instead of one `k`.

Around that observation model sit:

- a two-level **class → subclass** hierarchy on the infection rate `rho` and,
  optionally, on the activity `gamma`;
- an informative **nanopore library prior** on the per-cCRE abundance `a`, mean-
  centred to fix the otherwise unidentifiable shared scale with `rho`.

`k` is marginalised analytically by `logsumexp` over a truncated grid `0..kmax`,
so the target is a smooth continuous-parameter model that NumPyro fits by SVI
(production) or NUTS (calibration).

**Scalability.** The 408,621 × 400 observation array collapses to weighted unique
`(group, cre, t7, cre)` rows: the all-zero pairs share an identical marginal
within a cell type, so they become a single weighted row. The likelihood enters
as one `numpyro.factor`, not a plate.

## Install

```bash
pip install -e .            # core
pip install -e '.[gpu]'     # + CUDA 12 wheels for fitting
pip install -e '.[dev]'     # + pytest, statsmodels (test references)
pip install -e '.[repro]'   # exact pins of the published fits
```

`envs/baystarrfish-gpu.yml` and `envs/baystarrfish-gpu.lock.txt` reproduce the
environment that produced every published fit: python 3.11.15, jax 0.10.1 with
CUDA 12.9, numpyro 0.21.0.

Double precision is mandatory — the marginal mixes `~1e-6` rates with `-inf`
point masses inside a `logsumexp`, and in float32 the gradients go `nan`.
Importing `baystarrfish` sets `JAX_ENABLE_X64=1` before JAX is imported at all.

## Use

```python
import baystarrfish as bsf

data = bsf.data.CountData.from_h5ad(section="all", negative_control_mode="ordinary")

fit = bsf.fit(
    **data.to_run_kwargs(),
    level="subclass",
    channel="joint",
    infection_model="copy_number_dropout",
    activity_model="direct",
    num_steps=30_000,
    num_posterior=1_000,
)

bsf.io.write_fit(fit, "results/bayesian", tag="subclass_joint_copy_number_dropout_svi",
                 data=data)
```

Calling activity, from the posterior draws:

```python
from baystarrfish.io import load_posterior_samples
from baystarrfish.stats import negative_control_test

post = load_posterior_samples("results/bayesian", sites=["log_gamma"])
calls = negative_control_test(post["log_gamma"], ..., t7_threshold=50.0,
                              effect_threshold=0.0, method="joint+dropout")
```

`negative_control_test` contrasts each target against the mean of the negative
controls **inside each posterior draw**, so the reference's own uncertainty is
subtracted rather than treated as a fixed offset, then reports `p_right` (the
posterior tail probability) and its BH-adjusted `q_right`.

## Recovering the copy number: how many AAV genomes reached each cell

`k_ij` is what the model is built around and the one thing it never samples — it
is marginalised so inference stays in continuous parameters. It is recoverable
exactly afterwards, because the grid the fit summed over renormalises into a
distribution:

```python
from baystarrfish.data import CountData
from baystarrfish.inference import infer_copy_number_from_fit

data = CountData.from_h5ad(section="all", negative_control_mode="ordinary")
copies = infer_copy_number_from_fit(data, "results/bayesian_full_posterior", return_sd=True)

copies.copies             # (n_cells, n_cre) float32: E[k_ij | t7_ij, cre_ij]
copies.sd                 # same shape: posterior sd
copies.p_infected         # P(k >= 1 | obs), with return_probability=True
copies.total_per_cell()   # expected genomes per cell, summed over cCREs
copies.to_frame("copies") # labelled DataFrame (dense — 1.3 GB at full size)
copies.write_npz("results/copies.npz")
copies.write_csv("results/copies.csv.gz", "copies")
```

`copies` is the expected **count**; `p_infected` is a **probability**. They answer
different questions and can disagree — a pair can be certainly infected yet
expected to carry few copies. Thresholding `copies` is not an infection call.

From the command line, which is what you want for the full matrix:

```bash
python -m baystarrfish copy-number \
    --fit-dir revision/Bayes_OldData/bayesian \
    --out results/copy_number.npz \
    --csv results/copy_number --csv-which copies \
    --with-sd --max-draws 200
```

or, for the two refit datasets, the runner that wraps exactly that:

```bash
sbatch --job-name=cn_origin \
    revision/run_Bayes/submit_copy_number_matrix.slurm origin
```

CSV is written in row blocks so 159 million values never exist as text in
memory, gzipped when the path ends in `.gz`, and rounded (5 decimals is already
far finer than the posterior is sharp). Each full matrix is ~1.3 GB of text, so
`--csv-which` defaults to the posterior mean alone; `sd` still lands in the npz.

`kmax`, the cell-type granularity and the infection model are read from the fit's
manifest, so the reconstruction cannot silently disagree with the posterior it
came from.

**This is not the counts.** A cell reading `t7 = 0, cre = 0` is not known to be
uninfected — under dropout, or simply a low per-copy rate, an infected cell often
reads zero, so those pairs get a nonzero baseline that depends on the cell type's
infection rate and that cCRE's library abundance. A cell reading `t7 = 12` did
not receive twelve genomes. And `copies` is a posterior **mean over an integer**:
`0.02` means "almost certainly zero copies", not a fiftieth of a virus.
Thresholding it is not an infection call — `P(k >= 1)` is a different quantity.

**It needs `log_rho` and `log_a`, which fits before 2026-08 did not save.** The
copy number depends on infection and abundance, not just activity, and neither
site can be reconstructed after the fact: `rho` survives as a per-subclass
summary in `<tag>_rho.csv`, but `log_a = centre(lib_size_centered + tau_a *
eps_a_raw)` and `eps_a_raw` was not stored. With `tau_a = 1.48` the posterior
abundance sits within a factor of ~19 of its nanopore prior mean, so substituting
the prior is not an approximation of `a`, it is a different `a`.

`run_bayes.py` now requests all three by default — they add ~3 MB to a 444 MB
file. Fits that carry them:

| fit | dataset | model |
|---|---|---|
| `revision/Bayes_OldData/bayesian/` | 5/28 NEWNEW | joint+dropout, direct activity |
| `revision/Bayes_NewData/bayesian/` | 07/29 SFv8 low dose | joint+dropout, direct activity |
| `revision/bayesian_vs_fold_change/results/ablation/bayesian_full_posterior/` | 5/28 NEWNEW | no dropout |

The first two replicate the earlier `log_gamma`-only fits exactly — same settings,
same seed — and reproduce their activity tables to 1e-12 relative, differing only
in what was written out. Because they are strict supersets, they have since
replaced those fits under the canonical `bayesian/` name: the superseded
`Bayes_NewData` fit was deleted, and `bayesian_vs_fold_change/results/bayesian/`
is retained only as the original published artifact — every plotting and testing
entry point now reads `Bayes_OldData/bayesian/` via
`analysis_utils.OLD_DATA_BAYES`.

A production fit is ~13 minutes on an L40S (the collapsed representation is only
~300k rows), so regenerating one is cheap:

```bash
sbatch --job-name=bayes_cn_origin \
    revision/run_Bayes/submit_bayesian_copy_number.slurm origin
```

**Sanity-check the scale before trusting absolute copy numbers.** `k` and
`beta_t7` are only weakly separated by the likelihood — "many copies, tiny
per-copy rate" fits about as well as "few copies, large rate", and the
mean-centering of `log a` fixes the `rho`–`a` scale but not this one. Both fits
land in the many-copies corner:

| | `bayesian_full_posterior` (no dropout) | `results/bayesian` (production, joint+dropout) |
|---|---|---|
| `beta_t7` | 0.03 | 0.034 |
| `phi_cre` | 0.01 | 0.012 |
| `p_drop_t7` / `p_drop_cre` | — | 0.0004 / 0.0006 |
| max `rho * a` | 39.5 | 37.5 |
| rows truncated at `kmax=60` | — | 2,686 (tail mass 2.6e-4) |

One genome yielding 0.03 T7 transcripts is what lets `rho * a` reach ~38 while
the data stays 99.2% zeros. Note the production model **had** dropout available
and drove it to ~0.0005: it did not use dropout to explain the zeros, it used
many copies at a tiny per-copy rate, exactly as the no-dropout ablation did. So
this is a property of the model and data, not of one fit's configuration.

Treat ratios and rankings across cells as far more robust than the absolute
level. `infer_copy_number` warns when `rho * a` approaches `kmax`, where
truncation biases `E[k]` low on top of this.

**Cost.** `P(k | obs)` depends on the observation only through
`(cell type, cCRE, t7, cre)`, so the 99.85% all-zero pairs collapse to one
baseline per (cell type, cCRE) and the rest to ~10⁶ unique patterns instead of
1.6 × 10⁸ evaluations. The result is exact, not sampled or approximated.

Even so, the all-zero baseline alone is 328 × 389 = 128k patterns, and that cost
is paid whatever the cell count — the work scales with the *posterior*, not the
data. Runtime is linear in the number of draws, so `--max-draws 200` is the main
lever; a posterior **mean** over 200 evenly-spaced draws is within Monte Carlo
error of one over 1,000. Run it on a compute node, not a login node.

## Spatial maps

Five views of the section, one visual grammar, on black:

```python
from baystarrfish.data import CountData
from baystarrfish.plotting import plot_spatial

data = CountData.from_h5ad(section="sec1")          # fills .spatial from obsm

plot_spatial(data, "celltype", celltypes=["CB Granule Glut", "Oligo NN"])
plot_spatial(data, "cre", cre="CRE155")             # raw enhancer counts
plot_spatial(data, "t7", cre="CRE155")              # raw constitutive counts
plot_spatial(data, "copy_number", cre="CRE155", copies=copies, log=True)
plot_spatial(data, "activity", cre="CRE155", copies=copies)   # cCRE / E[k]

# Any value mode: colour by cell-type identity instead of by value. Size (and
# opacity) still track the value, so one figure answers "how much" and "where".
plot_spatial(data, "activity_posterior_normalized", cre="CRE155", activity=m,
             celltypes=["CB Granule Glut", "Oligo NN"])
```

### Three per-cell activity estimators

Both answer "how much enhancer output per virus copy", which is what makes a cell
that received one copy comparable to one that received thirty.

`activity` — the **moment estimator** `cre / E[k | obs]`. The model says
`E[cre | k] = k · gamma`, so the quotient estimates `gamma` directly. It cannot
blow up: `k = 0` is a point mass forcing both channels to zero, so any cell with
`cre > 0` has `P(k = 0 | obs) = 0` and hence `E[k] ≥ 1`, bounding the ratio by the
raw count (observed minimum on the old dataset: `E[k] = 1.0003` across all 23,111
cCRE-positive cells).

`activity_posterior` — the **Gamma-conjugate posterior mean**, which is a
posterior of a quantity the model contains rather than a ratio formed afterwards.
`NB2(k·gamma, phi)` *is* `Poisson(k·gamma·G)` with `G ~ Gamma(phi, phi)`, so the
cell's multiplicative deviation from its cell type's activity is a latent
variable already in the model, and it is conjugate:

```
G | cre, k  ~  Gamma(phi_cre + cre,  phi_cre + k·gamma)
activity    =  gamma · E[G | cre, k],  marginalised over P(k | obs)
```

It shrinks toward the cell-type activity where evidence is thin, relaxes to
`cre / k` when counts are large, and stays finite at `cre = 0` — returning a small
positive number rather than the ratio's hard zero, which is the difference
between "silent" and "no information".

```python
m = infer_copy_number_from_fit(data, fit_dir, return_activity=True)
plot_spatial(data, "activity", cre="CRE007", copies=m)              # moment
plot_spatial(data, "activity_posterior", cre="CRE007", activity=m)  # posterior
```

`activity_posterior_normalized` — the same posterior activity divided by the cell
type's **negative-control reference**, so **1 is background**. `gamma` has no
absolute scale (it is set jointly by the infection rate and the library
abundance), which means a raw activity map partly shows which cell types capture
better. Dividing by the controls of the *same* cell type removes that and puts
every cell on the fold-change scale the published calls use:

```
log b[d, s] = mean_{j' ∈ controls(s)} log_gamma[d, s, j']     # geometric mean
R_ij        = E_d[ (gamma[d,s,j] / b[d,s]) · E[G | cre, k] ]  # divided per draw
```

Two things are load-bearing. The reference is the mean of *logs* — the geometric
mean — because that is the centre a log contrast is symmetric about and what the
published tables used; a corollary is that no individual control's target is 1,
only their geometric mean is. And the division happens **inside** each posterior
draw: `gamma` and `b` share the scale factors that make either one arbitrary, so
their ratio is far better determined than either term (on the production fit the
per-draw and two-stage forms differ by 0.8% at the median, up to 17%).

`baystarrfish.stats.negative_control_log_baseline` builds that reference and is
the same code `negative_control_test` uses, so a map and the table beside it
cannot disagree about what background means. Cell types whose pooled control T7
falls below the threshold (50, as published) have no reference and come back
`NaN` — on the old dataset that is 44 of 328 subclasses eligible, covering
289,802 of 408,621 cells; the rest stay grey rather than being scored against a
background the data cannot support.

```python
m = infer_copy_number_from_fit(data, fit_dir, return_activity_normalized=True)
plot_spatial(data, "activity_posterior_normalized", cre="CRE007", activity=m)
```

**Calibration.** Running each estimator on the 7 control columns and comparing
the cell-mean of `R` against its parameter-level target (`E_d[gamma/b]`, *not* 1)
gives, on rows with at least 10 control counts: conjugate 0.994, moment
ratio-of-sums 0.975 — both unbiased. But the moment estimator's *mean of
per-cell ratios* is **0.070**, because the median fraction of cCRE-positive cells
is 0.0007 and a cell with `cre = 0` contributes exactly 0. So `R = 1` is
background only for the conjugate estimator at the individual-cell level; that is
why only that mode anchors its colour ramp at 1, and why the moment map is left
un-normalised.

A fourth option not implemented: `cre · E[1/k]` instead of `cre / E[k]` fixes a
Jensen bias in the moment estimator (it is low by ~8% at the median, up to 23%,
always low). Ranking is unaffected — Spearman 0.9999 — so it matters only for
absolute values. A per-*cell* background (dividing by that cell's own control
reads) is not implemented either, and would need a refit: each `(cell, cCRE)`
pair has an independent `G` in the model, so a cell's control reads carry no
information about its target pair without a shared per-cell factor. Even as a
deliberate plug-in it fails — with `phi_cre ≈ 0.0124`, `E[1/G]` needs
`phi + controls > 1` and most cells have zero control reads, so the mean does not
exist.

Every mode draws **all** cells first as small grey dots, so the section outline
is always visible and a sparse signal is never mistaken for a sparse tissue. The
three value modes then overlay the cells carrying signal, encoding magnitude
three ways at once — dot size, opacity and colour — because on black at the dot
sizes a 400,000-cell section forces, one channel is not enough. Each value mode
has its own hue (cCRE red, T7 cyan, copy number amber, moment activity green,
posterior activity teal, normalised activity orchid) so the maps stay
distinguishable when cropped out of their titles.

To render one map from a fit without recomputing the full matrix:

```bash
python revision/run_Bayes/plot_spatial_cre.py \
    --fit-dir revision/Bayes_OldData/bayesian \
    --cre CRE007 --mode activity_posterior_normalized \
    --outdir revision/Bayes_OldData/visualization
```

That reconstructs only the requested cCRE plus the negative-control columns —
exact, because pairs are conditionally independent given the parameters and the
collapse is keyed on `(cell type, cCRE, t7, cre)` — which turns a ~25 minute
whole-matrix pass into a couple of minutes.

Conventions follow `STARRFISH.utils.STARRFISH.plot_gene` — black facecolor, grey
background at size 3, values from 5 to 30, alpha ramp, equal aspect, no ticks,
the 7-dot scale bar — so these sit alongside the existing figures. Three
deliberate departures:

- value cells are drawn in **ascending** order, so the strongest are never buried
  under whichever neighbour came later in the array;
- the scale bar uses the same **linear** size ramp as the plot (`plot_gene` cubes
  it, so its legend disagrees with the figure it documents);
- `vmax` defaults to the **99th percentile** of the drawn cells, not the maximum,
  because one 300-count cell otherwise flattens everything else to invisibility.

**`copy_number` is not like the count modes.** `E[k | obs]` is defined for every
cell, so "overlay cells with signal" would paint the entire tissue one colour.
It therefore overlays only cells with a nonzero read in either channel — 7.4% of
the section for a typical cCRE — leaving the rest grey, because everywhere else
the estimate is the cell-type baseline `E[k | 0, 0]` rather than a measurement.
`evidence_mask` exposes that rule; `min_value` overrides it.

Needs the `plots` extra. `baystarrfish.plotting` is never imported unless asked
for, so the inference path stays free of matplotlib.

## Layout

```
baystarrfish/
  model/       priors, collapsed statistics, likelihood, sampling blocks,
               the twelve model functions, the registry, the forward sampler
  inference/   moment initialisation, SVI/NUTS, summaries, posterior predictive
               checks, the run_model / run_decoupled_model drivers, and the
               latent copy-number reconstruction (posterior_k, copy_number)
  data/        AnnData -> arrays: paths, label standardisation, blacklisting,
               negative-control modes, nanopore library prior, CountData
  stats/       BH-FDR and the negative-control contrast
  io/          write_fit / read_fit / load_gamma / load_posterior_samples
  simulate/    simulation-recovery test
```

`model/` and `inference/` need NumPyro. `data/`, `stats/` and `io/` deliberately
do not, so the plotting environment can reload inputs and read fits without the
inference stack.

Three infection families, selected by `infection_model`: `copy_number`,
`copy_number_dropout` (production), and `binary` — a shared Bernoulli infection
gate with probability `1 - exp(-rho_s a_j)` instead of a copy count. Two activity
parameterisations, selected by `activity_model`: `hierarchical`
(`log_gamma = alpha_j + eta_cj + delta_sj`) and `direct` (exchangeable over the
whole subclass × cCRE matrix, used for the production fit).

## Priors

| site | prior | meaning |
|---|---|---|
| `mu_rho` | `Normal(-6, 2)` | global log infection baseline |
| `sigma_u`, `sigma_w` | `HalfNormal(1)` | class- and subclass-level infection sd |
| `tau_a` | `HalfNormal(0.5)` | slack around the centred log nanopore abundance |
| `beta_t7` | `LogNormal(0, 1)` | T7 counts per infected copy |
| `phi_t7`, `phi_cre` | `HalfNormal(5)` | NB2 dispersions |
| `mu_alpha` | `Normal(0, 3)` | activity baseline (hierarchical) |
| `sigma_alpha` | `HalfNormal(2)` | per-cCRE baseline sd |
| `sigma_eta`, `sigma_delta` | `HalfNormal(1)` | class- and subclass-level activity sd |
| `mu_gamma`, `sigma_gamma` | `Normal(0, 3)`, `HalfNormal(2)` | activity (direct) |
| `p_drop_t7`, `p_drop_cre` | `Beta(1, 9)` | measurement dropout |

All non-centred (`*_raw ~ Normal(0, 1)`, scaled by `sigma_*`). Sensitivity to the
dropout prior is reported in
[`revision/bayesian_vs_fold_change/Bayesian_priors.md`](revision/bayesian_vs_fold_change/Bayesian_priors.md).

## Reading the output

`<tag>_gamma.csv` carries a **`prior_dominated`** flag marking pairs with no
double-positive (T7 > 0 **and** cCRE > 0) support. Those posteriors are the
prior, not evidence; they are not activity estimates and must not be read as
such. The same eligibility logic drives the T7 filter in
`negative_control_test`.

## Tests

```bash
pytest tests -m 'not slow'    # 127 tests, ~55 s
pytest tests -m slow          # end-to-end SVI recovery, ~60 s
python -m baystarrfish recovery --classes 3 --cres 5 --cells 400 --steps 4000
```

The suite checks the model against outside references rather than against itself:
the negative binomial against `scipy.stats.nbinom`, the marginal against an
explicit sum over the latent grid, BH-FDR against `statsmodels`, and the NumPy
copy-number posterior against the JAX likelihood.

`testpaths = ["tests"]` in `pyproject.toml` is load-bearing:
`revision/bayesian_vs_fold_change/code/` contains six files named `test_*.py`
that are analysis entry points launching multi-hour jobs, not tests.

## Citation

Gibbs, Z., Zhong, G. et al. (2026). STARRFISH [under submission].
Data: <https://doi.org/10.5061/dryad.3bk3j9m0p>
