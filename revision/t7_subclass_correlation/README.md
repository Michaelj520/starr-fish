# T7 recovery vs the input AAV library

Does the per-cCRE signal we recover in single cells track the abundance of that
cCRE in the injected AAV library (`SFv8_400CRE_nanopore_counts.csv`)?

## Scripts

| script | what it produces |
|---|---|
| `plot_t7_subclass_correlation.py` | subclass x subclass T7 correlation heatmaps |
| `plot_t7_vs_nanopore.py` | raw T7 counts vs nanopore, pooled and per subclass |
| `plot_copies_vs_nanopore.py` | inferred AAV copy number vs nanopore, per subclass |

`plot_t7_vs_nanopore.py` needs `--data`, because two 5/28 AnnData files now match
the default glob; the published figures use `..._CRE_T7CRE_NEW.h5ad`.
`plot_copies_vs_nanopore.py` reads only `revision/Bayes_OldData/` plus the
nanopore CSV — no AnnData load.

## Why the copy-number version does not use the fit's own copies

In the production model the expected copy number is rank-one,
`lambda_{s,c} = rho_s * a_c` (`baystarrfish/model/models.py`), and the abundance
prior *is* the nanopore library:
`log a = centred log1p(nanopore) + tau_a * eps`, `tau_a ~ HalfNormal(0.5)`
(`baystarrfish/model/blocks.py:32`). Two consequences:

* `a_c` is shared by every subclass, so a per-subclass correlation against
  nanopore is the same number for all 328 subclasses, shifted vertically by
  `log rho_s`. There is no per-subclass information in it.
* it is centred on the quantity being validated, so the correlation partly
  reports the prior. For the record the posterior did move: `tau_a` posterior
  mean 1.48 against a prior scale of 0.5, and `log a` regresses on its own prior
  mean with slope 0.51 (r = 0.79) — see
  `figures/abundance_posterior_vs_nanopore.pdf`.

## What is fitted instead

Per subclass, a two-parameter maximum-likelihood regression of the **T7
detection pattern** on nanopore abundance, keeping the fit's measurement
parameters (`beta_t7 = 0.034`, `phi_t7 = 1.55`, `p_drop_t7 = 0.0004`, `kmax = 60`)
and discarding its abundance prior:

```
lambda_{s,c} = exp(b_s + m_s * z_c),   z_c = centred log1p(nanopore_c)
k | lambda   ~ Poisson(lambda)                            latent AAV copies
t7 | k       ~ NB2(mean = k * beta_t7, disp = phi_t7), zero-inflated p_drop_t7
n_pos_{s,c}  ~ Binomial(n_cells_s, q(lambda_{s,c}))
q(lambda)    = (1 - p_drop_t7) * sum_{k>=1} Pois(k|lambda) *
               (1 - (phi_t7 / (phi_t7 + beta_t7 k))^phi_t7)
```

`m_s = 1` is exactly "inferred copy number proportional to nanopore count".
Analytic gradient, expected-Fisher standard errors, and the zero pairs (the
large majority: `P(T7 > 0 | 1 copy) = 0.033`) enter the likelihood instead of
being filtered out.

## Correlation (copies vs nanopore)

* the fit's own copies, `lambda = rho_s * a_c`, log-log against nanopore:
  **r = 0.791** over 381 cCREs — and, because the model is rank-one, that is the
  identical value for all 328 subclasses. It is also partly the prior (above).
* detection-derived copies, per subclass: Pearson r (log10, over the cCREs with
  at least one T7-positive cell) median **0.605**, IQR 0.496-0.689; Spearman rho
  over all cCREs (undetected = zero copies) median **0.673**. Both rise steeply
  with subclass size (Spearman rho vs cell number 0.85 / 0.87) and level off at
  **~0.79-0.80** for subclasses above ~3,000 cells — i.e. they converge on the
  fit's value, so the spread across subclasses is sampling noise, not biology.

See `figures/copies_vs_nanopore_pearson.pdf` and
`figures/copies_vs_nanopore_pearson.csv`.

## Result

Proportionality is rejected, but the *same* relation holds in every cell type.

* pooled exponent (shared `m`, free offset per subclass): **0.430 +- 0.001**
* per-subclass: median **0.427**, IQR 0.413-0.435, range 0.330-0.489, over the
  280/328 subclasses with >= 50 T7-positive observations; **0/280** 95% CIs
  cover 1, and there is no meaningful depth trend (Spearman rho 0.19 vs cell
  number)
* range-resolved (95-cCRE sliding windows): `m` = 0.75-0.88 between ~35 and ~400
  reads, falling to **0.17** above ~1000 reads, and 0.33-0.40 below ~90 reads

So the relation is monotone but curved: near-proportional in the mid-abundance
range, saturating at the top, attenuated at the bottom. The low-end attenuation
is expected from measurement error in the covariate (nanopore counts of 1-50 are
Poisson-noisy, which dilutes a log-log slope). The high-end saturation is **not
identified** by this analysis: true infection saturation and competition for
capture in the assay predict the same flattening, and the model has no term for
either.

## Figures

| file | content |
|---|---|
| `copies_vs_nanopore_pearson` | per-subclass Pearson r / Spearman rho of copies vs nanopore, against subclass depth |
| `copies_vs_nanopore_slope_vs_depth` | `m_s` +- 95% CI vs subclass depth, its histogram, and the range-resolved exponent |
| `copies_vs_nanopore_examples` | per-cCRE inferred copies vs nanopore for 6 subclasses spanning the depth range, with the zero-aware binned curve and the 1-positive-cell censoring floor |
| `abundance_posterior_vs_nanopore` | the fit's `log a` against its nanopore prior (diagnostic only) |
| `copies_vs_nanopore_slopes.csv` | per-subclass exponent table |
| `copies_vs_nanopore_local_exponents.csv` | range-resolved exponent table |
| `t7_nanopore_corr_vs_depth[_filtered]` | earlier raw-count correlation vs depth, unfiltered and with the per-subclass T7 >= 50 / >= 50 cCRE filter |
