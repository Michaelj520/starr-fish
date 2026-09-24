# CRE182 diagnosis

The point near -6 is **CRE182_bar186 in the RNA/DNA plot** (ln fold change
-6.1817). Its log GLM activity is -12.8337. The two plots use different scales.

| Barcode | RNA counts | DNA counts | Cells with RNA | ln RNA/DNA | ln GLM activity |
|---|---:|---:|---:|---:|---:|
| bar010 | 39,740 | 21,899 | 50.56% | -0.90 | -7.97 |
| bar086 | 1,992 | 8,632 | 6.33% | -2.96 | -10.94 |
| bar129 | 3,214 | 736 | 7.35% | -0.02 | -9.87 |
| bar145 | 51,916 | 12,400 | 56.32% | -0.07 | -7.53 |
| bar186 | 336 | 36,342 | 1.03% | -6.18 | -12.83 |

`CRE182_rna_vs_total_rna.png` / `.pdf` show all five barcodes, with one point
per cell, global OLS lines, and decile means. The bottom row excludes the
highest-total-RNA 1% of cells for a closer view; its regression line remains
the original global fit. Each barcode has its own y-axis range. Total RNA
means the sum over the 300 reporter features, not whole-transcriptome RNA.

`CRE182_diagnostic_summary.png` / `.pdf` compare RNA/DNA abundance,
regression sensitivity, and the supplied STARR-seq replicate activities.

## Findings

- bar186 has the largest DNA count of these five barcodes, but only 336 FISH
  RNA counts in 286 of 27,889 cells; no cell has more than 3 counts. Its
  normalized RNA/DNA ratio is 0.00207 (about 484-fold below parity).
- Its RNA is detected in 175 of 283 imaging fields, so it is not restricted
  to one anomalous field. This does not establish equal detection across fields.
- No single cell changes its slope by more than 2.60% when omitted. Removing
  the highest-total-RNA 1% changes log GLM activity from -12.834 to -12.752.
  Excluding the barcode from total RNA changes it to -12.837; excluding all
  CRE182 barcodes changes it to -12.821. These checks support a broadly weak
  signal rather than a high-leverage-cell or target-in-predictor explanation.
- The CSV's reported STARR-seq activities for bar186 are 0.7266 and 0.7796
  (RNA counts 2,126 and 2,097). Those values do not show the extreme depletion
  observed in STARR-FISH. bar145's corresponding activities are 0.7693 and
  0.7650, while its FISH ratio is approximately 0.936.

The evidence prioritizes checking bar186's FISH probe/readout performance,
decoding or feature assignment, and CRE–barcode mapping. It does not by
itself distinguish technical detection failure from a barcode-dependent
biological or experimental difference. The supplied sequencing replicates
are a comparison, not evidence that the experiments have identical conditions.

## Audit tables

- `cell_counts.csv.gz`: cell IDs, five barcode counts, total reporter RNA,
  summed CRE182 RNA, Leiden cluster, and imaging field parsed before `--`.
- `barcode_diagnostics.csv`: coefficients, count support, DNA normalization,
  correlation, field coverage, and influence statistics.
- `sensitivity_refits.csv`: original and diagnostic fits with retained cell
  and positive-cell counts. Trimming by Cook's distance is outcome-selected
  and removes most positive cells for sparse barcodes; it is retained for
  audit but omitted from the sensitivity plot, and is not a recommended filter.
- `influential_cells.csv`: the 20 largest Cook's distances for each barcode.
- `fov_summary.csv`: count totals and detection fractions per imaging field.
- `sequencing_comparison.csv`: FISH results alongside the supplied replicate
  counts and activities, without changing the underlying activity definition.

Regenerate with `python revision/barcode_shuffling/diagnose_cre.py --cre CRE182`.
