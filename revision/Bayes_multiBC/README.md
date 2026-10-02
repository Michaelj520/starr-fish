# Bayes_multiBC

Fits the `Bayes_OldData` model to the SFv6 multi-barcode data
(`../Data/scdata_260821_SFv6_multiBC_FINAL.h5ad`), with one activity per barcode.

## Model

The model is the one in `../Bayes_OldData/bayesian/run_manifest.json`: subclass level,
joint CRE + T7, copy number with zero-inflated dropout (Beta(1, 9) on both channels),
`direct` activity, ordinary negative controls, SVI with `AutoNormal`, 30,000 steps,
lr 0.005, kmax 60, 1,000 posterior draws. `run_bayes_multibc.py` fixes these as
constants; only optimiser settings are flags.

## How this input maps onto the model

| model column | SFv6 input |
|---|---|
| cCRE (`cre_names`) | each of the 300 barcodes, named `<CRE>_barNNN` from `var_names` |
| subclass / class | one label, `all`, for the 14,919 cells with `saturation_fail == False` |
| CRE channel | `obsm['CRE']` |
| T7 channel | `obsm['T7']` |
| abundance prior `lib_size_log` | `log1p(DNA_count)` per barcode from `../Data/STARR_seq_Counts_wReps_DNA.csv` |
| negative controls | the 5 `barcode only_barNNN` barcodes, fitted like the rest |
| blacklist | none |

- `obsm['CRE']` / `obsm['T7']` columns are `barNNN`. The loader matches them to
  `var_names` by barcode, not by position.
- The fit drops the 14,003 of 28,922 cells with `obs['saturation_fail'] == True`.
  `run_bayes_multibc.py` sets this in `DROP_SATURATION_FAIL` and records it as
  `cell_filter` in `run_manifest.json`. `--max-cells` subsamples the cells that remain.
- There is no blacklist because the SFv8 blacklist comes from AAV barcode-mismatch
  rates in a different library.
- The RNA-to-DNA matching comes from
  `../barcode_shuffling/compute_activities.py` (`feature_metadata`, `load_dna_counts`),
  so this fit and the fold-change activities pair each barcode with the same DNA count.

## Files

- `multibc_data.py`: `load_multibc_counts` builds a `baystarrfish.data.CountData`.
  It reads only the `obs`/`var` indices and `obsm['CRE' | 'T7' | 'X_spatial']`.
- `run_bayes_multibc.py`: the fit script. It refuses an outdir that already holds a
  `run_manifest.json` unless you pass `--overwrite`.
- `submit_bayes_multibc.slurm`: the GPU job. It writes to `bayesian/` and logs to `logs/`.
- `test_multibc_data.py`: loader tests on a synthetic h5ad.

## Run

From the repository root:

```bash
# loader tests
/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python -m pytest revision/Bayes_multiBC/test_multibc_data.py -q

# smoke fit on CPU (2,000 cells, 500 steps) into a scratch dir
/gpfs/commons/home/guojiezhong/miniconda3/envs/bayes-jax/bin/python revision/Bayes_multiBC/run_bayes_multibc.py \
  --cpu --max-cells 2000 --steps 500 --num-posterior 100 --outdir /tmp/multibc_smoke --overwrite

# full fit
sbatch revision/Bayes_multiBC/submit_bayes_multibc.slurm
```

## Outputs (`bayesian/`)

The layout matches `../Bayes_OldData/bayesian/`, with tag `subclass_joint_copy_number_dropout_svi`.
Posterior arrays have shape `(draws, 1, 300)` for `log_gamma`, with `group_names = ['all']`
and `cre_names` = the 300 barcode features. `cre_info.csv` records each barcode's
element, DNA count, DNA CPM and control flag.
