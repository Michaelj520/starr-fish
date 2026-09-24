#!/usr/bin/env python3
"""Reproduce glm_fit_total's OLS / log(nanopore DNA count) activity for SFv6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from compute_activities import DEFAULT_DNA, DEFAULT_INPUT, HERE, load_counts, load_dna_counts


def fit_total_ols(rna: np.ndarray, total_rna: np.ndarray,
                  dna_counts: np.ndarray) -> pd.DataFrame:
    """Fit y = intercept + slope * total_RNA for each column, at full precision.

    Centered least squares is equivalent to statsmodels OLS with an intercept.
    Standard errors are the classical homoscedastic OLS standard errors.
    DNA is treated as fixed, just as in the original glm_fit_total.
    """
    y = np.asarray(rna, dtype=float)
    x = np.asarray(total_rna, dtype=float)
    dna = np.asarray(dna_counts, dtype=float)
    if y.ndim != 2 or x.shape != (y.shape[0],) or dna.shape != (y.shape[1],):
        raise ValueError("RNA, per-cell total RNA, and DNA count axes must match")
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (y, x, dna)):
        raise ValueError("RNA and DNA inputs must be finite and nonnegative")
    n, n_features = y.shape
    intercept, slope, slope_se = (np.full(n_features, np.nan) for _ in range(3))
    status = np.full(n_features, "ok", dtype=object)
    if n < 3:
        status[:] = "insufficient_cells"
    else:
        xc = x - x.mean()
        sxx = xc @ xc
        if sxx <= 0:
            status[:] = "constant_total_rna"
        else:
            yc = y - y.mean(axis=0)
            slope = (xc @ yc) / sxx
            intercept = y.mean(axis=0) - slope * x.mean()
            residuals = yc - xc[:, None] * slope[None, :]
            slope_se = np.sqrt(np.sum(residuals ** 2, axis=0) / (n - 2) / sxx)
            status[slope <= 0] = "nonpositive_slope"
    valid_dna = dna > 1
    status[~valid_dna] = "dna_count_le1"
    denominator = np.full(n_features, np.nan)
    denominator[valid_dna] = np.log(dna[valid_dna])
    activity = slope / denominator
    log_activity = np.full(n_features, np.nan)
    positive = activity > 0
    log_activity[positive] = np.log(activity[positive])
    return pd.DataFrame({
        "n_cells": n,
        "total_counts": y.sum(axis=0),
        "n_positive_cells": (y > 0).sum(axis=0),
        "dna_count": dna,
        "intercept": intercept,
        "slope": slope,
        "slope_se": slope_se,
        "activity": activity,
        "activity_se": slope_se / denominator,
        "log_activity": log_activity,
        "fit_status": status,
        "passes_legacy_dna_filter": dna > 10,
    })


def compute_glm_tables(counts, labels: pd.Series, metadata: pd.DataFrame,
                       dna_counts: pd.Series) -> dict[str, pd.DataFrame]:
    """Fit barcodes and pooled CREs globally or within each supplied group."""
    if counts.shape != (len(labels), len(metadata)) or labels.isna().any():
        raise ValueError("Counts, labels, and metadata axes must match with no missing labels")
    if dna_counts.index.has_duplicates or set(dna_counts.index) != set(metadata.index):
        raise ValueError("DNA counts must have exactly one value per RNA feature")
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    dna = dna_counts.reindex(metadata.index).to_numpy(dtype=float)
    cre_codes, cres = pd.factorize(metadata.cre, sort=True)
    mapping = sparse.csr_matrix(
        (np.ones(len(metadata)), (np.arange(len(metadata)), cre_codes)),
        shape=(len(metadata), len(cres)),
    )
    pooled_counts = counts @ mapping
    pooled_dna = np.asarray(dna @ mapping).ravel()
    n_barcodes = np.bincount(cre_codes, minlength=len(cres))
    total_rna = counts.sum(axis=1)  # Includes the target barcode, as in glm.py.
    group_codes, groups = pd.factorize(labels, sort=True)
    output = {"barcode": [], "cre": []}
    for group_index, group in enumerate(groups):
        cells = group_codes == group_index
        for level, y, d, names, multiplicity in (
            ("barcode", counts, dna, metadata.index, np.ones(len(metadata), dtype=int)),
            ("cre", pooled_counts, pooled_dna, pd.Index(cres), n_barcodes),
        ):
            table = fit_total_ols(y[cells], total_rna[cells], d)
            table.insert(0, "feature", names.to_numpy())
            table.insert(0, "group", str(group))
            table["n_barcodes"] = multiplicity
            if level == "barcode":
                table = table.merge(metadata.reset_index(), on="feature", validate="one_to_one")
            else:
                table["cre"] = table.feature
            table["is_control"] = table.cre.eq("barcode only")
            output[level].append(table)
    tables = {level: pd.concat(frames, ignore_index=True) for level, frames in output.items()}
    means = tables["barcode"].groupby(["group", "cre"], sort=False).log_activity.agg(
        lambda values: np.mean(values.to_numpy()))
    tables["cre"] = tables["cre"].merge(means.rename("mean_barcode_log_activity"),
                                         on=["group", "cre"], validate="one_to_one")
    return tables


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dna-csv", type=Path, default=DEFAULT_DNA)
    parser.add_argument("--outdir", type=Path, default=HERE / "results/glm")
    parser.add_argument("--groupby", default="leiden")
    args = parser.parse_args(argv)
    print(f"Reading {args.h5ad}", flush=True)
    counts, labels, metadata = load_counts(args.h5ad, args.groupby)
    dna = load_dna_counts(args.dna_csv, metadata.index)
    tables = {}
    for scope, grouping in (("global", pd.Series("all", index=labels.index)), ("grouped", labels)):
        print(f"Fitting {scope} OLS models", flush=True)
        tables[scope] = compute_glm_tables(counts, grouping, metadata, dna.DNA_count)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for scope, levels in tables.items():
        for level, table in levels.items():
            table.to_csv(args.outdir / f"{scope}_{level}_activities.csv", index=False)
            for value in ("activity", "log_activity"):
                table.pivot(index="group", columns="feature", values=value).to_csv(
                    args.outdir / f"{scope}_{level}_{value}.csv")
    metadata.join(dna).to_csv(args.outdir / "feature_metadata.csv")
    manifest = {
        "method": "glm_total_ols_nanopore",
        "reference_source": "STARRFISH_in_vitro/glm.py:glm_fit_total",
        "input": str(args.h5ad.resolve()),
        "input_size_bytes": args.h5ad.stat().st_size,
        "input_mtime_ns": args.h5ad.stat().st_mtime_ns,
        "count_source": "obsm/X_raw",
        "dna_input": str(args.dna_csv.resolve()),
        "dna_input_size_bytes": args.dna_csv.stat().st_size,
        "dna_input_mtime_ns": args.dna_csv.stat().st_mtime_ns,
        "dna_count_source": "DNA_count",
        "n_cells": len(labels), "n_features": len(metadata),
        "n_cre_labels": int(metadata.cre.nunique()),
        "groupby": args.groupby, "n_groups": int(labels.nunique()),
        "model": "RNA[cell, barcode] = intercept[barcode] + slope[barcode] * total_RNA[cell] + error",
        "family": "Gaussian identity-link, unweighted ordinary least squares",
        "use_fov_covariate": False, "norm_by_vector": False, "norm_by_nanopore": True,
        "total_rna_includes_target": True,
        "definition": "activity = slope / ln(DNA_count); log_activity = ln(activity)",
        "log_base": float(np.e),
        "cre_aggregation": "Fit pooled CRE RNA on original cell total RNA, divide its slope by ln(pooled CRE DNA); also export mean barcode log activity",
        "standard_errors": "Classical OLS; activity_se = slope_se / ln(DNA_count), treating DNA as fixed",
        "invalid_policy": "DNA <= 1, nonpositive slopes, constant total RNA, or fewer than 3 cells have no log activity; flags and raw slopes are retained",
        "plot_filter": "DNA_count > 10 and finite log_activity, matching the legacy plotting threshold",
        "global_fit_status_counts": tables["global"]["barcode"].fit_status.value_counts().to_dict(),
        "grouped_fit_status_counts": tables["grouped"]["barcode"].fit_status.value_counts().to_dict(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote OLS activities for {len(metadata)} barcodes to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
