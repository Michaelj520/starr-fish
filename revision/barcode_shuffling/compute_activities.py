#!/usr/bin/env python3
"""Compute SFv6 barcode and CRE activities as log(RNA CPM / nanopore DNA CPM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "Data/scdata_260821_SFv6_multiBC.h5ad"
DEFAULT_DNA = HERE.parent / "Data/STARR_seq_Counts_wReps_DNA.csv"


def feature_metadata(names: pd.Index) -> pd.DataFrame:
    """Parse the embedded CRE–barcode mapping without merging barcode identities."""
    names = pd.Index(names.astype(str), name="feature")
    if names.has_duplicates:
        raise ValueError("Feature names must be unique")
    parsed = names.to_series().str.extract(r"^(?P<cre>.+)_(?P<barcode>bar\d+)$")
    if parsed.isna().any().any():
        raise ValueError("Every feature must have the form <CRE>_bar<digits>")
    if parsed.barcode.duplicated().any():
        raise ValueError("Barcode IDs must be unique")
    return parsed


def load_counts(path: Path, groupby: str):
    """Use X_raw: X in this dataset is already transformed for clustering."""
    data = ad.read_h5ad(path)
    if "X_raw" not in data.obsm:
        raise ValueError("Missing obsm['X_raw']; refusing to use transformed X")
    if groupby not in data.obs:
        raise ValueError(f"Unknown grouping column {groupby!r}; available: {list(data.obs)}")
    counts = data.obsm["X_raw"]
    if isinstance(counts, pd.DataFrame):
        if not counts.index.equals(data.obs_names) or not counts.columns.equals(data.var_names):
            raise ValueError("X_raw labels do not match the AnnData axes")
        counts = counts.to_numpy()
    counts = counts.astype(np.float64)
    values = counts.data if sparse.issparse(counts) else counts
    if counts.shape != data.shape or data.n_obs == 0 or data.n_vars == 0:
        raise ValueError("X_raw must be a nonempty cells × features matrix matching var_names")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Raw counts must be finite and nonnegative")
    if not np.allclose(values, np.round(values), rtol=0, atol=1e-8):
        raise ValueError("X_raw contains noninteger values; expected untransformed counts")
    if data.obs_names.has_duplicates or data.obs[groupby].isna().any():
        raise ValueError("Cell IDs must be unique and group labels must not be missing")
    return counts, data.obs[groupby].copy(), feature_metadata(data.var_names)


def load_dna_counts(path: Path, features: pd.Index) -> pd.DataFrame:
    """Match full element identities, allowing only the known control-name alias."""
    dna = pd.read_csv(path)
    required = {"element", "DNA_count"}
    if not required.issubset(dna.columns) or dna.element.isna().any():
        raise ValueError("DNA table requires nonmissing element IDs and DNA_count")
    dna = dna.loc[:, ["element", "DNA_count"]].copy()
    dna["feature"] = dna.element.str.replace(r"^barcode_only_", "barcode only_", regex=True)
    if dna.feature.duplicated().any():
        raise ValueError("Duplicate DNA element IDs after normalizing control names")
    missing = set(features) - set(dna.feature)
    extra = set(dna.feature) - set(features)
    if missing or extra:
        raise ValueError(f"RNA/DNA feature sets differ: missing DNA={sorted(missing)}, extra DNA={sorted(extra)}")
    dna = dna.set_index("feature").loc[features]
    values = pd.to_numeric(dna.DNA_count, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.all(values == np.round(values)):
        raise ValueError("DNA_count must contain finite, nonnegative integers")
    if values.sum() <= 0:
        raise ValueError("DNA library has no reads")
    dna["DNA_count"] = values
    dna["DNA_CPM"] = values / values.sum() * 1e6
    return dna


def log_fold_change(rna_cpm: np.ndarray, dna_cpm: np.ndarray, pseudocount: float,
                    log_base: float) -> np.ndarray:
    """Zero RNA yields -inf at p=0; zero DNA or an empty RNA library yields NaN."""
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("Pseudocount must be finite and nonnegative")
    if not np.isfinite(log_base) or log_base <= 1:
        raise ValueError("Log base must be finite and greater than one")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (np.log(rna_cpm + pseudocount) - np.log(dna_cpm[None, :] + pseudocount)) / np.log(log_base)
    result[:, dna_cpm <= 0] = np.nan
    return result


def compute_tables(counts, labels: pd.Series, metadata: pd.DataFrame,
                   dna_counts: pd.Series, pseudocount: float = 0,
                   log_base: float = np.e) -> dict[str, pd.DataFrame]:
    """Normalize RNA within each group and DNA over the same feature universe."""
    if counts.shape != (len(labels), len(metadata)) or labels.isna().any():
        raise ValueError("Counts, labels, and feature metadata must have matching axes")
    if dna_counts.index.has_duplicates or set(dna_counts.index) != set(metadata.index):
        raise ValueError("DNA counts must have exactly one value per RNA feature")
    dna = dna_counts.reindex(metadata.index).to_numpy(dtype=float)
    if not np.isfinite(dna).all() or (dna < 0).any() or dna.sum() <= 0:
        raise ValueError("DNA counts must be finite and nonnegative with a positive library total")
    group_codes, groups = pd.factorize(labels, sort=True)
    n_groups = len(groups)
    membership = sparse.csr_matrix(
        (np.ones(len(labels)), (group_codes, np.arange(len(labels)))),
        shape=(n_groups, len(labels)),
    )
    n_cells = np.bincount(group_codes, minlength=n_groups)

    def aggregate(matrix):
        totals = membership @ matrix
        positive = membership @ (matrix > 0)
        return tuple(x.toarray() if sparse.issparse(x) else np.asarray(x)
                     for x in (totals, positive))

    barcode_totals, barcode_positive = aggregate(counts)
    rna_library_total = barcode_totals.sum(axis=1)
    dna_library_total = dna.sum()
    cre_codes, cres = pd.factorize(metadata.cre, sort=True)
    n_barcodes = np.bincount(cre_codes, minlength=len(cres))
    mapping = sparse.csr_matrix(
        (np.ones(len(metadata)), (np.arange(len(metadata)), cre_codes)),
        shape=(len(metadata), len(cres)),
    )
    cre_totals, cre_positive = aggregate(counts @ mapping)
    cre_dna = np.asarray(dna @ mapping).ravel()
    tables = {}
    for level, names, totals, positive, multiplicity, dna_total in (
        ("barcode", metadata.index, barcode_totals, barcode_positive, np.ones(len(metadata), dtype=int), dna),
        ("cre", pd.Index(cres), cre_totals, cre_positive, n_barcodes, cre_dna),
    ):
        mean = totals / n_cells[:, None] / multiplicity[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rna_cpm = totals / rna_library_total[:, None] * 1e6
        dna_cpm = dna_total / dna_library_total * 1e6
        activity = log_fold_change(rna_cpm, dna_cpm, pseudocount, log_base)
        with np.errstate(divide="ignore", invalid="ignore"):
            fold_change = (rna_cpm + pseudocount) / (dna_cpm[None, :] + pseudocount)
        fold_change[:, dna_total <= 0] = np.nan
        index = pd.MultiIndex.from_product([groups.astype(str), names], names=["group", "feature"])
        table = pd.DataFrame({
            "n_cells": np.repeat(n_cells, len(names)),
            "n_barcodes": np.tile(multiplicity, n_groups),
            "total_counts": totals.ravel(),
            "dna_count": np.tile(dna_total, n_groups),
            "rna_library_total": np.repeat(rna_library_total, len(names)),
            "dna_library_total": dna_library_total,
            "rna_cpm": rna_cpm.ravel(),
            "dna_cpm": np.tile(dna_cpm, n_groups),
            "n_positive_cells": positive.ravel().astype(int),
            "mean_count_per_cell_per_barcode": mean.ravel(),
            "fold_change": fold_change.ravel(),
            "log_fold_change": activity.ravel(),
            "dna_observed": np.tile(dna_total > 0, n_groups),
        }, index=index).reset_index()
        if level == "barcode":
            table = table.merge(metadata.reset_index(), on="feature", validate="many_to_one")
        else:
            table["cre"] = table.feature
        table["is_control"] = table.cre.eq("barcode only")
        tables[level] = table
    # Equal-barcode mean of log activities, also used to order the boxplot.
    # np.mean deliberately propagates missing DNA and -inf from zero RNA.
    averages = tables["barcode"].groupby(["group", "cre"], sort=False).log_fold_change.agg(
        lambda x: np.mean(x.to_numpy()))
    tables["cre"] = tables["cre"].merge(
        averages.rename("mean_barcode_log_fold_change"), on=["group", "cre"], validate="one_to_one")
    return tables


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dna-csv", type=Path, default=DEFAULT_DNA)
    parser.add_argument("--outdir", type=Path, default=HERE / "results")
    parser.add_argument("--groupby", default="leiden")
    parser.add_argument("--log-base", type=float, default=np.e)
    parser.add_argument("--pseudocount", type=float, default=0,
                        help="Added to both RNA CPM and DNA CPM after library normalization")
    args = parser.parse_args(argv)
    # Validate numeric settings before loading the data.
    log_fold_change(np.ones((1, 1)), np.ones(1), args.pseudocount, args.log_base)
    print(f"Reading {args.h5ad}", flush=True)
    counts, labels, metadata = load_counts(args.h5ad, args.groupby)
    dna = load_dna_counts(args.dna_csv, metadata.index)
    print(f"Matched {len(dna)} DNA barcodes ({dna.DNA_count.sum():,.0f} reads)", flush=True)
    results = {}
    for scope, grouping in (
        ("global", pd.Series("all", index=labels.index)),
        ("grouped", labels),
    ):
        results[scope] = compute_tables(counts, grouping, metadata, dna.DNA_count,
                                        args.pseudocount, args.log_base)
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata.assign(is_control=metadata.cre.eq("barcode only")).join(dna).to_csv(args.outdir / "feature_metadata.csv")
    for scope, tables in results.items():
        for level, table in tables.items():
            table.to_csv(args.outdir / f"{scope}_{level}_activities.csv", index=False)
            table.pivot(index="group", columns="feature", values="log_fold_change").to_csv(
                args.outdir / f"{scope}_{level}_log_fold_change.csv")
    manifest = {
        "input": str(args.h5ad.resolve()),
        "input_size_bytes": args.h5ad.stat().st_size,
        "input_mtime_ns": args.h5ad.stat().st_mtime_ns,
        "count_source": "obsm/X_raw",
        "dna_input": str(args.dna_csv.resolve()),
        "dna_input_size_bytes": args.dna_csv.stat().st_size,
        "dna_input_mtime_ns": args.dna_csv.stat().st_mtime_ns,
        "dna_count_source": "DNA_count",
        "dna_library_total": float(dna.DNA_count.sum()),
        "dna_name_alias": "barcode_only_barNNN -> barcode only_barNNN",
        "n_cells": len(labels), "n_features": len(metadata), "n_cre_labels": int(metadata.cre.nunique()),
        "groupby": args.groupby, "n_groups": int(labels.nunique()),
        "control_features": metadata.index[metadata.cre.eq("barcode only")].tolist(),
        "log_base": args.log_base, "pseudocount_on_cpm": args.pseudocount,
        "definition": "log_base((RNA_CPM + p) / (DNA_CPM + p))",
        "cre_aggregation": "Pool RNA and DNA counts over barcodes before computing CPM ratio; also export equal-barcode mean log activity",
        "zero_policy": "Zero DNA or empty RNA library: NaN even with pseudocount; zero RNA with p=0: -inf",
        "normalization": "RNA CPM within each group; DNA CPM over matching features; no negative-control centering",
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote global and {args.groupby} activities for {len(metadata)} barcodes / "
          f"{metadata.cre.nunique()} CRE labels to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
