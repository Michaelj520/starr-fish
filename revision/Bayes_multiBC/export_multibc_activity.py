#!/usr/bin/env python3
"""Reduce the multiBC posterior to one control-centred activity per barcode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baystarrfish._log import log
from baystarrfish.io import fit_tag, input_fingerprint, load_posterior_samples, write_json

from multibc_data import GROUP_LABEL, HERE

DEFAULT_BAYES = HERE / "bayesian"
DEFAULT_OUTDIR = HERE / "tables"
#: Read by barcode_shuffling/plot_activities.py to confirm the table it was given.
METHOD = "bayes_direct_activity_control_centered"
ACTIVITY_DEFINITION = (
    "per draw: log_gamma of the barcode minus the mean log_gamma of the barcode-only "
    "controls; activity is the posterior mean, activity_lo/hi the 2.5%/97.5% quantiles"
)
COUNT_COLUMNS = ["n_t7_pos", "n_cre_pos", "n_double_pos", "n_total"]


def barcode_activities(
    log_gamma: np.ndarray,
    cre_names: np.ndarray,
    cre_info: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """One row per barcode; ``log_gamma`` is ``(draws, n_barcodes)``."""
    if log_gamma.ndim != 2 or log_gamma.shape[1] != len(cre_names):
        raise ValueError(
            f"log_gamma has shape {log_gamma.shape}; expected (draws, {len(cre_names)})"
        )
    names = pd.Index(cre_names, dtype=str)
    missing = names.difference(cre_info.index)
    if len(missing):
        raise ValueError(f"cre_info lacks fitted barcodes: {missing[:5].tolist()}")
    info = cre_info.loc[names]
    control = info["is_control"].to_numpy(dtype=bool)
    if not control.any():
        raise ValueError("no control barcodes in cre_info; activity has no reference")

    draws = log_gamma.astype(np.float64)
    activity = draws - draws[:, control].mean(axis=1, keepdims=True)
    lo, hi = np.quantile(activity, [0.025, 0.975], axis=0)
    table = pd.DataFrame(
        {
            "group": GROUP_LABEL,
            "feature": names,
            "cre": info["element"].to_numpy(),
            "barcode": info["barcode"].to_numpy(),
            "is_control": control,
            "dna_count": info["DNA_count"].to_numpy(),
            "log_gamma_mean": draws.mean(axis=0),
            "log_gamma_sd": draws.std(axis=0, ddof=1),
            "activity": activity.mean(axis=0),
            "activity_sd": activity.std(axis=0, ddof=1),
            "activity_lo": lo,
            "activity_hi": hi,
        }
    )
    return table.join(counts.loc[names, COUNT_COLUMNS].reset_index(drop=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes-dir", type=Path, default=DEFAULT_BAYES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)

    tag = fit_tag(args.bayes_dir)
    fit_manifest = json.loads((args.bayes_dir / "run_manifest.json").read_text())
    posterior = load_posterior_samples(args.bayes_dir, tag, sites=["log_gamma"])
    groups = posterior["group_names"].tolist()
    if groups != [GROUP_LABEL]:
        raise ValueError(f"expected the single group {GROUP_LABEL!r}, got {groups}")
    cre_info = pd.read_csv(args.bayes_dir / "cre_info.csv", index_col="cre")
    counts = pd.read_csv(args.bayes_dir / f"{tag}_gamma.csv").set_index("cre")

    table = barcode_activities(
        posterior["log_gamma"][:, 0, :], posterior["cre_names"], cre_info, counts
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    table_path = args.outdir / "global_barcode_activities.csv"
    table.to_csv(table_path, index=False)

    posterior_path = args.bayes_dir / f"{tag}_posterior_samples.npz"
    write_json(
        args.outdir / "run_manifest.json",
        {
            "method": METHOD,
            "activity_definition": ACTIVITY_DEFINITION,
            "log_base": float(np.e),
            "n_cells": fit_manifest["n_cells"],
            "cell_filter": fit_manifest["cell_filter"],
            "n_draws": int(posterior["log_gamma"].shape[0]),
            "control_features": table.loc[table.is_control, "feature"].tolist(),
            "fit_tag": tag,
            "fit_method_variant": fit_manifest["method_variant"],
            "input": fit_manifest["input"],
            "dna_input": fit_manifest["config"]["dna_input"],
            "posterior": input_fingerprint(posterior_path),
            "outputs": {"barcode_activities": str(table_path)},
        },
    )
    log(
        f"[export] {len(table)} barcodes ({int(table.is_control.sum())} controls), "
        f"{fit_manifest['n_cells']:,} cells -> {table_path}"
    )


if __name__ == "__main__":
    main()
