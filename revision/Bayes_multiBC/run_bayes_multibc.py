#!/usr/bin/env python3
"""Fit the Bayes_OldData model to the SFv6 multi-barcode data, one activity per barcode."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

# Importing baystarrfish costs nothing: the facade is lazy and pulls in neither
# JAX nor NumPyro until a model symbol is touched, which is what lets --cpu set
# JAX_PLATFORMS below before the backend is chosen.
import baystarrfish as bsf
from baystarrfish._log import log
from baystarrfish.io import input_fingerprint, write_fit

from multibc_data import (
    DEFAULT_DNA_CSV,
    DEFAULT_H5AD,
    GROUP_LABEL,
    HERE,
    SATURATION_FLAG,
    load_multibc_counts,
)

# The model of Bayes_OldData/bayesian/run_manifest.json. These are pinned rather
# than exposed as flags: changing any one of them fits a different model.
LEVEL = "subclass"
CHANNEL = "joint"
METHOD = "svi"
INFECTION_MODEL = "copy_number_dropout"
ACTIVITY_MODEL = "direct"
NEGATIVE_CONTROL_MODE = "ordinary"
GUIDE = "AutoNormal"
METHOD_VARIANT = "bayesian_joint_dropout_direct_activity_ordinary_negative_controls"
DROPOUT_PRIOR = {
    "label": "default_beta_1_9",
    "p_drop_t7_alpha": 1.0,
    "p_drop_t7_beta": 9.0,
    "p_drop_t7_mean": 0.1,
    "p_drop_cre_alpha": 1.0,
    "p_drop_cre_beta": 9.0,
    "p_drop_cre_mean": 0.1,
}
TAG = f"{LEVEL}_{CHANNEL}_{INFECTION_MODEL}_{METHOD}"

# Cell selection for this dataset: cells whose images saturated are not fitted.
DROP_SATURATION_FAIL = True
CELL_FILTER = f"obs[{SATURATION_FLAG!r}] == False" if DROP_SATURATION_FAIL else "none"
DEFAULT_OUTDIR = HERE / "bayesian"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--dna-csv",
        type=Path,
        default=DEFAULT_DNA_CSV,
        help="Per-barcode plasmid DNA counts; log1p(DNA_count) is the abundance prior.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    # Optimiser settings, defaulting to the Bayes_OldData run.
    parser.add_argument("--kmax", type=int, default=60)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num-posterior", type=int, default=1_000)
    parser.add_argument(
        "--posterior-sites",
        nargs="+",
        default=["log_gamma", "log_rho", "log_a"],
        help="Posterior sites to save in *_posterior_samples.npz; 'all' saves every site.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a completed fit. Without it, an outdir holding a run_manifest.json is refused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_manifest = args.outdir / "run_manifest.json"
    if existing_manifest.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite the completed fit in {args.outdir}\n"
            f"  {existing_manifest} already exists\n"
            "pass --overwrite, or point --outdir somewhere else"
        )
    if args.cpu:
        # Must precede the first touch of a model symbol, which is what triggers
        # the JAX import and therefore the backend choice.
        os.environ["JAX_PLATFORMS"] = "cpu"

    priors = dataclasses.replace(
        bsf.ModelPriors(),
        p_drop_t7_alpha=DROPOUT_PRIOR["p_drop_t7_alpha"],
        p_drop_t7_beta=DROPOUT_PRIOR["p_drop_t7_beta"],
        p_drop_cre_alpha=DROPOUT_PRIOR["p_drop_cre_alpha"],
        p_drop_cre_beta=DROPOUT_PRIOR["p_drop_cre_beta"],
    )
    data = load_multibc_counts(
        args.h5ad,
        args.dna_csv,
        drop_saturation_fail=DROP_SATURATION_FAIL,
        max_cells=args.max_cells,
        seed=args.seed,
    )

    log(f"[bayesian] fitting {data.n_cre} barcodes, {data.n_subclasses} group, tag={TAG}")
    posterior_sites = (
        ["all"] if any(site.lower() == "all" for site in args.posterior_sites)
        else args.posterior_sites
    )
    result = bsf.fit(
        **data.to_run_kwargs(),
        level=LEVEL,
        channel=CHANNEL,
        method=METHOD,
        kmax=args.kmax,
        num_steps=args.steps,
        lr=args.lr,
        guide=GUIDE,
        num_posterior=args.num_posterior,
        seed=args.seed,
        infection_model=INFECTION_MODEL,
        activity_model=ACTIVITY_MODEL,
        priors=priors,
        posterior_sites_to_return=posterior_sites,
    )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "dna_input": input_fingerprint(args.dna_csv),
            "activity_unit": "barcode",
            "group_label": GROUP_LABEL,
            "cell_filter": CELL_FILTER,
            "blacklist_cre": data.blacklist,
            "max_cells": args.max_cells,
            "section": data.section,
            "posterior_sites_to_return": posterior_sites,
            "dropout_model": "zero_inflated",
            "dropout_prior": DROPOUT_PRIOR,
            "negative_control_mode": NEGATIVE_CONTROL_MODE,
            "activity_model": ACTIVITY_MODEL,
            "annotated_negative_control_cre": data.negative_controls,
            "pooled_negative_control": data.pooled_negative_control,
        }
    )

    write_fit(
        result,
        args.outdir,
        TAG,
        data=data,
        input_path=args.h5ad,
        manifest_extra={
            "method_variant": METHOD_VARIANT,
            "activity_model": ACTIVITY_MODEL,
            "dropout_prior": DROPOUT_PRIOR,
            "activity_unit": "barcode",
            "cell_filter": CELL_FILTER,
        },
    )
    log(f"[bayesian] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
