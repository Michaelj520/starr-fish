#!/usr/bin/env python3
"""Run Bayesian ordinary-control mean tests across matched T7 thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The shared analysis layer (analysis_utils and the plot_* modules that other
# scripts import) stays in the parent code/ directory.
import sys as _sys
from pathlib import Path as _Path
_CODE_DIR = _Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_CODE_DIR))

from analysis_utils import (
    ANALYSIS_DIR,
    DEFAULT_H5AD,
    FIGURES_WORK,
    OLD_DATA_BAYES,
    write_json,
)
from plot_method_activity_correlation import read_cre_blacklist
from test_individual_negative_control_loo_empirical_fdr import (
    POOLED_NAME,
    load_grouped_t7,
)
from test_mean_negative_control_activity import METHOD, compute_tests


FILTERED_METHOD = "Joint+dropout mean controls (filtered)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayes-dir",
        type=Path,
        default=OLD_DATA_BAYES,
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--t7-thresholds", type=float, nargs="+", default=[5, 10, 20, 50, 100]
    )
    parser.add_argument("--effect-threshold", type=float, default=0.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--skip-filtered",
        action="store_true",
        help=(
            "Compute only the all-seven-control reference and skip the "
            "control-T7-matched series."
        ),
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--unfiltered-stem",
        default="joint_dropout_direct_activity_mean_negative_control_tests_t7_series",
    )
    parser.add_argument(
        "--filtered-stem",
        default=(
            "joint_dropout_direct_activity_mean_negative_control_tests_"
            "control_t7_matched_series"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(value) for value in args.t7_thresholds})
    run_manifest = json.loads((args.bayes_dir / "run_manifest.json").read_text())
    posterior_path = args.bayes_dir / f"{run_manifest['tag']}_posterior_samples.npz"
    negative_controls = pd.read_csv(args.bayes_dir / "negative_controls.csv").iloc[
        :, 0
    ].astype(str).tolist()
    blacklist = read_cre_blacklist(args.bayes_dir)

    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre_names = posterior["cre_names"].astype(str)
        ordinary_mask = all_cre_names != POOLED_NAME
        cre_names = all_cre_names[ordinary_mask]
        log_gamma = posterior["log_gamma"][:, :, ordinary_mask].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cre_names, negative_controls))
    if len(control_indices) != 7:
        raise ValueError(f"Expected seven ordinary negative controls; found {len(control_indices)}")
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, negative_controls) & ~np.isin(cre_names, list(blacklist))
    )
    t7_totals, group_classes, group_cell_counts = load_grouped_t7(
        args.h5ad, groups, cre_names
    )

    unfiltered_frames = []
    filtered_frames = []
    for threshold in thresholds:
        unfiltered_frames.append(
            compute_tests(
                log_gamma,
                groups,
                cre_names,
                target_indices,
                control_indices,
                t7_totals,
                group_classes,
                group_cell_counts,
                threshold,
                args.effect_threshold,
                None,
                METHOD,
            )
        )
        if args.skip_filtered:
            continue
        filtered_frames.append(
            compute_tests(
                log_gamma,
                groups,
                cre_names,
                target_indices,
                control_indices,
                t7_totals,
                group_classes,
                group_cell_counts,
                threshold,
                args.effect_threshold,
                threshold,
                FILTERED_METHOD,
            )
        )

    unfiltered = pd.concat(unfiltered_frames, ignore_index=True)
    unfiltered["significant_q"] = unfiltered["q_right"].le(args.q_cutoff)
    unfiltered_path = args.tables_dir / f"{args.unfiltered_stem}.csv.gz"
    unfiltered.to_csv(unfiltered_path, index=False)
    emitted = [unfiltered]
    filtered_path = None
    if not args.skip_filtered:
        filtered = pd.concat(filtered_frames, ignore_index=True)
        filtered["significant_q"] = filtered["q_right"].le(args.q_cutoff)
        filtered_path = args.tables_dir / f"{args.filtered_stem}.csv.gz"
        filtered.to_csv(filtered_path, index=False)
        emitted.append(filtered)

    summary = pd.concat(emitted, ignore_index=True).groupby(
        ["method", "t7_threshold"], sort=False
    ).agg(
        eligible_tests=("q_right", "size"),
        significant_tests=("significant_q", "sum"),
        tested_cell_types=("group", "nunique"),
        minimum_controls=("n_negative_controls", "min"),
        maximum_controls=("n_negative_controls", "max"),
    ).reset_index()
    manifest_path = args.figures_dir / "joint_dropout_mean_control_t7_series_manifest.json"
    write_json(
        manifest_path,
        {
            "model": "Joint+dropout ordinary-and-pooled negative controls",
            "posterior": str(posterior_path),
            "thresholds": thresholds,
            "unfiltered_reference": "mean of all seven ordinary controls",
            "skip_filtered": bool(args.skip_filtered),
            "filtered_reference": (
                "mean of controls with individual cell-type T7 >= the panel threshold; "
                "at least one retained control required"
            ),
            "activity_definition": "raw posterior log_gamma; alpha is not subtracted",
            "q_cutoff": args.q_cutoff,
            "summary": summary.to_dict(orient="records"),
            "outputs": {
                "unfiltered_tests": str(unfiltered_path),
                "filtered_tests": (
                    None if filtered_path is None else str(filtered_path)
                ),
                "manifest": str(manifest_path),
            },
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
