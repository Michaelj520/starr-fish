#!/usr/bin/env python3
"""Run bootstrap ordinary-control mean tests across matched T7 thresholds."""

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
    OLD_DATA_BOOTSTRAP,
    write_json,
)
from compute_t7_filter_negative_control_stats import (
    aligned_t7_totals,
    bh_fdr,
    read_negative_controls,
)
from plot_method_activity_correlation import read_cre_blacklist
from test_bootstrap_mean_negative_control_activity import METHOD, compute_statistics


FILTERED_METHOD = "Bootstrap mean controls (filtered)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=OLD_DATA_BOOTSTRAP,
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--t7-thresholds", type=float, nargs="+", default=[5, 10, 20, 50, 100]
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--skip-filtered",
        action="store_true",
        help=(
            "Compute only the all-seven-control reference. The filtered series "
            "needs one extra full pass over the bootstrap array per threshold."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--unfiltered-stem", default="bootstrap_mean_negative_control_tests_t7_series"
    )
    parser.add_argument(
        "--filtered-stem",
        default="bootstrap_mean_negative_control_tests_control_t7_matched_series",
    )
    return parser.parse_args()


def build_tests(
    statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    threshold: float,
    method: str,
    groups: pd.Index,
    cres: pd.Index,
    negative_controls: list[str],
    blacklist: set[str],
    pair_t7: pd.DataFrame,
    control_include: np.ndarray,
    q_cutoff: float,
) -> pd.DataFrame:
    (
        activity_mean,
        effect_mean,
        p_right,
        n_testable,
        mean_controls_used,
        reference_count,
    ) = statistics
    control_t7 = pair_t7.loc[:, negative_controls].sum(axis=1).to_numpy(float)
    n_controls_passing_t7 = control_include.sum(axis=1)
    group_grid, cre_grid = np.meshgrid(
        groups.to_numpy(str), cres.to_numpy(str), indexing="ij"
    )
    is_negative = np.isin(cre_grid.ravel(), negative_controls)
    is_blacklisted = np.isin(cre_grid.ravel(), list(blacklist))
    target_t7 = pair_t7.to_numpy(float).ravel()
    control_t7_flat = np.repeat(control_t7, len(cres))
    valid = (
        (target_t7 >= threshold)
        & (control_t7_flat >= threshold)
        & ~is_negative
        & ~is_blacklisted
        & np.isfinite(p_right.ravel())
    )
    tests = pd.DataFrame(
        {
            "t7_threshold": float(threshold),
            "method": method,
            "group": group_grid.ravel()[valid],
            "cre": cre_grid.ravel()[valid],
            "target_t7_total": target_t7[valid],
            "negative_control_t7_total": control_t7_flat[valid],
            "activity_mean": activity_mean.ravel()[valid],
            "effect_vs_mean_control_mean": effect_mean.ravel()[valid],
            "posterior_probability_above_mean_control": 1.0
            - p_right.ravel()[valid],
            "p_right": p_right.ravel()[valid],
            "n_testable_bootstraps": n_testable.ravel()[valid],
            "mean_controls_used_per_reference": np.repeat(
                mean_controls_used, len(cres)
            )[valid],
            "n_controls_passing_t7": np.repeat(
                n_controls_passing_t7, len(cres)
            )[valid],
            "n_bootstraps_with_control_reference": np.repeat(
                reference_count, len(cres)
            )[valid],
        }
    )
    tests["q_right"] = bh_fdr(tests["p_right"].to_numpy(float))
    tests["significant_q"] = tests["q_right"].le(q_cutoff)
    return tests


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted({float(value) for value in args.t7_thresholds})
    axes = json.loads((args.bootstrap_dir / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    negative_controls = read_negative_controls(args.bootstrap_dir)
    negative_mask = cres.isin(negative_controls)
    if int(negative_mask.sum()) != 7:
        raise ValueError(f"Expected seven negative controls; found {negative_mask.sum()}")
    blacklist = set(read_cre_blacklist(args.bootstrap_dir))
    activity_array = np.load(
        args.bootstrap_dir / "celltype_activity_array.npy", mmap_mode="r"
    )
    if activity_array.shape[1:] != (len(groups), len(cres)):
        raise ValueError("Bootstrap activity array does not match saved axes")
    pair_t7 = aligned_t7_totals(args.h5ad, groups, cres)
    control_pair_t7 = pair_t7.loc[:, negative_controls].to_numpy(float)

    print("[bootstrap-series] computing all-seven-control reference")
    unfiltered_include = np.ones(control_pair_t7.shape, dtype=bool)
    unfiltered_statistics = compute_statistics(
        activity_array, negative_mask, unfiltered_include, args.chunk_size
    )
    unfiltered_frames = [
        build_tests(
            unfiltered_statistics,
            threshold=threshold,
            method=METHOD,
            groups=groups,
            cres=cres,
            negative_controls=negative_controls,
            blacklist=blacklist,
            pair_t7=pair_t7,
            control_include=unfiltered_include,
            q_cutoff=args.q_cutoff,
        )
        for threshold in thresholds
    ]

    filtered_frames = []
    for threshold in [] if args.skip_filtered else thresholds:
        print(
            "[bootstrap-series] computing reference from controls with "
            f"individual T7 >= {threshold:g}"
        )
        control_include = control_pair_t7 >= threshold
        statistics = compute_statistics(
            activity_array, negative_mask, control_include, args.chunk_size
        )
        filtered_frames.append(
            build_tests(
                statistics,
                threshold=threshold,
                method=FILTERED_METHOD,
                groups=groups,
                cres=cres,
                negative_controls=negative_controls,
                blacklist=blacklist,
                pair_t7=pair_t7,
                control_include=control_include,
                q_cutoff=args.q_cutoff,
            )
        )

    unfiltered = pd.concat(unfiltered_frames, ignore_index=True)
    unfiltered_path = args.tables_dir / f"{args.unfiltered_stem}.csv.gz"
    unfiltered.to_csv(unfiltered_path, index=False)
    emitted = [unfiltered]
    filtered_path = None
    if not args.skip_filtered:
        filtered = pd.concat(filtered_frames, ignore_index=True)
        filtered_path = args.tables_dir / f"{args.filtered_stem}.csv.gz"
        filtered.to_csv(filtered_path, index=False)
        emitted.append(filtered)
    summary = pd.concat(emitted, ignore_index=True).groupby(
        ["method", "t7_threshold"], sort=False
    ).agg(
        eligible_tests=("q_right", "size"),
        significant_tests=("significant_q", "sum"),
        tested_cell_types=("group", "nunique"),
        minimum_controls=("n_controls_passing_t7", "min"),
        maximum_controls=("n_controls_passing_t7", "max"),
    ).reset_index()
    manifest_path = args.figures_dir / "bootstrap_mean_control_t7_series_manifest.json"
    write_json(
        manifest_path,
        {
            "method": "Bootstrap mean ordinary-control tests",
            "bootstrap_dir": str(args.bootstrap_dir),
            "thresholds": thresholds,
            "unfiltered_reference": "mean of all seven ordinary controls",
            "skip_filtered": bool(args.skip_filtered),
            "filtered_reference": (
                "mean of controls with individual cell-type T7 >= the panel threshold; "
                "at least one retained control required"
            ),
            "activity_definition": "log bootstrap activity; alpha is not subtracted",
            "contrast_definition": (
                "target log activity minus the arithmetic mean of finite log activities "
                "among retained controls within each bootstrap replicate"
            ),
            "p_right_definition": "fraction of valid bootstrap contrasts <= 0",
            "multiple_testing": "BH separately within each method and T7 threshold",
            "q_cutoff": args.q_cutoff,
            "n_bootstraps": int(activity_array.shape[0]),
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
