#!/usr/bin/env python3
"""Precision/recall of replicate-concordant calls across matched T7 thresholds.

This is the published T7-threshold precision-recall figure
(`revision/bayesian_vs_fold_change/results/figures/final/`
`method_activity_t7_filter_precision_recall.pdf`) restricted, for both methods,
to the cCRE-cell-type pairs whose call agrees between the two experiments.

For every T7 threshold and each of the two methods

    Bayesian  : Joint+dropout draw-wise mean-of-seven-controls test
    Bootstrap : bootstrap replicate-wise mean-of-seven-controls test

the original and new low-dose test tables are intersected on their eligible
pairs, BH is recomputed inside that shared universe separately per experiment
(the definition behind `overlap_t7_ge50_significant_call_concordance_bh_q`), and
only the concordant pairs -- `both_significant` or `neither_significant` at
`q <= --q-cutoff` -- are handed to `benchmark_assay`. The tested universe of the
figure is therefore the concordant set, so the assay-positive denominator of
recall and the naive-precision baseline are restricted to it as well.

`--common-pairs` (the default, matching the published figure) first intersects
the eligible pairs of all four test tables at each threshold, so both methods
start from one identical universe before their own concordance filter is
applied. `--no-common-pairs` lets each method use its own origin/new
intersection; at T7 >= 50 that reproduces the saved Bayesian concordance table
exactly, which `--validate-calls` checks.

Precision, recall, the naive-precision baseline, the one-sided Fisher test and
the panel layout are all reused from
`revision/bayesian_vs_fold_change/code/plot_t7_filter_precision_recall.py`, so
the definitions are identical to the published figure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

HERE: Final[Path] = Path(__file__).resolve()
ANALYSIS_DIR: Final[Path] = HERE.parent.parent
REVISION_DIR: Final[Path] = ANALYSIS_DIR.parent
REPO_ROOT: Final[Path] = REVISION_DIR.parent
ORIGIN_CODE: Final[Path] = REVISION_DIR / "bayesian_vs_fold_change" / "code"
if str(ORIGIN_CODE) not in sys.path:
    sys.path.insert(0, str(ORIGIN_CODE))

from analysis_utils import write_json  # noqa: E402
from plot_t7_filter_precision_recall import (  # noqa: E402
    ASSAYS,
    assay_positive_for_tests,
    benchmark_assay,
    plot_precision_recall,
    read_assay,
)
from baystarrfish.stats import bh_fdr  # noqa: E402


ORIGIN_TABLES: Final[Path] = (
    REVISION_DIR / "bayesian_vs_fold_change" / "results" / "tables"
)
NEW_TABLES: Final[Path] = REVISION_DIR / "Bayes_NewData" / "tables"
DEFAULT_COMPARISON_DIR: Final[Path] = ANALYSIS_DIR / "results" / "comparison"

KEY: Final[list[str]] = ["group", "cre"]
BAYES_METHOD: Final[str] = "Joint+dropout mean controls"
BOOTSTRAP_METHOD: Final[str] = "Bootstrap mean controls"
CONCORDANT_STATUSES: Final[tuple[str, str]] = (
    "both_significant",
    "neither_significant",
)
STEM: Final[str] = "replicate_concordant_bh_call_precision_recall"


@dataclass(frozen=True)
class MethodSpec:
    """One method's pair of origin/new T7-threshold-series test tables."""

    label: str
    origin: Path
    new: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-bayesian",
        type=Path,
        default=ORIGIN_TABLES
        / "joint_dropout_direct_activity_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--new-bayesian",
        type=Path,
        default=NEW_TABLES / "new_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--origin-bootstrap",
        type=Path,
        default=ORIGIN_TABLES / "bootstrap_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--new-bootstrap",
        type=Path,
        default=NEW_TABLES
        / "new_bootstrap_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--t7-thresholds", type=float, nargs="+", default=[5, 10, 20, 50, 100]
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--common-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Intersect the eligible pairs of all four test tables at each T7 "
            "threshold before recomputing BH, so both methods share one "
            "pre-concordance universe (default: on)."
        ),
    )
    parser.add_argument(
        "--validate-calls",
        type=Path,
        default=None,
        help=(
            "Saved *_significant_call_concordance_calls.csv.gz to reproduce "
            "exactly with the Bayesian arm. Requires --no-common-pairs."
        ),
    )
    parser.add_argument("--validate-threshold", type=float, default=50.0)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--stem", default=STEM)
    return parser.parse_args()


def read_series(path: Path, label: str, thresholds: list[float]) -> pd.DataFrame:
    """Load one method's threshold series, keeping only `label`'s rows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing threshold-series table: {path}. Run the matching "
            "test_*_mean_negative_control_activity_threshold_series.py first."
        )
    frame = pd.read_csv(
        path, usecols=["t7_threshold", "method", "group", "cre", "p_right"]
    )
    frame = frame.loc[frame["method"].astype(str).eq(label)].copy()
    if frame.empty:
        raise ValueError(f"{path} holds no rows for method {label!r}")
    frame["t7_threshold"] = frame["t7_threshold"].astype(float).round(6)
    frame[KEY] = frame[KEY].astype(str)
    frame = frame.loc[frame["t7_threshold"].isin(thresholds)]
    missing = sorted(set(thresholds) - set(frame["t7_threshold"]))
    if missing:
        raise ValueError(f"{path} is missing T7 thresholds {missing}")
    if frame.duplicated(["t7_threshold", *KEY]).any():
        raise ValueError(f"{path} contains duplicated threshold/group/cre rows")
    return frame[["t7_threshold", *KEY, "p_right"]].reset_index(drop=True)


def pair_index(frame: pd.DataFrame, threshold: float) -> pd.MultiIndex:
    selected = frame.loc[frame["t7_threshold"].eq(threshold), KEY]
    return pd.MultiIndex.from_frame(selected)


def common_pair_index(frames: list[pd.DataFrame], threshold: float) -> pd.MultiIndex:
    index = pair_index(frames[0], threshold)
    for frame in frames[1:]:
        index = index.intersection(pair_index(frame, threshold))
    if len(index) == 0:
        raise ValueError(f"No pairs are shared by all tables at T7 >= {threshold:g}")
    return index


def concordance_calls(
    spec: MethodSpec,
    origin: pd.DataFrame,
    new: pd.DataFrame,
    threshold: float,
    q_cutoff: float,
    restrict: pd.MultiIndex | None,
) -> pd.DataFrame:
    """Per-pair origin/new BH calls and their concordance at one threshold."""
    merged = (
        origin.loc[origin["t7_threshold"].eq(threshold), [*KEY, "p_right"]]
        .rename(columns={"p_right": "origin_p_right"})
        .merge(
            new.loc[new["t7_threshold"].eq(threshold), [*KEY, "p_right"]].rename(
                columns={"p_right": "new_p_right"}
            ),
            on=KEY,
            how="inner",
            validate="one_to_one",
        )
    )
    if restrict is not None:
        keep = pd.MultiIndex.from_frame(merged[KEY]).isin(restrict)
        merged = merged.loc[keep].reset_index(drop=True)
    if merged.empty:
        raise ValueError(
            f"{spec.label} has no shared pairs at T7 >= {threshold:g}"
        )
    merged["origin_q_bh"] = bh_fdr(merged["origin_p_right"].to_numpy(float))
    merged["new_q_bh"] = bh_fdr(merged["new_p_right"].to_numpy(float))
    origin_significant = merged["origin_q_bh"].le(q_cutoff)
    new_significant = merged["new_q_bh"].le(q_cutoff)
    merged["origin_significant_bh_q"] = origin_significant
    merged["new_significant_bh_q"] = new_significant
    merged["bh_q_call_status"] = np.select(
        [
            origin_significant & new_significant,
            ~origin_significant & ~new_significant,
            origin_significant,
        ],
        ["both_significant", "neither_significant", "origin_only_significant"],
        default="new_only_significant",
    )
    # The concordant call is significant in both experiments or in neither, so
    # the per-pair maximum of the two q-values reproduces that call on its own
    # and is the conservative statistic to carry into benchmark_assay.
    merged["q_right"] = merged[["origin_q_bh", "new_q_bh"]].max(axis=1)
    merged.insert(0, "t7_threshold", float(threshold))
    merged.insert(1, "method", spec.label)
    return merged


def validate_against_calls(
    calls: pd.DataFrame, saved_path: Path, threshold: float
) -> dict[str, object]:
    """Require exact agreement with a saved concordance table."""
    saved = pd.read_csv(saved_path)
    required = [*KEY, "origin_q_bh", "new_q_bh", "bh_q_call_status"]
    missing = [column for column in required if column not in saved.columns]
    if missing:
        raise ValueError(f"{saved_path} lacks required columns: {missing}")
    saved[KEY] = saved[KEY].astype(str)
    subject = calls.loc[calls["t7_threshold"].eq(threshold), required]
    if len(subject) != len(saved):
        raise ValueError(
            f"{saved_path} has {len(saved)} pairs but the recomputed universe at "
            f"T7 >= {threshold:g} has {len(subject)}"
        )
    merged = subject.merge(saved, on=KEY, how="inner", suffixes=("", "_saved"))
    if len(merged) != len(saved):
        raise ValueError(f"{saved_path} and the recomputed universe differ in pairs")
    for column in ("origin_q_bh", "new_q_bh"):
        if not np.allclose(merged[column], merged[f"{column}_saved"], atol=1e-12):
            raise ValueError(f"{column} does not reproduce {saved_path}")
    if not merged["bh_q_call_status"].eq(merged["bh_q_call_status_saved"]).all():
        raise ValueError(f"Call statuses do not reproduce {saved_path}")
    return {
        "path": str(saved_path),
        "t7_threshold": float(threshold),
        "pairs": int(len(saved)),
        "reproduced": True,
    }


def status_counts(calls: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    counted = (
        calls.groupby(["method", "t7_threshold", "bh_q_call_status"], sort=True)
        .size()
        .rename("pairs")
        .reset_index()
    )
    out: dict[str, dict[str, dict[str, int]]] = {}
    for (method, threshold), frame in counted.groupby(["method", "t7_threshold"]):
        out.setdefault(str(method), {})[f"{threshold:g}"] = {
            str(row.bh_q_call_status): int(row.pairs)
            for row in frame.itertuples(index=False)
        }
    return out


def main() -> None:
    args = parse_args()
    thresholds = sorted({round(float(value), 6) for value in args.t7_thresholds})
    if args.validate_calls is not None and args.common_pairs:
        raise SystemExit("--validate-calls requires --no-common-pairs")

    specs = (
        MethodSpec(BAYES_METHOD, args.origin_bayesian, args.new_bayesian),
        MethodSpec(BOOTSTRAP_METHOD, args.origin_bootstrap, args.new_bootstrap),
    )
    series = {
        spec.label: (
            read_series(spec.origin, spec.label, thresholds),
            read_series(spec.new, spec.label, thresholds),
        )
        for spec in specs
    }

    common_counts: dict[str, int] = {}
    call_frames = []
    for threshold in thresholds:
        restrict = None
        if args.common_pairs:
            restrict = common_pair_index(
                [frame for pair in series.values() for frame in pair], threshold
            )
            common_counts[f"{threshold:g}"] = int(len(restrict))
        for spec in specs:
            origin, new = series[spec.label]
            call_frames.append(
                concordance_calls(
                    spec, origin, new, threshold, args.q_cutoff, restrict
                )
            )
    calls = pd.concat(call_frames, ignore_index=True)

    validation = None
    if args.validate_calls is not None:
        validation = validate_against_calls(
            calls.loc[calls["method"].eq(BAYES_METHOD)],
            args.validate_calls,
            round(float(args.validate_threshold), 6),
        )

    concordant = calls.loc[
        calls["bh_q_call_status"].isin(CONCORDANT_STATUSES)
    ].reset_index(drop=True)
    disagreeing = concordant.loc[
        concordant["origin_significant_bh_q"].ne(concordant["new_significant_bh_q"])
    ]
    if not disagreeing.empty:
        raise ValueError("Concordant universe contains pairs whose calls differ")

    tables_dir = args.comparison_dir / "tables"
    figures_dir = args.comparison_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    assays = {name: read_assay(path) for name, path in ASSAYS.items()}
    methods = tuple(spec.label for spec in specs)
    summary = pd.concat(
        [
            benchmark_assay(concordant, name, assay, args.q_cutoff)
            for name, assay in assays.items()
        ],
        ignore_index=True,
    )
    method_order = {method: index for index, method in enumerate(methods)}
    assay_order = {assay: index for index, assay in enumerate(assays)}
    summary = (
        summary.assign(
            assay_order=summary["assay"].map(assay_order),
            method_order=summary["method"].map(method_order),
        )
        .sort_values(["assay_order", "t7_threshold", "method_order"])
        .drop(columns=["assay_order", "method_order"])
        .reset_index(drop=True)
    )

    summary_path = tables_dir / f"{args.stem}.csv"
    calls_path = tables_dir / f"{args.stem}_calls.csv.gz"
    summary.to_csv(summary_path, index=False)
    calls.to_csv(calls_path, index=False)

    title_note = (
        "Replicate-concordant pairs only: original and new low-dose calls agree "
        f"at BH q <= {args.q_cutoff:g}"
    )
    figure_paths = []
    for suffix in ("pdf", "png"):
        output = figures_dir / f"{args.stem}.{suffix}"
        plot_precision_recall(
            summary,
            output,
            methods=methods,
            common_pairs=args.common_pairs,
            title_note=title_note,
        )
        figure_paths.append(str(output))

    coverage: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for name, assay in assays.items():
        positive = assay_positive_for_tests(concordant, assay)
        covered = concordant["group"].isin(assay.index).to_numpy(bool)
        for (method, threshold), index in concordant.groupby(
            ["method", "t7_threshold"], sort=True
        ).groups.items():
            rows = concordant.index.get_indexer(pd.Index(index))
            coverage.setdefault(name, {}).setdefault(str(method), {})[
                f"{threshold:g}"
            ] = {
                "n_concordant_pairs": int(len(rows)),
                "n_assay_positive_concordant_pairs": int(positive[rows].sum()),
                "n_concordant_pairs_in_groups_absent_from_assay": int(
                    (~covered[rows]).sum()
                ),
            }
    write_json(
        tables_dir / f"{args.stem}_manifest.json",
        {
            "figure_stem": args.stem,
            "figures": figure_paths,
            "metrics_table": str(summary_path),
            "calls_table": str(calls_path),
            "sources": {
                spec.label: {"origin": str(spec.origin), "new": str(spec.new)}
                for spec in specs
            },
            "definition_source": str(
                ORIGIN_CODE / "plot_t7_filter_precision_recall.py"
            ),
            "methods": list(methods),
            "t7_thresholds": thresholds,
            "q_cutoff": float(args.q_cutoff),
            "common_pairs": bool(args.common_pairs),
            "common_pair_counts_by_t7_threshold": common_counts or None,
            "call_status_counts": status_counts(calls),
            "assay_coverage_concordant_pairs": coverage,
            "assays": {name: str(path) for name, path in ASSAYS.items()},
            "validation": validation,
            "definitions": {
                "eligible_pair": (
                    "target T7 >= threshold and combined seven-control T7 >= "
                    "threshold within the experiment that produced the table"
                ),
                "shared_universe_bh": (
                    "At each T7 threshold the origin and new eligible pairs are "
                    "intersected and BH is recomputed inside that intersection "
                    "separately for each experiment"
                ),
                "common_pairs": (
                    "With --common-pairs the intersection is taken across all "
                    "four test tables, so both methods share one universe before "
                    "the concordance filter"
                ),
                "concordant_pair": (
                    "both_significant or neither_significant at BH q <= q_cutoff "
                    "in the shared universe"
                ),
                "tested": "concordant pairs; the figure evaluates only these",
                "q_right": (
                    "per-pair max(origin_q_bh, new_q_bh); on concordant pairs it "
                    "reproduces the shared call"
                ),
                "TP": "significant concordant pairs with assay value > 0.5",
                "precision": "TP / significant; 0 when there are no calls",
                "recall": "TP / assay-positive concordant pairs",
                "naive_precision_baseline": (
                    "assay-positive concordant pairs / concordant pairs"
                ),
            },
        },
    )
    print(
        summary[
            [
                "assay",
                "t7_threshold",
                "method",
                "TP",
                "significant",
                "tested",
                "assay_positive",
                "precision",
                "recall",
                "fisher_oddsratio",
                "fisher_p",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
