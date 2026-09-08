#!/usr/bin/env python3
"""Precision/recall of T7-filtered significant cCRE calls against assays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score, precision_recall_curve

from analysis_utils import (
    ANALYSIS_DIR,
    FIGURES_WORK,
    PANEL_SIDE_INCHES,
    STARRFISH_DATA,
    display_label,
    fit_panel_size,
    write_json,
)
from test_individual_negative_control_loo_empirical_fdr import assign_empirical_fdr
from baystarrfish.stats import bh_fdr


METHODS = (
    "Bootstrap",
    "Joint",
    "Decoupled",
    "Joint+dropout",
    "Decoupled+dropout",
    "Metacell Bayesian",
)
LOO_METHOD = "Joint+dropout LOO"
LOO_SCORE_BH_METHOD = "LOO score BH (exploratory)"
MAX_CONTROL_METHOD = "Joint+dropout max control"
MEAN_CONTROL_METHOD = "Joint+dropout mean controls"
BOOTSTRAP_MEAN_CONTROL_METHOD = "Bootstrap mean controls"
FILTERED_MEAN_CONTROL_METHOD = "Joint+dropout mean controls (filtered)"
FILTERED_BOOTSTRAP_MEAN_CONTROL_METHOD = "Bootstrap mean controls (filtered)"
METHOD_COLORS = {
    "Bootstrap": "#f58518",
    "Joint": "#4c78a8",
    "Decoupled": "#54a24b",
    "Joint+dropout": "#b279a2",
    "Decoupled+dropout": "#e45756",
    "Metacell Bayesian": "#72b7b2",
    LOO_METHOD: "#2f2f2f",
    LOO_SCORE_BH_METHOD: "#8c564b",
    MAX_CONTROL_METHOD: "#2f2f2f",
    MEAN_CONTROL_METHOD: "C0",
    BOOTSTRAP_MEAN_CONTROL_METHOD: "C1",
    FILTERED_MEAN_CONTROL_METHOD: "#7f7f7f",
    FILTERED_BOOTSTRAP_MEAN_CONTROL_METHOD: "#bcbd22",
}
#: Width reserved for the method legend, which sits outside the top-right panel.
LEGEND_INCHES = 2.2

ASSAYS = {
    "ATAC peak": STARRFISH_DATA / "cre_atac_peaks.csv",
    "Chromatin-a": STARRFISH_DATA / "cre_chromatin_state_a.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "method_activity_t7_filter_negative_control_tests.csv.gz",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_WORK
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=ANALYSIS_DIR / "results" / "tables"
    )
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--loo-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / (
            "joint_dropout_direct_activity_individual_negative_control_"
            "loo_empirical_fdr_target_tests.csv"
        ),
        help="Target tests from the individual-negative-control LOO empirical-FDR analysis.",
    )
    parser.add_argument(
        "--loo-null",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / (
            "joint_dropout_direct_activity_individual_negative_control_"
            "loo_empirical_fdr_loo_null.csv"
        ),
        help="LOO negative-control null scores used to recompute empirical q-values on common pairs.",
    )
    parser.add_argument("--loo-threshold", type=float, default=50.0)
    parser.add_argument(
        "--skip-loo",
        action="store_true",
        help="Do not add the individual-negative-control LOO method.",
    )
    parser.add_argument(
        "--include-loo-score-bh",
        action="store_true",
        help="Also treat 1 - LOO score as an exploratory p-value and apply BH.",
    )
    parser.add_argument(
        "--max-control-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "joint_dropout_direct_activity_max_negative_control_tests.csv.gz",
    )
    parser.add_argument(
        "--include-max-control",
        action="store_true",
        help="Add the posterior test against the draw-wise maximum ordinary control.",
    )
    parser.add_argument(
        "--mean-control-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "joint_dropout_direct_activity_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--include-mean-control",
        action="store_true",
        help="Add the posterior test against the draw-wise mean ordinary control.",
    )
    parser.add_argument(
        "--bootstrap-mean-control-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "bootstrap_mean_negative_control_tests_t7_series.csv.gz",
    )
    parser.add_argument(
        "--include-bootstrap-mean-control",
        action="store_true",
        help="Add the bootstrap test against the replicate-wise mean ordinary control.",
    )
    parser.add_argument(
        "--filtered-mean-control-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / (
            "joint_dropout_direct_activity_mean_negative_control_tests_"
            "control_t7_matched_series.csv.gz"
        ),
    )
    parser.add_argument(
        "--include-filtered-mean-control",
        action="store_true",
    )
    parser.add_argument(
        "--filtered-bootstrap-mean-control-tests",
        type=Path,
        default=ANALYSIS_DIR
        / "results"
        / "tables"
        / "bootstrap_mean_negative_control_tests_control_t7_matched_series.csv.gz",
    )
    parser.add_argument(
        "--include-filtered-bootstrap-mean-control",
        action="store_true",
    )
    parser.add_argument(
        "--common-pairs",
        action="store_true",
        help="Evaluate every method on the intersection of tested pairs and recompute BH q-values.",
    )
    parser.add_argument(
        "--pr-threshold",
        type=float,
        default=50.0,
        help="T7 threshold shown in the full precision-recall curves.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=METHODS,
        default=list(METHODS),
        help="Legacy methods to include in the comparison; pass no values to omit them.",
    )
    parser.add_argument("--stem", default="method_activity_t7_filter")
    return parser.parse_args()


def read_assay(path: Path) -> pd.DataFrame:
    assay = pd.read_csv(path, index_col=0)
    assay.index = assay.index.astype(str).str.replace("/", "-", regex=False)
    assay.columns = assay.columns.astype(str)
    return assay


def read_loo_tests(path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "group",
            "cre",
            "test_statistic",
            "empirical_p",
            "empirical_q",
        ],
    )
    return frame.assign(
        t7_threshold=float(threshold),
        method=LOO_METHOD,
        p_right=frame["empirical_p"].to_numpy(float),
        q_right=frame["empirical_q"].to_numpy(float),
        ranking_score=frame["test_statistic"].to_numpy(float),
        q_source="loo_empirical_fdr",
    )


def make_loo_score_bh_tests(loo_tests: pd.DataFrame) -> pd.DataFrame:
    frame = loo_tests.copy()
    frame["method"] = LOO_SCORE_BH_METHOD
    frame["p_right"] = 1.0 - frame["ranking_score"].to_numpy(float)
    frame["q_right"] = bh_fdr(frame["p_right"])
    frame["q_source"] = "bh"
    return frame


def read_max_control_tests(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "t7_threshold",
            "group",
            "cre",
            "posterior_probability_above_max_control",
            "p_right",
            "q_right",
        ],
    )
    return frame.assign(
        method=MAX_CONTROL_METHOD,
        ranking_score=frame[
            "posterior_probability_above_max_control"
        ].to_numpy(float),
        q_source="bh",
    )


def read_mean_control_tests(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "t7_threshold",
            "group",
            "cre",
            "posterior_probability_above_mean_control",
            "p_right",
            "q_right",
        ],
    )
    return frame.assign(
        method=MEAN_CONTROL_METHOD,
        ranking_score=frame[
            "posterior_probability_above_mean_control"
        ].to_numpy(float),
        q_source="bh",
    )


def read_bootstrap_mean_control_tests(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "t7_threshold",
            "group",
            "cre",
            "posterior_probability_above_mean_control",
            "p_right",
            "q_right",
        ],
    )
    return frame.assign(
        method=BOOTSTRAP_MEAN_CONTROL_METHOD,
        ranking_score=frame[
            "posterior_probability_above_mean_control"
        ].to_numpy(float),
        q_source="bh",
    )


def read_filtered_mean_control_tests(path: Path) -> pd.DataFrame:
    frame = read_mean_control_tests(path)
    frame["method"] = FILTERED_MEAN_CONTROL_METHOD
    return frame


def read_filtered_bootstrap_mean_control_tests(path: Path) -> pd.DataFrame:
    frame = read_bootstrap_mean_control_tests(path)
    frame["method"] = FILTERED_BOOTSTRAP_MEAN_CONTROL_METHOD
    return frame


def assay_positive_for_tests(tests: pd.DataFrame, assay: pd.DataFrame) -> np.ndarray:
    index = pd.MultiIndex.from_frame(tests[["group", "cre"]].astype(str))
    common_groups = pd.Index(tests["group"].astype(str).unique()).intersection(assay.index)
    common_cres = pd.Index(tests["cre"].astype(str).unique()).intersection(assay.columns)
    if len(common_groups) == 0 or len(common_cres) == 0:
        return np.zeros(len(tests), dtype=bool)
    assay_stack = (
        assay.reindex(index=common_groups, columns=common_cres, fill_value=0.0)
        .gt(0.5)
        .rename_axis(index="group", columns="cre")
        .stack(future_stack=True)
    )
    return assay_stack.reindex(index, fill_value=False).to_numpy(bool)




def restrict_to_common_pairs(
    tests: pd.DataFrame,
    methods: tuple[str, ...],
    loo_null_scores: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    records = []
    common_counts = {}
    method_thresholds = {
        method: set(group["t7_threshold"].astype(float))
        for method, group in tests.groupby("method", sort=False)
    }
    for threshold, frame in tests.groupby("t7_threshold", sort=True):
        present = set(frame["method"].astype(str))
        active_methods = tuple(
            method
            for method in methods
            if float(threshold) in method_thresholds.get(method, set())
        )
        missing = set(active_methods) - present
        if missing:
            raise ValueError(f"T7 >= {threshold:g} is missing methods: {sorted(missing)}")
        pair_method_counts = frame.groupby(["group", "cre"])["method"].nunique()
        common_index = pair_method_counts[
            pair_method_counts.eq(len(active_methods))
        ].index
        pair_index = pd.MultiIndex.from_frame(frame[["group", "cre"]].astype(str))
        common = frame.loc[pair_index.isin(common_index)].copy()
        common_counts[str(float(threshold))] = int(len(common_index))
        records.append(common)
    output = pd.concat(records, ignore_index=True)
    output["q_right_input"] = output["q_right"]
    bh_mask = output["q_source"].eq("bh")
    output.loc[bh_mask, "q_right"] = output.loc[bh_mask].groupby(
        ["t7_threshold", "method"], sort=False
    )["p_right"].transform(bh_fdr)
    loo_mask = output["q_source"].eq("loo_empirical_fdr")
    if loo_mask.any():
        if loo_null_scores is None:
            raise ValueError("LOO null scores are required for common-pair empirical FDR")
        for threshold in output.loc[loo_mask, "t7_threshold"].unique():
            selected = loo_mask & output["t7_threshold"].eq(threshold)
            recomputed, _ = assign_empirical_fdr(
                output.loc[selected].copy(), loo_null_scores
            )
            output.loc[selected, "p_right"] = recomputed["empirical_p"].to_numpy()
            output.loc[selected, "q_right"] = recomputed["empirical_q"].to_numpy()
    return output, common_counts


def benchmark_assay(
    tests: pd.DataFrame,
    assay_name: str,
    assay: pd.DataFrame,
    q_cutoff: float,
) -> pd.DataFrame:
    frame = tests.copy()
    frame["assay_positive"] = assay_positive_for_tests(frame, assay)
    frame["significant"] = frame["q_right"].le(q_cutoff)
    rows = []
    for (threshold, method), group in frame.groupby(["t7_threshold", "method"], sort=False):
        significant = group["significant"].to_numpy(bool)
        assay_positive = group["assay_positive"].to_numpy(bool)
        tp = int((significant & assay_positive).sum())
        n_significant = int(significant.sum())
        n_assay = int(assay_positive.sum())
        n_tested = int(len(group))
        fp = n_significant - tp
        fn = n_assay - tp
        tn = n_tested - tp - fp - fn
        odds, pvalue = (
            fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
            if n_tested and min(tp, fp, fn, tn) >= 0
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "assay": assay_name,
                "t7_threshold": float(threshold),
                "method": method,
                "TP": tp,
                "significant": n_significant,
                "tested": n_tested,
                "assay_positive": n_assay,
                "precision": tp / n_significant if n_significant else 0.0,
                "recall": tp / n_assay if n_assay else np.nan,
                "fisher_oddsratio": odds,
                "fisher_p": pvalue,
                "q_cutoff": q_cutoff,
            }
        )
    return pd.DataFrame(rows)


def precision_recall_tables(
    tests: pd.DataFrame,
    assays: dict[str, pd.DataFrame],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = tests[np.isclose(tests["t7_threshold"].to_numpy(float), threshold)].copy()
    if selected.empty:
        raise ValueError(f"No tests found for PR threshold T7 >= {threshold:g}")
    curve_frames = []
    metric_rows = []
    for assay_name, assay in assays.items():
        frame = selected.copy()
        frame["assay_positive"] = assay_positive_for_tests(frame, assay)
        for method, group in frame.groupby("method", sort=False):
            labels = group["assay_positive"].to_numpy(bool)
            scores = group["ranking_score"].to_numpy(float)
            valid = np.isfinite(scores)
            labels = labels[valid]
            scores = scores[valid]
            if len(labels) == 0 or not labels.any():
                continue
            precision, recall, score_thresholds = precision_recall_curve(labels, scores)
            curve_frames.append(
                pd.DataFrame(
                    {
                        "assay": assay_name,
                        "method": method,
                        "t7_threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "score_threshold": np.r_[score_thresholds, np.nan],
                    }
                )
            )
            metric_rows.append(
                {
                    "assay": assay_name,
                    "method": method,
                    "t7_threshold": threshold,
                    "average_precision": average_precision_score(labels, scores),
                    "prevalence": labels.mean(),
                    "tested": len(labels),
                    "assay_positive": int(labels.sum()),
                }
            )
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(metric_rows)


def _adaptive_ylim(values: pd.Series) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return (0.0, 1.0)
    ymax = float(finite.max())
    if ymax <= 0:
        return (0.0, 0.05)
    return (0.0, min(1.0, ymax * 1.45))


def _threshold_for_bar(bar: plt.Rectangle, thresholds: list[float]) -> float | None:
    category_index = int(round(bar.get_x() + bar.get_width() / 2.0))
    if 0 <= category_index < len(thresholds):
        return float(thresholds[category_index])
    return None


def _label_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    thresholds: list[float],
    *,
    metric: str,
    methods: tuple[str, ...],
) -> None:
    denominator = "significant" if metric == "precision" else "assay_positive"
    lookup = {
        (str(row.method), float(row.t7_threshold)): f"{int(row.TP)}/{int(getattr(row, denominator))}"
        for row in data.itertuples(index=False)
    }
    for container, method in zip(ax.containers, methods):
        labels = []
        for bar in container:
            threshold = _threshold_for_bar(bar, thresholds)
            labels.append(lookup.get((method, threshold), ""))
        ax.bar_label(
            container,
            labels=labels,
            padding=1.5,
            rotation=90,
            fontsize=5,
            color="#333333",
        )


def plot_precision_recall(
    summary: pd.DataFrame,
    output: Path,
    *,
    methods: tuple[str, ...],
    common_pairs: bool = False,
    title_note: str | None = None,
) -> None:
    """Draw the T7-threshold precision/recall panel grid.

    ``title_note`` is appended to the caption as its own line, which is how
    callers that narrow the tested universe (for example the replicate-
    concordant figure in ``revision/origin_vs_new``) record that restriction
    without duplicating this layout.
    """
    sns.set_theme(context="paper", style="whitegrid")
    thresholds = sorted(summary["t7_threshold"].unique())
    threshold_labels = {threshold: f">={threshold:g}" for threshold in thresholds}
    plot_data = summary.copy()
    plot_data["T7 filter"] = plot_data["t7_threshold"].map(threshold_labels)
    assays = list(ASSAYS)
    metrics = ["precision", "recall"]
    fig, axes = plt.subplots(
        len(metrics),
        len(assays),
        figsize=(
            PANEL_SIDE_INCHES * len(assays) + LEGEND_INCHES,
            PANEL_SIDE_INCHES * len(metrics),
        ),
        sharey=False,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = np.array([axes])
    if len(assays) == 1:
        axes = axes.reshape(len(metrics), 1)
    for row_idx, metric in enumerate(metrics):
        for col_idx, assay in enumerate(assays):
            ax = axes[row_idx, col_idx]
            data = plot_data[plot_data["assay"].eq(assay)]
            sns.barplot(
                data=data,
                x="T7 filter",
                y=metric,
                hue="method",
                order=[threshold_labels[t] for t in thresholds],
                hue_order=list(methods),
                palette=METHOD_COLORS,
                errorbar=None,
                ax=ax,
            )
            ax.set_title(assay if row_idx == 0 else "")
            ax.set_xlabel("T7 filter" if row_idx == len(metrics) - 1 else "")
            ax.set_ylabel(metric.capitalize())
            ylim_values = data[metric]
            if metric == "precision":
                naive = data.assign(
                    naive_precision=data["assay_positive"] / data["tested"]
                ).pivot(
                    index="t7_threshold",
                    columns="method",
                    values="naive_precision",
                )
                naive = naive.reindex(index=thresholds, columns=list(methods))
                baseline_y = []
                baseline_xmin = []
                baseline_xmax = []
                for container, method in zip(ax.containers, methods):
                    for bar in container:
                        threshold = _threshold_for_bar(bar, thresholds)
                        if threshold is None:
                            continue
                        value = float(naive.loc[threshold, method])
                        if not np.isfinite(value):
                            continue
                        inset = 0.04 * bar.get_width()
                        baseline_y.append(value)
                        baseline_xmin.append(bar.get_x() + inset)
                        baseline_xmax.append(bar.get_x() + bar.get_width() - inset)
                ax.hlines(
                    baseline_y,
                    baseline_xmin,
                    baseline_xmax,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=5,
                    label="All cCREs significant",
                )
                ylim_values = pd.concat(
                    [ylim_values, pd.Series(baseline_y, name="precision")],
                    ignore_index=True,
                )
            ax.set_ylim(*_adaptive_ylim(ylim_values))
            ax.tick_params(axis="x", rotation=0)
            _label_bars(ax, data, thresholds, metric=metric, methods=methods)
            if row_idx != 0 or col_idx != len(assays) - 1:
                ax.get_legend().remove()
            else:
                handles, legend_labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    [display_label(text) for text in legend_labels],
                    title="Method",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                )
    fig.suptitle(
        "Precision and recall of T7-filtered significant calls at "
        f"q <= {summary['q_cutoff'].iloc[0]:g}\nLabels show "
        "TP/significant for precision and TP/#peaks for recall"
        + (
            "; common tested pairs within each T7 group"
            if common_pairs
            else ""
        )
        + (f"\n{title_note}" if title_note else ""),
        fontsize=12,
    )
    sns.despine(fig=fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fit_panel_size(fig, axes)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
    threshold: float,
    methods: tuple[str, ...],
) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    fig, axes = plt.subplots(
        1,
        len(ASSAYS),
        figsize=(PANEL_SIDE_INCHES * len(ASSAYS), PANEL_SIDE_INCHES),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, assay_name in zip(axes[0], ASSAYS):
        assay_curves = curves[curves["assay"].eq(assay_name)]
        assay_metrics = metrics[metrics["assay"].eq(assay_name)].set_index("method")
        for method in methods:
            curve = assay_curves[assay_curves["method"].eq(method)]
            if curve.empty:
                continue
            ap = float(assay_metrics.loc[method, "average_precision"])
            ax.step(
                curve["recall"],
                curve["precision"],
                where="post",
                color=METHOD_COLORS[method],
                linewidth=1.35,
                label=f"{display_label(method)} (AP={ap:.3f})",
            )
        prevalence = float(assay_metrics["prevalence"].iloc[0])
        ax.axhline(
            prevalence, color="0.35", linestyle="--", linewidth=0.9,
            label=f"Prevalence ({prevalence:.3f})",
        )
        visible_precision = assay_curves.loc[
            assay_curves["recall"].gt(0)
            & assay_curves["precision"].gt(0)
            & np.isfinite(assay_curves["precision"]),
            "precision",
        ].to_numpy(float)
        if visible_precision.size:
            precision_min = min(prevalence, float(visible_precision.min()))
            precision_max = max(prevalence, float(visible_precision.max()))
            y_min = max(1e-4, precision_min * 0.8)
            y_max = min(1.0, precision_max * 1.25)
        else:
            y_min = max(1e-4, prevalence / 10.0)
            y_max = 1.0
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision (zoomed log scale)")
        ax.set_title(assay_name)
        ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.suptitle(
        f"Precision-recall curves on common tested pairs, T7 >= {threshold:g}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fit_panel_size(fig, axes)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["t7_threshold", "method", "group", "cre", "p_right", "q_right"]
    tests = pd.read_csv(args.tests, usecols=usecols)
    tests["method"] = tests["method"].astype(str)
    methods = tuple(dict.fromkeys(args.methods))
    tests = tests[tests["method"].isin(methods)].copy()
    missing_methods = set(methods) - set(tests["method"])
    if missing_methods:
        raise ValueError(f"Tests are missing requested methods: {sorted(missing_methods)}")
    tests["ranking_score"] = 1.0 - tests["p_right"].to_numpy(float)
    tests["q_source"] = "bh"
    loo_null_scores = None
    if not args.skip_loo:
        loo_tests = read_loo_tests(args.loo_tests, args.loo_threshold)
        tests = pd.concat([tests, loo_tests], ignore_index=True, sort=False)
        methods = (*methods, LOO_METHOD)
        if args.include_loo_score_bh:
            tests = pd.concat(
                [tests, make_loo_score_bh_tests(loo_tests)],
                ignore_index=True,
                sort=False,
            )
            methods = (*methods, LOO_SCORE_BH_METHOD)
        loo_null_scores = pd.read_csv(
            args.loo_null, usecols=["test_statistic"]
        )["test_statistic"].to_numpy(float)
    if args.include_max_control:
        tests = pd.concat(
            [tests, read_max_control_tests(args.max_control_tests)],
            ignore_index=True,
            sort=False,
        )
        methods = (*methods, MAX_CONTROL_METHOD)
    if args.include_mean_control:
        tests = pd.concat(
            [tests, read_mean_control_tests(args.mean_control_tests)],
            ignore_index=True,
            sort=False,
        )
        methods = (*methods, MEAN_CONTROL_METHOD)
    if args.include_bootstrap_mean_control:
        tests = pd.concat(
            [
                tests,
                read_bootstrap_mean_control_tests(
                    args.bootstrap_mean_control_tests
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        methods = (*methods, BOOTSTRAP_MEAN_CONTROL_METHOD)
    if args.include_filtered_mean_control:
        tests = pd.concat(
            [tests, read_filtered_mean_control_tests(args.filtered_mean_control_tests)],
            ignore_index=True,
            sort=False,
        )
        methods = (*methods, FILTERED_MEAN_CONTROL_METHOD)
    if args.include_filtered_bootstrap_mean_control:
        tests = pd.concat(
            [
                tests,
                read_filtered_bootstrap_mean_control_tests(
                    args.filtered_bootstrap_mean_control_tests
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        methods = (*methods, FILTERED_BOOTSTRAP_MEAN_CONTROL_METHOD)
    common_counts = None
    if args.common_pairs:
        tests, common_counts = restrict_to_common_pairs(
            tests, methods, loo_null_scores=loo_null_scores
        )
    evaluated_tests_path = args.tables_dir / f"{args.stem}_evaluated_tests.csv.gz"
    tests.to_csv(evaluated_tests_path, index=False)
    assays = {name: read_assay(path) for name, path in ASSAYS.items()}
    summaries = []
    for assay_name, assay in assays.items():
        summaries.append(
            benchmark_assay(tests, assay_name, assay, args.q_cutoff)
        )
    summary = pd.concat(summaries, ignore_index=True)
    method_order = {method: idx for idx, method in enumerate(methods)}
    assay_order = {assay: idx for idx, assay in enumerate(ASSAYS)}
    summary = (
        summary.assign(
            assay_order=summary["assay"].map(assay_order),
            method_order=summary["method"].map(method_order),
        )
        .sort_values(["assay_order", "t7_threshold", "method_order"])
        .drop(columns=["assay_order", "method_order"])
    )
    summary_path = args.tables_dir / f"{args.stem}_precision_recall_summary.csv"
    summary.to_csv(summary_path, index=False)
    combined_pdf = args.figures_dir / f"{args.stem}_precision_recall.pdf"
    plot_precision_recall(
        summary, combined_pdf, methods=methods, common_pairs=args.common_pairs
    )
    curves, ap_summary = precision_recall_tables(tests, assays, args.pr_threshold)
    curves_path = args.tables_dir / f"{args.stem}_pr_curves.csv.gz"
    ap_path = args.tables_dir / f"{args.stem}_average_precision.csv"
    curves.to_csv(curves_path, index=False)
    ap_summary.to_csv(ap_path, index=False)
    pr_curve_pdf = args.figures_dir / f"{args.stem}_pr_curves_t7_ge{args.pr_threshold:g}.pdf"
    plot_pr_curves(curves, ap_summary, pr_curve_pdf, args.pr_threshold, methods)
    write_json(
        args.figures_dir / f"{args.stem}_precision_recall_manifest.json",
        {
            "tests": str(args.tests),
            "loo_tests": None if args.skip_loo else str(args.loo_tests),
            "loo_null": None if args.skip_loo else str(args.loo_null),
            "include_loo_score_bh": args.include_loo_score_bh,
            "max_control_tests": (
                str(args.max_control_tests) if args.include_max_control else None
            ),
            "mean_control_tests": (
                str(args.mean_control_tests) if args.include_mean_control else None
            ),
            "bootstrap_mean_control_tests": (
                str(args.bootstrap_mean_control_tests)
                if args.include_bootstrap_mean_control
                else None
            ),
            "filtered_mean_control_tests": (
                str(args.filtered_mean_control_tests)
                if args.include_filtered_mean_control
                else None
            ),
            "filtered_bootstrap_mean_control_tests": (
                str(args.filtered_bootstrap_mean_control_tests)
                if args.include_filtered_bootstrap_mean_control
                else None
            ),
            "summary": str(summary_path),
            "evaluated_tests": str(evaluated_tests_path),
            "combined_figure": str(combined_pdf),
            "pr_curves": str(curves_path),
            "average_precision": str(ap_path),
            "pr_curve_figure": str(pr_curve_pdf),
            "q_cutoff": args.q_cutoff,
            "pr_threshold": args.pr_threshold,
            "common_pairs": args.common_pairs,
            "common_pair_counts_by_t7_threshold": common_counts,
            "methods": list(methods),
            "assays": {name: str(path) for name, path in ASSAYS.items()},
            "definitions": {
                "TP": "significant tested cCRE-celltype pairs with assay value > 0.5",
                "precision": "TP / significant; defined as 0 when there are no significant calls",
                "recall": "TP / #peaks, where #peaks is assay_positive among tested pairs",
                "bar_labels_precision": "TP / significant",
                "bar_labels_recall": "TP / #peaks",
                "average_precision": "area-equivalent summary of the precision-recall ranking; conventional methods use 1 - p_right and the LOO method uses its posterior comparison statistic",
                "common_pair_fdr": "At each T7 threshold, methods available at that threshold are restricted to their common tested-pair intersection. BH q-values are recomputed for conventional methods; LOO empirical q-values are recomputed against the saved LOO negative-control null.",
                "loo_score_bh": "Exploratory only: p = 1 - LOO posterior comparison score, followed by BH across the common tested pairs. This posterior score transformation is not established as a frequentist-calibrated p-value.",
                "max_control_test": "For each posterior draw, target raw log_gamma minus the maximum raw log_gamma among seven ordinary negative controls; p_right is the posterior fraction of contrasts <= 0, followed by BH.",
                "mean_control_test": "For each posterior draw, target raw log_gamma minus the mean raw log_gamma among seven ordinary negative controls; p_right is the posterior fraction of contrasts <= 0, followed by BH.",
                "bootstrap_mean_control_test": "Within each bootstrap replicate, target log activity minus the arithmetic mean of finite log activities among seven ordinary controls; p_right is the fraction of valid contrasts <= 0, followed by BH.",
                "filtered_control_test": "Cell-type-specific reference averages only ordinary controls with observed individual T7 >= the current panel threshold, requiring at least one retained control; Bayesian posterior or bootstrap tail fractions are followed by BH.",
                "naive_precision_baseline": "assay-positive common tested pairs / all common tested pairs, equivalent to calling every eligible cCRE significant",
            },
        },
    )


if __name__ == "__main__":
    main()
