#!/usr/bin/env python3
"""Plot old/new activity concordance across shared T7 cutoffs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve()
ANALYSIS_DIR = HERE.parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
ORIGIN_ANALYSIS = REPO_ROOT / "revision" / "bayesian_vs_fold_change"
ORIGIN_CODE = ORIGIN_ANALYSIS / "code"
# the *_threshold_series helpers live one level down in figure_work/
for _origin_path in (ORIGIN_CODE, ORIGIN_CODE / "figure_work"):
    if str(_origin_path) not in sys.path:
        sys.path.insert(0, str(_origin_path))

from compute_t7_filter_negative_control_stats import (  # noqa: E402
    aligned_t7_totals,
    read_negative_controls,
)
from plot_method_activity_correlation import read_cre_blacklist  # noqa: E402
from test_bootstrap_mean_negative_control_activity import (  # noqa: E402
    METHOD as BOOTSTRAP_METHOD,
    compute_statistics,
)
from test_bootstrap_mean_negative_control_activity_threshold_series import (  # noqa: E402
    build_tests as build_bootstrap_tests,
)
from test_individual_negative_control_loo_empirical_fdr import (  # noqa: E402
    POOLED_NAME,
    load_grouped_t7,
)
from test_mean_negative_control_activity import (  # noqa: E402
    METHOD as BAYESIAN_METHOD,
    compute_tests as compute_bayesian_tests,
)


DEFAULT_ORIGIN_H5AD = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEWNEW.h5ad"
)
DEFAULT_NEW_H5AD = (
    REPO_ROOT
    / "revision"
    / "Data"
    / "scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad"
)
NEW_RESULTS = REPO_ROOT / "revision" / "Bayes_NewData"
DEFAULT_ORIGIN_BAYES = ORIGIN_ANALYSIS / "results" / "bayesian"
DEFAULT_NEW_BAYES = NEW_RESULTS / "bayesian"
DEFAULT_ORIGIN_BOOTSTRAP = REPO_ROOT / "revision" / "Bootstrap_OldData"
DEFAULT_NEW_BOOTSTRAP = REPO_ROOT / "revision" / "Bootstrap_NewData"
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "results" / "comparison"
DEFAULT_CUTOFFS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
# highest cutoff drawn on the CCC-vs-cutoff x-axis (computation keeps them all)
DEFAULT_PLOT_MAX_CUTOFF = 100.0
KEY = ["group", "cre"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-h5ad", type=Path, default=DEFAULT_ORIGIN_H5AD)
    parser.add_argument("--new-h5ad", type=Path, default=DEFAULT_NEW_H5AD)
    parser.add_argument("--origin-bayes", type=Path, default=DEFAULT_ORIGIN_BAYES)
    parser.add_argument("--new-bayes", type=Path, default=DEFAULT_NEW_BAYES)
    parser.add_argument(
        "--origin-bootstrap", type=Path, default=DEFAULT_ORIGIN_BOOTSTRAP
    )
    parser.add_argument("--new-bootstrap", type=Path, default=DEFAULT_NEW_BOOTSTRAP)
    parser.add_argument(
        "--t7-cutoffs", type=float, nargs="+", default=DEFAULT_CUTOFFS
    )
    parser.add_argument(
        "--plot-max-t7-cutoff",
        type=float,
        default=DEFAULT_PLOT_MAX_CUTOFF,
        help="highest T7 cutoff drawn on the CCC-vs-cutoff x-axis (pass a large "
             "value to keep every computed cutoff)",
    )
    parser.add_argument("--minimum-unit-pairs", type=int, default=10)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient using population moments."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if len(x) < 2:
        return np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    covariance = np.mean((x - x_mean) * (y - y_mean))
    denominator = x.var() + y.var() + (x_mean - y_mean) ** 2
    if denominator == 0:
        return 1.0 if np.array_equal(x, y) else np.nan
    return float(2.0 * covariance / denominator)


def run_controls(directory: Path) -> list[str]:
    return pd.read_csv(directory / "negative_controls.csv").iloc[:, 0].astype(str).tolist()


def validate_inputs(args: argparse.Namespace) -> tuple[list[str], set[str]]:
    directories = [
        args.origin_bayes,
        args.new_bayes,
        args.origin_bootstrap,
        args.new_bootstrap,
    ]
    controls = [run_controls(directory) for directory in directories]
    if any(value != controls[0] for value in controls[1:]):
        raise ValueError("Bayesian/bootstrap old/new negative-control lists differ")
    blacklists = [set(read_cre_blacklist(directory)) for directory in directories]
    if any(value != blacklists[0] for value in blacklists[1:]):
        raise ValueError("Bayesian/bootstrap old/new cCRE blacklists differ")
    return controls[0], blacklists[0]


def bayesian_base_tests(
    bayes_dir: Path,
    h5ad: Path,
    controls: list[str],
    blacklist: set[str],
) -> pd.DataFrame:
    manifest = json.loads((bayes_dir / "run_manifest.json").read_text())
    posterior_path = bayes_dir / f"{manifest['tag']}_posterior_samples.npz"
    print(f"[bayesian] loading {posterior_path}", flush=True)
    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cres = posterior["cre_names"].astype(str)
        ordinary = all_cres != POOLED_NAME
        cres = all_cres[ordinary]
        log_gamma = posterior["log_gamma"][:, :, ordinary].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cres, controls))
    if len(control_indices) != len(controls):
        raise ValueError(
            f"{bayes_dir} contains {len(control_indices)} of {len(controls)} controls"
        )
    target_indices = np.flatnonzero(
        ~np.isin(cres, controls) & ~np.isin(cres, list(blacklist))
    )
    t7_totals, group_classes, group_cell_counts = load_grouped_t7(
        h5ad, groups, cres
    )
    tests = compute_bayesian_tests(
        log_gamma,
        groups,
        cres,
        target_indices,
        control_indices,
        t7_totals,
        group_classes,
        group_cell_counts,
        0.0,
        0.0,
        None,
        BAYESIAN_METHOD,
    )
    return tests[
        KEY
        + [
            "target_t7_total",
            "negative_control_t7_total",
            "effect_vs_mean_control_mean",
        ]
    ]


def bootstrap_base_tests(
    bootstrap_dir: Path,
    h5ad: Path,
    controls: list[str],
    blacklist: set[str],
    chunk_size: int,
) -> pd.DataFrame:
    axes = json.loads((bootstrap_dir / "bootstrap_axes.json").read_text())
    groups = pd.Index(axes["subclasses"], dtype=str)
    cres = pd.Index(axes["cres"], dtype=str)
    saved_controls = read_negative_controls(bootstrap_dir)
    if saved_controls != controls:
        raise ValueError(f"{bootstrap_dir} negative controls differ")
    negative_mask = cres.isin(controls)
    activity_array = np.load(
        bootstrap_dir / "celltype_activity_array.npy", mmap_mode="r"
    )
    if activity_array.shape[1:] != (len(groups), len(cres)):
        raise ValueError(f"{bootstrap_dir} activity array does not match its axes")
    pair_t7 = aligned_t7_totals(h5ad, groups, cres)
    control_include = np.ones((len(groups), len(controls)), dtype=bool)
    print(
        f"[bootstrap] computing centered effects from {bootstrap_dir} ",
        f"({activity_array.shape[0]:,} replicates)",
        flush=True,
    )
    statistics = compute_statistics(
        activity_array,
        negative_mask,
        control_include,
        chunk_size,
    )
    tests = build_bootstrap_tests(
        statistics,
        threshold=0.0,
        method=BOOTSTRAP_METHOD,
        groups=groups,
        cres=cres,
        negative_controls=controls,
        blacklist=blacklist,
        pair_t7=pair_t7,
        control_include=control_include,
        q_cutoff=0.05,
    )
    return tests[
        KEY
        + [
            "target_t7_total",
            "negative_control_t7_total",
            "effect_vs_mean_control_mean",
        ]
    ]


def cutoff_metrics(
    method: str,
    origin: pd.DataFrame,
    new: pd.DataFrame,
    cutoffs: list[float],
) -> pd.DataFrame:
    origin = origin.rename(
        columns={
            "target_t7_total": "origin_target_t7_total",
            "negative_control_t7_total": "origin_control_t7_total",
            "effect_vs_mean_control_mean": "origin_activity",
        }
    )
    new = new.rename(
        columns={
            "target_t7_total": "new_target_t7_total",
            "negative_control_t7_total": "new_control_t7_total",
            "effect_vs_mean_control_mean": "new_activity",
        }
    )
    shared = origin.merge(new, on=KEY, how="inner", validate="one_to_one")
    rows = []
    for cutoff in cutoffs:
        selected = shared.loc[
            shared["origin_target_t7_total"].ge(cutoff)
            & shared["origin_control_t7_total"].ge(cutoff)
            & shared["new_target_t7_total"].ge(cutoff)
            & shared["new_control_t7_total"].ge(cutoff)
            & np.isfinite(shared["origin_activity"])
            & np.isfinite(shared["new_activity"])
        ]
        x = selected["origin_activity"].to_numpy(float)
        y = selected["new_activity"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "t7_cutoff": float(cutoff),
                "n_pairs": int(len(selected)),
                "lins_ccc": lins_ccc(x, y),
                "pearson": float(pearsonr(x, y).statistic) if len(x) >= 2 else np.nan,
                "spearman": float(spearmanr(x, y).statistic)
                if len(x) >= 2
                else np.nan,
                "origin_mean_activity": float(np.mean(x)) if len(x) else np.nan,
                "new_mean_activity": float(np.mean(y)) if len(y) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def unit_cutoff_metrics(
    method: str,
    origin: pd.DataFrame,
    new: pd.DataFrame,
    cutoffs: list[float],
    *,
    unit_column: str,
    minimum_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute within-unit CCCs and their unweighted mean at every cutoff."""
    origin = origin.rename(
        columns={
            "target_t7_total": "origin_target_t7_total",
            "negative_control_t7_total": "origin_control_t7_total",
            "effect_vs_mean_control_mean": "origin_activity",
        }
    )
    new = new.rename(
        columns={
            "target_t7_total": "new_target_t7_total",
            "negative_control_t7_total": "new_control_t7_total",
            "effect_vs_mean_control_mean": "new_activity",
        }
    )
    shared = origin.merge(new, on=KEY, how="inner", validate="one_to_one")
    detail_rows = []
    summary_rows = []
    for cutoff in cutoffs:
        selected = shared.loc[
            shared["origin_target_t7_total"].ge(cutoff)
            & shared["origin_control_t7_total"].ge(cutoff)
            & shared["new_target_t7_total"].ge(cutoff)
            & shared["new_control_t7_total"].ge(cutoff)
            & np.isfinite(shared["origin_activity"])
            & np.isfinite(shared["new_activity"])
        ]
        cutoff_rows = []
        for unit, frame in selected.groupby(unit_column, sort=True):
            if len(frame) < minimum_pairs:
                continue
            ccc = lins_ccc(
                frame["origin_activity"].to_numpy(float),
                frame["new_activity"].to_numpy(float),
            )
            row = {
                "method": method,
                "t7_cutoff": float(cutoff),
                "unit_axis": unit_column,
                "unit": str(unit),
                "n_supported_pairs": int(len(frame)),
                "lins_ccc": ccc,
            }
            cutoff_rows.append(row)
            detail_rows.append(row)
        finite_values = np.asarray(
            [row["lins_ccc"] for row in cutoff_rows if np.isfinite(row["lins_ccc"])],
            dtype=float,
        )
        summary_rows.append(
            {
                "method": method,
                "t7_cutoff": float(cutoff),
                "unit_axis": unit_column,
                "minimum_supported_pairs": int(minimum_pairs),
                "n_units_meeting_support": int(len(cutoff_rows)),
                "n_units_with_finite_ccc": int(len(finite_values)),
                "n_pairs_in_supported_units": int(
                    sum(row["n_supported_pairs"] for row in cutoff_rows)
                ),
                "mean_lins_ccc": float(finite_values.mean())
                if len(finite_values)
                else np.nan,
                "std_lins_ccc": float(finite_values.std(ddof=1))
                if len(finite_values) >= 2
                else np.nan,
                "median_lins_ccc": float(np.median(finite_values))
                if len(finite_values)
                else np.nan,
                "minimum_lins_ccc": float(finite_values.min())
                if len(finite_values)
                else np.nan,
                "maximum_lins_ccc": float(finite_values.max())
                if len(finite_values)
                else np.nan,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def validate_t7_50_reference(metrics: pd.DataFrame) -> None:
    expected = {
        "Bayesian": {"n_pairs": 1170, "pearson": 0.653},
        "Bootstrap": {"n_pairs": 1090, "pearson": 0.564},
    }
    at_50 = metrics.loc[np.isclose(metrics["t7_cutoff"], 50.0)].set_index("method")
    if set(at_50.index) != set(expected):
        raise ValueError("T7=50 validation rows are missing")
    for method, values in expected.items():
        observed = at_50.loc[method]
        if int(observed["n_pairs"]) != values["n_pairs"] or not np.isclose(
            observed["pearson"], values["pearson"], atol=5e-4
        ):
            raise ValueError(
                f"{method} T7=50 does not reproduce the existing comparison: "
                f"n={int(observed['n_pairs'])}, Pearson={observed['pearson']:.6f}"
            )


def plot_metrics(
    metrics: pd.DataFrame,
    output_stem: Path,
    *,
    max_cutoff: float | None = DEFAULT_PLOT_MAX_CUTOFF,
) -> None:
    """Plot CCC against T7 cutoff, drawing only cutoffs <= ``max_cutoff``.

    The table keeps every computed cutoff; the cap trims the x-axis where the
    sparse high-cutoff points (200, 500) add noise rather than signal.
    """
    if max_cutoff is not None:
        metrics = metrics.loc[metrics["t7_cutoff"].le(max_cutoff)]
        if metrics.empty:
            raise ValueError(f"no cutoffs at or below {max_cutoff}")
    cutoffs = metrics["t7_cutoff"].drop_duplicates().tolist()
    positions = np.arange(len(cutoffs))
    styles = {
        "Bayesian": {"color": "#4477AA", "marker": "o"},
        "Bootstrap": {"color": "#CC6677", "marker": "s"},
    }
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    for method, style in styles.items():
        frame = metrics.loc[metrics["method"].eq(method)].set_index("t7_cutoff").loc[
            cutoffs
        ]
        values = frame["lins_ccc"].to_numpy(float)
        ax.plot(
            positions,
            values,
            label=method,
            linewidth=2.0,
            markersize=6.5,
            **style,
        )
        offset = 7 if method == "Bayesian" else -12
        for x_position, value in zip(positions, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.3f}",
                    (x_position, value),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    fontsize=7.5,
                    color=style["color"],
                )

    labels = []
    for cutoff in cutoffs:
        counts = metrics.loc[np.isclose(metrics["t7_cutoff"], cutoff)].set_index(
            "method"
        )["n_pairs"]
        labels.append(
            f"{cutoff:g}\nBayes n={int(counts['Bayesian']):,}"
            f"\nBoot n={int(counts['Bootstrap']):,}"
        )
    finite_ccc = metrics["lins_ccc"].to_numpy(float)
    finite_ccc = finite_ccc[np.isfinite(finite_ccc)]
    lower = max(-1.0, min(0.0, float(finite_ccc.min()) - 0.08))
    upper = min(1.0, max(0.0, float(finite_ccc.max()) + 0.08))
    if lower == upper:
        lower, upper = -0.05, 1.0
    ax.set_ylim(lower, upper)
    ax.set_xticks(positions, labels)
    ax.set_xlabel("T7 cutoff in both datasets (pair counts after exact overlap filter)")
    ax.set_ylabel("Lin's concordance correlation coefficient")
    ax.set_title(
        "Original versus new low-dose activity concordance across T7 cutoffs\n"
        "Target and pooled seven-control T7 must meet the cutoff in both datasets"
    )
    ax.axhline(0.0, color="0.55", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def plot_unit_mean_metrics(
    metrics: pd.DataFrame,
    output_stem: Path,
    *,
    unit_label: str,
    within_label: str,
) -> None:
    cutoffs = metrics["t7_cutoff"].drop_duplicates().tolist()
    positions = np.arange(len(cutoffs))
    styles = {
        "Bayesian": {"color": "#4477AA", "marker": "o"},
        "Bootstrap": {"color": "#CC6677", "marker": "s"},
    }
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    for method, style in styles.items():
        frame = metrics.loc[metrics["method"].eq(method)].set_index("t7_cutoff").loc[
            cutoffs
        ]
        values = frame["mean_lins_ccc"].to_numpy(float)
        errors = frame["std_lins_ccc"].to_numpy(float)
        ax.errorbar(
            positions,
            values,
            yerr=errors,
            label=method,
            linewidth=2.0,
            markersize=6.5,
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
            **style,
        )
        offset = 7 if method == "Bayesian" else -12
        for x_position, value, error in zip(positions, values, errors):
            if np.isfinite(value):
                error = error if np.isfinite(error) else 0.0
                label_y = value + error if offset > 0 else value - error
                ax.annotate(
                    f"{value:.3f}",
                    (x_position, label_y),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    fontsize=7.5,
                    color=style["color"],
                )

    labels = []
    for cutoff in cutoffs:
        counts = metrics.loc[np.isclose(metrics["t7_cutoff"], cutoff)].set_index(
            "method"
        )["n_units_with_finite_ccc"]
        labels.append(
            f"{cutoff:g}\nBayes n={int(counts['Bayesian']):,}"
            f"\nBoot n={int(counts['Bootstrap']):,}"
        )
    means = metrics["mean_lins_ccc"].to_numpy(float)
    errors = metrics["std_lins_ccc"].fillna(0.0).to_numpy(float)
    valid = np.isfinite(means)
    lower = max(-1.0, min(0.0, float((means[valid] - errors[valid]).min()) - 0.05))
    upper = min(1.0, max(0.0, float((means[valid] + errors[valid]).max()) + 0.05))
    if lower == upper:
        lower, upper = -0.05, 1.0
    minimum_pairs = int(metrics["minimum_supported_pairs"].iloc[0])
    ax.set_ylim(lower, upper)
    ax.set_xticks(positions, labels)
    ax.set_xlabel(
        f"T7 cutoff in both datasets (n = {unit_label} with finite CCC)"
    )
    ax.set_ylabel(f"Mean {within_label} Lin's CCC (±1 SD)")
    ax.set_title(
        f"Original versus new low-dose mean CCC across {unit_label}\n"
        f"Each {unit_label.rstrip('s')} requires at least {minimum_pairs} supported pairs; "
        "points show mean ± 1 SD"
    )
    ax.axhline(0.0, color="0.55", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cutoffs = sorted({float(value) for value in args.t7_cutoffs})
    controls, blacklist = validate_inputs(args)

    bayesian_origin = bayesian_base_tests(
        args.origin_bayes, args.origin_h5ad, controls, blacklist
    )
    bayesian_new = bayesian_base_tests(
        args.new_bayes, args.new_h5ad, controls, blacklist
    )
    bootstrap_origin = bootstrap_base_tests(
        args.origin_bootstrap,
        args.origin_h5ad,
        controls,
        blacklist,
        args.bootstrap_chunk_size,
    )
    bootstrap_new = bootstrap_base_tests(
        args.new_bootstrap,
        args.new_h5ad,
        controls,
        blacklist,
        args.bootstrap_chunk_size,
    )

    metrics = pd.concat(
        [
            cutoff_metrics(
                "Bayesian", bayesian_origin, bayesian_new, cutoffs
            ),
            cutoff_metrics(
                "Bootstrap", bootstrap_origin, bootstrap_new, cutoffs
            ),
        ],
        ignore_index=True,
    )
    validate_t7_50_reference(metrics)

    celltype_results = [
        unit_cutoff_metrics(
            "Bayesian",
            bayesian_origin,
            bayesian_new,
            cutoffs,
            unit_column="group",
            minimum_pairs=args.minimum_unit_pairs,
        ),
        unit_cutoff_metrics(
            "Bootstrap",
            bootstrap_origin,
            bootstrap_new,
            cutoffs,
            unit_column="group",
            minimum_pairs=args.minimum_unit_pairs,
        ),
    ]
    ccre_results = [
        unit_cutoff_metrics(
            "Bayesian",
            bayesian_origin,
            bayesian_new,
            cutoffs,
            unit_column="cre",
            minimum_pairs=args.minimum_unit_pairs,
        ),
        unit_cutoff_metrics(
            "Bootstrap",
            bootstrap_origin,
            bootstrap_new,
            cutoffs,
            unit_column="cre",
            minimum_pairs=args.minimum_unit_pairs,
        ),
    ]
    celltype_summary = pd.concat(
        [result[0] for result in celltype_results], ignore_index=True
    )
    celltype_detail = pd.concat(
        [result[1] for result in celltype_results], ignore_index=True
    )
    ccre_summary = pd.concat(
        [result[0] for result in ccre_results], ignore_index=True
    )
    ccre_detail = pd.concat(
        [result[1] for result in ccre_results], ignore_index=True
    )

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "activity_concordance_ccc_vs_t7_cutoff.csv"
    figure_stem = figures_dir / "activity_concordance_ccc_vs_t7_cutoff"
    metrics.to_csv(table_path, index=False)
    plot_metrics(metrics, figure_stem, max_cutoff=args.plot_max_t7_cutoff)
    celltype_summary_path = tables_dir / "mean_celltype_ccc_vs_t7_cutoff.csv"
    celltype_detail_path = tables_dir / "celltype_ccc_by_t7_cutoff.csv.gz"
    ccre_summary_path = tables_dir / "mean_ccre_ccc_vs_t7_cutoff.csv"
    ccre_detail_path = tables_dir / "ccre_ccc_by_t7_cutoff.csv.gz"
    celltype_summary.to_csv(celltype_summary_path, index=False)
    celltype_detail.to_csv(celltype_detail_path, index=False)
    ccre_summary.to_csv(ccre_summary_path, index=False)
    ccre_detail.to_csv(ccre_detail_path, index=False)
    celltype_figure_stem = figures_dir / "mean_celltype_ccc_vs_t7_cutoff"
    ccre_figure_stem = figures_dir / "mean_ccre_ccc_vs_t7_cutoff"
    plot_unit_mean_metrics(
        celltype_summary,
        celltype_figure_stem,
        unit_label="cell types",
        within_label="within-cell-type",
    )
    plot_unit_mean_metrics(
        ccre_summary,
        ccre_figure_stem,
        unit_label="cCREs",
        within_label="within-cCRE",
    )
    manifest = {
        "metric": (
            "Lin's concordance correlation coefficient: 2*cov(x,y) / "
            "(var(x)+var(y)+(mean(x)-mean(y))^2), using population moments"
        ),
        "filter": (
            "target T7 >= cutoff and combined seven-control T7 >= cutoff in both "
            "original and new datasets; common subclass-cCRE pair; finite centered "
            "activity in both; negative controls and blacklist excluded"
        ),
        "cutoffs": cutoffs,
        "minimum_supported_pairs_per_celltype_or_ccre": args.minimum_unit_pairs,
        "unit_curve_error_bars": (
            "plus or minus one sample standard deviation across retained finite "
            "unit-level CCC values; omitted when fewer than two units remain"
        ),
        "negative_controls": controls,
        "blacklist": sorted(blacklist),
        "outputs": {
            "table": str(table_path.resolve()),
            "pdf": str(figure_stem.with_suffix(".pdf").resolve()),
            "png": str(figure_stem.with_suffix(".png").resolve()),
            "celltype_summary_table": str(celltype_summary_path.resolve()),
            "celltype_detail_table": str(celltype_detail_path.resolve()),
            "celltype_pdf": str(celltype_figure_stem.with_suffix(".pdf").resolve()),
            "celltype_png": str(celltype_figure_stem.with_suffix(".png").resolve()),
            "ccre_summary_table": str(ccre_summary_path.resolve()),
            "ccre_detail_table": str(ccre_detail_path.resolve()),
            "ccre_pdf": str(ccre_figure_stem.with_suffix(".pdf").resolve()),
            "ccre_png": str(ccre_figure_stem.with_suffix(".png").resolve()),
        },
    }
    (tables_dir / "activity_concordance_ccc_vs_t7_cutoff_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(metrics.to_string(index=False))
    print("\nMean within-cell-type CCC:\n", celltype_summary.to_string(index=False))
    print("\nMean within-cCRE CCC:\n", ccre_summary.to_string(index=False))


if __name__ == "__main__":
    main()
