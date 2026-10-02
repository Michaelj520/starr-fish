#!/usr/bin/env python3
"""Replicate-concordant left-tail heatmap with ChromHMM silencer/insulator boxes.

Same layout as ``revision/origin_vs_new/code/plot_origin_vs_new_heatmap.py
--control-reference mean``: one panel per experiment coloured by the posterior
mean target log_gamma minus the mean of the seven ordinary controls, a star for
BH significance, a control-spread strip beside each panel and count tracks under
it. It differs in three places:

* significance is the left tail (``p_left`` from ``test_silencer_left_tail.py``),
  with BH redone over the pairs eligible at the T7 threshold in *both* runs, and
  the pairs are restricted to those whose calls agree;
* boxes are the ChromHMM labels -- silencer (Chr-R + Hc-P fraction > cutoff) and
  insulator (Chr-O > 0.9 and Chr-A < 0.1) -- instead of ATAC peaks;
* only pairs with a ChromHMM annotation are drawn, so rows are the annotated
  cell types and chrX cCREs drop out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE: Final[Path] = Path(__file__).resolve()
SILENCER_DIR: Final[Path] = HERE.parents[1]
REVISION_DIR: Final[Path] = HERE.parents[2]
for code_dir in (
    HERE.parent,
    REVISION_DIR / "insulator" / "code",
    REVISION_DIR / "origin_vs_new" / "code",
):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

# plot_origin_vs_new_heatmap puts revision/run_Bayes on sys.path for activity_matrix_io.
from plot_origin_vs_new_heatmap import (  # noqa: E402
    DEFAULT_LIBRARY_COUNTS,
    DEFAULT_ORIGIN_H5AD,
    DEFAULT_TABLES,
    RUN_LABELS,
    RUNS,
    add_markers,
    pivot,
    read_library_counts,
    subclass_numeric_order,
)
from activity_matrix_io import load_dataset  # noqa: E402
from baystarrfish.stats import bh_fdr  # noqa: E402
from insulator_truth import (  # noqa: E402
    DEFAULT_ACTIVE_CUTOFF,
    DEFAULT_OPEN_CUTOFF,
    load_insulator_truth,
)
from silencer_truth import load_silencer_truth  # noqa: E402

DEFAULT_TESTS: Final[dict[str, Path]] = {
    "origin": SILENCER_DIR / "results" / "silencer_left_tail_tests.csv.gz",
    "new": SILENCER_DIR / "results" / "silencer_left_tail_tests_newdata.csv.gz",
}
DEFAULT_STEM: Final[str] = "silencer_left_tail_replicate_concordant_heatmap_t7_ge50"
DEFAULT_SILENCER_CUTOFF: Final[float] = 0.1
N_CONTROLS: Final[int] = 7
ACTIVITY_TOLERANCE: Final[float] = 1e-6
TEST_COLUMNS: Final[tuple[str, ...]] = (
    "t7_threshold",
    "group",
    "cre",
    "n_cells",
    "target_t7_total",
    "negative_control_t7_total",
    "activity_mean",
    "mean_negative_control_activity_mean",
    "effect_vs_control_reference_mean",
    "effect_vs_control_reference_lo90",
    "effect_vs_control_reference_hi90",
    "p_left",
    "q_left",
)
CALL_STATUSES: Final[tuple[str, ...]] = (
    "both_significant",
    "origin_only_significant",
    "new_only_significant",
    "neither_significant",
)
BOX_STYLES: Final[dict[str, dict[str, object]]] = {
    "silencer": {"edgecolor": "black", "linewidth": 0.7},
    "insulator": {"edgecolor": "#00b000", "linewidth": 0.9},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--origin-tests", type=Path, default=DEFAULT_TESTS["origin"])
    parser.add_argument("--new-tests", type=Path, default=DEFAULT_TESTS["new"])
    parser.add_argument(
        "--origin-tables",
        type=Path,
        default=DEFAULT_TABLES["origin"],
        help="Exported matrices of the original run; source of the control strip.",
    )
    parser.add_argument("--new-tables", type=Path, default=DEFAULT_TABLES["new"])
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=SILENCER_DIR / "results",
        help="Directory holding the cre_by_celltype_*_fraction.csv matrices.",
    )
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--nominal-p-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--silencer-cutoff",
        type=float,
        default=DEFAULT_SILENCER_CUTOFF,
        help="Silencer = Chr-R + Hc-P fraction strictly above this.",
    )
    parser.add_argument("--open-cutoff", type=float, default=DEFAULT_OPEN_CUTOFF)
    parser.add_argument("--active-cutoff", type=float, default=DEFAULT_ACTIVE_CUTOFF)
    parser.add_argument(
        "--restrict-status",
        default="both_significant,neither_significant",
        help=f"Comma-separated call statuses kept, from {', '.join(CALL_STATUSES)}.",
    )
    parser.add_argument("--origin-h5ad", type=Path, default=DEFAULT_ORIGIN_H5AD)
    parser.add_argument("--library-counts", type=Path, default=DEFAULT_LIBRARY_COUNTS)
    parser.add_argument("--figures-dir", type=Path, default=SILENCER_DIR / "figures")
    parser.add_argument("--results-dir", type=Path, default=SILENCER_DIR / "results")
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_arm(path: Path, run: str, t7_threshold: float) -> pd.DataFrame:
    """One run's left-tail tests at ``t7_threshold``, columns prefixed by run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run test_silencer_left_tail.py first")
    frame = pd.read_csv(path, usecols=list(TEST_COLUMNS))
    frame = frame[np.isclose(frame["t7_threshold"].astype(float), t7_threshold)]
    if frame.empty:
        raise ValueError(f"{path} has no rows at T7 >= {t7_threshold:g}")
    frame = frame.drop(columns="t7_threshold").astype({"group": str, "cre": str})
    if frame.duplicated(["group", "cre"]).any():
        raise ValueError(f"{path} has duplicate group-cCRE pairs at T7 >= {t7_threshold:g}")
    frame["centered"] = frame["activity_mean"] - frame["mean_negative_control_activity_mean"]
    return frame.rename(
        columns={c: f"{run}_{c}" for c in frame.columns if c not in ("group", "cre")}
    )


def load_shared(
    tests: dict[str, Path], t7_threshold: float, q_cutoff: float
) -> pd.DataFrame:
    """Inner-join both runs and redo BH over that shared family, per run.

    Each run's shipped ``q_left`` is BH over its own eligible pairs; the family
    this figure tests is their intersection, so ``q`` is recomputed from
    ``p_left`` exactly as the right-tail reference heatmap does.
    """
    origin, new = (load_arm(tests[run], run, t7_threshold) for run in RUNS)
    shared = origin.merge(new, on=["group", "cre"], how="inner", validate="one_to_one")
    if shared.empty:
        raise ValueError(f"No pairs are eligible at T7 >= {t7_threshold:g} in both runs")
    for run in RUNS:
        shared[f"{run}_q_left_common"] = bh_fdr(shared[f"{run}_p_left"].to_numpy(float))
        shared[f"{run}_significant_common_q"] = shared[f"{run}_q_left_common"].le(q_cutoff)
    origin_sig = shared["origin_significant_common_q"]
    new_sig = shared["new_significant_common_q"]
    shared["call_status"] = np.select(
        [origin_sig & new_sig, origin_sig & ~new_sig, ~origin_sig & new_sig],
        list(CALL_STATUSES[:3]),
        default=CALL_STATUSES[3],
    )
    return shared


def add_annotation(
    shared: pd.DataFrame,
    annotation_dir: Path,
    silencer_cutoff: float,
    open_cutoff: float,
    active_cutoff: float,
) -> pd.DataFrame:
    """Attach ChromHMM fractions and both labels; NaN coverage is unannotated."""
    silencer = load_silencer_truth(annotation_dir)
    insulator = load_insulator_truth(annotation_dir)
    index = pd.MultiIndex.from_frame(shared[["group", "cre"]])
    out = shared.copy()
    for name, matrix in (
        ("repressive_fraction", silencer.repressive_fraction),
        ("open_fraction", insulator.open_fraction),
        ("active_fraction", insulator.active_fraction),
        ("nd_fraction", silencer.nd_fraction),
    ):
        out[name] = matrix.stack(future_stack=True).reindex(index).to_numpy(float)
    out["annotated"] = (
        out[["repressive_fraction", "open_fraction", "active_fraction"]].notna().all(axis=1)
    )
    out["silencer"] = out["annotated"] & out["repressive_fraction"].gt(silencer_cutoff)
    out["insulator"] = (
        out["annotated"]
        & out["open_fraction"].gt(open_cutoff)
        & out["active_fraction"].lt(active_cutoff)
    )
    both = int((out["silencer"] & out["insulator"]).sum())
    if both:
        raise ValueError(f"{both} pairs carry both labels; the box colours would overlap")
    return out


def control_spread(
    tables: dict[str, Path], shared: pd.DataFrame, row_order: list[str]
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    """SD across the seven control posterior means per subclass, from the exports.

    The exported activity must equal the tests' centred activity on every shown
    pair, which pins the strip and the heatmap to the same posterior.
    """
    spread: dict[str, pd.Series] = {}
    max_diff: dict[str, float] = {}
    index = pd.MultiIndex.from_frame(shared[["group", "cre"]])
    for run in RUNS:
        dataset = load_dataset(tables[run])
        if dataset.negative_control_activity.shape[1] != N_CONTROLS:
            raise ValueError(
                f"{run} control matrix has {dataset.negative_control_activity.shape[1]} "
                f"controls, expected {N_CONTROLS}"
            )
        exported = dataset.activity.stack(future_stack=True).reindex(index).to_numpy(float)
        diff = np.abs(exported - shared[f"{run}_centered"].to_numpy(float))
        if not np.all(np.isfinite(diff)) or float(diff.max()) > ACTIVITY_TOLERANCE:
            raise ValueError(
                f"{run}: exported activity in {tables[run]} does not match the left-tail "
                f"tests (max |diff| = {np.nanmax(diff):.3g}, {int(np.isnan(diff).sum())} "
                "missing); regenerate one from the other's posterior"
            )
        max_diff[run] = float(diff.max())
        spread[run] = dataset.control_spread.reindex(row_order).astype(float)
        missing = spread[run][spread[run].isna()].index.tolist()
        if missing:
            raise ValueError(f"{run} has no control spread for subclasses: {missing}")
    return spread, max_diff


def add_boxes(ax: plt.Axes, mask: np.ndarray, edgecolor: str, linewidth: float) -> int:
    rows, cols = np.nonzero(mask)
    if len(rows):
        ax.add_collection(
            PatchCollection(
                [Rectangle((c - 0.5, r - 0.5), 1.0, 1.0) for r, c in zip(rows, cols)],
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=4,
            )
        )
    return int(len(rows))


def run_summary(
    frame: pd.DataFrame,
    run: str,
    q_cutoff: float,
    nominal_p_cutoff: float,
    spread: pd.Series,
) -> dict[str, float | int]:
    significant = frame[f"{run}_q_left_common"].le(q_cutoff)
    return {
        "n_significant_common_bh": int(significant.sum()),
        "n_nominal_p_left_le_cutoff": int(frame[f"{run}_p_left"].le(nominal_p_cutoff).sum()),
        "n_significant_silencer": int((significant & frame["silencer"]).sum()),
        "n_significant_insulator": int((significant & frame["insulator"]).sum()),
        "median_target_t7": float(frame[f"{run}_target_t7_total"].median()),
        "median_negative_control_spread": float(frame["group"].map(spread).median()),
        "mean_activity_minus_control_mean": float(frame[f"{run}_centered"].mean()),
    }


def plot_heatmap(
    frame: pd.DataFrame,
    row_order: list[str],
    col_order: list[str],
    library_counts: pd.Series,
    spread: dict[str, pd.Series],
    args: argparse.Namespace,
    output_pdf: Path,
    output_png: Path,
) -> dict[str, object]:
    centered = {run: pivot(frame, f"{run}_centered", row_order, col_order) for run in RUNS}
    finite = np.concatenate(
        [m.to_numpy(float)[np.isfinite(m.to_numpy(float))] for m in centered.values()]
    )
    activity_limit = max(float(np.percentile(np.abs(finite), 99)), 1e-6)
    sd_max = max(
        float(np.percentile(np.concatenate([s.to_numpy(float) for s in spread.values()]), 99)),
        1e-6,
    )
    library_log = np.log10(1.0 + library_counts.clip(lower=0)).reindex(col_order)
    count_tracks = {
        run: np.vstack(
            [
                library_log.to_numpy(float),
                np.log10(
                    1.0
                    + frame.groupby("cre")[f"{run}_target_t7_total"]
                    .sum()
                    .reindex(col_order)
                    .clip(lower=0)
                ).to_numpy(float),
            ]
        )
        for run in RUNS
    }
    finite_counts = np.concatenate([t[np.isfinite(t)] for t in count_tracks.values()])
    activity_norm = Normalize(vmin=-activity_limit, vmax=activity_limit)
    sd_norm = Normalize(vmin=0.0, vmax=sd_max)
    count_norm = Normalize(vmin=float(finite_counts.min()), vmax=float(finite_counts.max()))
    label_masks = {
        label: pivot(frame, label, row_order, col_order).eq(True).to_numpy(bool)
        for label in BOX_STYLES
    }

    n_rows, n_cols = max(len(row_order), 1), max(len(col_order), 1)
    fig = plt.figure(
        figsize=(max(14.0, 5.2 + 0.105 * n_cols), max(10.0, 4.6 + 0.19 * n_rows * 2)),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        4, 4, height_ratios=[n_rows, 2.1, n_rows, 2.1], width_ratios=[n_cols, 2.2, 3.0, 3.0]
    )
    activity_cax = fig.add_subplot(grid[:, 2])
    sd_cax = fig.add_subplot(grid[:2, 3])
    count_cax = fig.add_subplot(grid[2:, 3])
    activity_cmap = plt.get_cmap("coolwarm").copy()
    activity_cmap.set_bad("0.9")
    annotation_cmap = plt.get_cmap("viridis").copy()
    annotation_cmap.set_bad("0.9")
    panels: dict[str, dict[str, int]] = {}

    for panel, run in enumerate(RUNS):
        ax = fig.add_subplot(grid[panel * 2, 0])
        sd_ax = fig.add_subplot(grid[panel * 2, 1], sharey=ax)
        count_ax = fig.add_subplot(grid[panel * 2 + 1, 0], sharex=ax)
        fig.add_subplot(grid[panel * 2 + 1, 1]).axis("off")

        matrix = centered[run].to_numpy(float)
        present = np.isfinite(matrix)
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            cmap=activity_cmap,
            norm=activity_norm,
            interpolation="nearest",
        )
        n_boxes = {
            label: add_boxes(ax, label_masks[label] & present, **style)
            for label, style in BOX_STYLES.items()
        }
        nominal = (
            pivot(frame, f"{run}_p_left", row_order, col_order)
            .le(args.nominal_p_cutoff)
            .to_numpy(bool)
            & present
        )
        significant = (
            pivot(frame, f"{run}_q_left_common", row_order, col_order)
            .le(args.q_cutoff)
            .to_numpy(bool)
            & present
        )
        n_nominal_only, n_significant = add_markers(ax, nominal, significant)
        starred = {
            label: int(np.count_nonzero(significant & label_masks[label])) for label in BOX_STYLES
        }

        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=6.2)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_ylabel("Cell subclass (numeric-prefix order)", fontsize=7)
        ax.set_title(
            f"{RUN_LABELS[run]}: {n_significant} BH-significant (left tail); "
            f"{n_nominal_only + n_significant} nominal p_left≤{args.nominal_p_cutoff:g}; "
            f"starred silencers {starred['silencer']}/{n_boxes['silencer']}, "
            f"insulators {starred['insulator']}/{n_boxes['insulator']}; "
            f"median T7={frame[f'{run}_target_t7_total'].median():.0f}; "
            f"median control SD={frame['group'].map(spread[run]).median():.2f}",
            fontsize=8.5,
        )

        sd_image = sd_ax.imshow(
            spread[run].to_numpy(float)[:, None],
            aspect="auto",
            cmap=annotation_cmap,
            norm=sd_norm,
            interpolation="nearest",
        )
        sd_ax.tick_params(axis="y", left=False, labelleft=False)
        sd_ax.set_xticks([0])
        sd_ax.set_xticklabels(["Control\nspread"], fontsize=6)
        sd_ax.tick_params(axis="x", bottom=False, labelbottom=True)

        count_image = count_ax.imshow(
            np.ma.masked_invalid(count_tracks[run]),
            aspect="auto",
            cmap=annotation_cmap,
            norm=count_norm,
            interpolation="nearest",
        )
        count_ax.set_yticks([0, 1])
        count_ax.set_yticklabels(["Nanopore library", f"{RUN_LABELS[run]} T7"], fontsize=5.7)
        count_ax.set_xticks(np.arange(len(col_order)))
        if panel == len(RUNS) - 1:
            count_ax.set_xticklabels(col_order, rotation=90, fontsize=4.2)
            count_ax.set_xlabel(
                "cCRE ordered by original + new T7 counts across displayed shared pairs "
                f"(n={len(col_order)})",
                fontsize=7,
            )
        else:
            count_ax.tick_params(axis="x", bottom=False, labelbottom=False)

        panels[run] = {
            "significant_common_bh": int(n_significant),
            "nominal_p_left_le_cutoff": int(n_nominal_only + n_significant),
            "silencer_boxes": n_boxes["silencer"],
            "insulator_boxes": n_boxes["insulator"],
            "significant_silencers": starred["silencer"],
            "significant_insulators": starred["insulator"],
        }

    fig.colorbar(
        image,
        cax=activity_cax,
        label="posterior mean [target log_gamma − mean(log_gamma of 7 controls)]",
    )
    fig.colorbar(
        sd_image, cax=sd_cax, label="SD across the 7 control posterior-mean activities"
    )
    count_colorbar = fig.colorbar(count_image, cax=count_cax, label="log10(1 + count)")
    count_colorbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        f"Left-tail (silencer) test, original versus new low-dose run: shared "
        f"T7≥{args.t7_threshold:g} universe\n"
        f"Restricted to {args.restrict_status.replace(',', ' or ')} pairs with a ChromHMM "
        f"annotation; n={len(frame):,} subclass–cCRE pairs; "
        f"★ shared-universe BH q_left≤{args.q_cutoff:g}\n"
        f"black box = silencer (Chr-R + Hc-P > {args.silencer_cutoff:g}); "
        f"green box = insulator (Chr-O > {args.open_cutoff:g}, "
        f"Chr-A < {args.active_cutoff:g})",
        fontsize=10,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "activity_color_limit": activity_limit,
        "control_spread_color_max": sd_max,
        "count_color_min": float(count_norm.vmin),
        "count_color_max": float(count_norm.vmax),
        "panels": panels,
    }


def main() -> None:
    args = parse_args()
    for name in ("q_cutoff", "nominal_p_cutoff", "silencer_cutoff", "open_cutoff",
                 "active_cutoff"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    statuses = [s.strip() for s in args.restrict_status.split(",") if s.strip()]
    unknown = sorted(set(statuses) - set(CALL_STATUSES))
    if not statuses or unknown:
        raise ValueError(f"--restrict-status must name statuses from {CALL_STATUSES}; "
                         f"unknown: {unknown}")

    tests = {"origin": args.origin_tests, "new": args.new_tests}
    tables = {"origin": args.origin_tables, "new": args.new_tables}
    shared = add_annotation(
        load_shared(tests, args.t7_threshold, args.q_cutoff),
        args.annotation_dir,
        args.silencer_cutoff,
        args.open_cutoff,
        args.active_cutoff,
    )
    frame = shared[shared["call_status"].isin(statuses) & shared["annotated"]].reset_index(
        drop=True
    )
    if frame.empty:
        raise ValueError("No annotated pairs survive the call-status restriction")

    row_order = subclass_numeric_order(
        args.origin_h5ad, pd.Index(frame["group"].drop_duplicates(), dtype=str)
    )
    col_order = (
        frame.assign(combined_t7=frame["origin_target_t7_total"] + frame["new_target_t7_total"])
        .groupby("cre")["combined_t7"]
        .sum()
        .sort_index(kind="stable")
        .sort_values(ascending=False, kind="stable")
        .index.astype(str)
        .tolist()
    )
    spread, activity_diff = control_spread(tables, frame, row_order)
    library_counts = read_library_counts(args.library_counts, col_order)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = args.figures_dir / f"{args.stem}.pdf"
    output_png = args.figures_dir / f"{args.stem}.png"
    values_path = args.results_dir / f"{args.stem}_values.csv.gz"
    manifest_path = args.results_dir / f"{args.stem}_manifest.json"

    details = plot_heatmap(
        frame, row_order, col_order, library_counts, spread, args, output_pdf, output_png
    )
    frame.to_csv(values_path, index=False)

    def status_counts(subset: pd.DataFrame) -> dict[str, int]:
        return {s: int(subset["call_status"].eq(s).sum()) for s in CALL_STATUSES}

    manifest = {
        "inputs": {
            "left_tail_tests": {run: str(tests[run].resolve()) for run in RUNS},
            "exported_tables_for_control_spread": {
                run: str(tables[run].resolve()) for run in RUNS
            },
            "annotation_dir": str(args.annotation_dir.resolve()),
            "origin_h5ad_for_row_order": str(args.origin_h5ad.resolve()),
            "library_counts": str(args.library_counts.resolve()),
            "t7_threshold": float(args.t7_threshold),
            "q_cutoff": float(args.q_cutoff),
            "nominal_p_cutoff": float(args.nominal_p_cutoff),
            "silencer_cutoff": float(args.silencer_cutoff),
            "open_cutoff": float(args.open_cutoff),
            "active_cutoff": float(args.active_cutoff),
            "retained_statuses": statuses,
        },
        "outputs": {
            "pdf": str(output_pdf.resolve()),
            "png": str(output_png.resolve()),
            "values": str(values_path.resolve()),
        },
        "universe": {
            "definition": (
                f"pairs with target and pooled-control T7 >= {args.t7_threshold:g} in both "
                "runs; BH over p_left within that set per run; shown pairs are those with "
                f"call status in {statuses} and a ChromHMM annotation"
            ),
            "common_pairs": int(len(shared)),
            "common_call_status": status_counts(shared),
            "common_annotated_pairs": int(shared["annotated"].sum()),
            "shown_pairs": int(len(frame)),
            "shown_call_status": status_counts(frame),
            "shown_cell_types": int(len(row_order)),
            "shown_ccres": int(len(col_order)),
            "shown_silencers": int(frame["silencer"].sum()),
            "shown_insulators": int(frame["insulator"].sum()),
            "max_abs_activity_diff_tests_vs_exports": activity_diff,
        },
        "plot": {
            "activity": "posterior mean target log_gamma minus mean of the seven "
                        "ordinary negative controls (left-tail test table)",
            "significance_marker": f"star: BH q_left <= {args.q_cutoff:g} within the shared "
                                   "universe; nominal-only pairs counted in titles",
            "silencer_box": f"black: Chr-R + Hc-P fraction > {args.silencer_cutoff:g}",
            "insulator_box": f"green: Chr-O fraction > {args.open_cutoff:g} and Chr-A "
                             f"fraction < {args.active_cutoff:g}",
            "row_order": "subclass numeric prefix from original H5AD",
            "column_order": "descending original + new target T7 summed over shown pairs",
            **details,
        },
        "summary": {
            run: run_summary(frame, run, args.q_cutoff, args.nominal_p_cutoff, spread[run])
            for run in RUNS
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"universe": manifest["universe"], "summary": manifest["summary"]},
                     indent=2))


if __name__ == "__main__":
    main()
