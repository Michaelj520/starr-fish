#!/usr/bin/env python3
"""Plot global GLM, RNA/DNA or Bayesian barcode activities as cCRE boxes with jittered points."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BAYES_TABLES = HERE.parent / "Bayes_multiBC" / "tables"
#: Written by Bayes_multiBC/export_multibc_activity.py.
BAYES_METHOD = "bayes_direct_activity_control_centered"

# (results, figures) defaults per method.
DEFAULT_DIRS = {
    "glm": (HERE / "results/glm", HERE / "figures/glm"),
    "rna-dna": (HERE / "results", HERE / "figures"),
    "bayes": (BAYES_TABLES, HERE / "figures/bayes"),
}

# Point colours pass the dataviz palette checks on a light surface; box fills and
# edges are 18% and 50% tints of the same hues.
TARGET_POINT, TARGET_FACE, TARGET_EDGE = "#2A78D6", "#D6E2F7", "#95BCEB"
CONTROL_POINT, CONTROL_FACE, CONTROL_EDGE = "#EB6834", "#FBE4DB", "#F5B49A"


@dataclass(frozen=True)
class MethodView:
    """What one activity method plots and how the figure describes it."""

    value_column: str
    title: str
    ylabel: str
    note: str
    stem: str
    #: Legend label of the dashed y = 0 reference; None draws no reference.
    zero_label: str | None
    #: GLM keeps nonfinite log activities in the CSV and leaves them off the plot.
    allow_nonfinite: bool


def method_view(method: str, manifest: dict, table: pd.DataFrame, min_dna: float) -> MethodView:
    """Validate the results against ``method`` and describe its figure."""
    if method == "glm":
        if manifest.get("method") != "glm_total_ols_nanopore":
            raise ValueError("Expected GLM results; run compute_glm_activities.py first")
        required = {"log_activity", "dna_count"}
        view = MethodView(
            value_column="log_activity",
            title="SFv6 multi-barcode GLM activity",
            ylabel="Log GLM activity\nln(slope / ln(DNA count))",
            note=f"RNA ~ total RNA + intercept; slope / ln(DNA) | DNA > {min_dna:g}",
            stem="global_ccre_barcode_glm_activity_boxplot",
            zero_label=None,
            allow_nonfinite=True,
        )
    elif method == "rna-dna":
        required = {"rna_cpm", "dna_cpm", "log_fold_change"}
        base = manifest["log_base"]
        log_label = "ln" if np.isclose(base, np.e) else f"log{base:g}"
        view = MethodView(
            value_column="log_fold_change",
            title="SFv6 multi-barcode activity",
            ylabel=f"Log fold change activity\n{log_label}(RNA CPM / DNA CPM)",
            note="normalized by nanopore DNA abundance",
            stem="global_ccre_barcode_activity_boxplot",
            zero_label="RNA CPM = DNA CPM",
            allow_nonfinite=False,
        )
    elif method == "bayes":
        if manifest.get("method") != BAYES_METHOD:
            raise ValueError(
                "Expected Bayesian activities; run Bayes_multiBC/export_multibc_activity.py first"
            )
        required = {"activity", "activity_lo", "activity_hi"}
        view = MethodView(
            value_column="activity",
            title="SFv6 multi-barcode Bayesian activity",
            ylabel="Bayesian activity\nln γ(barcode) − mean ln γ(controls)",
            note="saturation_fail cells removed | posterior mean per barcode",
            stem="global_ccre_barcode_bayes_activity_boxplot",
            zero_label="Mean of barcode-only controls",
            allow_nonfinite=False,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    if "dna_input" not in manifest or not required.issubset(table.columns):
        raise ValueError("Recompute activities with nanopore DNA counts before plotting")
    return view


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(DEFAULT_DIRS), default="glm")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--min-dna", type=float, default=10,
                        help="GLM plot includes DNA counts strictly above this threshold")
    parser.add_argument("--order", choices=["name", "mean", "median"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.min_dna) or args.min_dna < 0:
        raise ValueError("--min-dna must be finite and nonnegative")
    default_results, default_outdir = DEFAULT_DIRS[args.method]
    args.results = args.results or default_results
    args.outdir = args.outdir or default_outdir
    table = pd.read_csv(args.results / "global_barcode_activities.csv")
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    view = method_view(args.method, manifest, table, args.min_dna)
    if table.empty or set(table.group) != {"all"} or table.feature.duplicated().any():
        raise ValueError("Expected one global activity per barcode")
    value_column = view.value_column
    finite = np.isfinite(table[value_column])
    if not view.allow_nonfinite and not finite.all():
        raise ValueError("Plot requires finite activities; check zero DNA counts and RNA counts")
    all_values = table.copy()
    all_values["plot_exclusion_reason"] = np.where(finite, "", "nonfinite_log_activity")
    if args.method == "glm":
        all_values.loc[all_values.dna_count.le(args.min_dna), "plot_exclusion_reason"] = "low_dna_count"
    all_values["plot_included"] = all_values.plot_exclusion_reason.eq("")
    table = all_values.loc[all_values.plot_included].copy()
    if table.empty:
        raise ValueError("No barcodes pass the plot filters")
    controls = set(table.loc[table.is_control, "cre"])
    targets = sorted(set(table.cre) - controls)
    if args.order in {"mean", "median"}:
        averages = table.groupby("cre")[value_column].agg(args.order)
        targets.sort(key=lambda name: (averages[name], name))
    order = targets + sorted(controls)
    values = [table.loc[table.cre.eq(cre), value_column].to_numpy() for cre in order]
    rng = np.random.default_rng(args.seed)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.31), 5.6))
    boxes = ax.boxplot(values, positions=np.arange(len(order)), widths=0.58,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": "#17394D", "linewidth": 1.3},
                       whiskerprops={"color": "#65747D", "linewidth": 0.8},
                       capprops={"color": "#65747D", "linewidth": 0.8})
    table["plot_x"] = np.nan
    table["plot_order"] = table.cre.map({cre: i for i, cre in enumerate(order)})
    for i, (cre, box) in enumerate(zip(order, boxes["boxes"])):
        control = cre in controls
        box.set(facecolor=CONTROL_FACE if control else TARGET_FACE,
                edgecolor=CONTROL_EDGE if control else TARGET_EDGE, linewidth=0.8)
        mask = table.cre.eq(cre)
        jitter = i + rng.uniform(-0.20, 0.20, int(mask.sum()))
        table.loc[mask, "plot_x"] = jitter
        ax.scatter(jitter, table.loc[mask, value_column], s=20,
                   color=CONTROL_POINT if control else TARGET_POINT, alpha=0.85,
                   edgecolors="white", linewidths=0.35, zorder=3)
    if view.zero_label is not None:
        ax.axhline(0, color="#858585", linestyle="--", linewidth=0.9, zorder=0)
    if controls and targets:
        ax.axvline(len(targets) - 0.5, color="#DDDDDD", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)), order, rotation=90, fontsize=8)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xlabel("cCRE", labelpad=10)
    ax.set_ylabel(view.ylabel)
    ax.set_title(view.title, loc="left", fontsize=15, weight="bold", pad=29)
    multiplicity = table.groupby("cre").size()
    barcode_note = (f"{multiplicity.iloc[0]} barcodes per construct" if multiplicity.nunique() == 1
                    else f"{len(table)} barcodes")
    ax.text(0, 1.035, f"{manifest['n_cells']:,} cells | {barcode_note} | {view.note}",
            transform=ax.transAxes, color="#555555", fontsize=10)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=TARGET_POINT, markersize=5, label="One barcode"),
        Line2D([], [], marker="o", linestyle="none", color=CONTROL_POINT, markersize=5, label="Barcode-only control"),
    ]
    if view.zero_label is not None:
        handles.append(Line2D([], [], linestyle="--", color="#858585", linewidth=0.9, label=view.zero_label))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=len(handles))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.16)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / view.stem
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    export = all_values.merge(table[["feature", "plot_order", "plot_x"]], on="feature", validate="one_to_one", how="left")
    export.sort_values(["plot_order", "barcode"]).to_csv(stem.with_suffix(".csv"), index=False)
    print(f"Plotted {len(table)} barcode points across {len(targets)} cCREs and "
          f"{len(controls)} control group(s); excluded {len(export) - len(table)} barcodes "
          f"(reasons in plot CSV): {stem}.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
