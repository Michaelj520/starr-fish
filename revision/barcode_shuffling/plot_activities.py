#!/usr/bin/env python3
"""Plot global GLM or RNA/DNA barcode activities as cCRE boxes with jittered points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["glm", "rna-dna"], default="glm")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--min-dna", type=float, default=10,
                        help="GLM plot includes DNA counts strictly above this threshold")
    parser.add_argument("--order", choices=["name", "mean", "median"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    is_glm = args.method == "glm"
    if not np.isfinite(args.min_dna) or args.min_dna < 0:
        raise ValueError("--min-dna must be finite and nonnegative")
    if args.results is None:
        args.results = HERE / ("results/glm" if is_glm else "results")
    if args.outdir is None:
        args.outdir = HERE / ("figures/glm" if is_glm else "figures")
    table = pd.read_csv(args.results / "global_barcode_activities.csv")
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    required = {"log_activity", "dna_count"} if is_glm else {"rna_cpm", "dna_cpm", "log_fold_change"}
    if "dna_input" not in manifest or not required.issubset(table.columns):
        raise ValueError("Recompute activities with nanopore DNA counts before plotting")
    if is_glm and manifest.get("method") != "glm_total_ols_nanopore":
        raise ValueError("Expected GLM results; run compute_glm_activities.py first")
    if table.empty or set(table.group) != {"all"} or table.feature.duplicated().any():
        raise ValueError("Expected one global activity per barcode")
    value_column = "log_activity" if is_glm else "log_fold_change"
    finite = np.isfinite(table[value_column])
    if not is_glm and not finite.all():
        raise ValueError("Plot requires finite activities; check zero DNA counts and RNA counts")
    all_values = table.copy()
    all_values["plot_exclusion_reason"] = np.where(finite, "", "nonfinite_log_activity")
    if is_glm:
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
    base = manifest["log_base"]
    log_label = "ln" if np.isclose(base, np.e) else f"log{base:g}"
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
        box.set(facecolor="#F4D5B6" if control else "#D9E8F0",
                edgecolor="#BA7E44" if control else "#7FA1B5", linewidth=0.8)
        mask = table.cre.eq(cre)
        jitter = i + rng.uniform(-0.20, 0.20, int(mask.sum()))
        table.loc[mask, "plot_x"] = jitter
        ax.scatter(jitter, table.loc[mask, value_column], s=20,
                   color="#BD6B28" if control else "#276687", alpha=0.85,
                   edgecolors="white", linewidths=0.35, zorder=3)
    if not is_glm:
        ax.axhline(0, color="#858585", linestyle="--", linewidth=0.9, zorder=0)
    if controls and targets:
        ax.axvline(len(targets) - 0.5, color="#DDDDDD", linewidth=0.8)
    ax.set_xticks(np.arange(len(order)), order, rotation=90, fontsize=8)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xlabel("cCRE", labelpad=10)
    ax.set_ylabel("Log GLM activity\nln(slope / ln(DNA count))" if is_glm
                  else f"Log fold change activity\n{log_label}(RNA CPM / DNA CPM)")
    ax.set_title("SFv6 multi-barcode GLM activity" if is_glm else "SFv6 multi-barcode activity",
                 loc="left", fontsize=15, weight="bold", pad=29)
    multiplicity = table.groupby("cre").size()
    barcode_note = (f"{multiplicity.iloc[0]} barcodes per construct" if multiplicity.nunique() == 1
                    else f"{len(table)} barcodes")
    method_note = (f"RNA ~ total RNA + intercept; slope / ln(DNA) | DNA > {args.min_dna:g}"
                   if is_glm else "normalized by nanopore DNA abundance")
    ax.text(0, 1.035, f"{manifest['n_cells']:,} cells | {barcode_note} | {method_note}",
            transform=ax.transAxes, color="#555555", fontsize=10)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color="#276687", markersize=5, label="One barcode"),
        Line2D([], [], marker="o", linestyle="none", color="#BD6B28", markersize=5, label="Barcode-only control"),
    ]
    if not is_glm:
        handles.append(Line2D([], [], linestyle="--", color="#858585", linewidth=0.9, label="RNA CPM = DNA CPM"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=len(handles))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.16)
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / ("global_ccre_barcode_glm_activity_boxplot" if is_glm
                         else "global_ccre_barcode_activity_boxplot")
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
