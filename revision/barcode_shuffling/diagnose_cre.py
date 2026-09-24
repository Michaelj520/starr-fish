#!/usr/bin/env python3
"""Plot per-cell barcode RNA versus total reporter RNA and diagnose OLS sensitivity."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from compute_activities import DEFAULT_INPUT, DEFAULT_DNA, HERE, load_counts, load_dna_counts
from compute_glm_activities import fit_total_ols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cre", default="CRE182")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dna-csv", type=Path, default=DEFAULT_DNA)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    out = args.outdir or HERE / "diagnostics" / args.cre
    counts, labels, metadata = load_counts(args.h5ad, "leiden")
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    dna = load_dna_counts(args.dna_csv, metadata.index)
    names = metadata.index[metadata.cre.eq(args.cre)]
    if not len(names):
        raise ValueError(f"No barcodes for {args.cre}")
    y = counts[:, metadata.index.get_indexer(names)]
    total = counts.sum(1)
    selected_dna = dna.loc[names, "DNA_count"].to_numpy()
    fits = fit_total_ols(y, total, selected_dna)
    n = len(total)
    cutoff = np.quantile(total, 0.99)
    typical = total <= cutoff
    cells = pd.DataFrame(y, columns=names, index=labels.index)
    cells.index.name = "cell"
    cells["total_rna"] = total
    cells["cre_total_rna"] = y.sum(1)
    cells["leiden"] = labels.astype(str)
    cells["fov"] = cells.index.str.split("--").str[0]
    xc = total - total.mean()
    sxx = xc @ xc
    leverage = 1 / n + xc ** 2 / sxx
    summaries, influence, sensitivity = [], [], []
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, len(names), figsize=(3.7 * len(names), 8.2), squeeze=False)
    for j, name in enumerate(names):
        response = y[:, j]
        fit = fits.iloc[j]
        residual = response - fit.intercept - fit.slope * total
        mse = np.sum(residual ** 2) / (n - 2)
        cook = residual ** 2 / (2 * mse) * leverage / (1 - leverage) ** 2 if mse > 0 else np.zeros(n)
        top = np.argsort(cook)[-20:][::-1]
        leave_one_out_slope = fit.slope - residual * xc / (sxx * (1 - leverage))
        lfc = np.log((response.sum() / total.sum()) / (selected_dna[j] / dna.DNA_count.sum()))
        summary = {"feature": name, **fit.to_dict(),
                   "rna_dna_log_fold_change": lfc,
                   "fraction_positive": float((response > 0).mean()),
                   "mean_rna_per_cell": response.mean(), "max_rna_per_cell": response.max(),
                   "rna_cpm": response.sum() / total.sum() * 1e6,
                   "dna_cpm": selected_dna[j] / dna.DNA_count.sum() * 1e6,
                   "pearson_r": np.corrcoef(total, response)[0, 1],
                   "rna_fraction_in_top1pct_total_rna_cells": response[~typical].sum() / response.sum(),
                   "max_cooks_distance": cook.max(),
                   "max_single_cell_relative_slope_change": np.max(np.abs(leave_one_out_slope / fit.slope - 1)),
                   "n_fov_with_rna": int(cells.loc[response > 0, "fov"].nunique()),
                   "n_fov": int(cells.fov.nunique())}
        summaries.append(summary)
        top_cells = cells.iloc[top][[name, "total_rna", "fov", "leiden"]].rename(columns={name: "barcode_rna"}).reset_index()
        top_cells["feature"] = name
        top_cells["cooks_distance"] = cook[top]
        top_cells["leverage"] = leverage[top]
        influence.append(top_cells)
        omit_influential = np.ones(n, dtype=bool)
        omit_influential[np.argsort(cook)[-max(1, int(np.ceil(n * .01))):]] = False
        for method, keep, predictor in (
            ("Original", np.ones(n, dtype=bool), total),
            ("Exclude barcode from total", np.ones(n, dtype=bool), total - response),
            ("Exclude CRE from total", np.ones(n, dtype=bool), total - y.sum(1)),
            ("Omit top 1% total RNA", typical, total),
            ("Omit top 1% Cook distance", omit_influential, total),
        ):
            refit = fit_total_ols(response[keep, None], predictor[keep], selected_dna[j:j+1]).iloc[0]
            sensitivity.append({"feature": name, "method": method, "n_cells": int(keep.sum()),
                                "n_positive_cells_retained": int((response[keep] > 0).sum()),
                                "slope": refit.slope, "log_activity": refit.log_activity,
                                "delta_log_activity": refit.log_activity - fit.log_activity})
        for row, keep in enumerate((np.ones(n, dtype=bool), typical)):
            ax = axes[row, j]
            ax.scatter(total[keep], response[keep], s=4, alpha=.17, color="#276687", rasterized=True, linewidths=0)
            xline = np.array([0, total[keep].max()])
            ax.plot(xline, fit.intercept + fit.slope * xline, color="#B33D32", lw=1.2, label="Global OLS")
            bins = pd.qcut(total[keep], q=10, duplicates="drop")
            means = pd.DataFrame({"x": total[keep], "y": response[keep], "bin": bins}).groupby("bin", observed=True)[["x", "y"]].mean()
            ax.plot(means.x, means.y, "o-", color="#D18A18", ms=3.5, lw=1, label="Decile mean")
            ax.set_xlim(-total[keep].max() * .025, total[keep].max() * 1.025)
            ax.set_ylim(-.3, max(1, response[keep].max()) * 1.08)
            ax.set_xlabel("Total reporter RNA counts per cell")
            ax.set_ylabel("Barcode RNA counts per cell")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.15)
            if row == 0:
                ax.set_title(f"{name}\nRNA/DNA ln FC = {lfc:.2f}; GLM ln activity = {fit.log_activity:.2f}\n"
                             f"RNA = {response.sum():,.0f}; DNA = {selected_dna[j]:,.0f}\n"
                             f"Detected in {(response > 0).mean():.2%} of cells", fontsize=9)
            else:
                ax.set_title(f"Zoom: total RNA ≤ {cutoff:,.0f} (99th percentile)", fontsize=9)
        if j == 0:
            axes[0, j].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{args.cre}: per-cell RNA versus total reporter RNA | {n:,} cells\n"
                 "Each point is one cell; zeros included. Y-axis ranges differ by barcode. Total includes the target barcode.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.cre}_rna_vs_total_rna.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    summary = pd.DataFrame(summaries)
    refits = pd.DataFrame(sensitivity)
    sequencing = pd.read_csv(args.dna_csv)
    sequencing["feature"] = sequencing.element.str.replace(r"^barcode_only_", "barcode only_", regex=True)
    sequencing = sequencing.set_index("feature").reindex(names)
    has_replicates = {"activity_1", "activity_2"}.issubset(sequencing.columns)
    if has_replicates:
        comparison = summary[["feature", "rna_dna_log_fold_change", "log_activity"]].merge(
            sequencing[["count1", "count2", "activity_1", "activity_2"]], on="feature", validate="one_to_one")
        comparison.to_csv(out / "sequencing_comparison.csv", index=False)
    summary.to_csv(out / "barcode_diagnostics.csv", index=False)
    refits.to_csv(out / "sensitivity_refits.csv", index=False)
    pd.concat(influence).to_csv(out / "influential_cells.csv", index=False)
    cells.to_csv(out / "cell_counts.csv.gz")
    fov = cells.groupby("fov").size().rename("n_cells").to_frame()
    for name in names:
        fov[name + "_rna_total"] = cells.groupby("fov")[name].sum()
        fov[name + "_positive_fraction"] = cells.assign(positive=cells[name] > 0).groupby("fov").positive.mean()
    fov.to_csv(out / "fov_summary.csv")
    fig, axes = plt.subplots(1, 3 if has_replicates else 2, figsize=(17.5 if has_replicates else 12, 4.8))
    positions = np.arange(len(names))
    axes[0].bar(positions - .18, summary.rna_cpm, .36, label="RNA CPM", color="#276687")
    axes[0].bar(positions + .18, summary.dna_cpm, .36, label="DNA CPM", color="#D18A18")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Library abundance (CPM, log scale)")
    axes[0].set_title("RNA signal and input DNA abundance")
    axes[0].set_xticks(positions, [name.rsplit('_', 1)[1] for name in names])
    axes[0].legend(frameon=False)
    for method, frame in refits.groupby("method", sort=False):
        # Outcome-selected trimming removes nearly all positives for sparse barcodes;
        # retain it in the audit table, but do not frame it as an unbiased refit.
        if method == "Omit top 1% Cook distance":
            continue
        axes[1].plot(positions, frame.set_index("feature").loc[names, "log_activity"], "o-", label=method, ms=4)
    axes[1].set_xticks(positions, [name.rsplit('_', 1)[1] for name in names])
    axes[1].set_ylabel("Log GLM activity")
    axes[1].set_title("Regression sensitivity (diagnostic refits)")
    axes[1].legend(fontsize=8, frameon=False)
    if has_replicates:
        axes[2].plot(positions, summary.rna_dna_log_fold_change, "o-", label="STARR-FISH", color="#276687")
        for column, label in (("activity_1", "STARR-seq replicate 1"), ("activity_2", "STARR-seq replicate 2")):
            axes[2].plot(positions, np.log(sequencing[column]), "o--", label=label)
        axes[2].set_xticks(positions, [name.rsplit('_', 1)[1] for name in names])
        axes[2].set_ylabel("ln(RNA/DNA activity)")
        axes[2].set_title("Comparison with supplied replicate activities")
        axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(args.cre, fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.cre}_diagnostic_summary.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(refits.to_string(index=False))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
