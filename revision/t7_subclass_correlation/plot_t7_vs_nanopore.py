#!/usr/bin/env python3
"""T7 recovery vs the nanopore input library.

Two figures relate per-cCRE T7 barcode counts (``obsm['T7CRE']`` in the
5/28/2025 single-cell dataset) to the ONT/nanopore read counts of the input
AAV library (``SFv8_400CRE_nanopore_counts.csv``):

  1. Pooled scatter: for every cCRE, total T7 counts summed across *all* cells
     (y, log) vs nanopore read counts (x, log), with the Pearson r annotated.
  2. Per-cell-type recovery: for each subclass, the Pearson r between its
     per-cCRE total T7 counts and the nanopore counts (both log10(x+1)), drawn
     against the subclass cell number (left) and total T7 counts (right).
  3. The same per-cell-type recovery restricted to well-covered cCREs: within
     each subclass only cCREs with at least ``--min-cre-t7`` T7 counts are
     correlated, and subclasses left with fewer than ``--min-cres`` such cCREs
     are dropped. This removes the zero-inflated floor that otherwise drives
     the correlation in shallow subclasses.

Both counts are analyzed on log10(count + 1) so the correlations match the
convention used in the T7 subclass-correlation heatmap.

Example
-------
    python revision/t7_subclass_correlation/plot_t7_vs_nanopore.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_t7_subclass_correlation import (
    SectionPseudobulk,
    default_h5ad,
    load_pseudobulk,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
NANOPORE_CSV = os.path.join(
    REPO_ROOT, "STARRFISH_in_vivo", "Data", "SFv8_400CRE_nanopore_counts.csv"
)
BLACKLIST_CSV = os.path.join(REPO_ROOT, "revision", "Data", "cre_blacklist.csv")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_nanopore_counts(path: str) -> pd.Series:
    """Return per-cCRE nanopore read counts indexed by the CRE001... convention."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"nanopore counts CSV not found: {path}")
    df = pd.read_csv(path)
    if not {"CRE", "counts"}.issubset(df.columns):
        raise KeyError(f"expected columns CRE,counts; got {list(df.columns)}")
    return df.set_index("CRE")["counts"].astype(float)


def load_blacklist(path: str) -> list[str]:
    """Return the list of blacklisted cCRE ids (empty if the file is absent)."""
    if not path or not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if "cre" not in df.columns:
        raise KeyError(f"expected a 'cre' column in {path}; got {list(df.columns)}")
    return df["cre"].astype(str).tolist()


def _row_pearson_vs(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson r of each row of ``matrix`` against ``target`` (NaN if degenerate)."""
    mc = matrix - matrix.mean(axis=1, keepdims=True)
    tc = target - target.mean()
    num = mc @ tc
    denom = np.sqrt((mc**2).sum(axis=1)) * np.sqrt((tc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, num / denom, np.nan)


def _row_pearson_vs_masked(
    counts: np.ndarray,
    target: np.ndarray,
    min_count: float,
    min_cres: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row Pearson r against ``target`` using only well-covered cCREs.

    For each row of ``counts`` (raw T7 counts, n_sub x n_cre) the correlation is
    computed on the columns with ``counts >= min_count`` only, on
    log10(count + 1) for both variables. Rows keeping fewer than ``min_cres``
    columns get NaN.

    Returns
    -------
    (r, n_kept)
        Both length n_sub; ``n_kept`` is the per-row surviving cCRE count.
    """
    n_sub = counts.shape[0]
    r = np.full(n_sub, np.nan, dtype=float)
    n_kept = np.zeros(n_sub, dtype=int)
    log_target = np.log10(target + 1.0)
    for i in range(n_sub):
        mask = counts[i] >= min_count
        n_kept[i] = int(mask.sum())
        if n_kept[i] < max(min_cres, 3):
            continue
        x = log_target[mask]
        y = np.log10(counts[i][mask] + 1.0)
        if x.std() == 0.0 or y.std() == 0.0:
            continue
        r[i] = float(np.corrcoef(x, y)[0, 1])
    return r, n_kept


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_pooled_scatter(
    nano: np.ndarray, total_t7: np.ndarray, n_cre: int, outpath: str
) -> None:
    """Total T7 across all cells vs nanopore read counts, one point per cCRE."""
    log_nano = np.log10(nano + 1.0)
    log_t7 = np.log10(total_t7 + 1.0)
    r, p = stats.pearsonr(log_nano, log_t7)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(nano, total_t7, s=20, alpha=0.5, color="#4c6ef5", edgecolor="none")
    # least-squares fit in log-log space
    slope, intercept = np.polyfit(log_nano, log_t7, 1)
    xs = np.linspace(log_nano.min(), log_nano.max(), 100)
    ax.plot(10.0**xs, 10.0 ** (slope * xs + intercept), color="0.3",
            linewidth=1.2, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nanopore read counts (input AAV library)")
    ax.set_ylabel("Total T7 counts across all cells")
    ax.set_title("Per-cCRE T7 recovery vs input library abundance", fontsize=11)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\np = {p:.1e}\nn = {n_cre} cCREs",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"))
    ax.grid(True, which="both", color="0.9", linewidth=0.4)
    ax.set_axisbelow(True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_corr_vs_depth(
    r_per_subclass: pd.Series,
    n_cells: pd.Series,
    total_t7: pd.Series,
    classes: pd.Series,
    outpath: str,
    n_cre: pd.Series | None = None,
    suptitle: str = ("Per-subclass T7-vs-nanopore recovery correlation "
                     "vs sequencing depth"),
    ylabel: str = "Pearson r\n(per-cCRE log T7 vs log nanopore)",
) -> None:
    """Per-subclass T7-vs-nanopore r against cell number and total T7 counts.

    When ``n_cre`` is given a third panel plots r against the number of cCREs
    that survived the per-subclass coverage filter.
    """
    subs = r_per_subclass.index
    finite = r_per_subclass.to_numpy()
    class_labels = classes.reindex(subs).astype(str).to_numpy()
    class_levels = sorted(pd.unique(class_labels))
    cmap = plt.get_cmap("tab20", max(len(class_levels), 1))
    color_of = {c: cmap(i) for i, c in enumerate(class_levels)}
    point_colors = [color_of[c] for c in class_labels]

    n_panels = 3 if n_cre is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 6.0),
                             constrained_layout=True, sharey=True)
    panels = [
        (axes[0], n_cells.reindex(subs).to_numpy(), "Number of cells in subclass"),
        (axes[1], total_t7.reindex(subs).to_numpy(), "Total T7 counts in subclass"),
    ]
    if n_cre is not None:
        panels.append((axes[2], n_cre.reindex(subs).to_numpy().astype(float),
                       "Number of cCREs passing the T7 filter"))
    for ax, xvals, xlabel in panels:
        ax.scatter(xvals, finite, s=28, c=point_colors, alpha=0.85,
                   edgecolor="0.3", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", color="0.9", linewidth=0.4)
        ax.set_axisbelow(True)
        # Spearman trend between the x-axis depth and the recovery r
        mask = np.isfinite(xvals) & np.isfinite(finite) & (xvals > 0)
        if mask.sum() >= 3:
            rho, p = stats.spearmanr(np.log10(xvals[mask]), finite[mask])
            ax.text(0.05, 0.05, f"Spearman ρ = {rho:.3f}\np = {p:.1e}\n"
                    f"n = {int(mask.sum())} subclasses",
                    transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                              edgecolor="0.7"))
    axes[0].set_ylabel(ylabel)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor=color_of[c], markeredgecolor="0.3",
                          label=c) for c in class_levels]
    fig.legend(handles=handles, title="Class", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=6, title_fontsize=8,
               frameon=False)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="input AnnData (default: sole 5/28 file)")
    ap.add_argument("--nanopore", default=NANOPORE_CSV, help="nanopore counts CSV")
    ap.add_argument("--blacklist", default=BLACKLIST_CSV,
                    help="cCRE blacklist CSV to exclude (pass '' to keep all)")
    ap.add_argument("--subclass-col", default="subclass_name")
    ap.add_argument("--class-col", default="class_name")
    ap.add_argument("--obsm-key", default=None, help="T7 obsm key (default: auto)")
    ap.add_argument("--min-cre-t7", type=float, default=50.0,
                    help="filtered figure: minimum per-subclass T7 counts for a "
                         "cCRE to enter the correlation (default: 50)")
    ap.add_argument("--min-cres", type=int, default=50,
                    help="filtered figure: minimum surviving cCREs for a "
                         "subclass to be plotted (default: 50)")
    ap.add_argument("--outdir", default=os.path.join(THIS_DIR, "figures"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    nano = load_nanopore_counts(args.nanopore)
    data_path = os.path.abspath(args.data) if args.data else default_h5ad()
    print(f"Loading {data_path}")
    pbk: SectionPseudobulk = load_pseudobulk(
        "all", args.subclass_col, args.class_col, args.obsm_key, data_path
    )

    # cCREs measured by both assays (drops blanks and unmeasured CREs)
    blacklist = set(load_blacklist(args.blacklist))
    cres = [c for c in pbk.pb.columns if c in nano.index and c not in blacklist]
    if len(cres) < 3:
        raise ValueError(f"only {len(cres)} cCREs shared between T7 and nanopore")
    if blacklist:
        print(f"excluded {len(blacklist)} blacklisted cCREs "
              f"({sorted(blacklist)})")
    nano_v = nano.reindex(cres).to_numpy()
    pb = pbk.pb.reindex(columns=cres)
    print(f"{len(cres)} cCREs shared between T7CRE and nanopore counts; "
          f"{pbk.pb.shape[0]} subclasses")

    # ---- plot 1: pooled per-cCRE scatter ---- #
    total_per_cre = pb.sum(axis=0).to_numpy()
    out1 = os.path.join(args.outdir, "t7_vs_nanopore_pooled.png")
    plot_pooled_scatter(nano_v, total_per_cre, len(cres), out1)

    # ---- plot 2: per-subclass recovery r vs depth ---- #
    log_t7 = np.log10(pb.to_numpy() + 1.0)          # n_sub x n_cre
    log_nano = np.log10(nano_v + 1.0)
    r_vals = _row_pearson_vs(log_t7, log_nano)
    r_series = pd.Series(r_vals, index=pb.index)
    # total over the shared cCREs only (blanks + blacklist already excluded)
    total_clean = pb.sum(axis=1)
    out2 = os.path.join(args.outdir, "t7_nanopore_corr_vs_depth.png")
    plot_corr_vs_depth(r_series, pbk.n_cells, total_clean, pbk.cls, out2)

    n_ok = int(np.isfinite(r_vals).sum())
    print(f"per-subclass recovery r: {n_ok}/{len(r_vals)} finite, "
          f"median {np.nanmedian(r_vals):.3f}, "
          f"range {np.nanmin(r_vals):.3f}..{np.nanmax(r_vals):.3f}")

    # ---- plot 3: same, restricted to well-covered cCREs per subclass ---- #
    counts = pb.to_numpy(dtype=float)
    r_f, n_kept = _row_pearson_vs_masked(
        counts, nano_v, args.min_cre_t7, args.min_cres
    )
    r_f_series = pd.Series(r_f, index=pb.index).dropna()
    n_kept_series = pd.Series(n_kept, index=pb.index)
    if r_f_series.empty:
        raise ValueError(
            f"no subclass keeps >= {args.min_cres} cCREs with T7 >= "
            f"{args.min_cre_t7:g}; relax --min-cre-t7/--min-cres"
        )
    out3 = os.path.join(args.outdir, "t7_nanopore_corr_vs_depth_filtered.png")
    plot_corr_vs_depth(
        r_f_series, pbk.n_cells, total_clean, pbk.cls, out3,
        n_cre=n_kept_series,
        suptitle=(f"Per-subclass T7-vs-nanopore recovery correlation "
                  f"(cCREs with T7 >= {args.min_cre_t7:g} in the subclass; "
                  f"subclasses with >= {args.min_cres} such cCREs)"),
        ylabel=(f"Pearson r\n(log T7 vs log nanopore, "
                f"T7 >= {args.min_cre_t7:g} cCREs)"),
    )
    r_f_vals = r_f_series.to_numpy()
    print(f"filtered recovery r (T7 >= {args.min_cre_t7:g}, "
          f">= {args.min_cres} cCREs): {len(r_f_vals)}/{len(r_f)} subclasses "
          f"kept, median {np.median(r_f_vals):.3f}, "
          f"range {r_f_vals.min():.3f}..{r_f_vals.max():.3f}, "
          f"cCREs kept median {int(np.median(n_kept_series[r_f_series.index]))}")
    print(f"wrote {out1}, {out2} and {out3} (+.pdf)")


if __name__ == "__main__":
    main()
