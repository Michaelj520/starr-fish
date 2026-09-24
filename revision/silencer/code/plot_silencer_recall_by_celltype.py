#!/usr/bin/env python
"""Per-cell-type silencer annotation coverage and left-tail recall.

Breaks the pooled ``silencer_left_tail_precision_recall.csv`` row at one truth
cutoff down by cell type (Allen subclass) and draws three bars per cell type:

    n_silencer   number of annotated silencers among the cCREs tested there
    prevalence   n_silencer / n_cre_tested
    recall       n_silencer_called / n_silencer     (calls are q_left <= cutoff)

Counts live on the left axis, the two fractions on the right axis. Cell types
with no annotated silencer carry no recall and are dropped by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baystarrfish.data.paths import repo_root

COUNT_COLOR: Final[str] = "#e45756"
PREVALENCE_COLOR: Final[str] = "#9e9e9e"
RECALL_COLOR: Final[str] = "#4c78a8"
BAR_WIDTH: Final[float] = 0.27
#: Recall below this many annotated silencers rests on too few positives to read.
LOW_N_SILENCER: Final[int] = 5


def parse_args() -> argparse.Namespace:
    root = repo_root()
    results = root / "revision/silencer/results"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tests", type=Path,
                        default=results / "silencer_left_tail_tests_annotated.csv.gz")
    parser.add_argument("--out-dir", type=Path, default=results)
    parser.add_argument("--figures-dir", type=Path,
                        default=root / "revision/silencer/figures")
    parser.add_argument("--silencer-cutoff", type=float, default=0.1,
                        help="annotated silencer when repressive fraction > this")
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--stem", type=str, default="silencer_recall_by_celltype")
    parser.add_argument("--bars", choices=("all", "annotation"), default="all",
                        help="'annotation' drops the recall bar (count + prevalence only)")
    parser.add_argument("--keep-zero-silencer", action="store_true",
                        help="keep cell types with no annotated silencer (recall is NaN)")
    return parser.parse_args()


def summarize(tests: pd.DataFrame, silencer_cutoff: float,
              q_cutoff: float) -> pd.DataFrame:
    """One row per cell type: tested / annotated / called counts and rates."""
    frame = tests.assign(
        is_silencer=tests["repressive_fraction"] > silencer_cutoff,
        called=tests["q_left"] <= q_cutoff,
    )
    grouped = frame.groupby("group", sort=False)
    table = pd.DataFrame({
        "n_cre_tested": grouped.size(),
        "n_silencer": grouped["is_silencer"].sum().astype(int),
        "n_called": grouped["called"].sum().astype(int),
        "n_silencer_called": grouped.apply(
            lambda d: int((d["is_silencer"] & d["called"]).sum()), include_groups=False
        ),
    })
    table["prevalence"] = table["n_silencer"] / table["n_cre_tested"]
    table["call_rate"] = table["n_called"] / table["n_cre_tested"]
    table["recall"] = np.where(
        table["n_silencer"] > 0, table["n_silencer_called"] / table["n_silencer"], np.nan
    )
    table["precision"] = np.where(
        table["n_called"] > 0, table["n_silencer_called"] / table["n_called"], np.nan
    )
    return (table.sort_values(["n_silencer", "n_cre_tested"], ascending=False)
            .rename_axis("cell_type").reset_index())


def plot(table: pd.DataFrame, path: Path, silencer_cutoff: float,
         pooled_recall: float, with_recall: bool = True) -> None:
    positions = np.arange(len(table), dtype=float)
    low_n = table["n_silencer"] < LOW_N_SILENCER
    # Two bars straddle the tick; three need the outer pair pushed out by a width.
    width = BAR_WIDTH if with_recall else BAR_WIDTH * 1.3
    offsets = (-width, 0.0) if with_recall else (-width / 2, width / 2)
    fig, ax_count = plt.subplots(figsize=(max(6.0, 0.62 * len(table) + 2.0), 4.6),
                                 constrained_layout=True)
    ax_rate = ax_count.twinx()
    ax_count.set_zorder(ax_rate.get_zorder() + 1)
    ax_count.patch.set_visible(False)

    bars_count = ax_count.bar(positions + offsets[0], table["n_silencer"],
                              width=width, color=COUNT_COLOR,
                              label="annotated silencers (n)")
    bars_prev = ax_rate.bar(positions + offsets[1], table["prevalence"], width=width,
                            color=PREVALENCE_COLOR,
                            label="silencers / cCREs tested")
    rate_bars = [bars_prev]
    handles = [bars_count, bars_prev]
    if with_recall:
        bars_recall = ax_rate.bar(positions + width, table["recall"], width=width,
                                  color=RECALL_COLOR,
                                  label="recall (called / annotated)")
        # Recall on a handful of positives is noise; hatch it rather than hide it.
        for bar, flag in zip(bars_recall, low_n):
            if flag:
                bar.set_hatch("///")
                bar.set_edgecolor("white")
        rate_bars.append(bars_recall)
        handles.append(bars_recall)
        ax_rate.axhline(pooled_recall, color=RECALL_COLOR, linestyle="--", linewidth=1,
                        label=f"pooled recall = {pooled_recall:.2f}")
        handles.append(ax_rate.lines[0])

    ax_count.bar_label(bars_count, fmt="%d", fontsize=6.5, padding=2, rotation=90)
    for bars in rate_bars:
        ax_rate.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2, rotation=90)

    ax_count.set_xticks(positions)
    labels = [f"{name} *" if (flag and with_recall) else name
              for name, flag in zip(table["cell_type"], low_n)]
    ax_count.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_count.set_ylabel("annotated silencers (count)", color=COUNT_COLOR)
    ax_count.tick_params(axis="y", colors=COUNT_COLOR)
    ax_count.set_ylim(0, table["n_silencer"].max() * 1.45)
    ax_rate.set_ylabel("silencers / cCREs tested" if not with_recall else "fraction")
    rate_top = 1.45 if with_recall else float(table["prevalence"].max()) * 1.45
    ax_rate.set_ylim(0, rate_top)
    if with_recall:
        ax_count.set_xlabel(
            f"* fewer than {LOW_N_SILENCER} annotated silencers - hatched recall bar "
            f"rests on too few positives to read", fontsize=8)
        title = (f"Annotated silencers (Chr-R + Hc-P > {silencer_cutoff:g}) and "
                 f"left-tail recall per cell type")
    else:
        title = (f"Annotated silencers (Chr-R + Hc-P > {silencer_cutoff:g}) "
                 f"per cell type")
    ax_count.set_title(title, fontsize=11, pad=26)

    ax_count.legend(handles, [h.get_label() for h in handles], frameon=False,
                    fontsize=8, ncol=len(handles), loc="lower center",
                    bbox_to_anchor=(0.5, 1.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tests = pd.read_csv(args.tests)
    table = summarize(tests, args.silencer_cutoff, args.q_cutoff)

    pooled_silencer = int(table["n_silencer"].sum())
    pooled_recall = float(table["n_silencer_called"].sum() / pooled_silencer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{args.stem}.csv"
    table.to_csv(out_csv, index=False)

    plotted = table if args.keep_zero_silencer else table[table["n_silencer"] > 0]
    with_recall = args.bars == "all"
    fig_path = args.figures_dir / f"{args.stem}.pdf"
    plot(plotted.reset_index(drop=True), fig_path, args.silencer_cutoff, pooled_recall,
         with_recall=with_recall)

    print(f"{len(table)} cell types, {len(plotted)} plotted; "
          f"{pooled_silencer} annotated silencers, pooled recall {pooled_recall:.3f}")
    print(f"wrote {out_csv}")
    print(f"wrote {fig_path} (+ .png)")


if __name__ == "__main__":
    main()
