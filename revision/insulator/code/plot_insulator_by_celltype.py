#!/usr/bin/env python
"""Per-cell-type annotated insulator counts and proportions.

An insulator is ``Chr-O > --open-cutoff`` and ``Chr-A < --active-cutoff``. Two
bars per cell type: the number of annotated insulators (left axis) and that
number over the cCREs scored in the cell type (right axis).

The denominator is the cCREs the assay tested in that cell type at
``--t7-threshold`` (``--scope tested``, the default) -- the same denominator as
``revision/silencer/code/plot_silencer_recall_by_celltype.py``, so the insulator
and silencer figures are directly comparable. ``--scope annotation`` instead
scores the whole panel, where every cell type shares the identical 376-cCRE
denominator and the proportion carries no per-cell-type information.
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
from insulator_truth import (DEFAULT_ACTIVE_CUTOFF, DEFAULT_OPEN_CUTOFF,
                             load_insulator_truth)

COUNT_COLOR: Final[str] = "#f58518"
FRACTION_COLOR: Final[str] = "#9e9e9e"
BAR_WIDTH: Final[float] = 0.35


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotation-dir", type=Path,
                        default=root / "revision/silencer/results",
                        help="where cre_by_celltype_<state>_fraction.csv live")
    parser.add_argument("--tests", type=Path,
                        default=root / "revision/silencer/results"
                                     / "silencer_left_tail_tests.csv.gz",
                        help="eligible assay pairs, used by --scope tested")
    parser.add_argument("--out-dir", type=Path, default=root / "revision/insulator/results")
    parser.add_argument("--figures-dir", type=Path,
                        default=root / "revision/insulator/figures")
    parser.add_argument("--scope", choices=("tested", "annotation"), default="tested")
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--open-cutoff", type=float, default=DEFAULT_OPEN_CUTOFF)
    parser.add_argument("--active-cutoff", type=float, default=DEFAULT_ACTIVE_CUTOFF)
    parser.add_argument("--top-n", type=int, default=30,
                        help="cell types to draw, by insulator count (0 = all)")
    parser.add_argument("--stem", type=str, default=None)
    return parser.parse_args()


def eligible_mask(measured: pd.DataFrame, tests: Path,
                  t7_threshold: float) -> pd.DataFrame:
    """Cell type x cCRE mask of pairs the assay tested at the T7 filter."""
    frame = pd.read_csv(tests, usecols=["t7_threshold", "group", "cre"])
    frame = frame[frame["t7_threshold"] == t7_threshold]
    tested = (pd.crosstab(frame["group"], frame["cre"]) > 0)
    return tested.reindex(index=measured.index, columns=measured.columns, fill_value=False)


def summarize(mask: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """One row per cell type: cCREs in the denominator, insulators, proportion."""
    table = pd.DataFrame({
        "n_cre_tested": scored.sum(axis=1).astype(int),
        "n_insulator": (mask & scored).sum(axis=1).astype(int),
    })
    table = table[table["n_cre_tested"] > 0]
    table["fraction_insulator"] = table["n_insulator"] / table["n_cre_tested"]
    return (table.sort_values(["n_insulator", "n_cre_tested"], ascending=False)
            .rename_axis("cell_type").reset_index())


def plot(table: pd.DataFrame, path: Path, open_cutoff: float, active_cutoff: float,
         scope: str, pooled: tuple[int, int]) -> None:
    positions = np.arange(len(table), dtype=float)
    fig, ax_count = plt.subplots(figsize=(max(6.0, 0.55 * len(table) + 2.0), 4.4),
                                 constrained_layout=True)
    ax_rate = ax_count.twinx()
    ax_count.set_zorder(ax_rate.get_zorder() + 1)
    ax_count.patch.set_visible(False)

    bars_count = ax_count.bar(positions - BAR_WIDTH / 2, table["n_insulator"],
                              width=BAR_WIDTH, color=COUNT_COLOR,
                              label="annotated insulators (n)")
    bars_frac = ax_rate.bar(positions + BAR_WIDTH / 2, table["fraction_insulator"],
                            width=BAR_WIDTH, color=FRACTION_COLOR,
                            label="insulators / cCREs tested")
    ax_count.bar_label(bars_count, fmt="%d", fontsize=6.5, padding=2, rotation=90)
    ax_rate.bar_label(bars_frac, fmt="%.3f", fontsize=6.5, padding=2, rotation=90)

    ax_count.set_xticks(positions)
    ax_count.set_xticklabels(table["cell_type"], rotation=45, ha="right", fontsize=8)
    ax_count.set_ylabel("annotated insulators (count)", color=COUNT_COLOR)
    ax_count.tick_params(axis="y", colors=COUNT_COLOR)
    ax_count.set_ylim(0, max(table["n_insulator"].max(), 1) * 1.45)
    ax_rate.set_ylabel("insulators / cCREs tested")
    ax_rate.set_ylim(0, max(float(table["fraction_insulator"].max()), 1e-3) * 1.45)
    insulators, scored = pooled
    ax_count.set_title(
        f"Annotated insulators (Chr-O > {open_cutoff:g}, Chr-A < {active_cutoff:g}) "
        f"per cell type [{scope} scope: {insulators} of {scored} pairs]",
        fontsize=11, pad=24)
    handles = [bars_count, bars_frac]
    ax_count.legend(handles, [h.get_label() for h in handles], frameon=False,
                    fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    truth = load_insulator_truth(args.annotation_dir)
    mask = truth.mask(args.open_cutoff, args.active_cutoff)
    scored = truth.measured
    if args.scope == "tested":
        scored = scored & eligible_mask(scored, args.tests, args.t7_threshold)

    table = summarize(mask, scored)
    pooled = (int(table["n_insulator"].sum()), int(table["n_cre_tested"].sum()))

    stem = args.stem or f"insulator_by_celltype_{args.scope}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{stem}.csv"
    table.to_csv(out_csv, index=False)

    plotted = table[table["n_insulator"] > 0]
    if args.top_n > 0:
        plotted = plotted.head(args.top_n)
    fig_path = args.figures_dir / f"{stem}.pdf"
    plot(plotted.reset_index(drop=True), fig_path, args.open_cutoff, args.active_cutoff,
         args.scope, pooled)

    print(f"scope={args.scope}: {len(table)} cell types, {pooled[0]} insulators "
          f"of {pooled[1]} scored pairs; {len(plotted)} cell types plotted")
    print(f"wrote {out_csv}")
    print(f"wrote {fig_path} (+ .png)")


if __name__ == "__main__":
    main()
