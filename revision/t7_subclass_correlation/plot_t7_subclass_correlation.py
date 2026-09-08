#!/usr/bin/env python3
"""Per-CRE T7-count concordance between matched subclasses.

Using the 5/28/2025 single-cell dataset, we:
  1. Pseudobulk the per-cell T7 barcode counts (``obsm['T7CRE']``) to a
     subclass x CRE matrix of summed counts.
  2. Retain every subclass whose total T7 counts exceed ``--min-total-t7``
     (1,000 by default) and that has at least ``--min-cells`` cells (no cell
     filter by default).
  3. Calculate all pairwise Pearson correlations on log10(count+1) and
     Spearman correlations on raw counts, and draw both correlation matrices
     as heatmaps.

By default, the script loads the sole
``revision/Data/scdata_5_28_2025_*.h5ad`` file.

Example
-------
    python revision/t7_subclass_correlation/plot_t7_subclass_correlation.py
    python revision/t7_subclass_correlation/plot_t7_subclass_correlation.py --k 4 --min-cells 1000
"""

from __future__ import annotations

import argparse
import glob
import os
import textwrap
from dataclasses import dataclass
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REVISION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REVISION_DIR, "Data")
DEFAULT_DATA_GLOB = os.path.join(DATA_DIR, "scdata_5_28_2025_*.h5ad")
BLACKLIST_CSV = os.path.join(DATA_DIR, "cre_blacklist.csv")
_T7_KEYS = ("T7CRE", "T7")  # obsm key preference order


def load_blacklist(path: str) -> list[str]:
    """Return blacklisted cCRE ids (empty if the file is absent or path is '')."""
    if not path or not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if "cre" not in df.columns:
        raise KeyError(f"expected a 'cre' column in {path}; got {list(df.columns)}")
    return df["cre"].astype(str).tolist()


# --------------------------------------------------------------------------- #
# data loading / pseudobulk
# --------------------------------------------------------------------------- #
def default_h5ad() -> str:
    """Return the sole 5/28/2025 AnnData file in ``revision/Data``."""
    paths = sorted(path for path in glob.glob(DEFAULT_DATA_GLOB) if os.path.isfile(path))
    if not paths:
        raise FileNotFoundError(f"no AnnData matched {DEFAULT_DATA_GLOB}")
    if len(paths) > 1:
        raise RuntimeError(
            f"multiple AnnData files matched {DEFAULT_DATA_GLOB}; choose one with --data:\n"
            + "\n".join(paths)
        )
    return paths[0]


@dataclass(frozen=True)
class SectionPseudobulk:
    """Subclass-level T7 pseudobulk for one section."""

    section: str
    pb: pd.DataFrame        # subclass x CRE summed T7 counts
    n_cells: pd.Series      # cells per subclass (index = subclass)
    cls: pd.Series          # parent class label per subclass (index = subclass)

    @property
    def total_t7(self) -> pd.Series:
        return self.pb.sum(axis=1)

    @property
    def mean_t7_per_cell(self) -> pd.Series:
        """Total summed T7 counts divided by cells in the subclass."""
        return self.total_t7 / self.n_cells.reindex(self.pb.index)


def _resolve_t7_key(obsm_keys: Sequence[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in obsm_keys:
            raise KeyError(f"obsm key {requested!r} absent; have {list(obsm_keys)}")
        return requested
    for k in _T7_KEYS:
        if k in obsm_keys:
            return k
    raise KeyError(f"no T7 obsm key {_T7_KEYS} present; have {list(obsm_keys)}")


def load_pseudobulk(section: str, subclass_col: str, class_col: str,
                    obsm_key: str | None, path: str) -> SectionPseudobulk:
    """Load AnnData and return the subclass x CRE summed-T7 matrix.

    AnnData is opened in backed mode so only the T7 obsm and subclass/class
    labels are loaded, not the unrelated expression matrix.
    """
    import anndata as ad

    if not os.path.isfile(path):
        raise FileNotFoundError(f"AnnData file not found: {path}")
    A = ad.read_h5ad(path, backed="r")
    try:
        key = _resolve_t7_key(list(A.obsm.keys()), obsm_key)
        t7 = A.obsm[key]
        if not isinstance(t7, pd.DataFrame):
            array = np.asarray(t7)
            t7 = pd.DataFrame(
                array,
                index=A.obs_names,
                columns=[f"CRE{i + 1:03d}" for i in range(array.shape[1])],
            )
        if subclass_col not in A.obs:
            raise KeyError(f"obs[{subclass_col!r}] missing; have {list(A.obs.columns)}")
        if class_col not in A.obs:
            raise KeyError(f"obs[{class_col!r}] missing; have {list(A.obs.columns)}")
        sub = A.obs[subclass_col].astype(str).copy()
        cls_labels = A.obs[class_col].astype(str).copy()
    finally:
        A.file.close()

    pb = t7.groupby(sub.to_numpy(), sort=True).sum()
    pb.index.name = subclass_col
    n_cells = sub.value_counts().reindex(pb.index).astype(int)
    # subclass -> parent class (strict hierarchy; mode guards against stray labels)
    cls = (pd.Series(cls_labels.to_numpy(), index=sub.to_numpy())
           .groupby(level=0).agg(lambda s: s.mode().iat[0])
           .reindex(pb.index))
    return SectionPseudobulk(section=section, pb=pb, n_cells=n_cells, cls=cls)



# --------------------------------------------------------------------------- #
# cell-type selection
# --------------------------------------------------------------------------- #
def select_eligible(pbk: SectionPseudobulk, min_cells: int,
                    min_total_t7: int) -> list[str]:
    """Return eligible subclasses ranked by total T7 counts (desc).

    A subclass qualifies when it has at least ``min_cells`` cells *and* more
    than ``min_total_t7`` total T7 counts. Total counts reflect both the cell
    number and the per-cell infection rate.
    """
    metadata = pd.DataFrame({
        "subclass": pbk.n_cells.index,
        "class": pbk.cls.reindex(pbk.n_cells.index).to_numpy(),
        "n_cells": pbk.n_cells.to_numpy(),
        "total_t7": pbk.total_t7.reindex(pbk.n_cells.index).to_numpy(),
    })
    metadata = metadata.loc[
        (metadata["n_cells"] >= min_cells) & (metadata["total_t7"] > min_total_t7)
    ]
    if len(metadata) < 2:
        raise ValueError(
            f"[{pbk.section}] only {len(metadata)} subclasses pass "
            f"n_cells >= {min_cells} and total_t7 > {min_total_t7}; "
            "at least two are required."
        )
    return metadata.sort_values("total_t7", ascending=False)["subclass"].tolist()


# --------------------------------------------------------------------------- #
# correlation + plotting
# --------------------------------------------------------------------------- #
def pearson_matrix(
    pbk: SectionPseudobulk, subclasses: list[str], log_counts: bool
) -> pd.DataFrame:
    """Pearson correlation across subclasses on raw or log10(count+1) T7 counts."""
    mat = pbk.pb.reindex(subclasses).to_numpy().astype(float)  # k x n_cre
    if log_counts:
        mat = np.log10(mat + 1.0)
    r = np.corrcoef(mat)
    return pd.DataFrame(r, index=subclasses, columns=subclasses)


def correlation_matrices(
    pbk: SectionPseudobulk, subclasses: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Pearson-log-count and Spearman raw-count correlation matrices."""
    mat = pbk.pb.reindex(subclasses).to_numpy().astype(float)  # k x n_cre
    spearman = np.asarray(stats.spearmanr(mat, axis=1).statistic)
    return (
        pearson_matrix(pbk, subclasses, log_counts=True),
        pd.DataFrame(spearman, index=subclasses, columns=subclasses),
    )


def plot_correlation_heatmap(
    pbk: SectionPseudobulk,
    pearson: pd.DataFrame,
    filter_desc: str,
    outpath: str,
    count_desc: str = "log10(T7 count + 1)",
    fig_width: float | None = None,
    fig_height: float | None = None,
) -> None:
    """Plot the Pearson correlation matrix with a T7-depth margin on the edge.

    Rows/columns are ordered top→bottom by total T7 counts (highest first);
    total counts capture both cell number and infection rate. A class color
    strip and a log-scaled horizontal bar of total T7 counts are drawn on the
    left edge, sharing the heatmap row axis.
    """
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    k = len(pearson)
    order = list(pearson.index)
    disp = pearson.to_numpy()
    vmin, vmax = 0.0, 1.0
    cbar_label = "Correlation"
    panel_title = f"Pearson r on {count_desc}"

    classes = pbk.cls.reindex(order).astype(str).to_numpy()
    class_levels = sorted(pd.unique(classes))
    cmap_cls = plt.get_cmap("tab20", max(len(class_levels), 1))
    class_color = {c: cmap_cls(i) for i, c in enumerate(class_levels)}
    class_idx = np.array([class_levels.index(c) for c in classes]).reshape(-1, 1)
    total_t7 = pbk.total_t7.reindex(order).to_numpy()

    # cap dimensions so an unfiltered run (hundreds of subclasses) stays renderable;
    # width runs wider than height to offset the left bar/class panels so the
    # heatmap itself stays roughly square. explicit sizes override the auto rule.
    if fig_width is None:
        fig_width = min(max(16.0, 0.42 * k), 70.0)
    if fig_height is None:
        fig_height = min(max(10.0, 0.34 * k), 46.0)

    # font sizes track the canvas so a small figure is not swamped by k tick
    # labels sized for a 26-inch one; clipped at the historical values on top
    row_pts = 72.0 * fig_height / max(k, 1)
    tick_fs = float(np.clip(0.55 * row_pts, 2.0, 6.0))
    label_fs = float(np.clip(0.9 * fig_width, 4.5, 11.0))
    title_fs = float(np.clip(1.1 * fig_width, 5.5, 13.0))

    fig, ax_pear = plt.subplots(figsize=(fig_width, fig_height))

    # ---- main plot: Pearson heatmap with T7-depth + class side panels ---- #
    # aspect="auto" makes the image fill its axes box so the divider-appended
    # marginal axes (which share this box's y-extent) align row-for-row.
    im_p = ax_pear.imshow(disp, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                          aspect="auto")
    ax_pear.set_xticks(range(k), labels=order, rotation=90, fontsize=tick_fs)
    ax_pear.tick_params(axis="y", left=False, labelleft=False)
    ax_pear.set_title(panel_title, fontsize=label_fs)
    ax_pear.set_xlabel("Subclass (ranked by total T7 counts)", fontsize=label_fs)

    # divider stacks appended axes outward: the class strip lands flush to the
    # heatmap, the T7 bar just beyond it, so left→right is bar | class | heatmap.
    divider = make_axes_locatable(ax_pear)
    ax_cls = divider.append_axes("left", size="4%", pad=0.08, sharey=ax_pear)
    ax_bar = divider.append_axes("left", size="22%", pad=0.1, sharey=ax_pear)
    cax_p = divider.append_axes("right", size="3%", pad=0.12)

    # total-T7 marginal; row labels live on this leftmost axis
    ax_bar.barh(np.arange(k), total_t7, height=0.8, color="#4c6ef5",
                edgecolor="none")
    ax_bar.set_xscale("log")
    ax_bar.set_yticks(range(k), labels=order, fontsize=tick_fs)
    ax_bar.tick_params(axis="y", left=True, labelleft=True)
    ax_bar.tick_params(axis="x", labelsize=tick_fs)
    ax_bar.set_xlabel("total T7\ncounts", fontsize=0.7 * label_fs)
    ax_bar.grid(axis="x", color="0.85", linewidth=0.4)
    ax_bar.set_axisbelow(True)

    # thin class color strip flush against the heatmap (no labels of its own)
    ax_cls.imshow(class_idx, aspect="auto", cmap=cmap_cls, vmin=0,
                  vmax=max(len(class_levels) - 1, 1), extent=(0, 1, k - 0.5, -0.5))
    ax_cls.set_xticks([])
    ax_cls.tick_params(axis="y", left=False, labelleft=False)
    cbar = fig.colorbar(im_p, cax=cax_p, label=cbar_label)
    cbar.set_label(cbar_label, fontsize=0.8 * label_fs)
    cbar.ax.tick_params(labelsize=0.7 * label_fs)

    legend_handles = [Patch(facecolor=class_color[c], label=c) for c in class_levels]
    fig.legend(
        handles=legend_handles, title="Class", loc="center left",
        bbox_to_anchor=(1.0, 0.5), fontsize=tick_fs, title_fontsize=0.8 * label_fs, frameon=False,
    )
    # wrap so a narrow canvas is not widened by the title under a tight bbox
    fig.suptitle(
        "\n".join(textwrap.wrap(
            f"{pbk.section}: per-CRE T7 concordance for all {k} subclasses "
            f"with {filter_desc} (ranked by total T7 counts)",
            width=max(40, int(7.5 * fig_width)),
        )),
        fontsize=title_fs,
    )
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help=f"input AnnData file (default: sole match for {DEFAULT_DATA_GLOB})")
    ap.add_argument("--label", default="all",
                    help="dataset label used in plot titles and output filenames")
    ap.add_argument("--subclass-col", default="subclass_name")
    ap.add_argument("--class-col", default="class_name",
                    help="obs column holding the parent class")
    ap.add_argument("--obsm-key", default=None, help="T7 obsm key (default: auto T7CRE/T7)")
    ap.add_argument("--min-cells", type=int, default=0,
                    help="minimum cells required to include a subclass (0 = no cell filter)")
    ap.add_argument("--min-total-t7", type=int, default=1000,
                    help="include a subclass only if total T7 counts exceed this (0 = off)")
    ap.add_argument("--blacklist", default=BLACKLIST_CSV,
                    help="cCRE blacklist CSV to exclude (pass '' to keep all)")
    ap.add_argument("--fig-width", type=float, default=None,
                    help="figure width in inches (default: scales with subclass count)")
    ap.add_argument("--fig-height", type=float, default=None,
                    help="figure height in inches (default: scales with subclass count)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    return ap.parse_args()


def _emit(pbk: SectionPseudobulk, args: argparse.Namespace) -> None:
    """Analyze all eligible subclasses and write the plots and tables."""
    section = pbk.section
    selected = select_eligible(pbk, min_cells=args.min_cells,
                               min_total_t7=args.min_total_t7)
    filters = []
    if args.min_cells > 0:
        filters.append(f"≥ {args.min_cells:,} cells")
    if args.min_total_t7 > 0:
        filters.append(f"total T7 > {args.min_total_t7:,}")
    filter_desc = " and ".join(filters) if filters else "no filter"

    sel = pd.DataFrame({
        "subclass": selected,
        "class": pbk.cls.reindex(selected).to_numpy(),
        "n_cells": pbk.n_cells.reindex(selected).to_numpy(),
        "total_t7": pbk.total_t7.reindex(selected).to_numpy().astype(np.int64),
        "mean_t7_per_cell": pbk.mean_t7_per_cell.reindex(selected).to_numpy(),
    })
    n_cre = pbk.pb.shape[1]
    n_eligible_classes = sel["class"].nunique()
    print(f"\n[{section}] {pbk.pb.shape[0]} subclasses, {n_cre} CREs; "
          f"{len(sel)} included ({filter_desc}) "
          f"across {n_eligible_classes} classes")
    print(sel.to_string(index=False))

    fig_path = os.path.join(args.outdir, f"{section}_t7_subclass_correlation.png")
    logcounts_path = os.path.join(
        args.outdir, f"{section}_t7_subclass_correlation_logcounts.png")
    pearson_log, spearman = correlation_matrices(pbk, selected)
    pearson_raw = pearson_matrix(pbk, selected, log_counts=False)
    # original: Pearson on raw counts; companion: Pearson on log10(count+1)
    plot_correlation_heatmap(pbk, pearson_raw, filter_desc, fig_path,
                             count_desc="T7 counts",
                             fig_width=args.fig_width, fig_height=args.fig_height)
    plot_correlation_heatmap(pbk, pearson_log, filter_desc, logcounts_path,
                             count_desc="log10(T7 count + 1)",
                             fig_width=args.fig_width, fig_height=args.fig_height)
    sel.to_csv(os.path.join(args.outdir, f"{section}_cell_types.csv"), index=False)
    pearson_raw.to_csv(os.path.join(args.outdir, f"{section}_pearson_rawcount.csv"))
    pearson_log.to_csv(os.path.join(args.outdir, f"{section}_pearson_logcount.csv"))
    spearman.to_csv(os.path.join(args.outdir, f"{section}_spearman_count.csv"))

    tri_idx = np.triu_indices(len(selected), k=1)
    for name, mtx in (("raw count", pearson_raw), ("log count", pearson_log)):
        tri = mtx.to_numpy()[tri_idx]
        print(f"[{section}] pairwise Pearson r ({name}): "
              f"min {tri.min():.3f}, median {np.median(tri):.3f}, max {tri.max():.3f}")
    print(f"[{section}] wrote {fig_path} and {logcounts_path} (+.pdf), "
          "cell_types.csv, pearson_rawcount.csv, pearson_logcount.csv, spearman_count.csv")


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data_path = os.path.abspath(args.data) if args.data else default_h5ad()
    print(f"Loading {data_path}")
    pbk = load_pseudobulk(
        args.label,
        args.subclass_col,
        args.class_col,
        args.obsm_key,
        data_path,
    )

    # drop blacklisted cCREs *and* blank barcodes so correlation, ranking, and
    # the margin bar all use the same filtered per-cCRE set
    blacklist = [c for c in load_blacklist(args.blacklist) if c in pbk.pb.columns]
    blanks = [c for c in pbk.pb.columns if str(c).lower().startswith("blank")]
    drop = sorted(set(blacklist) | set(blanks))
    if drop:
        print(f"excluded {len(blacklist)} blacklisted cCREs and {len(blanks)} "
              f"blank barcodes ({len(drop)} columns)")
        pbk = SectionPseudobulk(
            section=pbk.section,
            pb=pbk.pb.drop(columns=drop),
            n_cells=pbk.n_cells,
            cls=pbk.cls,
        )
    _emit(pbk, args)


if __name__ == "__main__":
    main()
