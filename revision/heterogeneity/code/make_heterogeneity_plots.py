#!/usr/bin/env python3
"""Bayesian heterogeneity analysis: subgroup activity vs whole-cell-type activity.

For each of the ten top-T7 cell types, compare every fitted cCRE's activity
estimated from the intact cell type with the mean activity across either five
random subsets or the annotated supertypes recorded by the fit. Annotated
supertypes use a cell-count-weighted mean for intact-fit agreement and retain
an unweighted SD as a heterogeneity measure. A second annotated-supertype
analysis calculates Lin's CCC across cCREs for every within-parent pair.

Outputs (under ``results/``):
  * ``tables/bayesian_subset_vs_whole.csv`` - pair-level agreement data;
  * ``tables/bayesian_subset_vs_whole_summary.csv`` - per-cell-type and overall metrics;
  * ``figures/bayesian_subset_mean_vs_whole.{pdf,png}`` - faceted agreement plot;
  * ``raw/combined_activity_bayesian.csv`` and ``raw/split_activity_bayesian.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, shapiro, spearmanr, ttest_rel, wilcoxon

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
REVISION_DIR = ANALYSIS_DIR.parent
REF_RESULTS = REVISION_DIR / "bayesian_vs_fold_change" / "results"
sys.path.insert(0, str(REVISION_DIR / "bayesian_vs_fold_change" / "code"))
sys.path.insert(0, str(CODE_DIR))

from analysis_utils import (  # noqa: E402
    LIBSIZE_CSV,
    OLD_DATA_BOOTSTRAP,
    jsonable,
    log,
    write_json,
)

MODELS = ("bayesian", "bootstrap")
# Groupings that produce many unequal-size subgroups per parent cell type and
# therefore share the cell-count-weighted mean and the pairwise-CCC analysis.
# 'supertype_like_random' is the size-matched random null for 'supertype'.
SUPERTYPE_LIKE_GROUPINGS = ("supertype", "supertype_like_random")
GROUPINGS = ("random",) + SUPERTYPE_LIKE_GROUPINGS
SUBGROUP_NOUN = {
    "random": "random subset",
    "supertype": "annotated supertype",
    "supertype_like_random": "size-matched random subset",
}
BOOT_ACTIVITY_FILE = "log_activity_vs_negative_control.csv"
GROUP_RE = re.compile(r"^(?P<subclass>.+)_group_(?P<group>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-bayes-dir", type=Path, default=REF_RESULTS / "bayesian"
    )
    parser.add_argument(
        "--split-bayes-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "split" / "bayesian",
    )
    parser.add_argument(
        "--combined-bootstrap-dir", type=Path, default=OLD_DATA_BOOTSTRAP
    )
    parser.add_argument(
        "--split-bootstrap-dir",
        type=Path,
        default=ANALYSIS_DIR / "results" / "split" / "bootstrap",
    )
    parser.add_argument(
        "--model", choices=["bayesian", "bootstrap"], default="bayesian"
    )
    parser.add_argument(
        "--null-split-bayes-dir",
        type=Path,
        default=None,
        help=(
            "Bayesian fit directory of the size-matched random null "
            "(--grouping supertype_like_random). When given, the pairwise-CCC "
            "figure gains a second box per cell type and the annotated pairs "
            "are tested against their size-matched counterparts with a paired "
            "t-test."
        ),
    )
    parser.add_argument("--outdir", type=Path, default=ANALYSIS_DIR / "results")
    parser.add_argument(
        "--calibration",
        choices=["self_cre_negctrl", "negctrl_only"],
        default="negctrl_only",
        help=(
            "Bayesian activity scale. 'negctrl_only' (default) preserves a common "
            "per-cCRE scale between the independently fitted intact and split data; "
            "'self_cre_negctrl' additionally centers each fit per cCRE."
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=5,
        help="Number of cell-type panel columns (default: 5).",
    )
    return parser.parse_args()


def subgroup_noun(grouping: str, plural: bool = False) -> str:
    """Human-readable name of one subgroup under ``grouping``."""
    if grouping not in SUBGROUP_NOUN:
        raise ValueError(f"unsupported grouping={grouping!r}")
    noun = SUBGROUP_NOUN[grouping]
    return f"{noun}s" if plural else noun


def discover_tag(bayes_dir: Path) -> str:
    manifest = json.loads((bayes_dir / "run_manifest.json").read_text())
    return str(manifest["tag"])


def bayesian_effect_matrix(
    posterior_path: Path, negative_controls: set[str], self_cre: bool = True
) -> pd.DataFrame:
    """Negative-control-centred log effect per group x cCRE.

    With ``self_cre=True`` (default) the posterior ``log_gamma`` is first
    self-cCRE calibrated (per-cCRE mean over all draws and groups subtracted),
    mirroring ``plot_results.bayesian_significance``. With ``self_cre=False``
    only the per-group negative-control mean is subtracted.
    """
    with np.load(posterior_path, allow_pickle=True) as posterior:
        log_gamma = posterior["log_gamma"].astype(np.float64)
        groups = posterior["group_names"].astype(str)
        cres = posterior["cre_names"].astype(str)
    if self_cre:
        log_gamma = log_gamma - log_gamma.mean(axis=(0, 1))[None, None, :]
    negative_mask = np.isin(cres, list(negative_controls))
    if not negative_mask.any():
        raise ValueError(f"{posterior_path} has no negative-control cCREs")
    negative_threshold = log_gamma[:, :, negative_mask].mean(axis=(0, 2))
    effect = log_gamma.mean(axis=0) - negative_threshold[:, None]
    return pd.DataFrame(effect, index=groups, columns=cres)


def bootstrap_effect_matrix(dir_: Path, self_cre: bool = True) -> pd.DataFrame:
    """Negative-control-centred bootstrap log activity per subclass x cCRE.

    ``self_cre=True`` reuses the run's saved ``log_activity_vs_negative_control``
    (self-cCRE calibrated + negative-control centred). ``self_cre=False``
    recomputes from the raw per-bootstrap activity array applying only
    negative-control centering, matching ``average_bootstrap_test_q`` with
    ``calibrate=None, threshold='neg_control_mean'`` (filter mask applied).
    """
    if self_cre:
        frame = pd.read_csv(dir_ / BOOT_ACTIVITY_FILE, index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        return frame

    axes = json.loads((dir_ / "bootstrap_axes.json").read_text())
    subclasses = [str(s) for s in axes["subclasses"]]
    cres = [str(c) for c in axes["cres"]]
    negatives = set(pd.read_csv(dir_ / "negative_controls.csv")["cre"].astype(str))
    neg_idx = [i for i, c in enumerate(cres) if c in negatives]
    if not neg_idx:
        raise ValueError(f"{dir_} bootstrap axes contain no negative controls")
    fmask = pd.read_csv(dir_ / "qvalue_filter_mask.csv", index_col=0)
    fmask.index = fmask.index.astype(str)
    fmask.columns = fmask.columns.astype(str)
    fmask = fmask.reindex(index=subclasses, columns=cres).fillna(True).astype(bool)

    arr = np.load(dir_ / "celltype_activity_array.npy", mmap_mode="r")
    n_boot, n_groups, n_cres = arr.shape
    if (n_groups, n_cres) != (len(subclasses), len(cres)):
        raise ValueError("bootstrap array axes do not match bootstrap_axes.json")
    mask = fmask.to_numpy()  # (G, C) bool, True => filtered
    neg_idx = np.asarray(neg_idx)

    # Stream contiguous bootstrap-chunks (arr is C-contiguous over (boot, g, cre),
    # so arr[b0:b1] is a sequential read) accumulating nan-aware sums/counts, so
    # the 11 GB array is read once sequentially rather than strided per group.
    res_sum = np.zeros((n_groups, n_cres))
    res_cnt = np.zeros((n_groups, n_cres), dtype=np.int64)
    neg_sum = np.zeros(n_groups)
    neg_cnt = np.zeros(n_groups, dtype=np.int64)
    chunk = 1000
    with np.errstate(invalid="ignore", divide="ignore"):
        for b0 in range(0, n_boot, chunk):
            block = np.log(np.asarray(arr[b0 : b0 + chunk], dtype=np.float64))
            block[~np.isfinite(block)] = np.nan
            block[:, mask] = np.nan  # broadcast filter over the chunk's bootstraps
            valid = ~np.isnan(block)
            res_sum += np.nansum(block, axis=0)
            res_cnt += valid.sum(axis=0)
            neg_per_boot = np.nanmean(block[:, :, neg_idx], axis=2)  # (nb, G)
            neg_valid = ~np.isnan(neg_per_boot)
            neg_sum += np.nansum(neg_per_boot, axis=0)
            neg_cnt += neg_valid.sum(axis=0)
    res_df = np.where(res_cnt > 0, res_sum / np.maximum(res_cnt, 1), np.nan)
    fdc = np.where(neg_cnt > 0, neg_sum / np.maximum(neg_cnt, 1), np.nan)
    effect = res_df - fdc[:, None]
    return pd.DataFrame(effect, index=subclasses, columns=cres)


def load_bayesian(dir_: Path, self_cre: bool = True) -> pd.DataFrame:
    tag = discover_tag(dir_)
    negatives = set(
        pd.read_csv(dir_ / "negative_controls.csv")["cre"].astype(str)
    )
    return bayesian_effect_matrix(
        dir_ / f"{tag}_posterior_samples.npz", negatives, self_cre=self_cre
    )


def load_bootstrap(dir_: Path, self_cre: bool = True) -> pd.DataFrame:
    return bootstrap_effect_matrix(dir_, self_cre=self_cre)


def negative_controls(dir_: Path) -> set[str]:
    path = dir_ / "negative_controls.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["cre"].astype(str))


def blacklist(dir_: Path) -> set[str]:
    path = dir_ / "cre_blacklist.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def split_targets(
    split_bayes_dir: Path,
) -> tuple[list[str], dict[str, list[str]], str]:
    """Return target order, subgroup membership, and grouping strategy.

    Existing random-split runs predate the explicit membership map, so retain
    a fallback that reconstructs their ``<subclass>_group_i`` labels.
    """
    manifest = json.loads((split_bayes_dir / "run_manifest.json").read_text())
    targets = [str(target) for target in manifest["split_subclasses"]]
    grouping = str(manifest.get("grouping", "random"))
    configured = manifest.get("subgroups_by_subclass")
    if configured is None:
        n_groups = int(manifest["n_groups"])
        members = {
            target: [
                f"{target}_group_{group}" for group in range(1, n_groups + 1)
            ]
            for target in targets
        }
    else:
        members = {
            target: [str(member) for member in configured[target]]
            for target in targets
        }
    empty = [target for target, labels in members.items() if not labels]
    if empty:
        raise ValueError(f"split manifest has targets without subgroups: {empty}")
    if grouping not in GROUPINGS:
        raise ValueError(f"unsupported grouping={grouping!r}")
    return targets, members, grouping


def artifact_names(grouping: str, model: str = "bayesian") -> dict[str, str]:
    """Stable filenames for the legacy random and new supertype analyses."""
    if grouping == "random" and model == "bayesian":
        return {
            "table": "bayesian_subset_vs_whole.csv",
            "summary": "bayesian_subset_vs_whole_summary.csv",
            "split_raw": "split_activity_bayesian.csv",
            "figure": "bayesian_subset_mean_vs_whole",
            "manifest": "heterogeneity_manifest.json",
        }
    if grouping in SUPERTYPE_LIKE_GROUPINGS and model == "bayesian":
        return {
            "table": "bayesian_supertype_vs_whole.csv",
            "summary": "bayesian_supertype_vs_whole_summary.csv",
            "split_raw": "supertype_activity_bayesian.csv",
            "figure": "bayesian_supertype_mean_vs_whole",
            "pairwise_table": "bayesian_supertype_pairwise_ccc.csv",
            "pairwise_summary": "bayesian_supertype_pairwise_ccc_summary.csv",
            "pairwise_figure": "bayesian_supertype_pairwise_ccc",
            "pairwise_support_figure": (
                "bayesian_supertype_pairwise_ccc_vs_min_cells"
            ),
            "manifest": "heterogeneity_manifest.json",
        }
    if grouping == "random" and model == "bootstrap":
        return {
            "table": "bootstrap_subset_vs_whole.csv",
            "summary": "bootstrap_subset_vs_whole_summary.csv",
            "split_raw": "split_activity_bootstrap.csv",
            "figure": "bootstrap_subset_mean_vs_whole",
            "manifest": "bootstrap_heterogeneity_manifest.json",
        }
    if grouping in SUPERTYPE_LIKE_GROUPINGS and model == "bootstrap":
        return {
            "table": "bootstrap_supertype_vs_whole.csv",
            "summary": "bootstrap_supertype_vs_whole_summary.csv",
            "split_raw": "supertype_activity_bootstrap.csv",
            "figure": "bootstrap_supertype_mean_vs_whole",
            "pairwise_table": "bootstrap_supertype_pairwise_ccc.csv",
            "pairwise_summary": "bootstrap_supertype_pairwise_ccc_summary.csv",
            "pairwise_figure": "bootstrap_supertype_pairwise_ccc",
            "pairwise_support_figure": (
                "bootstrap_supertype_pairwise_ccc_vs_min_cells"
            ),
            "manifest": "bootstrap_heterogeneity_manifest.json",
        }
    raise ValueError(f"unsupported grouping={grouping!r}, model={model!r}")


def assemble(
    combined: dict[str, pd.DataFrame],
    split: dict[str, pd.DataFrame],
    targets: list[str],
    n_groups: int,
    drop_negatives: set[str],
) -> tuple[pd.DataFrame, dict, dict]:
    """Return long activity table plus per-model variance/diff matrices."""
    long_rows = []
    variance = {model: {} for model in MODELS}
    signed_diff = {model: {} for model in MODELS}

    for model in MODELS:
        comb = combined[model]
        spl = split[model]
        # Common non-blacklist / non-negative-control cCREs across both fits.
        cres = comb.columns.intersection(spl.columns)
        cres = [c for c in cres if c not in drop_negatives]
        for subclass in targets:
            if subclass not in comb.index:
                raise KeyError(f"{subclass!r} missing from combined {model} activity")
            member_labels = [f"{subclass}_group_{g}" for g in range(1, n_groups + 1)]
            missing = [m for m in member_labels if m not in spl.index]
            if missing:
                raise KeyError(f"missing split groups for {subclass}: {missing}")

            comb_vec = comb.loc[subclass, cres].astype(float)
            group_mat = spl.loc[member_labels, cres].astype(float)

            variance[model][subclass] = group_mat.var(axis=0, ddof=1)
            signed_diff[model][subclass] = (group_mat - comb_vec).mean(axis=0)

            for cre in cres:
                long_rows.append(
                    (model, subclass, "combined", cre, float(comb_vec[cre]))
                )
            for g, label in enumerate(member_labels, start=1):
                gvec = group_mat.loc[label]
                for cre in cres:
                    long_rows.append(
                        (model, subclass, f"group_{g}", cre, float(gvec[cre]))
                    )

    long = pd.DataFrame(
        long_rows, columns=["model", "subclass", "member", "cre", "activity"]
    )
    var_mats = {
        model: pd.DataFrame(variance[model]).T.reindex(targets) for model in MODELS
    }
    diff_mats = {
        model: pd.DataFrame(signed_diff[model]).T.reindex(targets) for model in MODELS
    }
    return long, var_mats, diff_mats


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def order_cres(
    activity_row: pd.Series, cres: list[str], neg_set: set[str], mode: str
) -> tuple[list[str], int]:
    """Real cCREs first, negative controls appended at the end.

    Returns the ordered cCRE list and the number of real (non-control) cCREs,
    i.e. the index where the negative-control block starts.
    """
    real = [c for c in cres if c not in neg_set]
    negs = [c for c in cres if c in neg_set]
    if mode == "activity":
        key = lambda c: activity_row.get(c, -np.inf)
        real.sort(key=key, reverse=True)
        negs.sort(key=key, reverse=True)
    else:
        real.sort()
        negs.sort()
    return real + negs, len(real)


def plot_subclass(
    subclass: str,
    combined: dict[str, pd.DataFrame],
    split: dict[str, pd.DataFrame],
    n_groups: int,
    cre_order: list[str],
    n_real: int,
    figures_dir: Path,
) -> None:
    """Two stacked panels (Bayesian, bootstrap). Per cCRE: a box over the five
    subgroups and a point for the intact-subclass estimate beside it. Negative
    controls sit at the right end, shaded and labelled."""
    members = [f"{subclass}_group_{g}" for g in range(1, n_groups + 1)]
    n = len(cre_order)
    base = np.arange(n)
    width = max(24.0, n * 0.16)
    fig, axes = plt.subplots(
        len(MODELS), 1, figsize=(width, 6 * len(MODELS) + 1), sharex=True
    )
    box_offset, pt_offset = -0.12, 0.26
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        comb = combined[model]
        spl = split[model].reindex(index=members, columns=cre_order)
        comb_row = comb.reindex(index=[subclass], columns=cre_order).iloc[0]

        box_data, positions = [], []
        for i, cre in enumerate(cre_order):
            vals = spl[cre].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                box_data.append(vals)
                positions.append(i + box_offset)
        if box_data:
            bp = ax.boxplot(
                box_data, positions=positions, widths=0.4, showfliers=False,
                patch_artist=True, manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#f58518")
                patch.set_alpha(0.6)
            for whisk in bp["whiskers"] + bp["caps"]:
                whisk.set_linewidth(0.6)
            for med in bp["medians"]:
                med.set_color("#7a3d00")
                med.set_linewidth(0.8)

        cy = comb_row.to_numpy(dtype=float)
        finite = np.isfinite(cy)
        ax.scatter(
            base[finite] + pt_offset, cy[finite], marker="D", s=9,
            color="#4c78a8", zorder=5, linewidths=0,
        )

        if n_real < n:  # shade + separate the negative-control block
            ax.axvspan(n_real - 0.5, n - 0.5, color="#d62728", alpha=0.06, zorder=0)
            ax.axvline(n_real - 0.5, color="#d62728", ls="--", lw=0.8, zorder=1)
            ax.text(
                (n_real + n) / 2 - 0.5, 0.98, "negative\ncontrols",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#d62728",
            )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(model.capitalize(), loc="left")
        ax.set_ylabel("log activity above\nnegative-control mean")
        ax.set_xlim(-1, n)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#f58518", alpha=0.6),
        plt.Line2D([0], [0], marker="D", color="#4c78a8", linestyle="none", markersize=6),
    ]
    axes[0].legend(
        handles, [f"{n_groups} subgroups", "combined subclass"],
        frameon=False, loc="upper right", ncol=2,
    )
    ax_bottom = np.atleast_1d(axes)[-1]
    ax_bottom.set_xticks(base)
    ax_bottom.set_xticklabels(cre_order, rotation=90, fontsize=max(2.0, min(6.0, 900 / n)))
    for i, lbl in enumerate(ax_bottom.get_xticklabels()):
        if i >= n_real:
            lbl.set_color("#d62728")
    ax_bottom.set_xlabel("cCRE")
    fig.suptitle(f"{subclass}: intact subclass vs {n_groups} random subgroups", y=0.995)
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{subclass}_heterogeneity")


def plot_overview(
    var_mats: dict, diff_mats: dict, targets: list[str], figures_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(targets) * 1.1), 10))
    metrics = [("Per-cCRE subgroup variance", var_mats, False),
               ("Per-cCRE signed diff (subgroup - combined)", diff_mats, True)]
    offsets = {"bayesian": -0.18, "bootstrap": 0.18}
    colors = {"bayesian": "#4c78a8", "bootstrap": "#f58518"}
    x = np.arange(len(targets))
    for ax, (title, mats, center_line) in zip(axes, metrics):
        for model in MODELS:
            mat = mats[model].reindex(targets)
            data = [mat.loc[s].dropna().to_numpy() for s in targets]
            positions = x + offsets[model]
            box = ax.boxplot(
                data, positions=positions, widths=0.32, showfliers=False,
                patch_artist=True,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[model])
                patch.set_alpha(0.7)
        if center_line:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha="right")
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.7) for m in MODELS]
    axes[0].legend(handles, [m.capitalize() for m in MODELS], frameon=False)
    fig.tight_layout()
    save_figure(fig, figures_dir / "heterogeneity_overview")


def within_vs_across(
    combined: dict[str, pd.DataFrame],
    var_mats: dict[str, pd.DataFrame],
    targets: list[str],
) -> pd.DataFrame:
    """Per-cCRE within- vs across-cell-type variance for each model.

    within  = mean over the 10 subclasses of the 5-subgroup variance (x-axis).
    across  = variance of the combined estimate across the same 10 subclasses.
    """
    rows = []
    for model in MODELS:
        var_mat = var_mats[model]
        cres = list(var_mat.columns)
        within = var_mat.mean(axis=0, skipna=True)
        n_within = var_mat.notna().sum(axis=0)
        comb = combined[model].reindex(index=targets, columns=cres)
        across = comb.var(axis=0, ddof=1, skipna=True)
        n_across = comb.notna().sum(axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "cre": cres,
                    "within_ct_variance": within.to_numpy(),
                    "across_ct_variance": across.to_numpy(),
                    "n_subclasses_within": n_within.to_numpy(),
                    "n_celltypes_across": n_across.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_within_vs_across(
    table: pd.DataFrame, counts: pd.Series, figures_dir: Path
) -> None:
    log_counts = np.log10(counts.astype(float))
    vmin, vmax = float(log_counts.min()), float(log_counts.max())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS) + 1, 6.5))
    scatter = None
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = table[table["model"] == model]
        x = sub["within_ct_variance"].to_numpy(dtype=float)
        y = sub["across_ct_variance"].to_numpy(dtype=float)
        c = log_counts.reindex(sub["cre"]).to_numpy(dtype=float)
        # require >=2 cell types for a defined across-variance, a positive pair,
        # and a known nanopore count for the colour.
        ok = (
            np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & np.isfinite(c)
            & (sub["n_celltypes_across"].to_numpy() >= 2)
        )
        x, y, c = x[ok], y[ok], c[ok]
        scatter = ax.scatter(
            x, y, c=c, s=16, alpha=0.75, cmap="viridis", vmin=vmin, vmax=vmax,
            linewidths=0,
        )
        if x.size:
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="y = x")
            n_above = int((y > x).sum())
            rho = spearman(x, y)
            ax.set_title(
                f"{model.capitalize()}  (n={x.size}, ρ={rho:.2f}, "
                f"{n_above}/{x.size} above y=x)"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Within-cell-type variance\n(mean 5-subgroup variance over 10 subclasses)")
        ax.set_ylabel("Across-cell-type variance\n(combined estimate over 10 subclasses)")
        ax.legend(frameon=False, loc="upper left")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("Nanopore sequencing count (log$_{10}$)")
    fig.suptitle("Per-cCRE within- vs across-cell-type activity variance")
    save_figure(fig, figures_dir / "within_vs_across_variance")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return np.nan
    return float(spearmanr(x, y).statistic)


def anova_variance_components(
    split: dict[str, pd.DataFrame], targets: list[str], n_groups: int, cres: list[str]
) -> pd.DataFrame:
    """Per-cCRE one-way random-effects decomposition over the 50 subgroup
    estimates (10 cell types x n_groups subgroups), cell type as random factor.

    Returns, per (model, cCRE): pooled within-cell-type residual variance
    (sigma2_within = MS_within), the bias-corrected between-cell-type variance
    (sigma2_between = max(0, (MS_between - MS_within) / n0)), the intraclass
    correlation ICC, and the F statistic. Handles the unbalanced case (NaN
    subgroups) via the general moment estimator for n0.
    """
    members = [f"{t}_group_{g}" for t in targets for g in range(1, n_groups + 1)]
    g, k = len(targets), n_groups
    rows = []
    for model in MODELS:
        mat = split[model].reindex(index=members, columns=cres)
        y = mat.to_numpy(dtype=float).reshape(g, k, len(cres))  # (celltype, subgroup, cCRE)
        finite = ~np.isnan(y)
        n_t = finite.sum(axis=1)  # (g, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            sum_t = np.nansum(y, axis=1)  # (g, C)
            mean_t = sum_t / np.where(n_t > 0, n_t, np.nan)
            N = n_t.sum(axis=0)  # (C,)
            grand = np.nansum(sum_t, axis=0) / np.where(N > 0, N, np.nan)
            ss_within = np.nansum((y - mean_t[:, None, :]) ** 2, axis=(0, 1))
            ss_between = np.nansum(n_t * (mean_t - grand[None, :]) ** 2, axis=0)
            g_eff = (n_t > 0).sum(axis=0)  # (C,)
            df_within = np.where(n_t > 0, n_t - 1, 0).sum(axis=0)
            df_between = g_eff - 1
            ms_within = ss_within / np.where(df_within > 0, df_within, np.nan)
            ms_between = ss_between / np.where(df_between > 0, df_between, np.nan)
            sum_nt2 = (n_t ** 2).sum(axis=0)
            n0 = (N - sum_nt2 / np.where(N > 0, N, np.nan)) / np.where(
                g_eff > 1, g_eff - 1, np.nan
            )
            sigma2_within = ms_within
            sigma2_between = np.maximum(0.0, (ms_between - ms_within) / n0)
            icc = sigma2_between / (sigma2_between + sigma2_within)
            f_stat = ms_between / ms_within
        valid = (g_eff >= 2) & (df_within >= 1)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "cre": cres,
                    "sigma2_within": np.where(valid, sigma2_within, np.nan),
                    "sigma2_between": np.where(valid, sigma2_between, np.nan),
                    "icc": np.where(valid, icc, np.nan),
                    "f_stat": np.where(valid, f_stat, np.nan),
                    "n_celltypes": g_eff,
                    "n_obs": N,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_variance_components(
    table: pd.DataFrame, counts: pd.Series, figures_dir: Path
) -> None:
    log_counts = np.log10(counts.astype(float))
    vmin, vmax = float(log_counts.min()), float(log_counts.max())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS) + 1, 6.5))
    scatter = None
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = table[table["model"] == model]
        x = sub["sigma2_within"].to_numpy(dtype=float)
        y = sub["sigma2_between"].to_numpy(dtype=float)
        c = log_counts.reindex(sub["cre"]).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & np.isfinite(c)
        xf, yf, cf = x[ok], y[ok], c[ok]
        scatter = ax.scatter(
            xf, yf, c=cf, s=16, alpha=0.75, cmap="viridis", vmin=vmin, vmax=vmax,
            linewidths=0,
        )
        if xf.size:
            lo = float(min(xf.min(), yf.min()))
            hi = float(max(xf.max(), yf.max()))
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="y = x")
            med_icc = float(np.nanmedian(sub["icc"]))
            ax.set_title(
                f"{model.capitalize()}  (n={xf.size}, median ICC={med_icc:.2f}, "
                f"{int((yf > xf).sum())}/{xf.size} above y=x)"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Within-cell-type variance  σ²_within\n(pooled residual MS)")
        ax.set_ylabel("Between-cell-type variance  σ²_between\n(bias-corrected)")
        ax.legend(frameon=False, loc="upper left")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=np.atleast_1d(axes).tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("Nanopore sequencing count (log$_{10}$)")
    fig.suptitle("Per-cCRE random-effects variance decomposition (cell type as factor)")
    save_figure(fig, figures_dir / "variance_components_anova")


def within_sd_vs_nn_distance(
    combined: dict[str, pd.DataFrame],
    var_mats: dict[str, pd.DataFrame],
    targets: list[str],
    scope: str,
) -> pd.DataFrame:
    """Per (cCRE, cell type): within-cell-type SD vs distance to the nearest
    other cell type's combined estimate for that cCRE.

    within_sd  = sqrt of the 5-subgroup variance (log-activity scale).
    nn_distance = min over other cell types of |combined[c, t] - combined[c, t']|.
    scope='all' compares against every cell type in the combined run;
    'selected' compares only against the other 9 split subclasses.
    """
    rows = []
    for model in MODELS:
        cres = list(var_mats[model].columns)
        comb = combined[model]
        comb.columns = comb.columns.astype(str)
        pool = comb if scope == "all" else comb.reindex(index=targets)
        sd = np.sqrt(var_mats[model])
        for t in targets:
            if t not in comb.index:
                continue
            self_vals = comb.reindex(index=[t], columns=cres).iloc[0]
            others = pool.drop(index=t, errors="ignore").reindex(columns=cres)
            nn = (others - self_vals).abs().min(axis=0, skipna=True)
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "celltype": t,
                        "cre": cres,
                        "within_sd": sd.loc[t, cres].to_numpy(dtype=float),
                        "nn_distance": nn.reindex(cres).to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def plot_within_sd_vs_nn(table: pd.DataFrame, targets: list[str], figures_dir: Path) -> None:
    palette = {t: plt.cm.tab10(i % 10) for i, t in enumerate(targets)}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.5 * len(MODELS), 6.8))
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = table[table["model"] == model]
        x = sub["within_sd"].to_numpy(dtype=float)
        y = sub["nn_distance"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        colors = sub["celltype"].map(palette).to_numpy()
        ax.scatter(x[ok], y[ok], s=10, alpha=0.5, c=list(colors[ok]), linewidths=0)
        xf, yf = x[ok], y[ok]
        if xf.size:
            lo = float(min(xf.min(), yf.min()))
            hi = float(max(xf.max(), yf.max()))
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
            frac = float((yf > xf).mean())
            ax.set_title(
                f"{model.capitalize()}  (n={xf.size}, "
                f"{frac:.0%} with NN distance > within SD)"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Within-cell-type SD  (√ 5-subgroup variance)")
        ax.set_ylabel("Distance to nearest other cell type\n(|combined activity difference|)")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color=palette[t], markersize=6)
        for t in targets
    ]
    fig.legend(
        handles, targets, frameon=False, loc="center left",
        bbox_to_anchor=(1.0, 0.5), fontsize=8, title="cell type",
    )
    fig.suptitle("Within-cell-type SD vs nearest-neighbour cell-type distance")
    save_figure(fig, figures_dir / "within_sd_vs_nn_distance")


def bayesian_subset_agreement(
    combined: pd.DataFrame,
    split: pd.DataFrame,
    targets: list[str],
    subgroups_by_subclass: dict[str, list[str]],
    excluded_cres: set[str],
    subgroup_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Return intact activity and subgroup summaries.

    When ``subgroup_weights`` is supplied, ``mean_subgroup_activity`` is the
    cell-count-weighted mean used for agreement with the intact fit.  The
    unweighted mean and sample SD are retained because they describe a typical
    subgroup and the between-subgroup heterogeneity, respectively.
    """
    cres = [
        cre
        for cre in combined.columns.intersection(split.columns).astype(str)
        if cre not in excluded_cres
    ]
    rows = []
    for cell_type in targets:
        if cell_type not in combined.index:
            raise KeyError(f"{cell_type!r} missing from intact Bayesian activity")
        members = subgroups_by_subclass[cell_type]
        missing = [member for member in members if member not in split.index]
        if missing:
            raise KeyError(f"missing split groups for {cell_type}: {missing}")
        subset = split.reindex(index=members, columns=cres).astype(float)
        whole = combined.reindex(index=[cell_type], columns=cres).iloc[0].astype(float)
        unweighted_mean = subset.mean(axis=0).to_numpy(dtype=float)
        if subgroup_weights is None:
            weighted_mean = unweighted_mean.copy()
        else:
            if cell_type not in subgroup_weights:
                raise KeyError(f"missing subgroup weights for {cell_type!r}")
            missing_weights = [
                member
                for member in members
                if member not in subgroup_weights[cell_type]
            ]
            if missing_weights:
                raise KeyError(
                    f"missing subgroup weights for {cell_type}: {missing_weights}"
                )
            weights = np.asarray(
                [subgroup_weights[cell_type][member] for member in members],
                dtype=float,
            )
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError(
                    f"subgroup weights for {cell_type} must be finite and positive"
                )
            values = subset.to_numpy(dtype=float)
            finite = np.isfinite(values)
            denominators = (finite * weights[:, np.newaxis]).sum(axis=0)
            numerators = np.where(finite, values, 0.0) * weights[:, np.newaxis]
            weighted_mean = np.divide(
                numerators.sum(axis=0),
                denominators,
                out=np.full(values.shape[1], np.nan, dtype=float),
                where=denominators > 0,
            )
        rows.append(
            pd.DataFrame(
                {
                    "cell_type": cell_type,
                    "cre": cres,
                    "whole_activity": whole.to_numpy(dtype=float),
                    "mean_subgroup_activity": weighted_mean,
                    "unweighted_mean_subgroup_activity": unweighted_mean,
                    "subgroup_sd": subset.std(axis=0, ddof=1).to_numpy(dtype=float),
                    "n_subgroups": subset.notna().sum(axis=0).to_numpy(dtype=int),
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output["difference"] = (
        output["mean_subgroup_activity"] - output["whole_activity"]
    )
    output["absolute_difference"] = output["difference"].abs()
    output["unweighted_difference"] = (
        output["unweighted_mean_subgroup_activity"] - output["whole_activity"]
    )
    output["unweighted_absolute_difference"] = (
        output["unweighted_difference"].abs()
    )
    return output


def vector_agreement_metrics(
    x: np.ndarray, y: np.ndarray
) -> dict[str, float | int]:
    """Agreement metrics for two vectors, including Lin's CCC."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return {
            "n": 0,
            "pearson_r": np.nan,
            "concordance_correlation": np.nan,
            "mean_error": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }
    delta = y - x
    x_mean, y_mean = float(x.mean()), float(y.mean())
    x_var, y_var = float(x.var()), float(y.var())
    covariance = float(np.mean((x - x_mean) * (y - y_mean)))
    denominator = x_var + y_var + (x_mean - y_mean) ** 2
    concordance = 2 * covariance / denominator if denominator > 0 else np.nan
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if x.size > 1 and x_var > 0 and y_var > 0
        else np.nan
    )
    return {
        "n": int(x.size),
        "pearson_r": pearson,
        "concordance_correlation": float(concordance),
        "mean_error": float(delta.mean()),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def agreement_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Metrics of agreement with y=x, including Lin's concordance correlation."""
    x = frame["whole_activity"].to_numpy(dtype=float)
    metrics = vector_agreement_metrics(
        x, frame["mean_subgroup_activity"].to_numpy(dtype=float)
    )
    finite_x = x[np.isfinite(x)]
    metrics["whole_activity_range"] = (
        float(finite_x.max() - finite_x.min()) if finite_x.size else np.nan
    )
    return metrics


def pairwise_supertype_agreement(
    split: pd.DataFrame,
    targets: list[str],
    subgroups_by_subclass: dict[str, list[str]],
    subgroup_weights: dict[str, dict[str, float]],
    excluded_cres: set[str],
) -> pd.DataFrame:
    """Compute agreement across cCREs for every within-parent supertype pair."""
    cres = [cre for cre in split.columns.astype(str) if cre not in excluded_cres]
    rows: list[dict[str, float | int | str]] = []
    for cell_type in targets:
        members = subgroups_by_subclass[cell_type]
        missing = [member for member in members if member not in split.index]
        if missing:
            raise KeyError(f"missing split groups for {cell_type}: {missing}")
        missing_weights = [
            member
            for member in members
            if member not in subgroup_weights.get(cell_type, {})
        ]
        if missing_weights:
            raise KeyError(
                f"missing subgroup weights for {cell_type}: {missing_weights}"
            )
        for first_index, first in enumerate(members):
            first_values = split.reindex(index=[first], columns=cres).iloc[0]
            for second in members[first_index + 1 :]:
                second_values = split.reindex(index=[second], columns=cres).iloc[0]
                metrics = vector_agreement_metrics(
                    first_values.to_numpy(dtype=float),
                    second_values.to_numpy(dtype=float),
                )
                first_cells = int(subgroup_weights[cell_type][first])
                second_cells = int(subgroup_weights[cell_type][second])
                rows.append(
                    {
                        "cell_type": cell_type,
                        "supertype_1": first,
                        "supertype_2": second,
                        "n_cells_1": first_cells,
                        "n_cells_2": second_cells,
                        "minimum_pair_cells": min(first_cells, second_cells),
                        "geometric_mean_pair_cells": float(
                            np.sqrt(first_cells * second_cells)
                        ),
                        "n_cres": metrics["n"],
                        "pearson_r": metrics["pearson_r"],
                        "concordance_correlation": metrics[
                            "concordance_correlation"
                        ],
                        "mean_difference": metrics["mean_error"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                    }
                )
    columns = [
        "cell_type",
        "supertype_1",
        "supertype_2",
        "n_cells_1",
        "n_cells_2",
        "minimum_pair_cells",
        "geometric_mean_pair_cells",
        "n_cres",
        "pearson_r",
        "concordance_correlation",
        "mean_difference",
        "mae",
        "rmse",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_pairwise_supertype_agreement(
    pairwise: pd.DataFrame,
    targets: list[str],
    subgroups_by_subclass: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarize the pairwise-CCC distribution separately for each parent."""
    rows = []
    for cell_type in targets:
        frame = pairwise[pairwise["cell_type"] == cell_type]
        ccc = frame["concordance_correlation"].dropna()
        pearson = frame["pearson_r"].dropna()
        mae = frame["mae"].dropna()
        n_supertypes = len(subgroups_by_subclass[cell_type])
        expected_pairs = n_supertypes * (n_supertypes - 1) // 2
        if len(frame) != expected_pairs:
            raise ValueError(
                f"{cell_type} has {len(frame)} pair rows; expected {expected_pairs}"
            )
        rows.append(
            {
                "cell_type": cell_type,
                "n_supertypes": n_supertypes,
                "n_pairs": int(len(frame)),
                "median_pairwise_ccc": float(ccc.median()) if ccc.size else np.nan,
                "mean_pairwise_ccc": float(ccc.mean()) if ccc.size else np.nan,
                "q25_pairwise_ccc": float(ccc.quantile(0.25)) if ccc.size else np.nan,
                "q75_pairwise_ccc": float(ccc.quantile(0.75)) if ccc.size else np.nan,
                "min_pairwise_ccc": float(ccc.min()) if ccc.size else np.nan,
                "max_pairwise_ccc": float(ccc.max()) if ccc.size else np.nan,
                "median_pairwise_pearson_r": (
                    float(pearson.median()) if pearson.size else np.nan
                ),
                "median_pairwise_mae": float(mae.median()) if mae.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_agreement(
    table: pd.DataFrame, include_supertype_heterogeneity: bool = False
) -> pd.DataFrame:
    rows = []
    for cell_type, frame in table.groupby("cell_type", sort=False):
        row = {"cell_type": cell_type, **agreement_metrics(frame)}
        if include_supertype_heterogeneity:
            finite_sd = frame["subgroup_sd"].dropna()
            row.update(
                {
                    "median_supertype_sd": float(finite_sd.median()),
                    "mean_supertype_sd": float(finite_sd.mean()),
                    "n_cres_with_supertype_sd": int(finite_sd.size),
                }
            )
        rows.append(row)
    overall = {"cell_type": "ALL", **agreement_metrics(table)}
    if include_supertype_heterogeneity:
        finite_sd = table["subgroup_sd"].dropna()
        overall.update(
            {
                "median_supertype_sd": float(finite_sd.median()),
                "mean_supertype_sd": float(finite_sd.mean()),
                "n_cres_with_supertype_sd": int(finite_sd.size),
            }
        )
    rows.append(overall)
    return pd.DataFrame(rows)


def agreement_for_export(table: pd.DataFrame, grouping: str) -> pd.DataFrame:
    """Give exported columns terminology specific to the grouping strategy."""
    if grouping == "random":
        return table[
            [
                "cell_type",
                "cre",
                "whole_activity",
                "mean_subgroup_activity",
                "subgroup_sd",
                "n_subgroups",
                "difference",
                "absolute_difference",
            ]
        ].rename(
            columns={
                "mean_subgroup_activity": "mean_subset_activity",
                "subgroup_sd": "subset_sd",
                "n_subgroups": "n_subsets",
            }
        )
    if grouping in SUPERTYPE_LIKE_GROUPINGS:
        return table[
            [
                "cell_type",
                "cre",
                "whole_activity",
                "mean_subgroup_activity",
                "unweighted_mean_subgroup_activity",
                "subgroup_sd",
                "n_subgroups",
                "difference",
                "absolute_difference",
                "unweighted_difference",
                "unweighted_absolute_difference",
            ]
        ].rename(
            columns={
                "mean_subgroup_activity": "cell_weighted_mean_supertype_activity",
                "unweighted_mean_subgroup_activity": "unweighted_mean_supertype_activity",
                "subgroup_sd": "supertype_sd",
                "n_subgroups": "n_supertypes",
                "difference": "cell_weighted_difference",
                "absolute_difference": "cell_weighted_absolute_difference",
            }
        )
    raise ValueError(f"unsupported grouping={grouping!r}")


def plot_bayesian_subset_agreement(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    targets: list[str],
    cell_counts: dict[str, int],
    subgroups_by_subclass: dict[str, list[str]],
    grouping: str,
    figures_dir: Path,
    ncols: int,
    model: str = "bayesian",
) -> None:
    noun = subgroup_noun(grouping)
    noun_plural = subgroup_noun(grouping, plural=True)
    dispersion_note = (
        "between-supertype heterogeneity"
        if grouping == "supertype"
        else "estimation dispersion at matched cell support"
    )
    ncols = min(len(targets), ncols)
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False
    )
    metric_rows = summary.set_index("cell_type")

    for ax, cell_type in zip(axes.flat, targets):
        panel = table[table["cell_type"] == cell_type].copy()
        x = panel["whole_activity"].to_numpy(dtype=float)
        y = panel["mean_subgroup_activity"].to_numpy(dtype=float)
        yerr = panel["subgroup_sd"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y, yerr = x[valid], y[valid], yerr[valid]
        if x.size:
            finite_error = np.isfinite(yerr)
            error_low = y[finite_error] - yerr[finite_error]
            error_high = y[finite_error] + yerr[finite_error]
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            if finite_error.any():
                lo = min(lo, float(error_low.min()))
                hi = max(hi, float(error_high.max()))
            pad = max(0.08 * (hi - lo), 0.08)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=1.1, zorder=1)
            if finite_error.any():
                ax.vlines(
                    x[finite_error],
                    error_low,
                    error_high,
                    color="#4c78a8",
                    linewidth=0.35,
                    alpha=0.18,
                    rasterized=True,
                    zorder=2,
                )
            ax.scatter(
                x,
                y,
                color="#4c78a8",
                s=10,
                alpha=0.6,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        metrics = metric_rows.loc[cell_type]
        ccc = metrics["concordance_correlation"]
        mae = metrics["mae"]
        metric_text = f"CCC = {ccc:.2f}\nMAE = {mae:.2f}"
        if grouping in SUPERTYPE_LIKE_GROUPINGS:
            median_sd = metrics["median_supertype_sd"]
            metric_text += (
                f"\nMedian {noun} SD = {median_sd:.2f}"
                if np.isfinite(median_sd)
                else f"\nMedian {noun} SD = unavailable"
            )
        ax.text(
            0.04,
            0.96,
            metric_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.set_title(
            (
                f"{cell_type}\n(n = {cell_counts[cell_type]:,} cells)"
                if grouping == "random"
                else (
                    f"{cell_type}\n(n = {cell_counts[cell_type]:,} cells; "
                    f"k = {len(subgroups_by_subclass[cell_type])} "
                    f"{noun if len(subgroups_by_subclass[cell_type]) == 1 else noun_plural})"
                )
            ),
            fontsize=10,
        )
        ax.set_xlabel("Whole cell-type activity")
        ax.set_ylabel(
            "Mean activity across 5 subsets"
            if grouping == "random"
            else f"Cell-count-weighted mean activity\nacross {noun_plural}"
        )
        if (
            grouping in SUPERTYPE_LIKE_GROUPINGS
            and len(subgroups_by_subclass[cell_type]) == 1
        ):
            ax.text(
                0.04,
                0.04,
                f"One {noun}; SD unavailable",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
        ax.set_aspect("equal", adjustable="box")

    for ax in axes.flat[len(targets) :]:
        ax.set_visible(False)

    fig.suptitle(
        (
            f"{model.capitalize()} cCRE activity: mean of 5 random cell subsets "
            "vs whole cell type"
            if grouping == "random"
            else (
                f"{model.capitalize()} cCRE activity: cell-count-weighted mean of "
                f"{noun_plural} vs whole cell type"
            )
        ),
        fontsize=14,
    )
    if grouping in SUPERTYPE_LIKE_GROUPINGS:
        fig.text(
            0.5,
            0.01,
            "Points: cell-count-weighted mean; vertical bars: ±1 unweighted SD "
            f"across {noun_plural} ({dispersion_note}, not uncertainty).",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    fig.tight_layout(
        rect=(0, 0.04 if grouping in SUPERTYPE_LIKE_GROUPINGS else 0, 1, 0.96)
    )
    save_figure(fig, figures_dir / artifact_names(grouping, model)["figure"])


def ordinal_pair_key(
    pairwise: pd.DataFrame, subgroups_by_subclass: dict[str, list[str]]
) -> pd.DataFrame:
    """Key each pair by its parent and the ordinals of its two subgroups.

    Ordinals come from the manifest membership order, which the size-matched
    null preserves by construction: its ``i``-th random group carries the cell
    count of the ``i``-th annotated supertype. Keying on ordinals rather than
    labels is what makes the two runs comparable pair-for-pair.
    """
    keyed = pairwise.copy()
    ordinals = {
        cell_type: {member: index for index, member in enumerate(members)}
        for cell_type, members in subgroups_by_subclass.items()
    }
    for column, target in (("supertype_1", "ordinal_1"), ("supertype_2", "ordinal_2")):
        keyed[target] = [
            ordinals[cell_type][member]
            for cell_type, member in zip(keyed["cell_type"], keyed[column])
        ]
    return keyed


def paired_null_comparison(
    pairwise: pd.DataFrame,
    null_pairwise: pd.DataFrame,
    subgroups_by_subclass: dict[str, list[str]],
    null_subgroups_by_subclass: dict[str, list[str]],
) -> pd.DataFrame:
    """Match annotated pairs to their size-matched random counterparts.

    Every annotated pair is joined to the null pair built from the two random
    groups with the same cell counts, so the difference in CCC isolates
    biological divergence from estimation noise at fixed cell support. The cell
    counts are re-checked after the join rather than assumed.
    """
    left = ordinal_pair_key(pairwise, subgroups_by_subclass)
    right = ordinal_pair_key(null_pairwise, null_subgroups_by_subclass)
    merged = left.merge(
        right,
        on=["cell_type", "ordinal_1", "ordinal_2"],
        suffixes=("_annotated", "_null"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"pair alignment is incomplete: {len(left)} annotated and "
            f"{len(right)} null pairs matched {len(merged)} times"
        )
    for column in ("n_cells_1", "n_cells_2"):
        mismatch = merged[f"{column}_annotated"] != merged[f"{column}_null"]
        if mismatch.any():
            offending = merged.loc[mismatch, "cell_type"].unique().tolist()
            raise ValueError(
                f"null groups are not size-matched to the annotated supertypes "
                f"for {column} in {offending}"
            )
    merged["ccc_annotated"] = merged["concordance_correlation_annotated"]
    merged["ccc_null"] = merged["concordance_correlation_null"]
    merged["ccc_difference"] = merged["ccc_annotated"] - merged["ccc_null"]
    return merged


def hodges_lehmann(differences: np.ndarray) -> float:
    """Median of the Walsh averages: the estimator Wilcoxon actually tests."""
    if differences.size == 0:
        return float("nan")
    rows, cols = np.triu_indices(differences.size)
    return float(np.median((differences[rows] + differences[cols]) / 2.0))


def sign_flip_permutation_p(
    differences: np.ndarray, seed: int, max_exact: int = 20, draws: int = 20_000
) -> tuple[float, bool]:
    """Two-sided randomization p-value for a paired design.

    Under the null the annotated and null member of a pair are exchangeable, so
    flipping the sign of any subset of differences is equally likely. This
    assumes neither normality nor symmetry of the underlying distribution --
    only that the labelling within a pair carries no information. The test is
    enumerated exactly for small samples and sampled otherwise; the returned
    flag records which was used.
    """
    differences = differences[np.isfinite(differences)]
    n = differences.size
    if n == 0:
        return float("nan"), False
    observed = abs(float(differences.mean()))
    if n <= max_exact:
        signs = 1 - 2 * (
            (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
        )
        statistics = np.abs((signs * differences).mean(axis=1))
        return float((statistics >= observed - 1e-12).mean()), True
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, n))
    statistics = np.abs((signs * differences).mean(axis=1))
    # +1 in numerator and denominator: the observed labelling is itself one of
    # the equally likely outcomes, which keeps the p-value from reaching 0.
    return float((np.sum(statistics >= observed - 1e-12) + 1) / (draws + 1)), False


def paired_ccc_test(
    differences: np.ndarray, seed: int = 20260820
) -> dict[str, float | int | bool]:
    """Paired comparison of annotated vs null CCC.

    CCC is bounded and its marginal distribution is skewed, so the
    rank-based and randomization results are the ones to report; the t-test is
    retained only as a familiar reference and is accompanied by a normality
    check on the differences it assumes.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = int(differences.size)
    empty = {
        "n_pairs": n,
        "median_difference": float(np.median(differences)) if n else np.nan,
        "hodges_lehmann": hodges_lehmann(differences),
        "n_negative": int((differences < 0).sum()),
        "n_positive": int((differences > 0).sum()),
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p_value": np.nan,
        "rank_biserial": np.nan,
        "sign_test_p_value": np.nan,
        "permutation_p_value": np.nan,
        "permutation_exact": False,
        "mean_difference": float(differences.mean()) if n else np.nan,
        "sd_difference": np.nan,
        "t_statistic": np.nan,
        "t_p_value": np.nan,
        "cohens_dz": np.nan,
        "shapiro_p_value": np.nan,
    }
    if n < 2:
        return empty

    positive_rank_sum = float(
        wilcoxon(differences, alternative="two-sided").statistic
    )
    wilcoxon_result = wilcoxon(differences)
    total_rank_sum = n * (n + 1) / 2.0
    permutation_p, permutation_exact = sign_flip_permutation_p(differences, seed)
    non_zero = int((differences != 0).sum())
    sign_p = (
        float(
            binomtest(int((differences < 0).sum()), non_zero, 0.5).pvalue
        )
        if non_zero
        else np.nan
    )
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    t_statistic, t_p = ttest_rel(differences, np.zeros_like(differences))
    return {
        **empty,
        "wilcoxon_statistic": positive_rank_sum,
        "wilcoxon_p_value": float(wilcoxon_result.pvalue),
        # Rank-biserial: signed share of the total rank mass, in [-1, 1].
        "rank_biserial": float(2 * positive_rank_sum / total_rank_sum - 1),
        "sign_test_p_value": sign_p,
        "permutation_p_value": permutation_p,
        "permutation_exact": permutation_exact,
        "sd_difference": sd,
        "t_statistic": float(t_statistic),
        "t_p_value": float(t_p),
        "cohens_dz": mean / sd if sd > 0 else np.nan,
        "shapiro_p_value": float(shapiro(differences).pvalue) if n >= 3 else np.nan,
    }


def cell_type_level_test(
    merged: pd.DataFrame, seed: int = 20260820
) -> dict[str, float | int | bool]:
    """Test with the cell type, not the pair, as the independent unit.

    Pairs within a cell type share supertypes and are therefore not
    independent, which makes every pair-level p-value anticonservative.
    Collapsing each cell type to its median difference first gives one value
    per genuinely independent unit; with nine of them the sign-flip
    randomization is enumerated exactly.
    """
    per_cell_type = (
        merged.groupby("cell_type")["ccc_difference"].median().to_numpy(dtype=float)
    )
    per_cell_type = per_cell_type[np.isfinite(per_cell_type)]
    permutation_p, permutation_exact = sign_flip_permutation_p(per_cell_type, seed)
    non_zero = int((per_cell_type != 0).sum())
    return {
        "n_cell_types": int(per_cell_type.size),
        "median_of_cell_type_medians": (
            float(np.median(per_cell_type)) if per_cell_type.size else np.nan
        ),
        "n_negative": int((per_cell_type < 0).sum()),
        "n_positive": int((per_cell_type > 0).sum()),
        "permutation_p_value": permutation_p,
        "permutation_exact": permutation_exact,
        "sign_test_p_value": (
            float(binomtest(int((per_cell_type < 0).sum()), non_zero, 0.5).pvalue)
            if non_zero
            else np.nan
        ),
        "wilcoxon_p_value": (
            float(wilcoxon(per_cell_type).pvalue) if per_cell_type.size >= 2 else np.nan
        ),
    }


def summarize_paired_null_comparison(
    merged: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
    """Per-cell-type and pooled paired tests of annotated vs size-matched null."""
    rows = []
    for cell_type in targets + ["ALL"]:
        frame = (
            merged if cell_type == "ALL" else merged[merged["cell_type"] == cell_type]
        )
        if frame.empty:
            continue
        rows.append(
            {
                "cell_type": cell_type,
                "median_ccc_annotated": float(
                    frame["ccc_annotated"].median()
                ),
                "median_ccc_null": float(frame["ccc_null"].median()),
                **paired_ccc_test(frame["ccc_difference"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def significance_stars(p_value: float) -> str:
    """Conventional star notation; ``p_value`` is the rank-based p-value."""
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def _draw_ccc_distribution(
    ax: plt.Axes,
    position: float,
    values: np.ndarray,
    color: str,
    box_color: str,
    median_color: str,
    width: float,
    jitter: float,
    rng: np.random.Generator,
    label: str | None = None,
) -> None:
    """Draw one jittered point cloud plus its box at ``position``."""
    if not values.size:
        return
    ax.scatter(
        position + rng.uniform(-jitter, jitter, size=values.size),
        values,
        s=22,
        color=color,
        alpha=0.65,
        linewidth=0,
        zorder=3,
        label=label,
    )
    boxplot = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": median_color, "linewidth": 1.8},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"facecolor": box_color, "edgecolor": "#555555", "alpha": 0.65},
        zorder=2,
    )
    for artist in boxplot["boxes"]:
        artist.set_zorder(2)


def plot_pairwise_supertype_ccc(
    pairwise: pd.DataFrame,
    summary: pd.DataFrame,
    whole_agreement_summary: pd.DataFrame,
    targets: list[str],
    figures_dir: Path,
    grouping: str = "supertype",
    model: str = "bayesian",
    null_comparison: pd.DataFrame | None = None,
    null_tests: pd.DataFrame | None = None,
    cell_type_test: dict[str, float | int | bool] | None = None,
) -> None:
    """Plot distributions of within-parent pairwise subgroup CCC values.

    With ``null_comparison`` supplied, each cell type shows two boxes: the
    annotated supertype pairs and their size-matched random counterparts. The
    two are paired pair-for-pair, so the accompanying paired t-test in
    ``null_tests`` is annotated directly above each cell type.
    """
    noun = subgroup_noun(grouping)
    noun_plural = subgroup_noun(grouping, plural=True)
    paired = null_comparison is not None
    fig, ax = plt.subplots(figsize=(15.5 if paired else 14, 7.2))
    summary_by_cell_type = summary.set_index("cell_type")
    whole_by_cell_type = whole_agreement_summary.set_index("cell_type")
    test_by_cell_type = (
        null_tests.set_index("cell_type") if null_tests is not None else None
    )
    rng = np.random.default_rng(20260811)
    offset = 0.21 if paired else 0.0
    width = 0.34 if paired else 0.48
    jitter = 0.11 if paired else 0.16
    legend_drawn = False
    ceiling = 1.08

    for position, cell_type in enumerate(targets, start=1):
        if paired:
            frame = null_comparison[null_comparison["cell_type"] == cell_type]
            values = frame["ccc_annotated"].dropna().to_numpy(dtype=float)
            null_values = frame["ccc_null"].dropna().to_numpy(dtype=float)
        else:
            values = (
                pairwise.loc[
                    pairwise["cell_type"] == cell_type, "concordance_correlation"
                ]
                .dropna()
                .to_numpy(dtype=float)
            )
            null_values = np.array([], dtype=float)

        if not values.size:
            ax.text(
                position,
                0.0,
                f"one {noun}\n(no pairs)",
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )
            continue

        _draw_ccc_distribution(
            ax,
            position - offset,
            values,
            "#e45756",
            "#f2b8b5",
            "#8b1a1a",
            width,
            jitter,
            rng,
            label=None if legend_drawn else "Annotated supertype pairs",
        )
        if paired:
            _draw_ccc_distribution(
                ax,
                position + offset,
                null_values,
                "#4c78a8",
                "#b6cde4",
                "#1f3f66",
                width,
                jitter,
                rng,
                label=(
                    None
                    if legend_drawn
                    else "Size-matched random pairs (null)"
                ),
            )
            legend_drawn = True
            top = float(max(values.max(), null_values.max()))
            row = test_by_cell_type.loc[cell_type]
            ax.text(
                position,
                min(ceiling - 0.06, top + 0.05),
                f"{significance_stars(float(row['wilcoxon_p_value']))}\n"
                f"HL={float(row['hodges_lehmann']):+.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        else:
            legend_drawn = True
            median = summary_by_cell_type.loc[cell_type, "median_pairwise_ccc"]
            ax.text(
                position,
                min(1.02, float(values.max()) + 0.06),
                f"median {median:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#7a2e2d",
            )

    for position, cell_type in enumerate(targets, start=1):
        if cell_type not in whole_by_cell_type.index:
            raise KeyError(f"missing mean-vs-whole CCC baseline for {cell_type}")
        baseline = float(
            whole_by_cell_type.loc[cell_type, "concordance_correlation"]
        )
        if not np.isfinite(baseline):
            continue
        ax.scatter(
            position - offset,
            baseline,
            marker="D",
            s=58,
            facecolor="white",
            edgecolor="#1f77b4",
            linewidth=1.8,
            zorder=5,
        )
        if not paired:
            ax.text(
                position,
                min(1.055, baseline + 0.025),
                f"{baseline:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1f77b4",
                fontweight="bold",
            )

    labels = []
    for cell_type in targets:
        row = summary_by_cell_type.loc[cell_type]
        labels.append(f"{cell_type}\n({int(row['n_pairs'])} pairs)")
    ax.set_xticks(np.arange(1, len(targets) + 1), labels, rotation=32, ha="right")
    ax.axhline(1.0, color="#555555", ls="--", lw=1.0, zorder=1)
    ax.set_xlim(0.4, len(targets) + 0.6)
    finite_ccc = pairwise["concordance_correlation"].dropna().to_numpy(dtype=float)
    if paired:
        finite_ccc = np.concatenate(
            [finite_ccc, null_comparison["ccc_null"].dropna().to_numpy(dtype=float)]
        )
    lower_limit = (
        max(-1.08, min(-0.08, np.floor((finite_ccc.min() - 0.05) * 10) / 10))
        if finite_ccc.size
        else -0.08
    )
    ax.set_ylim(lower_limit, ceiling)
    ax.set_ylabel("Lin concordance correlation (CCC)\nacross fitted cCREs")
    if paired:
        overall = null_tests.loc[null_tests["cell_type"] == "ALL"].iloc[0]
        ax.set_title(
            f"{model.capitalize()} activity divergence among {noun_plural} "
            "versus size-matched random cell subsets",
            fontsize=14,
        )
        cluster_note = ""
        if cell_type_test is not None:
            cluster_note = (
                "\nCell type as the independent unit "
                f"(n = {int(cell_type_test['n_cell_types'])}): exact sign-flip "
                f"p = {float(cell_type_test['permutation_p_value']):.3g}"
            )
        ax.text(
            0.5,
            0.02,
            "Pooled Wilcoxon signed-rank (annotated - null): "
            f"Hodges-Lehmann = {float(overall['hodges_lehmann']):+.3f}, "
            f"p = {float(overall['wilcoxon_p_value']):.2g}, "
            f"r_rb = {float(overall['rank_biserial']):+.2f}, "
            f"n = {int(overall['n_pairs'])} pairs"
            + cluster_note,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )
        ax.legend(frameon=False, loc="upper left", fontsize=9, ncols=2)
    else:
        ax.set_title(
            f"{model.capitalize()} activity divergence among {noun_plural} "
            "within each cell type",
            fontsize=14,
        )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.012,
        (
            "Each point is one within-parent pair; the null pair uses two random cell "
            "subsets with the same two cell counts. Stars: per-cell-type Wilcoxon "
            "signed-rank (*** p<0.001, ** p<0.01, * p<0.05); HL is the Hodges-Lehmann "
            "shift. Blue diamonds: CCC of the cell-count-weighted supertype mean "
            "versus the intact whole cell type. Pair-level tests share supertypes and "
            "are anticonservative, hence the cell-type-level test above."
            if paired
            else (
                f"Red points/boxes: pairwise CCC among {noun_plural}. Blue diamonds: "
                f"CCC of the cell-count-weighted {noun} mean versus the intact whole "
                "cell type. Small groups may be noisier."
            )
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(
        fig,
        figures_dir / artifact_names(grouping, model)["pairwise_figure"],
    )


def plot_pairwise_ccc_vs_minimum_cells(
    pairwise: pd.DataFrame,
    targets: list[str],
    figures_dir: Path,
    grouping: str = "supertype",
    model: str = "bayesian",
) -> None:
    """Plot pairwise subgroup CCC against the smaller group's cell count."""
    noun = subgroup_noun(grouping)
    noun_plural = subgroup_noun(grouping, plural=True)
    valid = pairwise[
        np.isfinite(pairwise["concordance_correlation"])
        & np.isfinite(pairwise["minimum_pair_cells"])
        & (pairwise["minimum_pair_cells"] > 0)
    ].copy()
    if valid.empty:
        raise ValueError("no finite supertype pairs available for cell-support plot")

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    palette = {target: plt.cm.tab10(i % 10) for i, target in enumerate(targets)}
    for cell_type in targets:
        frame = valid[valid["cell_type"] == cell_type]
        if frame.empty:
            continue
        ax.scatter(
            frame["minimum_pair_cells"],
            frame["concordance_correlation"],
            s=36,
            alpha=0.72,
            color=palette[cell_type],
            linewidth=0,
            label=cell_type,
            rasterized=True,
        )

    x = valid["minimum_pair_cells"].to_numpy(dtype=float)
    y = valid["concordance_correlation"].to_numpy(dtype=float)
    log_x = np.log10(x)
    rho, p_value = spearmanr(log_x, y)
    if np.unique(log_x).size > 1:
        slope, intercept = np.polyfit(log_x, y, deg=1)
        x_guide = np.geomspace(x.min(), x.max(), 200)
        ax.plot(
            x_guide,
            intercept + slope * np.log10(x_guide),
            color="#222222",
            lw=1.8,
            ls="--",
            label="Linear guide on log cell count",
            zorder=4,
        )

    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho:.2f}\np = {p_value:.2g}\nn = {len(valid)} pairs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"Minimum cell count in the {noun} pair (log scale)")
    ax.set_ylabel("Pairwise Lin concordance correlation (CCC)\nacross fitted cCREs")
    ax.set_ylim(min(-0.05, float(y.min()) - 0.04), min(1.02, float(y.max()) + 0.08))
    ax.set_title(
        f"{model.capitalize()} pairwise {noun} agreement vs cell support",
        fontsize=14,
    )
    ax.grid(color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        title="Parent cell type",
    )
    fig.text(
        0.43,
        0.012,
        f"Each point is one within-parent pair of {noun_plural}; single-group parents "
        "have no pair. The dashed line is a visual guide, not a fitted model.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 0.82, 0.96))
    save_figure(
        fig,
        figures_dir
        / artifact_names(grouping, model)["pairwise_support_figure"],
    )


def main() -> None:
    args = parse_args()
    if args.ncols <= 0:
        raise ValueError("--ncols must be positive")

    tables = args.outdir / "tables"
    figures = args.outdir / "figures"
    raw = args.outdir / "raw"
    for d in (tables, figures, raw):
        d.mkdir(parents=True, exist_ok=True)

    if args.model == "bayesian":
        combined_dir = args.combined_bayes_dir
        split_dir = args.split_bayes_dir
        loader = load_bayesian
    else:
        combined_dir = args.combined_bootstrap_dir
        split_dir = args.split_bootstrap_dir
        loader = load_bootstrap

    targets, subgroups_by_subclass, grouping = split_targets(split_dir)
    names = artifact_names(grouping, args.model)
    self_cre = args.calibration == "self_cre_negctrl"
    log(
        f"[het] {len(targets)} split subclasses, grouping={grouping}, "
        f"{sum(map(len, subgroups_by_subclass.values()))} total groups, "
        f"model={args.model}, calibration={args.calibration}"
    )

    combined = loader(combined_dir, self_cre=self_cre)
    split = loader(split_dir, self_cre=self_cre)

    neg_set = negative_controls(split_dir) | negative_controls(combined_dir)
    black = blacklist(split_dir) | blacklist(combined_dir)
    # Blacklisted cCREs were not fitted; retain the fitted negative controls so
    # each cell-type panel shows every cCRE in the model.
    excluded = black
    assignment = pd.read_csv(split_dir / "cell_group_assignment.csv")
    subgroup_cell_counts = (
        assignment.groupby(["original_subclass", "new_subclass"], sort=False)
        .size()
        .astype(int)
        .rename("n_cells")
        .reset_index()
    )
    subgroup_weights = {
        cell_type: {
            str(row.new_subclass): int(row.n_cells)
            for row in subgroup_cell_counts[
                subgroup_cell_counts["original_subclass"] == cell_type
            ].itertuples(index=False)
        }
        for cell_type in targets
    }
    agreement = bayesian_subset_agreement(
        combined,
        split,
        targets,
        subgroups_by_subclass,
        excluded,
        subgroup_weights=(
            subgroup_weights if grouping in SUPERTYPE_LIKE_GROUPINGS else None
        ),
    )
    summary = summarize_agreement(
        agreement,
        include_supertype_heterogeneity=grouping in SUPERTYPE_LIKE_GROUPINGS,
    )
    cell_counts = (
        assignment.groupby("original_subclass", sort=False)
        .size()
        .astype(int)
        .to_dict()
    )
    missing_counts = [cell_type for cell_type in targets if cell_type not in cell_counts]
    if missing_counts:
        raise KeyError(f"missing cell counts for split cell types: {missing_counts}")
    panel_targets = sorted(
        targets, key=lambda cell_type: cell_counts[cell_type], reverse=True
    )

    combined.to_csv(raw / f"combined_activity_{args.model}.csv")
    split.to_csv(raw / names["split_raw"])
    agreement_for_export(agreement, grouping).to_csv(
        tables / names["table"], index=False
    )
    summary.to_csv(tables / names["summary"], index=False)
    plot_bayesian_subset_agreement(
        agreement,
        summary,
        panel_targets,
        cell_counts,
        subgroups_by_subclass,
        grouping,
        figures,
        args.ncols,
        model=args.model,
    )
    pairwise = None
    pairwise_summary = None
    pairwise_support_association = None
    if grouping in SUPERTYPE_LIKE_GROUPINGS:
        pairwise = pairwise_supertype_agreement(
            split,
            targets,
            subgroups_by_subclass,
            subgroup_weights,
            excluded,
        )
        pairwise_summary = summarize_pairwise_supertype_agreement(
            pairwise, targets, subgroups_by_subclass
        )
        valid_support = (
            pairwise[["minimum_pair_cells", "concordance_correlation"]]
            .dropna()
            .query("minimum_pair_cells > 0")
        )
        support_rho, support_p = spearmanr(
            np.log10(valid_support["minimum_pair_cells"].to_numpy(dtype=float)),
            valid_support["concordance_correlation"].to_numpy(dtype=float),
        )
        pairwise_support_association = {
            "predictor": "log10 minimum cell count of the two supertypes",
            "outcome": "pairwise Lin concordance correlation coefficient",
            "spearman_rho": float(support_rho),
            "p_value": float(support_p),
            "n_pairs": int(len(valid_support)),
            "interpretation": (
                "pairwise CCC depends on cell support; low-support pairs may mix "
                "biological heterogeneity with estimation noise"
            ),
        }
        null_comparison = None
        null_tests = None
        cell_type_test = None
        if args.null_split_bayes_dir is not None:
            if grouping != "supertype":
                raise ValueError(
                    "--null-split-bayes-dir applies to the annotated-supertype "
                    f"run; this run has grouping={grouping!r}"
                )
            null_targets, null_members, null_grouping = split_targets(
                args.null_split_bayes_dir
            )
            if null_grouping != "supertype_like_random":
                raise ValueError(
                    "--null-split-bayes-dir must point at a "
                    f"supertype_like_random fit; found {null_grouping!r}"
                )
            if null_targets != targets:
                raise ValueError(
                    "null and annotated runs cover different cell types"
                )
            null_split = loader(args.null_split_bayes_dir, self_cre=self_cre)
            null_assignment = pd.read_csv(
                args.null_split_bayes_dir / "cell_group_assignment.csv"
            )
            null_weights = {
                cell_type: (
                    null_assignment[
                        null_assignment["original_subclass"] == cell_type
                    ]
                    .groupby("new_subclass")
                    .size()
                    .astype(int)
                    .to_dict()
                )
                for cell_type in targets
            }
            shared_cres = set(split.columns.astype(str)) - excluded
            null_cres = set(null_split.columns.astype(str)) - excluded
            if shared_cres != null_cres:
                raise ValueError(
                    "annotated and null fits disagree on the fitted cCRE set: "
                    f"{len(shared_cres ^ null_cres)} cCREs differ"
                )
            null_pairwise = pairwise_supertype_agreement(
                null_split, targets, null_members, null_weights, excluded
            )
            null_comparison = paired_null_comparison(
                pairwise, null_pairwise, subgroups_by_subclass, null_members
            )
            null_tests = summarize_paired_null_comparison(
                null_comparison, panel_targets
            )
            cell_type_test = cell_type_level_test(null_comparison)
            null_pairwise.to_csv(
                tables / "bayesian_supertype_pairwise_ccc_null.csv", index=False
            )
            null_comparison[
                [
                    "cell_type",
                    "supertype_1_annotated",
                    "supertype_2_annotated",
                    "supertype_1_null",
                    "supertype_2_null",
                    "n_cells_1_annotated",
                    "n_cells_2_annotated",
                    "minimum_pair_cells_annotated",
                    "ccc_annotated",
                    "ccc_null",
                    "ccc_difference",
                ]
            ].rename(
                columns={
                    "supertype_1_annotated": "supertype_1",
                    "supertype_2_annotated": "supertype_2",
                    "supertype_1_null": "random_group_1",
                    "supertype_2_null": "random_group_2",
                    "n_cells_1_annotated": "n_cells_1",
                    "n_cells_2_annotated": "n_cells_2",
                    "minimum_pair_cells_annotated": "minimum_pair_cells",
                }
            ).to_csv(
                tables / "bayesian_supertype_pairwise_ccc_vs_null.csv", index=False
            )
            null_tests.to_csv(
                tables / "bayesian_supertype_pairwise_ccc_null_tests.csv",
                index=False,
            )
            pd.DataFrame([cell_type_test]).to_csv(
                tables / "bayesian_supertype_pairwise_ccc_cell_type_test.csv",
                index=False,
            )
        pairwise.to_csv(tables / names["pairwise_table"], index=False)
        pairwise_summary.to_csv(
            tables / names["pairwise_summary"], index=False
        )
        plot_pairwise_supertype_ccc(
            pairwise,
            pairwise_summary,
            summary,
            panel_targets,
            figures,
            grouping=grouping,
            model=args.model,
            null_comparison=null_comparison,
            null_tests=null_tests,
            cell_type_test=cell_type_test,
        )
        plot_pairwise_ccc_vs_minimum_cells(
            pairwise,
            panel_targets,
            figures,
            grouping=grouping,
            model=args.model,
        )

    n_subgroups_by_subclass = {
        target: len(subgroups_by_subclass[target]) for target in targets
    }
    noun = subgroup_noun(grouping)
    noun_plural = subgroup_noun(grouping, plural=True)

    write_json(
        tables / names["manifest"],
        {
            "split_subclasses": targets,
            "panel_cell_types": panel_targets,
            "panel_order": "descending original cell count",
            "n_groups": (
                next(iter(n_subgroups_by_subclass.values()))
                if len(set(n_subgroups_by_subclass.values())) == 1
                else None
            ),
            "grouping": grouping,
            "subgroup_obs_column": (
                None if grouping == "random" else "supertype_name"
            ),
            "subgroup_unit": noun,
            "subgroups_by_subclass": subgroups_by_subclass,
            "n_subgroups_by_subclass": n_subgroups_by_subclass,
            "subgroup_cell_counts": {
                row.new_subclass: int(row.n_cells)
                for row in subgroup_cell_counts.itertuples(index=False)
            },
            "calibration": args.calibration,
            "models": [args.model],
            "combined_run_dir": str(combined_dir),
            "split_run_dir": str(split_dir),
            "combined_bayes_dir": (
                str(combined_dir) if args.model == "bayesian" else None
            ),
            "split_bayes_dir": (
                str(split_dir) if args.model == "bayesian" else None
            ),
            "combined_bootstrap_dir": (
                str(combined_dir) if args.model == "bootstrap" else None
            ),
            "split_bootstrap_dir": (
                str(split_dir) if args.model == "bootstrap" else None
            ),
            "excluded_cres": sorted(excluded),
            "panel_unit": "cell type",
            "point_unit": "cCRE",
            "cell_counts": cell_counts,
            "n_cres_per_panel": int(agreement["cre"].nunique()),
            "negative_controls_included": sorted(neg_set),
            "x_metric": "negative-control-centered activity in intact cell type",
            "y_metric": (
                "cell-count-weighted mean negative-control-centered activity "
                f"across {noun_plural}"
                if grouping in SUPERTYPE_LIKE_GROUPINGS
                else (
                    "unweighted mean negative-control-centered activity across "
                    f"{noun_plural}"
                )
            ),
            "aggregation": (
                "cell-count-weighted"
                if grouping in SUPERTYPE_LIKE_GROUPINGS
                else "unweighted"
            ),
            "secondary_aggregation": (
                f"unweighted mean across {noun_plural}"
                if grouping in SUPERTYPE_LIKE_GROUPINGS
                else None
            ),
            "error_bar": (
                f"unweighted sample standard deviation across {noun_plural}"
                if grouping in SUPERTYPE_LIKE_GROUPINGS
                else f"sample standard deviation across {noun_plural}"
            ),
            "error_bar_interpretation": (
                "between-supertype biological heterogeneity, not posterior or fit "
                "uncertainty"
                if grouping == "supertype"
                else (
                    "estimation dispersion between random subsets at matched cell "
                    "support"
                    if grouping == "supertype_like_random"
                    else "between-random-subset dispersion"
                )
            ),
            "pairwise_supertype_analysis": (
                {
                    "table": names["pairwise_table"],
                    "summary": names["pairwise_summary"],
                    "figure": f"{names['pairwise_figure']}.pdf",
                    "cell_support_figure": (
                        f"{names['pairwise_support_figure']}.pdf"
                    ),
                    "metric": "Lin concordance correlation coefficient",
                    "comparison_unit": (
                        f"each unordered pair of {noun_plural} within a parent cell type"
                    ),
                    "feature_axis": "all fitted non-blacklisted cCREs",
                    "pair_weighting": "each supertype pair contributes once",
                    "baseline_marker": (
                        f"per-cell-type CCC of the cell-count-weighted {noun} "
                        "mean versus the intact whole-cell-type activity"
                    ),
                    "baseline_source": names["summary"],
                    "n_pairs_total": int(len(pairwise)),
                    "n_pairs_by_subclass": {
                        str(row.cell_type): int(row.n_pairs)
                        for row in pairwise_summary.itertuples(index=False)
                    },
                    "cell_support_association": pairwise_support_association,
                    "size_matched_null": (
                        {
                            "null_split_bayes_dir": str(args.null_split_bayes_dir),
                            "pair_matching": (
                                "annotated pair (i, j) matched to the random pair "
                                "(i, j) built from groups with the same two cell "
                                "counts"
                            ),
                            "primary_test": (
                                "Wilcoxon signed-rank on annotated - null CCC"
                            ),
                            "supporting_tests": [
                                "exact sign test",
                                "sign-flip randomization test",
                                "paired t-test with a Shapiro-Wilk check on the "
                                "differences",
                            ],
                            "effect_size": (
                                "Hodges-Lehmann shift and rank-biserial correlation"
                            ),
                            "caveat": (
                                "pairs share subgroups, so pair-level p-values are "
                                "anticonservative; the cell-type-level test uses one "
                                "median per cell type as the independent unit"
                            ),
                            "cell_type_level_test": jsonable(cell_type_test),
                            "table": "bayesian_supertype_pairwise_ccc_vs_null.csv",
                            "tests_table": (
                                "bayesian_supertype_pairwise_ccc_null_tests.csv"
                            ),
                            "overall": {
                                key: (
                                    float(value)
                                    if isinstance(value, (int, float, np.floating))
                                    else value
                                )
                                for key, value in null_tests.loc[
                                    null_tests["cell_type"] == "ALL"
                                ]
                                .iloc[0]
                                .to_dict()
                                .items()
                            },
                        }
                        if null_tests is not None
                        else None
                    ),
                }
                if grouping in SUPERTYPE_LIKE_GROUPINGS
                else None
            ),
            "agreement_metrics": [
                "Lin concordance correlation",
                "mean absolute error",
                "root mean squared error",
                "mean error",
                "Pearson correlation",
            ],
        },
    )
    overall = summary.loc[summary["cell_type"] == "ALL"].iloc[0]
    log(
        f"[het] {len(targets)} cell-type panels, "
        f"{agreement['cre'].nunique()} cCREs per panel; overall "
        f"CCC={overall['concordance_correlation']:.3f}, "
        f"MAE={overall['mae']:.3f}"
    )
    log(
        f"[het] wrote {grouping} {args.model} agreement table to {tables}, "
        f"figure to {figures}"
    )
    if grouping in SUPERTYPE_LIKE_GROUPINGS:
        log(
            f"[het] wrote {len(pairwise)} within-parent {noun}-pair CCCs "
            f"and distribution figure; CCC vs minimum pair cells "
            f"Spearman rho={pairwise_support_association['spearman_rho']:.3f}"
        )


if __name__ == "__main__":
    main()
