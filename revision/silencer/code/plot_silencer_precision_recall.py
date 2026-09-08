#!/usr/bin/env python
"""Benchmark left-tail silencer calls against the ChromHMM silencer annotation.

Evaluation is on every tested pair that carries an annotation (the "all
annotated" set) at the production T7 filter, sweeping how permissive the truth
label is: a pair is an annotated silencer when its ``Chr-R`` + ``Hc-P`` coverage
fraction exceeds ``--silencer-cutoffs`` (strictly greater, so ``0`` means "any
repressive coverage at all").

Calls are ``q_left <= --q-cutoff`` from ``test_silencer_left_tail.py``.
Precision and recall follow the convention of
``revision/bayesian_vs_fold_change/code/plot_t7_filter_precision_recall.py``:

    precision = TP / #significant          recall = TP / #annotated positives

with the annotation prevalence among tested pairs drawn as the no-skill
baseline, and a one-sided Fisher exact test for enrichment. The label sweep only
moves the truth set; the tested pairs, the calls and their scores are fixed, so
the whole sweep also collapses into one threshold-free statement: the rank
correlation between depletion and repressive fraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve

from baystarrfish.data.paths import repo_root
from silencer_truth import load_silencer_truth

PANEL_SIDE_INCHES: Final[float] = 4.0
CALL_COLOR: Final[str] = "#4c78a8"
POSITIVE_COLOR: Final[str] = "#e45756"
DEFAULT_CUTOFFS: Final[tuple[float, ...]] = (0.0, 0.1, 0.2, 0.5, 0.8)
#: Cutoff whose binary labels colour the q-value panels.
REFERENCE_CUTOFF: Final[float] = 0.5


def parse_args() -> argparse.Namespace:
    root = repo_root()
    results = root / "revision/silencer/results"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", type=Path,
                        default=results / "silencer_left_tail_tests.csv.gz")
    parser.add_argument("--annotation-dir", type=Path, default=results)
    parser.add_argument("--out-dir", type=Path, default=results)
    parser.add_argument("--figures-dir", type=Path,
                        default=root / "revision/silencer/figures")
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--silencer-cutoffs", type=float, nargs="+",
                        default=list(DEFAULT_CUTOFFS),
                        help="Repressive-fraction cutoffs; a positive is fraction > cutoff.")
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--stem", default="silencer_left_tail")
    return parser.parse_args()


def annotate_tests(
    tests: pd.DataFrame, annotation_dir: Path, t7_threshold: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach the repressive/ND fractions to the tests at one T7 threshold."""
    selected = tests[np.isclose(tests["t7_threshold"].to_numpy(float), t7_threshold)]
    if selected.empty:
        raise ValueError(f"no tests at T7 >= {t7_threshold:g} in the input table")
    truth = load_silencer_truth(annotation_dir)
    stacked = {
        name: (
            frame.rename_axis(index="group", columns="cre")
            .stack(future_stack=True)
            .rename(name)
        )
        for name, frame in (
            ("repressive_fraction", truth.repressive_fraction),
            ("nd_fraction", truth.nd_fraction),
        )
    }
    frame = selected.copy()
    frame["group"] = frame["group"].astype(str)
    frame["cre"] = frame["cre"].astype(str)
    index = pd.MultiIndex.from_frame(frame[["group", "cre"]])
    for name, series in stacked.items():
        frame[name] = series.reindex(index).to_numpy()
    annotated = frame[np.isfinite(frame["repressive_fraction"])].copy()
    coverage = {
        "t7_threshold": float(t7_threshold),
        "tests_total": int(len(frame)),
        "tests_with_annotation": int(len(annotated)),
        "cell_types_tested": int(frame["group"].nunique()),
        "cell_types_annotated": int(annotated["group"].nunique()),
        "ccres_tested": int(frame["cre"].nunique()),
        "ccres_annotated": int(annotated["cre"].nunique()),
        "unmapped_annotation_columns": list(truth.unmapped_columns),
        "mean_nd_fraction": float(annotated["nd_fraction"].mean()),
    }
    return annotated, coverage


def benchmark(
    frame: pd.DataFrame, cutoffs: Sequence[float], q_cutoff: float
) -> pd.DataFrame:
    """Precision/recall at a fixed call set, sweeping the truth cutoff."""
    significant = frame["q_left"].le(q_cutoff).to_numpy(bool)
    fraction = frame["repressive_fraction"].to_numpy(float)
    n_significant = int(significant.sum())
    n_tested = int(len(frame))
    rows = []
    for cutoff in cutoffs:
        positive = fraction > cutoff
        tp = int((significant & positive).sum())
        n_positive = int(positive.sum())
        fp = n_significant - tp
        fn = n_positive - tp
        tn = n_tested - tp - fp - fn
        precision = tp / n_significant if n_significant else np.nan
        recall = tp / n_positive if n_positive else np.nan
        prevalence = n_positive / n_tested if n_tested else np.nan
        odds, pvalue = (
            fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
            if n_positive
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "silencer_cutoff": float(cutoff),
                "q_cutoff": q_cutoff,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "significant": n_significant,
                "annotated_silencers": n_positive,
                "tested": n_tested,
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision and recall and np.isfinite(precision + recall)
                    else np.nan
                ),
                "naive_precision": prevalence,
                "enrichment_over_naive": (
                    precision / prevalence
                    if prevalence and np.isfinite(precision)
                    else np.nan
                ),
                "fisher_oddsratio": odds,
                "fisher_p": pvalue,
                "ties_at_cutoff": int(np.isclose(fraction, cutoff).sum()),
            }
        )
    return pd.DataFrame(rows)


def pr_curves(
    frame: pd.DataFrame, cutoffs: Sequence[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranking-based PR curves per truth cutoff; the score is the depletion."""
    scores = -frame["effect_vs_control_reference_mean"].to_numpy(float)
    fraction = frame["repressive_fraction"].to_numpy(float)
    valid = np.isfinite(scores)
    curve_frames, metric_rows = [], []
    for cutoff in cutoffs:
        labels = (fraction > cutoff)[valid]
        if not labels.any() or labels.all():
            continue
        precision, recall, score_thresholds = precision_recall_curve(
            labels, scores[valid]
        )
        curve_frames.append(
            pd.DataFrame(
                {
                    "silencer_cutoff": float(cutoff),
                    "precision": precision,
                    "recall": recall,
                    "score_threshold": np.r_[score_thresholds, np.nan],
                }
            )
        )
        metric_rows.append(
            {
                "silencer_cutoff": float(cutoff),
                "average_precision": float(
                    average_precision_score(labels, scores[valid])
                ),
                "prevalence": float(labels.mean()),
                "ap_over_prevalence": float(
                    average_precision_score(labels, scores[valid]) / labels.mean()
                ),
                "tested": int(labels.size),
                "annotated_silencers": int(labels.sum()),
            }
        )
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(metric_rows)


def rank_association(frame: pd.DataFrame) -> dict[str, float]:
    """Threshold-free view: does depletion track the repressive fraction at all?"""
    depletion = -frame["effect_vs_control_reference_mean"].to_numpy(float)
    fraction = frame["repressive_fraction"].to_numpy(float)
    evidence = 1.0 - frame["p_left"].to_numpy(float)
    valid = np.isfinite(depletion) & np.isfinite(fraction)
    rho_effect = spearmanr(depletion[valid], fraction[valid])
    rho_evidence = spearmanr(evidence[valid], fraction[valid])
    return {
        "spearman_depletion_vs_repressive": float(rho_effect.statistic),
        "spearman_depletion_p": float(rho_effect.pvalue),
        "spearman_evidence_vs_repressive": float(rho_evidence.statistic),
        "spearman_evidence_p": float(rho_evidence.pvalue),
        "n_pairs": int(valid.sum()),
    }


def _cutoff_colors(cutoffs: Sequence[float]) -> list[tuple[float, float, float, float]]:
    colormap = plt.get_cmap("viridis")
    return [colormap(i / max(len(cutoffs) - 1, 1)) for i in range(len(cutoffs))]


def plot_metrics(
    metrics: pd.DataFrame, t7_threshold: float, q_cutoff: float, path: Path
) -> None:
    """Precision, recall and enrichment as the truth cutoff is relaxed."""
    cutoffs = metrics["silencer_cutoff"].to_numpy(float)
    positions = np.arange(len(cutoffs))
    labels = [f"> {c:g}" for c in cutoffs]
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * PANEL_SIDE_INCHES, PANEL_SIDE_INCHES),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.bar(positions, metrics["precision"], color=CALL_COLOR, width=0.6,
           label="precision of calls")
    for x, y, naive in zip(positions, metrics["precision"], metrics["naive_precision"]):
        ax.plot([x - 0.3, x + 0.3], [naive, naive], color="k", linestyle="--",
                linewidth=1)
    for x, y, tp, n in zip(positions, metrics["precision"], metrics["TP"],
                           metrics["significant"]):
        ax.text(x, y, f"{y:.2f}\n{tp:.0f}/{n:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, min(1.05, float(metrics["precision"].max()) * 1.45))
    ax.set_ylabel("precision")
    ax.set_title("precision vs prevalence (dashed)")

    ax = axes[1]
    ax.bar(positions, metrics["recall"], color=POSITIVE_COLOR, width=0.6)
    for x, y, tp, n in zip(positions, metrics["recall"], metrics["TP"],
                           metrics["annotated_silencers"]):
        ax.text(x, y, f"{y:.2f}\n{tp:.0f}/{n:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, min(1.05, float(metrics["recall"].max()) * 1.45))
    ax.set_ylabel("recall")
    ax.set_title("recall")

    ax = axes[2]
    ax.bar(positions, metrics["enrichment_over_naive"], color="#54a24b", width=0.6)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    for x, y, p in zip(positions, metrics["enrichment_over_naive"], metrics["fisher_p"]):
        ax.text(x, y, f"{y:.2f}\np={p:.2g}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(1.15, float(metrics["enrichment_over_naive"].max()) * 1.35))
    ax.set_ylabel("precision / prevalence")
    ax.set_title("enrichment over chance (Fisher, one-sided)")

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("annotated-silencer cutoff (Chr-R + Hc-P fraction)")
    fig.suptitle(
        f"Left-tail calls (q_left <= {q_cutoff:g}) vs ChromHMM annotation, "
        f"all annotated pairs, T7 >= {t7_threshold:g}",
        fontsize=9,
    )
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_q_values(
    frame: pd.DataFrame, t7_threshold: float, q_cutoff: float, path: Path
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * PANEL_SIDE_INCHES, PANEL_SIDE_INCHES),
        constrained_layout=True,
    )
    fraction = frame["repressive_fraction"].to_numpy(float)
    q = frame["q_left"].to_numpy(float)

    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 41)
    positive = fraction > REFERENCE_CUTOFF
    for mask, label, color in (
        (positive, f"repressive > {REFERENCE_CUTOFF:g}", POSITIVE_COLOR),
        (~positive, f"repressive <= {REFERENCE_CUTOFF:g}", CALL_COLOR),
    ):
        ax.hist(q[mask], bins=bins, histtype="step", density=True, color=color,
                linewidth=1.5, label=f"{label} (n={int(mask.sum())})")
    ax.axvline(q_cutoff, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("BH q-value (left tail)")
    ax.set_ylabel("density")
    ax.set_title("q-value distribution")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    order = np.argsort(q)
    ax.plot(q[order], np.arange(1, len(q) + 1) / len(q), color="k", linewidth=1.5,
            label=f"all annotated (n={len(q)})")
    for cutoff, color in zip(DEFAULT_CUTOFFS, _cutoff_colors(DEFAULT_CUTOFFS)):
        subset = np.sort(q[fraction > cutoff])
        if not subset.size:
            continue
        ax.plot(subset, np.arange(1, subset.size + 1) / subset.size, color=color,
                linewidth=1.2, label=f"> {cutoff:g} (n={subset.size})")
    ax.axvline(q_cutoff, color="k", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("BH q-value (left tail)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("q-value ECDF by truth cutoff")
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = axes[2]
    scatter = ax.scatter(
        frame["effect_vs_control_reference_mean"].to_numpy(float),
        -np.log10(np.clip(q, 1e-4, None)),
        c=fraction, cmap="magma_r", vmin=0.0, vmax=1.0, s=8, alpha=0.8, linewidths=0,
    )
    ax.axhline(-np.log10(q_cutoff), color="k", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("activity minus control reference (log)")
    ax.set_ylabel("-log10 q (left tail, clipped at 1e-4)")
    ax.set_title("left-tail volcano")
    fig.colorbar(scatter, ax=ax, label="Chr-R + Hc-P fraction")

    fig.suptitle(f"Left-tail silencer test, T7 >= {t7_threshold:g}", fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    curves: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    t7_threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(PANEL_SIDE_INCHES * 1.4, PANEL_SIDE_INCHES), constrained_layout=True
    )
    cutoffs = curve_metrics["silencer_cutoff"].tolist()
    for cutoff, color in zip(cutoffs, _cutoff_colors(cutoffs)):
        curve = curves[np.isclose(curves["silencer_cutoff"], cutoff)]
        row = curve_metrics[np.isclose(curve_metrics["silencer_cutoff"], cutoff)].iloc[0]
        ax.step(curve["recall"], curve["precision"], where="post", color=color,
                linewidth=1.5,
                label=f"> {cutoff:g}: AP {row['average_precision']:.3f} "
                      f"(chance {row['prevalence']:.3f})")
        ax.axhline(row["prevalence"], color=color, linestyle="--", linewidth=0.8,
                   alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(
        f"Ranking by depletion below control reference, T7 >= {t7_threshold:g}",
        fontsize=9,
    )
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    tests = pd.read_csv(args.tests)
    if "q_left" not in tests:
        raise ValueError(f"{args.tests} has no q_left column; run the left-tail test")
    annotated, coverage = annotate_tests(tests, args.annotation_dir, args.t7_threshold)
    cutoffs = sorted(args.silencer_cutoffs)

    metrics = benchmark(annotated, cutoffs, args.q_cutoff)
    curves, curve_metrics = pr_curves(annotated, cutoffs)
    association = rank_association(annotated)

    metrics.to_csv(args.out_dir / f"{args.stem}_precision_recall.csv", index=False,
                   float_format="%.6g")
    curves.to_csv(args.out_dir / f"{args.stem}_pr_curves.csv.gz", index=False,
                  float_format="%.6g")
    curve_metrics.to_csv(args.out_dir / f"{args.stem}_pr_metrics.csv", index=False,
                         float_format="%.6g")
    annotated.to_csv(args.out_dir / f"{args.stem}_tests_annotated.csv.gz", index=False,
                     float_format="%.6g")

    plot_metrics(metrics, args.t7_threshold, args.q_cutoff,
                 args.figures_dir / f"{args.stem}_precision_recall.pdf")
    plot_q_values(annotated, args.t7_threshold, args.q_cutoff,
                  args.figures_dir / f"{args.stem}_qvalues.pdf")
    plot_pr_curves(curves, curve_metrics, args.t7_threshold,
                   args.figures_dir / f"{args.stem}_pr_curves.pdf")

    manifest = {
        "tests": str(args.tests),
        "annotation_dir": str(args.annotation_dir),
        "evaluation_set": "all annotated pairs (no ND stratification)",
        "silencer_definition": "Chr-R + Hc-P coverage fraction > cutoff",
        "silencer_cutoffs": cutoffs,
        "call_definition": f"q_left <= {args.q_cutoff:g}",
        "coverage": coverage,
        "rank_association": association,
        "metrics": metrics.to_dict(orient="records"),
        "pr_metrics": curve_metrics.to_dict(orient="records"),
    }
    (args.out_dir / f"{args.stem}_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float)
    )
    print(metrics.to_string(index=False))
    print(curve_metrics.to_string(index=False))
    print(json.dumps(association, indent=2))


if __name__ == "__main__":
    main()
