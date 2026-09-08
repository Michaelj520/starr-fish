#!/usr/bin/env python
"""Two-experiment silencer benchmark: agree first, then score against ChromHMM.

Both arms are the same Bayesian left-tail test from ``test_silencer_left_tail.py``
-- posterior ``p_left = P(activity >= control mean)``, BH over the eligible pairs
-- run on the two datasets: ``revision/Bayes_OldData`` (original 5/28 data) and
``revision/Bayes_NewData`` (7/29 SFv8 low-dose data). Same model, same null, same
T7 eligibility, so the arms differ only in the experiment behind them.

Pipeline:

1. restrict to (cell type, cCRE) pairs eligible in **both** experiments;
2. keep the pairs whose *call* agrees, i.e. ``q_left <= q_cutoff`` in both arms
   or in neither -- on that subset "significant in both" and "significant in
   either" are the same call, so the consistency filter needs no further choice;
3. compute precision/recall against the annotation, positives being a repressive
   (``Chr-R`` + ``Hc-P``) fraction above ``--silencer-cutoff``.

Each arm is scored on **its own** annotated eligible pairs -- the old-data arm on
everything eligible in the old data, the new-data arm on everything eligible in
the new data -- and only the consistent arm is restricted to the pairs eligible in
both. Prevalence therefore differs per arm, so read each bar against its own
dashed baseline; ``precision`` alone is not comparable across bars.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve

from baystarrfish.data.paths import repo_root
from baystarrfish.stats import bh_fdr
from silencer_truth import load_silencer_truth

PANEL_SIDE_INCHES: Final[float] = 4.0
ARM_A: Final[str] = "Bayesian, original data"
ARM_B: Final[str] = "Bayesian, new low-dose data"
CONSISTENT: Final[str] = "Consistent (both arms agree)"
ARM_COLORS: Final[dict[str, str]] = {
    ARM_A: "#4c78a8",
    ARM_B: "#f58518",
    CONSISTENT: "#54a24b",
}


def parse_args() -> argparse.Namespace:
    root = repo_root()
    results = root / "revision/silencer/results"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests-a", type=Path,
                        default=results / "silencer_left_tail_tests.csv.gz",
                        help="Left-tail tests of the first experiment.")
    parser.add_argument("--tests-b", type=Path,
                        default=results / "silencer_left_tail_tests_newdata.csv.gz",
                        help="Left-tail tests of the second experiment.")
    parser.add_argument("--annotation-dir", type=Path, default=results)
    parser.add_argument("--out-dir", type=Path, default=results)
    parser.add_argument("--figures-dir", type=Path,
                        default=root / "revision/silencer/figures")
    parser.add_argument("--t7-threshold", type=float, default=50.0,
                        help="T7 eligibility used for both arms.")
    parser.add_argument("--silencer-cutoff", type=float, default=0.1,
                        help="A positive is a repressive fraction > this cutoff.")
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--stem", default="silencer_two_experiment")
    return parser.parse_args()


def load_arm(tests: Path, t7_threshold: float) -> pd.DataFrame:
    """One arm's eligible pairs at a T7 threshold, plus a right-tail control q."""
    frame = pd.read_csv(tests)
    frame = frame[np.isclose(frame["t7_threshold"].to_numpy(float), t7_threshold)]
    if frame.empty:
        raise ValueError(f"{tests} has no rows at T7 >= {t7_threshold:g}")
    frame = frame[["group", "cre", "p_left", "q_left",
                   "effect_vs_control_reference_mean", "target_t7_total",
                   "n_cells"]].copy()
    frame[["group", "cre"]] = frame[["group", "cre"]].astype(str)
    # The right tail is the same posterior read from the other side (p_right =
    # 1 - p_left, verified exactly against the shipped right-tail table), BH'd
    # over the same eligible set. It serves as the positive control below.
    frame["q_right"] = bh_fdr(1.0 - frame["p_left"].to_numpy(float))
    return frame


def load_arms(
    tests_a: Path, tests_b: Path, t7_threshold: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Inner-join the two arms on (group, cre); one row per commonly tested pair."""
    arm_a = load_arm(tests_a, t7_threshold)
    arm_b = load_arm(tests_b, t7_threshold)
    merged = arm_a.merge(
        arm_b, on=["group", "cre"], suffixes=("_a", "_b"),
        how="inner", validate="one_to_one",
    )
    provenance = {
        "t7_threshold": float(t7_threshold),
        "evaluation_sets": {
            "arm A": "annotated pairs eligible in the old data",
            "arm B": "annotated pairs eligible in the new data",
            "consistent": "annotated pairs eligible in both, with agreeing calls",
        },
        "arm_a_tests": str(tests_a),
        "arm_a_eligible_pairs": int(len(arm_a)),
        "arm_a_cell_types": int(arm_a["group"].nunique()),
        "arm_b_tests": str(tests_b),
        "arm_b_eligible_pairs": int(len(arm_b)),
        "arm_b_cell_types": int(arm_b["group"].nunique()),
        "pairs_eligible_in_both": int(len(merged)),
        "cell_types_in_both": int(merged["group"].nunique()),
        "ccres_in_both": int(merged["cre"].nunique()),
    }
    return arm_a, arm_b, merged, provenance


def add_annotation(
    frame: pd.DataFrame, annotation_dir: Path, silencer_cutoff: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    truth = load_silencer_truth(annotation_dir)
    stacked = {
        name: (
            matrix.rename_axis(index="group", columns="cre")
            .stack(future_stack=True)
            .rename(name)
        )
        for name, matrix in (
            ("repressive_fraction", truth.repressive_fraction),
            ("nd_fraction", truth.nd_fraction),
        )
    }
    index = pd.MultiIndex.from_frame(frame[["group", "cre"]])
    out = frame.copy()
    for name, series in stacked.items():
        out[name] = series.reindex(index).to_numpy()
    annotated = out[np.isfinite(out["repressive_fraction"])].copy()
    annotated["silencer"] = annotated["repressive_fraction"] > silencer_cutoff
    coverage = {
        "silencer_cutoff": float(silencer_cutoff),
        "pairs_with_annotation": int(len(annotated)),
        "cell_types_annotated": int(annotated["group"].nunique()),
        "ccres_annotated": int(annotated["cre"].nunique()),
        "annotated_silencers": int(annotated["silencer"].sum()),
        "mean_nd_fraction": float(annotated["nd_fraction"].mean()),
    }
    return annotated, coverage


def concordance(frame: pd.DataFrame, q_cutoff: float) -> dict[str, object]:
    """How much do the two experiments agree, before any annotation is involved?"""
    a = frame["q_left_a"].le(q_cutoff).to_numpy(bool)
    b = frame["q_left_b"].le(q_cutoff).to_numpy(bool)
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    n = int(len(frame))
    observed = (both + neither) / n if n else np.nan
    expected = (
        (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())) if n else np.nan
    )
    odds, pvalue = (
        fisher_exact([[both, only_a], [only_b, neither]], alternative="greater")
        if n
        else (np.nan, np.nan)
    )
    effects = spearmanr(
        frame["effect_vs_control_reference_mean_a"],
        frame["effect_vs_control_reference_mean_b"],
    )
    return {
        "pairs": n,
        "significant_both": both,
        "significant_a_only": only_a,
        "significant_b_only": only_b,
        "significant_neither": neither,
        "significant_arm_a": int(a.sum()),
        "significant_arm_b": int(b.sum()),
        "agreement": observed,
        "agreement_expected_by_chance": expected,
        "cohen_kappa": (observed - expected) / (1 - expected)
        if np.isfinite(expected) and expected < 1
        else np.nan,
        "call_fisher_oddsratio": odds,
        "call_fisher_p": pvalue,
        "effect_spearman": float(effects.statistic),
        "effect_spearman_p": float(effects.pvalue),
    }


def right_tail_control(frame: pd.DataFrame, q_cutoff: float) -> dict[str, object]:
    """Do the arms agree in the *enhancer* direction on the same pairs?"""
    a = frame["q_right_a"].le(q_cutoff).to_numpy(bool)
    b = frame["q_right_b"].le(q_cutoff).to_numpy(bool)
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    odds, pvalue = fisher_exact([[both, only_a], [only_b, neither]],
                                alternative="greater")
    return {
        "pairs": int(len(frame)),
        "significant_both": both,
        "significant_a_only": only_a,
        "significant_b_only": only_b,
        "significant_neither": neither,
        "fisher_oddsratio": float(odds),
        "fisher_p": float(pvalue),
    }


def confusion(
    frame: pd.DataFrame, calls: np.ndarray, arm: str, q_cutoff: float
) -> dict[str, object]:
    positive = frame["silencer"].to_numpy(bool)
    tp = int((calls & positive).sum())
    n_calls = int(calls.sum())
    n_positive = int(positive.sum())
    n_tested = int(len(frame))
    fp = n_calls - tp
    fn = n_positive - tp
    tn = n_tested - tp - fp - fn
    precision = tp / n_calls if n_calls else np.nan
    recall = tp / n_positive if n_positive else np.nan
    prevalence = n_positive / n_tested if n_tested else np.nan
    odds, pvalue = (
        fisher_exact([[tp, fp], [fn, tn]], alternative="greater")
        if n_positive and n_calls
        else (np.nan, np.nan)
    )
    return {
        "arm": arm,
        "q_cutoff": q_cutoff,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "significant": n_calls,
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
            precision / prevalence if prevalence and np.isfinite(precision) else np.nan
        ),
        "fisher_oddsratio": odds,
        "fisher_p": pvalue,
    }


@dataclass(frozen=True)
class ArmEvaluation:
    """One bar: which pairs it is scored on, which of them it calls, how it ranks."""

    name: str
    frame: pd.DataFrame
    calls: np.ndarray
    scores: np.ndarray


def arm_evaluations(
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    consistent: pd.DataFrame,
    q_cutoff: float,
) -> list[ArmEvaluation]:
    """Each single arm on its own eligible pairs; the consistent arm on the overlap.

    The consistent arm ranks by the mean depletion of the two arms; a single arm
    ranks by its own.
    """
    consistent_scores = -0.5 * (
        consistent["effect_vs_control_reference_mean_a"].to_numpy(float)
        + consistent["effect_vs_control_reference_mean_b"].to_numpy(float)
    )
    return [
        ArmEvaluation(
            ARM_A, arm_a, arm_a["q_left"].le(q_cutoff).to_numpy(bool),
            -arm_a["effect_vs_control_reference_mean"].to_numpy(float),
        ),
        ArmEvaluation(
            ARM_B, arm_b, arm_b["q_left"].le(q_cutoff).to_numpy(bool),
            -arm_b["effect_vs_control_reference_mean"].to_numpy(float),
        ),
        ArmEvaluation(
            CONSISTENT, consistent,
            consistent["q_left_a"].le(q_cutoff).to_numpy(bool), consistent_scores,
        ),
    ]


def benchmark(arms: list[ArmEvaluation], q_cutoff: float) -> pd.DataFrame:
    return pd.DataFrame(
        [confusion(arm.frame, arm.calls, arm.name, q_cutoff) for arm in arms]
    )


def pr_curves(arms: list[ArmEvaluation]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_frames, metric_rows = [], []
    for evaluation in arms:
        arm, data, scores = evaluation.name, evaluation.frame, evaluation.scores
        labels = data["silencer"].to_numpy(bool)
        valid = np.isfinite(scores)
        labels, scores = labels[valid], scores[valid]
        if not labels.any() or labels.all():
            continue
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        curve_frames.append(
            pd.DataFrame(
                {
                    "arm": arm,
                    "precision": precision,
                    "recall": recall,
                    "score_threshold": np.r_[thresholds, np.nan],
                }
            )
        )
        average_precision = float(average_precision_score(labels, scores))
        metric_rows.append(
            {
                "arm": arm,
                "average_precision": average_precision,
                "prevalence": float(labels.mean()),
                "ap_over_prevalence": average_precision / float(labels.mean()),
                "tested": int(labels.size),
                "annotated_silencers": int(labels.sum()),
            }
        )
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(metric_rows)


def plot_concordance(
    frame: pd.DataFrame, stats: dict[str, object], q_cutoff: float, path: Path
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * PANEL_SIDE_INCHES, PANEL_SIDE_INCHES),
        constrained_layout=True,
    )
    a = frame["q_left_a"].le(q_cutoff).to_numpy(bool)
    b = frame["q_left_b"].le(q_cutoff).to_numpy(bool)

    ax = axes[0]
    for mask, label, color in (
        (a & b, "significant in both", ARM_COLORS[CONSISTENT]),
        (a & ~b, "original data only", ARM_COLORS[ARM_A]),
        (~a & b, "new data only", ARM_COLORS[ARM_B]),
        (~a & ~b, "neither", "#bab0ac"),
    ):
        ax.scatter(
            frame.loc[mask, "effect_vs_control_reference_mean_a"],
            frame.loc[mask, "effect_vs_control_reference_mean_b"],
            s=10, alpha=0.7, linewidths=0, color=color,
            label=f"{label} (n={int(mask.sum())})",
        )
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("original-data effect vs control (log)")
    ax.set_ylabel("new-data effect vs control (log)")
    ax.set_title(f"effect agreement (rho = {stats['effect_spearman']:.2f})")
    ax.legend(fontsize=6, frameon=False, loc="upper left")

    ax = axes[1]
    table = np.array(
        [
            [stats["significant_both"], stats["significant_a_only"]],
            [stats["significant_b_only"], stats["significant_neither"]],
        ],
        dtype=float,
    )
    image = ax.imshow(table, cmap="Blues")
    for (row, col), value in np.ndenumerate(table):
        ax.text(col, row, f"{value:.0f}", ha="center", va="center",
                color="white" if value > table.max() / 2 else "black")
    ax.set_xticks([0, 1], ["new sig", "new ns"])
    ax.set_yticks([0, 1], ["original sig", "original ns"])
    ax.set_title(
        f"call agreement {stats['agreement']:.2f} "
        f"(kappa {stats['cohen_kappa']:.2f}, p = {stats['call_fisher_p']:.2g})",
        fontsize=9,
    )
    fig.colorbar(image, ax=ax, label="pairs")

    ax = axes[2]
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(frame["q_left_a"], bins=bins, histtype="step", density=True,
            color=ARM_COLORS[ARM_A], linewidth=1.5, label="original data q_left")
    ax.hist(frame["q_left_b"], bins=bins, histtype="step", density=True,
            color=ARM_COLORS[ARM_B], linewidth=1.5, label="new data q_left")
    ax.axvline(q_cutoff, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("BH q-value (left tail)")
    ax.set_ylabel("density")
    ax.set_title("q-values on commonly tested pairs")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(
        f"Left-tail agreement between the two experiments, "
        f"{stats['pairs']} commonly eligible pairs",
        fontsize=10,
    )
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(
    metrics: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    silencer_cutoff: float,
    q_cutoff: float,
    path: Path,
) -> None:
    arms = metrics["arm"].tolist()
    positions = np.arange(len(arms))
    colors = [ARM_COLORS[arm] for arm in arms]
    fig, axes = plt.subplots(
        1, 4, figsize=(4 * PANEL_SIDE_INCHES, PANEL_SIDE_INCHES),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.bar(positions, metrics["precision"], color=colors, width=0.6)
    for x, y, naive in zip(positions, metrics["precision"], metrics["naive_precision"]):
        ax.plot([x - 0.3, x + 0.3], [naive, naive], color="k", linestyle="--",
                linewidth=1)
    for x, y, tp, n in zip(positions, metrics["precision"], metrics["TP"],
                           metrics["significant"]):
        ax.text(x, y, f"{y:.2f}\n{tp:.0f}/{n:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("precision")
    ax.set_title("precision vs prevalence (dashed)")

    ax = axes[1]
    ax.bar(positions, metrics["recall"], color=colors, width=0.6)
    for x, y, tp, n in zip(positions, metrics["recall"], metrics["TP"],
                           metrics["annotated_silencers"]):
        ax.text(x, y, f"{y:.2f}\n{tp:.0f}/{n:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("recall")
    ax.set_title("recall")

    ax = axes[2]
    ax.bar(positions, metrics["enrichment_over_naive"], color=colors, width=0.6)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    for x, y, p in zip(positions, metrics["enrichment_over_naive"], metrics["fisher_p"]):
        if np.isfinite(y):
            ax.text(x, y, f"{y:.2f}\np={p:.2g}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("precision / prevalence")
    ax.set_title("enrichment over chance")

    ax = axes[3]
    ordered = curve_metrics.set_index("arm").reindex(arms)
    ax.bar(positions, ordered["ap_over_prevalence"], color=colors, width=0.6)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    for x, y, ap in zip(positions, ordered["ap_over_prevalence"],
                        ordered["average_precision"]):
        if np.isfinite(y):
            ax.text(x, y, f"{y:.2f}\nAP {ap:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("average precision / chance")
    ax.set_title("ranking by depletion")

    # Only the ratio panels get the reference line at 1 inside their range; the
    # precision/recall panels scale to their own data or they read as empty.
    panels = (
        (metrics["precision"], 0.0),
        (metrics["recall"], 0.0),
        (metrics["enrichment_over_naive"], 1.15),
        (ordered["ap_over_prevalence"], 1.15),
    )
    for ax, (values, floor) in zip(axes, panels):
        ax.set_xticks(positions)
        ax.set_xticklabels([arm.replace(", ", ",\n").replace(" (", "\n(")
                            for arm in arms], fontsize=7)
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        ax.set_ylim(0, max(floor, float(finite.max()) * 1.4) if finite.size else 1.0)

    fig.suptitle(
        f"Silencer calls (q_left <= {q_cutoff:g}) vs ChromHMM annotation "
        f"(repressive fraction > {silencer_cutoff:g}); each single arm on its own "
        "eligible pairs, consistent arm on the overlap",
        fontsize=9,
    )
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    curves: pd.DataFrame, curve_metrics: pd.DataFrame, silencer_cutoff: float, path: Path
) -> None:
    fig, ax = plt.subplots(
        figsize=(PANEL_SIDE_INCHES * 1.4, PANEL_SIDE_INCHES), constrained_layout=True
    )
    for arm in curve_metrics["arm"]:
        curve = curves[curves["arm"].eq(arm)]
        row = curve_metrics[curve_metrics["arm"].eq(arm)].iloc[0]
        ax.step(curve["recall"], curve["precision"], where="post",
                color=ARM_COLORS[arm], linewidth=1.5,
                label=f"{arm}: AP {row['average_precision']:.3f} "
                      f"(chance {row['prevalence']:.3f})")
        ax.axhline(row["prevalence"], color=ARM_COLORS[arm], linestyle="--",
                   linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(
        f"Ranking by depletion, annotation cutoff > {silencer_cutoff:g}", fontsize=9
    )
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    arm_a, arm_b, merged, provenance = load_arms(args.tests_a, args.tests_b,
                                                 args.t7_threshold)
    call_stats = concordance(merged, args.q_cutoff)
    right_tail = right_tail_control(merged, args.q_cutoff)
    # Each single arm keeps its own eligible pairs; only the consistent arm is
    # restricted to the overlap, so every bar has its own prevalence.
    annotated_a, coverage_a = add_annotation(arm_a, args.annotation_dir,
                                             args.silencer_cutoff)
    annotated_b, coverage_b = add_annotation(arm_b, args.annotation_dir,
                                             args.silencer_cutoff)
    annotated, coverage = add_annotation(merged, args.annotation_dir,
                                         args.silencer_cutoff)
    agree = annotated["q_left_a"].le(args.q_cutoff).eq(
        annotated["q_left_b"].le(args.q_cutoff)
    )
    consistent = annotated[agree].copy()
    annotated_stats = concordance(annotated, args.q_cutoff)

    arms = arm_evaluations(annotated_a, annotated_b, consistent, args.q_cutoff)
    metrics = benchmark(arms, args.q_cutoff)
    curves, curve_metrics = pr_curves(arms)

    annotated.to_csv(args.out_dir / f"{args.stem}_tests_annotated.csv.gz", index=False,
                     float_format="%.6g")
    metrics.to_csv(args.out_dir / f"{args.stem}_precision_recall.csv", index=False,
                   float_format="%.6g")
    curves.to_csv(args.out_dir / f"{args.stem}_pr_curves.csv.gz", index=False,
                  float_format="%.6g")
    curve_metrics.to_csv(args.out_dir / f"{args.stem}_pr_metrics.csv", index=False,
                         float_format="%.6g")

    plot_concordance(annotated, annotated_stats, args.q_cutoff,
                     args.figures_dir / f"{args.stem}_concordance.pdf")
    plot_metrics(metrics, curve_metrics, args.silencer_cutoff, args.q_cutoff,
                 args.figures_dir / f"{args.stem}_precision_recall.pdf")
    plot_pr_curves(curves, curve_metrics, args.silencer_cutoff,
                   args.figures_dir / f"{args.stem}_pr_curves.pdf")

    manifest = {
        "arms": {
            ARM_A: "posterior left tail on revision/Bayes_OldData, T7 >= "
                   f"{args.t7_threshold:g}, BH over eligible pairs",
            ARM_B: "posterior left tail on revision/Bayes_NewData, T7 >= "
                   f"{args.t7_threshold:g}, BH over eligible pairs",
        },
        "consistency_filter": (
            f"q_left <= {args.q_cutoff:g} in both arms or in neither; on that subset "
            "'significant in both' and 'significant in either' coincide"
        ),
        "silencer_definition": f"Chr-R + Hc-P fraction > {args.silencer_cutoff:g}",
        "provenance": provenance,
        "concordance_all_common_pairs": call_stats,
        "right_tail_positive_control": right_tail,
        "concordance_annotated_pairs": annotated_stats,
        "annotation_coverage": {
            ARM_A: coverage_a,
            ARM_B: coverage_b,
            "eligible in both": coverage,
        },
        "consistent_pairs": int(len(consistent)),
        "metrics": metrics.to_dict(orient="records"),
        "pr_metrics": curve_metrics.to_dict(orient="records"),
    }
    (args.out_dir / f"{args.stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float)
    )
    print(json.dumps({"provenance": provenance, "concordance": call_stats,
                      "right_tail_control": right_tail,
                      "annotation": {ARM_A: coverage_a, ARM_B: coverage_b,
                                     "eligible in both": coverage},
                      "consistent_pairs": int(len(consistent))},
                     indent=2, default=float))
    print(metrics.to_string(index=False))
    print(curve_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
