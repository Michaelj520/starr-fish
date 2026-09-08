#!/usr/bin/env python
"""Left-tail test: is a cCRE *less* active than its negative-control reference?

Mirror image of the production right-tail call in
``revision/bayesian_vs_fold_change/code/test_mean_negative_control_activity.py``.
The contrast is formed draw-wise against the mean of the ordinary negative
controls in the same cell type; the left tail ``p_left = P(contrast >= 0)`` is
the posterior evidence *against* silencing, and BH over all eligible pairs turns
it into ``q_left``.

Posterior: ``revision/Bayes_OldData/bayesian`` (production joint CRE/T7
copy-number + dropout fit on the original 5/28 dataset).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from baystarrfish.data import POOLED_NEGATIVE_CONTROL_NAME, read_grouped_counts
from baystarrfish.data.paths import default_h5ad, repo_root
from baystarrfish.stats import negative_control_test

METHOD: Final[str] = "Joint+dropout mean controls (left tail)"
N_ORDINARY_CONTROLS: Final[int] = 7


def read_single_column(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pd.read_csv(path).iloc[:, 0].astype(str).tolist()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bayes-dir", type=Path,
                        default=root / "revision/Bayes_OldData/bayesian")
    parser.add_argument("--h5ad", type=Path, default=default_h5ad())
    parser.add_argument("--t7-thresholds", type=float, nargs="+", default=[50.0],
                        help="T7 eligibility thresholds; BH is applied within each.")
    parser.add_argument("--effect-threshold", type=float, default=0.0,
                        help="Minimum log-fold *depletion* below the control "
                             "reference required to call (shifts the null).")
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path,
                        default=root / "revision/silencer/results")
    parser.add_argument("--stem", default="silencer_left_tail_tests")
    return parser.parse_args()


def run_tests(
    bayes_dir: Path,
    h5ad: Path,
    t7_thresholds: Sequence[float],
    effect_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = json.loads((bayes_dir / "run_manifest.json").read_text())
    posterior_path = bayes_dir / f"{manifest['tag']}_posterior_samples.npz"
    negative_controls = read_single_column(bayes_dir / "negative_controls.csv")
    blacklist = read_single_column(bayes_dir / "cre_blacklist.csv")

    with np.load(posterior_path, allow_pickle=True) as posterior:
        groups = posterior["group_names"].astype(str)
        all_cre_names = posterior["cre_names"].astype(str)
        ordinary = all_cre_names != POOLED_NEGATIVE_CONTROL_NAME
        cre_names = all_cre_names[ordinary]
        log_gamma = posterior["log_gamma"][:, :, ordinary].astype(np.float32)

    control_indices = np.flatnonzero(np.isin(cre_names, negative_controls))
    if len(control_indices) != N_ORDINARY_CONTROLS:
        raise ValueError(
            f"expected {N_ORDINARY_CONTROLS} ordinary negative controls, "
            f"found {len(control_indices)}"
        )
    target_indices = np.flatnonzero(
        ~np.isin(cre_names, negative_controls) & ~np.isin(cre_names, blacklist)
    )
    counts = read_grouped_counts(h5ad, groups, cre_names, keys=("T7CRE",))
    frames = [
        negative_control_test(
            log_gamma,
            groups,
            cre_names,
            target_indices,
            control_indices,
            counts.totals["T7CRE"],
            counts.group_classes,
            counts.group_cell_counts,
            float(threshold),
            effect_threshold,
            None,
            METHOD,
            0.0,
            alternative="less",
        )
        for threshold in t7_thresholds
    ]
    tests = pd.concat(frames, ignore_index=True)
    provenance = {
        "posterior": str(posterior_path),
        "h5ad": str(h5ad),
        "n_draws": int(log_gamma.shape[0]),
        "n_groups": int(len(groups)),
        "n_targets": int(len(target_indices)),
        "negative_controls": negative_controls,
        "cre_blacklist": blacklist,
    }
    return tests, provenance


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tests, provenance = run_tests(
        args.bayes_dir, args.h5ad, args.t7_thresholds, args.effect_threshold
    )
    tests["significant_q"] = tests["q_left"].le(args.q_cutoff)
    table_path = args.out_dir / f"{args.stem}.csv.gz"
    tests.to_csv(table_path, index=False)

    per_threshold = {
        f"{threshold:g}": {
            "eligible_tests": int(len(frame)),
            "significant_tests": int(frame["significant_q"].sum()),
            "significant_ccres": int(frame.loc[frame["significant_q"], "cre"].nunique()),
            "significant_cell_types": int(
                frame.loc[frame["significant_q"], "group"].nunique()
            ),
            "min_q_left": float(frame["q_left"].min()),
            "median_effect": float(frame["effect_vs_control_reference_mean"].median()),
        }
        for threshold, frame in tests.groupby("t7_threshold", sort=True)
    }
    manifest = {
        "method": METHOD,
        "alternative": "less (activity below the negative-control mean)",
        "contrast_definition": (
            "target log_gamma minus the draw-wise mean of the ordinary negative "
            "control log_gamma, plus the effect threshold"
        ),
        "p_left_definition": "posterior fraction of draw-wise contrasts >= 0",
        "multiple_testing": "BH across all eligible target-cell-type pairs, "
                            "within each T7 threshold",
        "t7_thresholds": list(map(float, args.t7_thresholds)),
        "effect_threshold": float(args.effect_threshold),
        "q_cutoff": float(args.q_cutoff),
        "counts": per_threshold,
        "outputs": {"tests": str(table_path)},
        **provenance,
    }
    manifest_path = args.out_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(per_threshold, indent=2))
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
