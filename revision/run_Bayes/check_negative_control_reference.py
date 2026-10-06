#!/usr/bin/env python3
"""Calibrate the per-cell activity estimators on the negative controls.

    python revision/run_Bayes/check_negative_control_reference.py \
        --fit-dir revision/Bayes_NewData/bayesian \
        --h5ad revision/Data/scdata_07_29_2026_SFv8_low_dose_final_CRE_T7.h5ad \
        --out revision/Bayes_NewData/copy_number/negative_control_reference_check.csv

Each estimator is run on the seven control columns and its cell mean, divided by
the cell type's control reference ``b[d, s]``, is compared with the value it
should recover. One row per (cell type, control):

    target_perdraw          E_d[gamma[d,s,j] / b[d,s]], the parameter-level
                            target. Not 1: only the controls' geometric mean is.
    target_point            E_d[gamma] / E_d[b], the two-stage form, the target
                            for estimators that divide by a point reference.
    R_conjugate_cellmean    cell mean of the Gamma-conjugate normalised activity
                            (``activity_normalized``), divided inside each draw.
    R_moment_meanratio      mean_i(cre_i / E[k_i]) / E_d[b]
    R_moment_ratioofsums    (sum_i cre_i / sum_i E[k_i]) / E_d[b]

and the three ``*_over_target`` ratios: the conjugate one against
``target_perdraw``, the two moment ones against ``target_point``.

The reference is built for every cell type (pooled control T7 threshold 0), so
rows exist even where the published threshold leaves the map NaN; ``eligible``
records whether the cell type passes that threshold.

Only the control columns are reconstructed. Pairs are conditionally independent
given the parameters and the collapse is keyed on ``(cell type, cCRE, t7, cre)``,
so this equals slicing the controls out of the full matrix at a fraction of the
cost. The posterior is thinned once and the same draws feed both the
reconstruction and the targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from baystarrfish._log import log
from baystarrfish.data import CountData
from baystarrfish.inference.copy_number import (
    DEFAULT_CONTROL_T7_THRESHOLD,
    infer_copy_number,
    load_copy_number_draws,
    thin_draws,
)
from baystarrfish.stats.baseline import negative_control_log_baseline

#: Rows at or above this cCRE total enter the second logged calibration summary.
SUMMARY_MIN_CRE_TOTAL = 10

COLUMNS = (
    "subclass", "control", "n_cells", "control_t7_total", "eligible",
    "target_perdraw", "target_point",
    "R_conjugate_cellmean", "R_moment_meanratio", "R_moment_ratioofsums",
    "cre_total", "cre_positive_cells",
    "conj_over_target", "mom_meanratio_over_target", "mom_ratioofsums_over_target",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fit-dir", type=Path, required=True,
                        help="directory holding run_manifest.json and the posterior draws")
    parser.add_argument("--tag", default=None, help="defaults to the manifest tag")
    parser.add_argument("--h5ad", type=Path, default=None,
                        help="input; defaults to the path recorded in the fit manifest")
    parser.add_argument("--out", type=Path, required=True, help="output .csv")
    parser.add_argument("--max-draws", type=int, default=200,
                        help="evenly-spaced posterior draws; 200 matches the "
                             "copy-number matrices this check sits beside")
    parser.add_argument("--t7-threshold", type=float,
                        default=DEFAULT_CONTROL_T7_THRESHOLD,
                        help="pooled control T7 behind the 'eligible' column "
                             "(default: the published 50)")
    parser.add_argument("--chunk", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _group_sum(index: np.ndarray, weights: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(index, weights=np.asarray(weights, dtype=np.float64),
                       minlength=n_groups)


def build_table(
    data: CountData,
    draws: dict[str, np.ndarray],
    controls: Sequence[str],
    *,
    kmax: int,
    level: str,
    infection_model: str,
    t7_threshold: float,
    chunk: int,
) -> pd.DataFrame:
    """The calibration table for ``draws``, which must already be thinned."""
    names = [str(name) for name in data.cre_names]
    missing = [name for name in controls if name not in names]
    if missing:
        raise ValueError(f"control(s) {missing} are not columns of the data")
    data_columns = np.array([names.index(name) for name in controls], dtype=np.int64)
    t7 = np.asarray(data.t7)[:, data_columns].astype(np.int64)
    cre = np.asarray(data.cre)[:, data_columns].astype(np.int64)
    labels = np.asarray(data.class_ if level == "class" else data.subclass).astype(str)

    # Threshold 0: every cell type gets a reference, so the check covers the
    # cell types the published maps leave NaN as well.
    matrix = infer_copy_number(
        t7, cre, labels, draws,
        kmax=kmax, cre_names=list(controls),
        return_activity_normalized=True,
        negative_control_cre=list(controls),
        negative_control_t7_threshold=0.0,
        chunk=chunk, max_draws=None, dtype=np.float32,
        level=level, infection_model=infection_model,
    )

    fit_groups = [str(name) for name in draws["group_names"]]
    fit_cre = [str(name) for name in draws["cre_names"]]
    n_groups = len(fit_groups)
    group_position = {name: index for index, name in enumerate(fit_groups)}
    group_index = np.array([group_position[label] for label in labels], dtype=np.int64)
    control_fit = np.array([fit_cre.index(name) for name in controls], dtype=np.int64)

    t7_totals = np.zeros((n_groups, len(fit_cre)), dtype=np.float64)
    for column, fit_column in enumerate(control_fit):
        t7_totals[:, fit_column] = _group_sum(group_index, t7[:, column], n_groups)
    everywhere = negative_control_log_baseline(
        draws["log_gamma"], control_fit, t7_totals=t7_totals, t7_threshold=0.0
    )
    published = negative_control_log_baseline(
        draws["log_gamma"], control_fit, t7_totals=t7_totals, t7_threshold=t7_threshold
    )

    gamma = np.exp(np.asarray(draws["log_gamma"])[:, :, control_fit])   # (D, S, C)
    reference = everywhere.reference()                                  # (D, S)
    target_perdraw = (gamma / reference[:, :, None]).mean(axis=0)       # (S, C)
    reference_mean = reference.mean(axis=0)                             # (S,)
    target_point = gamma.mean(axis=0) / reference_mean[:, None]         # (S, C)

    n_cells = np.bincount(group_index, minlength=n_groups)
    copies = np.asarray(matrix.copies, dtype=np.float64)
    normalized = np.asarray(matrix.activity_normalized, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("non-finite normalised activity at threshold 0; "
                           "every cell type should have a reference here")
    if np.any(copies <= 0):
        raise RuntimeError("E[k] <= 0 for some pair; the moment estimator is undefined")

    present = np.flatnonzero(n_cells > 0)
    frames = []
    for column, control in enumerate(controls):
        counts = cre[:, column].astype(np.float64)
        conj_sum = _group_sum(group_index, normalized[:, column], n_groups)
        ratio_sum = _group_sum(group_index, counts / copies[:, column], n_groups)
        cre_total = _group_sum(group_index, counts, n_groups)
        copies_total = _group_sum(group_index, copies[:, column], n_groups)
        positive = np.bincount(group_index[counts > 0], minlength=n_groups)

        s = present
        r_conj = conj_sum[s] / n_cells[s]
        r_meanratio = ratio_sum[s] / n_cells[s] / reference_mean[s]
        r_ratioofsums = cre_total[s] / copies_total[s] / reference_mean[s]
        frames.append(pd.DataFrame({
            "group": s,
            "subclass": np.asarray(fit_groups, dtype=object)[s],
            "control": control,
            "n_cells": n_cells[s],
            "control_t7_total": everywhere.control_t7_total[s],
            "eligible": published.eligible[s],
            "target_perdraw": target_perdraw[s, column],
            "target_point": target_point[s, column],
            "R_conjugate_cellmean": r_conj,
            "R_moment_meanratio": r_meanratio,
            "R_moment_ratioofsums": r_ratioofsums,
            "cre_total": cre_total[s].astype(np.int64),
            "cre_positive_cells": positive[s],
            "conj_over_target": r_conj / target_perdraw[s, column],
            "mom_meanratio_over_target": r_meanratio / target_point[s, column],
            "mom_ratioofsums_over_target": r_ratioofsums / target_point[s, column],
        }))
    table = (
        pd.concat(frames, ignore_index=True)
        .assign(_order=lambda f: f["control"].map({c: i for i, c in enumerate(controls)}))
        .sort_values(["group", "_order"], kind="stable")
        .reset_index(drop=True)
    )
    return table.loc[:, list(COLUMNS)]


def log_summary(table: pd.DataFrame) -> None:
    eligible = table[table["eligible"]]
    for label, rows in (
        ("eligible rows", eligible),
        (f"eligible rows, cre_total >= {SUMMARY_MIN_CRE_TOTAL}",
         eligible[eligible["cre_total"] >= SUMMARY_MIN_CRE_TOTAL]),
    ):
        if rows.empty:
            log(f"[check] {label}: none")
            continue
        log(
            f"[check] {label} (n={len(rows)}): median conj/target "
            f"{rows['conj_over_target'].median():.3f}, moment ratio-of-sums/target "
            f"{rows['mom_ratioofsums_over_target'].median():.3f}, moment "
            f"mean-ratio/target {rows['mom_meanratio_over_target'].median():.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out} (pass --overwrite)")

    manifest = json.loads((args.fit_dir / "run_manifest.json").read_text())
    config = manifest.get("config", {})
    controls = [str(name) for name in config.get("annotated_negative_control_cre") or []]
    if not controls:
        raise SystemExit(f"{args.fit_dir} manifest records no annotated_negative_control_cre")
    kmax = config.get("kmax")
    if kmax is None:
        raise SystemExit(f"{args.fit_dir} manifest records no kmax")

    h5ad = args.h5ad or Path(config["input"]["path"])
    data = CountData.from_h5ad(
        h5ad,
        section=manifest.get("section", "all"),
        negative_control_mode=manifest.get("negative_control_mode", "ordinary"),
    )
    draws, _ = load_copy_number_draws(args.fit_dir, args.tag)
    draws = thin_draws(draws, args.max_draws, verbose=True)

    table = build_table(
        data, draws, controls,
        kmax=int(kmax),
        level=config.get("level", "subclass"),
        infection_model=config.get("infection_model", "copy_number"),
        t7_threshold=args.t7_threshold,
        chunk=args.chunk,
    )
    log_summary(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    log(f"[check] wrote {len(table):,} rows "
        f"({table['subclass'].nunique()} cell types x {len(controls)} controls) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
