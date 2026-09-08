"""The negative-control contrast that turns a posterior into calls.

A fitted ``log_gamma`` is not directly a claim about activity: it is on an
arbitrary scale set jointly by the infection rate and the library abundance.
What is interpretable is the *contrast* between a target cCRE and the negative
controls in the same cell type, evaluated draw by draw so the uncertainty in
both terms is carried through::

    contrast_d = log_gamma[d, s, j] - mean_{j' in controls} log_gamma[d, s, j']
                 - effect_threshold

    p_right = P(contrast <= 0)          # posterior tail probability
    q_right = BH(p_right)

The mirror-image question -- is this cCRE *less* active than its controls, i.e.
a silencer -- is the same contrast read from the other side. With
``alternative="less"`` the threshold is added rather than subtracted and the
tail is taken on the other side::

    contrast_d = log_gamma[d, s, j] - mean_{j' in controls} log_gamma[d, s, j']
                 + effect_threshold

    p_left = P(contrast >= 0)
    q_left = BH(p_left)

Because the contrast is formed inside each posterior draw, the control mean's
own uncertainty is subtracted correctly rather than being treated as a fixed
offset.

Eligibility is a T7 filter, not a cCRE filter: a (cell type, cCRE) pair with too
little constitutive signal has no evidence either way, and including it would
report the prior as a result. Cell types whose controls collectively fail the
filter are dropped entirely -- there is no reference to contrast against.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .baseline import negative_control_log_baseline
from .fdr import bh_fdr

__all__ = ["negative_control_test"]


def negative_control_test(
    log_gamma: np.ndarray,
    groups: np.ndarray,
    cre_names: np.ndarray,
    target_indices: np.ndarray,
    control_indices: np.ndarray,
    t7_totals: np.ndarray,
    group_classes: np.ndarray,
    group_cell_counts: np.ndarray,
    t7_threshold: float,
    effect_threshold: float,
    individual_control_t7_threshold: float | None,
    method: str,
    control_sd_multiplier: float = 0.0,
    alternative: Literal["greater", "less"] = "greater",
) -> pd.DataFrame:
    """Contrast every eligible target cCRE against the negative-control mean.

    Parameters
    ----------
    log_gamma : (n_draws, n_groups, n_cre)
        Posterior draws of log activity.
    target_indices, control_indices : int arrays
        Columns of ``log_gamma`` to test and to use as the reference.
    t7_totals : (n_groups, n_cre)
        Per-pair constitutive-channel totals driving the eligibility filter.
    t7_threshold
        Minimum T7 total for a target pair, and for the pooled control total
        when ``individual_control_t7_threshold`` is None.
    effect_threshold
        Minimum log-fold effect over the control reference to call; shifts the
        null rather than filtering after the fact.
    individual_control_t7_threshold
        If given, controls are filtered individually at this threshold and the
        reference is the mean over the survivors, instead of requiring the pooled
        control total to pass ``t7_threshold``.
    control_sd_multiplier
        Raise the reference by ``k`` control standard deviations (per draw) for a
        stricter null. Requires at least two surviving controls.
    alternative
        ``"greater"`` (default) tests for activity above the control reference and
        emits ``p_right`` / ``q_right``; ``"less"`` tests for activity below it --
        the silencer direction -- and emits ``p_left`` / ``q_left``.

    Returns a tidy frame with one row per tested (cell type, cCRE) pair.
    """
    if control_sd_multiplier < 0:
        raise ValueError("control_sd_multiplier must be non-negative")
    if alternative not in ("greater", "less"):
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")
    right_tail = alternative == "greater"
    # The reference is built by the module the per-cell normalised activity also
    # uses, so a map and the table it accompanies cannot disagree about what
    # "background" means.
    baseline = negative_control_log_baseline(
        log_gamma,
        control_indices,
        t7_totals=t7_totals,
        t7_threshold=t7_threshold,
        individual_control_t7_threshold=individual_control_t7_threshold,
        control_sd_multiplier=control_sd_multiplier,
    )
    records = []
    for group_idx, group in enumerate(groups):
        if not baseline.eligible[group_idx]:
            continue
        selected_control_indices = baseline.control_indices[group_idx]
        mean_control_draws = baseline.log_mean[:, group_idx]
        control_sd_draws = baseline.log_sd[:, group_idx]
        control_reference_draws = baseline.log_reference[:, group_idx]
        control_t7_total = float(baseline.control_t7_total[group_idx])
        eligible = t7_totals[group_idx, target_indices] >= t7_threshold
        selected_indices = target_indices[eligible]
        if len(selected_indices) == 0:
            continue

        target_draws = log_gamma[:, group_idx, selected_indices].astype(
            np.float64, copy=False
        )
        # The threshold shifts the null towards the alternative being tested, so it
        # is subtracted for a right tail and added for a left one.
        signed_threshold = effect_threshold if right_tail else -effect_threshold
        contrasts = target_draws - control_reference_draws[:, None] - signed_threshold
        contrast_lo, contrast_hi = np.quantile(contrasts, [0.05, 0.95], axis=0)
        tail_probability = (
            (contrasts <= 0.0).mean(axis=0)
            if right_tail
            else (contrasts >= 0.0).mean(axis=0)
        )
        directional = (
            {
                "posterior_probability_above_control_reference": (
                    contrasts > 0.0
                ).mean(axis=0),
                # Backward-compatible aliases used by existing comparison code.
                "posterior_probability_above_mean_control": (contrasts > 0.0).mean(axis=0),
                "p_right": tail_probability,
            }
            if right_tail
            else {
                "posterior_probability_below_control_reference": (
                    contrasts < 0.0
                ).mean(axis=0),
                "p_left": tail_probability,
            }
        )
        records.append(
            pd.DataFrame(
                {
                    "t7_threshold": float(t7_threshold),
                    "method": method,
                    "group": group,
                    "class": group_classes[group_idx],
                    "cre": cre_names[selected_indices],
                    "n_cells": int(group_cell_counts[group_idx]),
                    "target_t7_total": t7_totals[group_idx, selected_indices],
                    "negative_control_t7_total": control_t7_total,
                    "n_negative_controls": len(selected_control_indices),
                    "negative_controls_used": ",".join(
                        cre_names[selected_control_indices]
                    ),
                    "control_sd_multiplier": float(control_sd_multiplier),
                    "activity_mean": target_draws.mean(axis=0),
                    "mean_negative_control_activity_mean": float(
                        mean_control_draws.mean()
                    ),
                    "negative_control_activity_sd_mean": float(
                        control_sd_draws.mean()
                    ),
                    "control_reference_activity_mean": float(
                        control_reference_draws.mean()
                    ),
                    "effect_vs_control_reference_mean": contrasts.mean(axis=0),
                    "effect_vs_control_reference_lo90": contrast_lo,
                    "effect_vs_control_reference_hi90": contrast_hi,
                    "effect_vs_mean_control_mean": contrasts.mean(axis=0),
                    "effect_vs_mean_control_lo90": contrast_lo,
                    "effect_vs_mean_control_hi90": contrast_hi,
                    **directional,
                }
            )
        )
    if not records:
        raise ValueError("No cCRE-cell-type pairs passed the T7 filters")
    output = pd.concat(records, ignore_index=True)
    p_column = "p_right" if right_tail else "p_left"
    q_column = "q_right" if right_tail else "q_left"
    output[q_column] = bh_fdr(output[p_column].to_numpy(float))
    return output
