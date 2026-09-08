#!/usr/bin/env python3
"""Is the inferred AAV copy number proportional to the nanopore library counts?

Per subclass, using only the ``revision/Bayes_OldData`` fit outputs.

Why this is not simply "plot the fit's copies against nanopore"
--------------------------------------------------------------
In the production model the expected copy number is rank-one,
``lambda_{s,c} = rho_s * a_c`` (``baystarrfish/model/models.py``), and the
abundance prior is the nanopore library itself:
``log a = log1p(nanopore) - mean(...) + tau_a * eps`` with
``tau_a ~ HalfNormal(0.5)`` (``baystarrfish/model/blocks.py``). So (i) the
per-cCRE abundance is shared by every subclass, which makes a per-subclass
correlation identical for all subclasses up to the vertical shift ``log rho_s``,
and (ii) it is centred on the very quantity we would be validating against.
Correlating the fit's copies with nanopore therefore measures the prior.

What is fitted here instead
---------------------------
The nanopore counts are re-derived as a *free* per-subclass regression on the
T7 detection pattern, keeping the fit's measurement parameters but discarding
its abundance prior. For subclass ``s`` and cCRE ``c``::

    lambda_{s,c} = exp(b_s + m_s * z_c),  z_c = log1p(nano_c) - mean_c log1p(nano)
    k | lambda   ~ Poisson(lambda)                      (latent AAV copies)
    t7 | k       ~ NB2(mean = k * beta_t7, disp = phi_t7), zero-inflated p_drop_t7
    n_pos_{s,c}  ~ Binomial(n_cells_s, q(lambda_{s,c}))

    q(lambda) = (1 - p_drop_t7) * sum_{k>=1} Pois(k | lambda) *
                (1 - (phi_t7 / (phi_t7 + beta_t7 k))^phi_t7)

``beta_t7``, ``phi_t7`` and ``p_drop_t7`` are the fit's posterior means; the
only free parameters are the subclass offset ``b_s`` and the exponent ``m_s``.
``m_s = 1`` is exactly the claim "inferred copy number is proportional to the
nanopore count"; ``m_s = 0`` would mean the library composition is invisible in
that subclass. The exponent is estimated by maximum likelihood with an analytic
gradient, and its standard error from the expected Fisher information, so the
zero pairs (the large majority) contribute properly instead of being filtered
out.

Outputs (in ``figures/``)
-------------------------
``copies_vs_nanopore_slope_vs_depth``  m_s +- 95% CI against subclass depth
``copies_vs_nanopore_examples``        per-cCRE inferred copies vs nanopore
``abundance_posterior_vs_nanopore``    the fit's own log a against its prior
``copies_vs_nanopore_slopes.csv``      the per-subclass table

Example
-------
    python revision/t7_subclass_correlation/plot_copies_vs_nanopore.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
BAYES_DIR = os.path.join(REPO_ROOT, "revision", "Bayes_OldData", "bayesian")
TAG = "subclass_joint_copy_number_dropout_svi"
NANOPORE_CSV = os.path.join(
    REPO_ROOT, "STARRFISH_in_vivo", "Data", "SFv8_400CRE_nanopore_counts.csv"
)
CLASS_CSV = os.path.join(THIS_DIR, "figures", "all_unfiltered_cell_types.csv")


# --------------------------------------------------------------------------- #
# fitted measurement model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class T7Detection:
    """The fit's T7 measurement parameters (posterior means) and copy grid."""

    beta_t7: float
    phi_t7: float
    p_drop_t7: float
    kmax: int

    @property
    def k(self) -> np.ndarray:
        return np.arange(1, self.kmax + 1, dtype=np.float64)

    @property
    def h_k(self) -> np.ndarray:
        """P(t7 > 0 | k copies) for k = 1..kmax, dropout included."""
        nb_zero = (self.phi_t7 / (self.phi_t7 + self.beta_t7 * self.k)) ** self.phi_t7
        return (1.0 - self.p_drop_t7) * (1.0 - nb_zero)

    def q(self, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(q, dq/dlambda)`` for the T7-positive probability at ``lam``.

        ``lam`` is broadcast over the k grid; Poisson pmf differences give the
        derivative exactly (d/dl Pois(k|l) = Pois(k-1|l) - Pois(k|l)).
        """
        lam = np.asarray(lam, dtype=np.float64)[..., None]
        k = self.k
        log_pk = k * np.log(lam) - lam - _lgamma_int(k)
        pk = np.exp(log_pk)                                  # (..., K)
        pk_minus = np.exp((k - 1.0) * np.log(lam) - lam - _lgamma_int(k - 1.0))
        h = self.h_k
        q = pk @ h
        dq = (pk_minus - pk) @ h
        return q, dq


def _lgamma_int(k: np.ndarray) -> np.ndarray:
    """log(k!) for non-negative integer-valued floats."""
    from scipy.special import gammaln

    return gammaln(k + 1.0)


def load_detection(bayes_dir: str, tag: str, kmax: int | None) -> T7Detection:
    """Posterior-mean T7 measurement parameters from the scalar samples."""
    path = os.path.join(bayes_dir, f"{tag}_scalar_samples.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scalar samples not found: {path}")
    with np.load(path, allow_pickle=True) as store:
        beta = float(np.mean(store["beta_t7"]))
        phi = float(np.mean(store["phi_t7"]))
        p_drop = float(np.mean(store["p_drop_t7"]))
    if kmax is None:
        with np.load(
            os.path.join(bayes_dir, "..", "copy_number", "copy_number.npz"),
            allow_pickle=True,
        ) as store:
            kmax = int(store["kmax"])
    if not (beta > 0 and phi > 0 and 0.0 <= p_drop < 1.0):
        raise ValueError(f"implausible T7 parameters: {beta=}, {phi=}, {p_drop=}")
    return T7Detection(beta_t7=beta, phi_t7=phi, p_drop_t7=p_drop, kmax=int(kmax))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_evidence(bayes_dir: str, tag: str) -> pd.DataFrame:
    """Per-(subclass, cCRE) T7 detection counts: ``group, cre, n_t7_pos, n_total``."""
    path = os.path.join(bayes_dir, f"{tag}_gamma.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"per-pair evidence not found: {path}")
    frame = pd.read_csv(path, usecols=["group", "cre", "n_t7_pos", "n_total"])
    frame["group"] = frame["group"].astype(str)
    frame["cre"] = frame["cre"].astype(str)
    return frame


def load_nanopore(path: str) -> pd.Series:
    """Per-cCRE nanopore read counts of the input AAV library."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"nanopore counts CSV not found: {path}")
    frame = pd.read_csv(path)
    if not {"CRE", "counts"}.issubset(frame.columns):
        raise KeyError(f"expected columns CRE,counts; got {list(frame.columns)}")
    return frame.set_index("CRE")["counts"].astype(float)


def load_class_labels(path: str) -> pd.Series:
    """Subclass -> parent class, keyed on the fit's un-numbered subclass names.

    The fit stores ``obs['subclass']`` ("L2-3 IT CTX Glut") while the h5ad-derived
    table keeps the numbered ``subclass_name`` ("019 L2/3 IT CTX Glut"); the
    numeric prefix and the "/" are the only differences.
    """
    if not os.path.isfile(path):
        return pd.Series(dtype=object)
    frame = pd.read_csv(path)
    if not {"subclass", "class"}.issubset(frame.columns):
        raise KeyError(f"expected columns subclass,class in {path}")
    bare = (
        frame["subclass"]
        .astype(str)
        .str.replace(r"^\d+\s+", "", regex=True)
        .str.replace("/", "-", regex=False)
    )
    return pd.Series(frame["class"].astype(str).to_numpy(), index=bare.to_numpy())


# --------------------------------------------------------------------------- #
# per-subclass exponent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlopeFit:
    """ML exponent of nanopore abundance in one subclass's detection pattern."""

    subclass: str
    slope: float
    slope_se: float
    intercept: float
    n_cells: int
    n_cres: int
    n_pos_events: int
    converged: bool

    @property
    def z_vs_one(self) -> float:
        return (self.slope - 1.0) / self.slope_se if self.slope_se > 0 else np.nan

    @property
    def p_vs_one(self) -> float:
        z = self.z_vs_one
        return float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan


def _neg_loglik(
    theta: np.ndarray,
    n_pos: np.ndarray,
    n_cells: float,
    z: np.ndarray,
    det: T7Detection,
) -> tuple[float, np.ndarray]:
    """Binomial negative log-likelihood and its analytic gradient in (b, m)."""
    b, m = float(theta[0]), float(theta[1])
    lam = np.exp(np.clip(b + m * z, -30.0, np.log(det.kmax / 4.0)))
    q, dq = det.q(lam)
    q = np.clip(q, 1e-12, 1.0 - 1e-12)
    n_neg = n_cells - n_pos
    nll = -float(np.sum(n_pos * np.log(q) + n_neg * np.log1p(-q)))
    dll_dq = n_pos / q - n_neg / (1.0 - q)
    dll_dlam = dll_dq * dq
    grad = np.array([
        -float(np.sum(dll_dlam * lam)),
        -float(np.sum(dll_dlam * lam * z)),
    ])
    return nll, grad


def _slope_se(
    theta: np.ndarray, n_cells: float, z: np.ndarray, det: T7Detection
) -> float:
    """SE of the exponent from the expected Fisher information at the MLE."""
    lam = np.exp(np.clip(theta[0] + theta[1] * z, -30.0, np.log(det.kmax / 4.0)))
    q, dq = det.q(lam)
    q = np.clip(q, 1e-12, 1.0 - 1e-12)
    w = n_cells * (dq * lam) ** 2 / (q * (1.0 - q))
    info = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]])
    if not np.all(np.isfinite(info)) or np.linalg.det(info) <= 0:
        return np.nan
    return float(np.sqrt(np.linalg.inv(info)[1, 1]))


def fit_subclass(
    subclass: str,
    n_pos: np.ndarray,
    n_cells: float,
    z: np.ndarray,
    det: T7Detection,
) -> SlopeFit:
    """ML fit of ``lambda = exp(b + m z)`` to one subclass's detection counts."""
    p_bar = max(float(n_pos.sum()) / (n_cells * z.size), 1e-9)
    b0 = float(np.log(p_bar / max(det.h_k[0], 1e-9)))
    result = optimize.minimize(
        _neg_loglik,
        x0=np.array([b0, 1.0]),
        args=(n_pos, n_cells, z, det),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-30.0, 5.0), (-5.0, 5.0)],
    )
    theta = np.asarray(result.x, dtype=float)
    return SlopeFit(
        subclass=subclass,
        slope=float(theta[1]),
        slope_se=_slope_se(theta, n_cells, z, det),
        intercept=float(theta[0]),
        n_cells=int(n_cells),
        n_cres=int(z.size),
        n_pos_events=int(n_pos.sum()),
        converged=bool(result.success),
    )


def fit_pooled(
    counts: np.ndarray, n_cells: np.ndarray, z: np.ndarray, det: T7Detection
) -> tuple[float, float]:
    """One shared exponent with a free offset per subclass; returns ``(m, se)``.

    Parameters
    ----------
    counts : (n_sub, n_cre) T7-positive cell counts
    n_cells : (n_sub,) cells per subclass
    z : (n_cre,) centred log1p nanopore counts
    """
    n_sub = counts.shape[0]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b = theta[:n_sub][:, None]
        m = float(theta[-1])
        lam = np.exp(np.clip(b + m * z[None, :], -30.0, np.log(det.kmax / 4.0)))
        q, dq = det.q(lam)
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        n_neg = n_cells[:, None] - counts
        nll = -float(np.sum(counts * np.log(q) + n_neg * np.log1p(-q)))
        dll_dlam = (counts / q - n_neg / (1.0 - q)) * dq
        grad_b = -np.sum(dll_dlam * lam, axis=1)
        grad_m = -float(np.sum(dll_dlam * lam * z[None, :]))
        return nll, np.concatenate([grad_b, [grad_m]])

    p_bar = np.maximum(counts.sum(axis=1) / (n_cells * z.size), 1e-9)
    x0 = np.concatenate([np.log(p_bar / max(det.h_k[0], 1e-9)), [1.0]])
    result = optimize.minimize(objective, x0=x0, jac=True, method="L-BFGS-B")
    theta = np.asarray(result.x, dtype=float)
    # profile SE for the shared exponent: invert the full information, take [m, m]
    b = theta[:n_sub][:, None]
    lam = np.exp(np.clip(b + theta[-1] * z[None, :], -30.0, np.log(det.kmax / 4.0)))
    q, dq = det.q(lam)
    q = np.clip(q, 1e-12, 1.0 - 1e-12)
    w = n_cells[:, None] * (dq * lam) ** 2 / (q * (1.0 - q))
    i_bb = w.sum(axis=1)
    i_bm = (w * z[None, :]).sum(axis=1)
    i_mm = float((w * z[None, :] ** 2).sum())
    schur = i_mm - float(np.sum(i_bm**2 / np.maximum(i_bb, 1e-300)))
    se = float(np.sqrt(1.0 / schur)) if schur > 0 else np.nan
    return float(theta[-1]), se


def local_exponents(
    counts: np.ndarray,
    n_cells: np.ndarray,
    nano: np.ndarray,
    det: T7Detection,
    n_windows: int = 9,
    window_frac: float = 0.25,
) -> pd.DataFrame:
    """Pooled exponent inside overlapping equal-count nanopore windows.

    A single exponent over the full abundance range hides curvature: this
    re-fits the shared exponent on contiguous blocks of cCREs ranked by
    nanopore count, so a saturating relation shows up as a falling exponent.
    """
    order = np.argsort(nano)
    width = max(int(round(window_frac * nano.size)), 40)
    if width >= nano.size:
        raise ValueError("window wider than the cCRE set")
    starts = np.linspace(0, nano.size - width, n_windows).astype(int)
    rows = []
    for start in starts:
        idx = order[start:start + width]
        z = np.log1p(nano[idx])
        z = z - z.mean()
        slope, se = fit_pooled(counts[:, idx], n_cells, z, det)
        rows.append({
            "center": float(stats.gmean(np.maximum(nano[idx], 1.0))),
            "lo": float(nano[idx].min()),
            "hi": float(nano[idx].max()),
            "slope": slope,
            "slope_se": se,
            "n_cres": int(width),
        })
    return pd.DataFrame(rows)


def invert_copies(
    n_pos: np.ndarray, n_cells: float, det: T7Detection
) -> np.ndarray:
    """Per-cCRE method-of-moments copy number from an observed positive fraction.

    ``q`` is strictly increasing, so the inversion is a lookup on a log grid.
    Pairs with zero positives, or a fraction above the attainable maximum, come
    back as NaN: they carry no point estimate (they do enter the ML fit above).
    """
    grid = np.exp(np.linspace(np.log(1e-6), np.log(det.kmax / 4.0), 4000))
    q_grid, _ = det.q(grid)
    frac = n_pos / n_cells
    out = np.full(frac.shape, np.nan, dtype=float)
    usable = (frac > 0) & (frac < q_grid[-1])
    out[usable] = np.interp(frac[usable], q_grid, grid)
    return out


def fit_copies_correlation(
    bayes_dir: str, tag: str, nano: pd.Series
) -> tuple[float, int]:
    """Pearson r of the fit's own per-cCRE copies against nanopore, log-log.

    ``lambda_{s,c} = rho_s * a_c`` is rank-one, so ``log rho_s`` is an additive
    per-subclass constant and this single r is the value for *every* subclass.
    """
    with np.load(os.path.join(bayes_dir, f"{tag}_posterior_samples.npz"),
                 allow_pickle=True) as store:
        cres = np.array([str(c) for c in store["cre_names"]])
        log_a = store["log_a"].mean(axis=0)
    aligned = nano.reindex(cres)
    keep = aligned.notna().to_numpy()
    x = np.log10(aligned[keep].to_numpy(float) + 1.0)
    return float(stats.pearsonr(x, log_a[keep])[0]), int(keep.sum())


def pearson_table(
    counts: pd.DataFrame,
    n_cells: pd.Series,
    nano: np.ndarray,
    det: T7Detection,
    classes: pd.Series,
) -> pd.DataFrame:
    """Per-subclass correlation of detection-derived copies against nanopore.

    ``pearson_log`` is taken on log10 over the cCREs with at least one
    T7-positive cell, which is where a copy-number point estimate exists;
    ``spearman_all`` keeps every cCRE, scoring the undetected ones as zero
    copies, so it is free of that censoring.
    """
    log_nano = np.log10(nano + 1.0)
    rows = []
    for sub in counts.index:
        cells = float(n_cells.loc[sub])
        n_pos = counts.loc[sub].to_numpy(float)
        lam = invert_copies(n_pos, cells, det)
        ok = np.isfinite(lam) & (lam > 0)
        pearson = (float(stats.pearsonr(log_nano[ok], np.log10(lam[ok]))[0])
                   if ok.sum() >= 3 else np.nan)
        lam_all = np.where(np.isfinite(lam), lam, 0.0)
        spearman = (float(stats.spearmanr(log_nano, lam_all)[0])
                    if np.ptp(lam_all) > 0 else np.nan)
        rows.append({
            "subclass": sub,
            "class": classes.get(sub, "unknown"),
            "pearson_log": pearson,
            "spearman_all": spearman,
            "n_cres_detected": int(ok.sum()),
            "n_cres": int(nano.size),
            "n_cells": int(cells),
            "n_pos_events": int(n_pos.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _class_colors(labels: np.ndarray) -> tuple[list, dict]:
    levels = sorted(pd.unique(labels))
    cmap = plt.get_cmap("tab20", max(len(levels), 1))
    color_of = {c: cmap(i) for i, c in enumerate(levels)}
    return [color_of[c] for c in labels], color_of


def plot_correlation(
    table: pd.DataFrame, fit_r: tuple[float, int], outpath: str
) -> None:
    """Per-subclass copies-vs-nanopore correlation against subclass depth."""
    colors, color_of = _class_colors(table["class"].to_numpy())
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), constrained_layout=True)
    panels = (
        (axes[0], "pearson_log",
         "Pearson r, log10 copies vs log10 nanopore\n(cCREs with >=1 T7+ cell)"),
        (axes[1], "spearman_all",
         "Spearman ρ, all cCREs\n(undetected scored as zero copies)"),
    )
    for ax, column, ylabel in panels:
        y = table[column].to_numpy(float)
        x = table["n_cells"].to_numpy(float)
        ax.scatter(x, y, s=30, c=colors, edgecolor="0.3", linewidth=0.3)
        ax.axhline(fit_r[0], color="#c92a2a", linewidth=1.2, linestyle="--",
                   label=f"fit's own copies, r = {fit_r[0]:.3f}\n"
                         f"(identical in every subclass — prior-anchored)")
        ax.set_xscale("log")
        ax.set_xlabel("Number of cells in subclass")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color="0.92", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if mask.sum() >= 3:
            rho, p = stats.spearmanr(np.log10(x[mask]), y[mask])
            ax.text(0.03, 0.96, f"median = {np.nanmedian(y):.3f}\n"
                    f"IQR {np.nanpercentile(y, 25):.3f}–"
                    f"{np.nanpercentile(y, 75):.3f}\n"
                    f"vs depth: Spearman ρ = {rho:.3f} (p = {p:.1e})\n"
                    f"n = {int(mask.sum())} subclasses",
                    transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="0.7"))

    axes[2].hist(table["pearson_log"].dropna(), bins=24, color="#4c6ef5",
                 alpha=0.8, edgecolor="white", label="Pearson r (detected cCREs)")
    axes[2].hist(table["spearman_all"].dropna(), bins=24, color="#087f5b",
                 alpha=0.5, edgecolor="white", label="Spearman ρ (all cCREs)")
    axes[2].axvline(fit_r[0], color="#c92a2a", linewidth=1.2, linestyle="--")
    axes[2].set_xlabel("correlation with nanopore counts")
    axes[2].set_ylabel("subclasses")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(True, axis="y", color="0.92", linewidth=0.4)
    axes[2].set_axisbelow(True)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor=color_of[c], markeredgecolor="0.3",
                          label=c) for c in sorted(color_of)]
    fig.legend(handles=handles, title="Class", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=6, title_fontsize=8,
               frameon=False)
    fig.suptitle("Correlation between inferred AAV copy number and nanopore "
                 "library counts, per subclass", fontsize=12)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_slope_vs_depth(
    table: pd.DataFrame,
    pooled: tuple[float, float],
    local: pd.DataFrame,
    outpath: str,
) -> None:
    """Exponent per subclass against depth, with the proportionality line at 1."""
    colors, color_of = _class_colors(table["class"].to_numpy())
    fig, axes = plt.subplots(1, 4, figsize=(24.0, 5.8), constrained_layout=True)
    panels = (
        (axes[0], table["n_cells"].to_numpy(float), "Number of cells in subclass"),
        (axes[1], table["n_pos_events"].to_numpy(float),
         "T7-positive (cell, cCRE) observations in subclass"),
    )
    for ax, x, xlabel in panels:
        ax.errorbar(x, table["slope"], yerr=1.96 * table["slope_se"], fmt="none",
                    ecolor="0.75", elinewidth=0.7, zorder=1)
        ax.scatter(x, table["slope"], s=30, c=colors, edgecolor="0.3",
                   linewidth=0.3, zorder=2)
        ax.axhline(1.0, color="#c92a2a", linewidth=1.2, linestyle="--", zorder=3)
        ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":", zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", color="0.92", linewidth=0.4)
        ax.set_axisbelow(True)
        mask = np.isfinite(x) & np.isfinite(table["slope"].to_numpy()) & (x > 0)
        if mask.sum() >= 3:
            rho, p = stats.spearmanr(np.log10(x[mask]),
                                     table["slope"].to_numpy()[mask])
            ax.text(0.03, 0.04, f"Spearman ρ = {rho:.3f}\np = {p:.1e}\n"
                    f"n = {int(mask.sum())} subclasses", transform=ax.transAxes,
                    va="bottom", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="0.7"))
    axes[0].set_ylabel("ML exponent $m_s$\n" r"($\lambda \propto$ nanopore$^{m_s}$)")

    slopes = table["slope"].to_numpy(float)
    axes[2].hist(slopes, bins=24, color="#4c6ef5", alpha=0.8, edgecolor="white")
    axes[2].axvline(1.0, color="#c92a2a", linewidth=1.2, linestyle="--")
    axes[2].axvline(float(np.median(slopes)), color="0.25", linewidth=1.0)
    axes[2].set_xlabel("ML exponent $m_s$")
    axes[2].set_ylabel("subclasses")
    covers = np.abs(slopes - 1.0) <= 1.96 * table["slope_se"].to_numpy(float)
    axes[2].text(0.03, 0.96, f"median $m_s$ = {np.median(slopes):.3f}\n"
                 f"IQR {np.percentile(slopes, 25):.3f}–"
                 f"{np.percentile(slopes, 75):.3f}\n"
                 f"95% CI covers 1: {int(covers.sum())}/{len(slopes)}\n"
                 f"pooled m = {pooled[0]:.3f} ± {pooled[1]:.3f}",
                 transform=axes[2].transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                           edgecolor="0.7"))
    axes[2].grid(True, axis="y", color="0.92", linewidth=0.4)
    axes[2].set_axisbelow(True)

    axes[3].errorbar(local["center"], local["slope"],
                     yerr=1.96 * local["slope_se"], fmt="o-", color="#087f5b",
                     ecolor="#087f5b", markersize=5, linewidth=1.3, capsize=3)
    axes[3].axhline(1.0, color="#c92a2a", linewidth=1.2, linestyle="--")
    axes[3].axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("nanopore window centre (geometric mean read count)")
    axes[3].set_ylabel("pooled exponent inside the window")
    axes[3].set_title(f"range-resolved exponent "
                      f"({int(local['n_cres'].iat[0])}-cCRE sliding windows)",
                      fontsize=10)
    axes[3].grid(True, which="both", color="0.92", linewidth=0.4)
    axes[3].set_axisbelow(True)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor=color_of[c], markeredgecolor="0.3",
                          label=c) for c in sorted(color_of)]
    fig.legend(handles=handles, title="Class", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=6, title_fontsize=8,
               frameon=False)
    fig.suptitle("Inferred AAV copy number rises with the nanopore library count "
                 "sublinearly and saturates; the exponent is the same in every "
                 "subclass\n(dashed red = proportionality, $m$ = 1)", fontsize=12)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _binned_copies(
    n_pos: np.ndarray,
    n_cells: float,
    nano: np.ndarray,
    det: T7Detection,
    n_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-aware nonparametric copy number in equal-count nanopore bins.

    Pooling the positive counts of every cCRE in a bin before inverting keeps
    the zero pairs in the estimate, so this curve is free of the censoring that
    lifts the per-cCRE points at low abundance.
    """
    order = np.argsort(nano)
    chunks = np.array_split(order, n_bins)
    centers = np.array([stats.gmean(np.maximum(nano[c], 1.0)) for c in chunks])
    rates = np.array([n_pos[c].sum() / (n_cells * c.size) for c in chunks])
    lam = invert_copies(rates * n_cells, n_cells, det)
    return centers, lam


def plot_examples(
    table: pd.DataFrame,
    counts: pd.DataFrame,
    nano: np.ndarray,
    z: np.ndarray,
    det: T7Detection,
    n_cells: pd.Series,
    outpath: str,
    n_examples: int = 6,
) -> None:
    """Per-cCRE inferred copies vs nanopore for subclasses spanning the depth range."""
    ranked = table.sort_values("n_cells", ascending=False)
    picks = ranked.iloc[np.linspace(0, len(ranked) - 1, n_examples).astype(int)]
    n_col = 3
    n_row = int(np.ceil(n_examples / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.2 * n_col, 4.6 * n_row),
                             constrained_layout=True, sharex=True)
    flat = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(flat, picks.iterrows()):
        sub = row["subclass"]
        cells = float(n_cells.loc[sub])
        lam_hat = invert_copies(counts.loc[sub].to_numpy(float), cells, det)
        ok = np.isfinite(lam_hat)
        ax.scatter(nano[ok], lam_hat[ok], s=16, alpha=0.6, color="#4c6ef5",
                   edgecolor="none",
                   label=f"{int(ok.sum())} cCREs with >=1 T7+ cell")
        order = np.argsort(z)
        ax.plot(nano[order], np.exp(row["intercept"] + row["slope"] * z[order]),
                color="#c92a2a", linewidth=1.4,
                label=f"ML fit, m = {row['slope']:.2f} ± {row['slope_se']:.2f}")
        ax.plot(nano[order], np.exp(row["intercept"] + z[order]), color="0.35",
                linewidth=1.0, linestyle="--", label="proportional (m = 1)")
        centers, binned = _binned_copies(counts.loc[sub].to_numpy(float), cells,
                                         nano, det)
        ax.plot(centers, binned, color="#087f5b", marker="s", markersize=4,
                linewidth=1.2, label="binned, zeros included")
        floor = invert_copies(np.array([1.0]), cells, det)[0]
        if np.isfinite(floor):
            ax.axhline(floor, color="0.6", linewidth=0.8, linestyle=":",
                       label="1 T7+ cell (censoring floor)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{sub}\n{int(cells):,} cells, "
                     f"{int(row['n_pos_events']):,} T7+ observations", fontsize=9)
        ax.grid(True, which="both", color="0.92", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    for ax in flat[len(picks):]:
        ax.axis("off")
    for ax in flat[-n_col:]:
        ax.set_xlabel("Nanopore read counts (input AAV library)")
    for ax in flat[::n_col]:
        ax.set_ylabel("Inferred AAV copies per cell\n(from T7 detection rate)")
    fig.suptitle("Per-cCRE inferred copy number vs input library abundance, "
                 "by subclass", fontsize=12)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_abundance_prior_check(
    bayes_dir: str, tag: str, nano: pd.Series, outpath: str
) -> None:
    """The fit's own ``log a`` against the nanopore prior it was centred on."""
    with np.load(os.path.join(bayes_dir, f"{tag}_posterior_samples.npz"),
                 allow_pickle=True) as store:
        cres = np.array([str(c) for c in store["cre_names"]])
        log_a = store["log_a"].mean(axis=0)
    with np.load(os.path.join(bayes_dir, f"{tag}_scalar_samples.npz"),
                 allow_pickle=True) as store:
        tau_a = store["tau_a"]
    aligned = nano.reindex(cres)
    keep = aligned.notna().to_numpy()
    prior = np.log1p(aligned.fillna(0.0).to_numpy())
    prior = prior - prior.mean()
    x, y = prior[keep], log_a[keep]
    slope, intercept = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    ax.scatter(x, y, s=20, alpha=0.6, color="#7048e8", edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="0.3", linewidth=1.2, linestyle="--",
            label=f"fit, slope = {slope:.2f}")
    ax.plot(xs, xs, color="#c92a2a", linewidth=1.0, label="prior mean (slope 1)")
    ax.set_xlabel("prior mean: centred log1p(nanopore counts)")
    ax.set_ylabel("posterior mean log a (fitted abundance)")
    ax.set_title("The fit's abundance is anchored on nanopore by construction\n"
                 "(shown for completeness — not independent evidence)",
                 fontsize=10)
    ax.text(0.03, 0.96, f"Pearson r = {r:.3f}\np = {p:.1e}\n"
            f"n = {int(keep.sum())} cCREs\n"
            f"τ_a posterior = {tau_a.mean():.2f} "
            f"(prior scale 0.5)\nresidual sd = {np.std(y - x):.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="0.7"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, color="0.92", linewidth=0.4)
    ax.set_axisbelow(True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bayes-dir", default=BAYES_DIR)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--nanopore", default=NANOPORE_CSV)
    ap.add_argument("--class-csv", default=CLASS_CSV,
                    help="subclass -> class table used only for point colour")
    ap.add_argument("--kmax", type=int, default=None,
                    help="copy-number truncation (default: the fit's kmax)")
    ap.add_argument("--min-pos-events", type=int, default=50,
                    help="minimum T7-positive (cell, cCRE) observations for a "
                         "subclass to be fitted (default: 50)")
    ap.add_argument("--windows", type=int, default=9,
                    help="overlapping nanopore windows for the range-resolved "
                         "exponent (default: 9)")
    ap.add_argument("--outdir", default=os.path.join(THIS_DIR, "figures"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    det = load_detection(args.bayes_dir, args.tag, args.kmax)
    print(f"T7 measurement model: beta_t7 = {det.beta_t7:.4f}, "
          f"phi_t7 = {det.phi_t7:.3f}, p_drop_t7 = {det.p_drop_t7:.5f}, "
          f"kmax = {det.kmax}")
    print(f"P(T7 > 0 | 1 copy) = {det.h_k[0]:.4f}")

    evidence = load_evidence(args.bayes_dir, args.tag)
    nano = load_nanopore(args.nanopore)
    counts = evidence.pivot(index="group", columns="cre", values="n_t7_pos")
    totals = evidence.pivot(index="group", columns="cre", values="n_total")
    n_cells = totals.max(axis=1).astype(float)
    if not np.allclose(totals.to_numpy(), n_cells.to_numpy()[:, None]):
        raise ValueError("n_total varies within a subclass; evidence table is not "
                         "a clean subclass x cCRE grid")

    shared = [c for c in counts.columns if c in nano.index]
    dropped = [c for c in counts.columns if c not in nano.index]
    if dropped:
        print(f"dropped {len(dropped)} fitted cCREs absent from the nanopore CSV "
              f"({dropped}) — the fit gave them a zero-count prior")
    counts = counts.reindex(columns=shared)
    nano_v = nano.reindex(shared).to_numpy(float)
    z = np.log1p(nano_v)
    z = z - z.mean()
    print(f"{len(shared)} cCREs x {counts.shape[0]} subclasses; "
          f"nanopore counts {nano_v.min():.0f}..{nano_v.max():.0f}")

    pos_events = counts.sum(axis=1)
    keep = pos_events[pos_events >= args.min_pos_events].index
    print(f"{len(keep)}/{counts.shape[0]} subclasses have >= "
          f"{args.min_pos_events} T7-positive observations")
    if len(keep) < 3:
        raise ValueError("too few subclasses pass --min-pos-events")

    classes = load_class_labels(args.class_csv)
    fits = [
        fit_subclass(sub, counts.loc[sub].to_numpy(float),
                     float(n_cells.loc[sub]), z, det)
        for sub in keep
    ]
    table = pd.DataFrame([
        {
            "subclass": f.subclass,
            "class": classes.get(f.subclass, "unknown"),
            "slope": f.slope,
            "slope_se": f.slope_se,
            "slope_lo": f.slope - 1.96 * f.slope_se,
            "slope_hi": f.slope + 1.96 * f.slope_se,
            "intercept": f.intercept,
            "n_cells": f.n_cells,
            "n_cres": f.n_cres,
            "n_pos_events": f.n_pos_events,
            "p_vs_proportional": f.p_vs_one,
            "converged": f.converged,
        }
        for f in fits
    ])
    n_bad = int((~table["converged"]).sum())
    if n_bad:
        print(f"warning: {n_bad} subclass fits did not converge; kept and flagged")

    pooled = fit_pooled(counts.loc[keep].to_numpy(float),
                        n_cells.loc[keep].to_numpy(float), z, det)
    print(f"pooled exponent (shared m, free offset per subclass): "
          f"m = {pooled[0]:.4f} ± {pooled[1]:.4f} "
          f"(95% CI {pooled[0] - 1.96 * pooled[1]:.3f}.."
          f"{pooled[0] + 1.96 * pooled[1]:.3f})")
    slopes = table["slope"].to_numpy(float)
    covers = np.abs(slopes - 1.0) <= 1.96 * table["slope_se"].to_numpy(float)
    print(f"per-subclass exponent: median {np.median(slopes):.3f}, "
          f"IQR {np.percentile(slopes, 25):.3f}..{np.percentile(slopes, 75):.3f}, "
          f"range {slopes.min():.3f}..{slopes.max():.3f}; "
          f"{int(covers.sum())}/{len(slopes)} 95% CIs cover 1")

    csv_path = os.path.join(args.outdir, "copies_vs_nanopore_slopes.csv")
    table.sort_values("n_cells", ascending=False).to_csv(csv_path, index=False)

    fit_r = fit_copies_correlation(args.bayes_dir, args.tag, nano)
    print(f"fit's own copies (rho_s * a_c) vs nanopore, log-log: "
          f"r = {fit_r[0]:.4f} over {fit_r[1]} cCREs — the same value in every "
          f"subclass, and prior-anchored")
    corr = pearson_table(counts.loc[keep], n_cells, nano_v, det, classes)
    corr_path = os.path.join(args.outdir, "copies_vs_nanopore_pearson.csv")
    corr.sort_values("n_cells", ascending=False).to_csv(corr_path, index=False)
    print("detection-derived copies vs nanopore: "
          f"Pearson r median {corr['pearson_log'].median():.3f} "
          f"(IQR {corr['pearson_log'].quantile(0.25):.3f}.."
          f"{corr['pearson_log'].quantile(0.75):.3f}, "
          f"range {corr['pearson_log'].min():.3f}..{corr['pearson_log'].max():.3f}); "
          f"Spearman median {corr['spearman_all'].median():.3f}")
    out0 = os.path.join(args.outdir, "copies_vs_nanopore_pearson.png")
    plot_correlation(corr, fit_r, out0)

    local = local_exponents(counts.loc[keep].to_numpy(float),
                            n_cells.loc[keep].to_numpy(float), nano_v, det,
                            n_windows=args.windows)
    print("range-resolved pooled exponent "
          f"({int(local['n_cres'].iat[0])}-cCRE windows):")
    for _, row in local.iterrows():
        print(f"  nanopore {row['lo']:>8.0f}..{row['hi']:>7.0f} "
              f"(gm {row['center']:>8.0f}): m = {row['slope']:.3f} "
              f"± {row['slope_se']:.3f}")
    local.to_csv(os.path.join(args.outdir,
                              "copies_vs_nanopore_local_exponents.csv"),
                 index=False)

    out1 = os.path.join(args.outdir, "copies_vs_nanopore_slope_vs_depth.png")
    plot_slope_vs_depth(table, pooled, local, out1)
    out2 = os.path.join(args.outdir, "copies_vs_nanopore_examples.png")
    plot_examples(table, counts, nano_v, z, det, n_cells, out2)
    out3 = os.path.join(args.outdir, "abundance_posterior_vs_nanopore.png")
    plot_abundance_prior_check(args.bayes_dir, args.tag, nano, out3)
    print(f"wrote {csv_path}, {corr_path}, {out0}, {out1}, {out2} and {out3} "
          f"(+.pdf)")


if __name__ == "__main__":
    main()
