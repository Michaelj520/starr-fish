"""How many AAV genomes infected each cell, for each cCRE.

The latent copy number ``k_ij`` is the quantity the model is built around and the
one thing it never samples: it is marginalised analytically so that inference
stays in continuous parameters. This module puts it back, producing the
**cell x cCRE matrix of posterior mean copies** ``E[k_ij | t7_ij, cre_ij]``.

Why this is not just "the counts"
---------------------------------
A cell with ``t7 = 0`` and ``cre = 0`` is not known to be uninfected -- with
dropout, or simply a low per-copy rate, an infected cell often reads zero. And a
cell with ``t7 = 12`` did not receive twelve genomes; it received some ``k`` whose
posterior depends on ``beta_t7``, the dispersion, that cCRE's library abundance
and that cell type's infection rate. The matrix returned here is the model's
answer to both, with the arbitrary-scale problem already resolved.

How it stays tractable
----------------------
``P(k | obs)`` depends on the observation only through ``(cell type, cCRE, t7,
cre)``. On the real data 99.85% of pairs are ``(0, 0)``, so:

1. compute ``E[k | 0, 0]`` once per (cell type, cCRE) -- a 328 x 389 grid;
2. broadcast that baseline across every cell of each type;
3. compute the unique nonzero ``(cell type, cCRE, t7, cre)`` patterns -- about a
   million rather than 160 million -- and scatter them over the baseline.

The result is exact, not approximate: identical observations in the same cell
type have identical posteriors.

Required posterior sites
------------------------
This needs ``log_rho`` and ``log_a`` in addition to ``log_gamma``, because the
copy number depends on the infection rate and the library abundance, not just on
activity. The production fit was written with the default
``--posterior-sites log_gamma`` and therefore **cannot** be used as-is; see
:func:`load_copy_number_draws` for the error message and the exact refit command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .._log import log
from .posterior_k import posterior_k_moments

__all__ = [
    "CopyNumberMatrix",
    "DEFAULT_CONTROL_T7_THRESHOLD",
    "MATRIX_NAMES",
    "MissingPosteriorSites",
    "REQUIRED_POSTERIOR_SITES",
    "infer_copy_number",
    "infer_copy_number_from_fit",
    "load_copy_number_draws",
    "thin_draws",
]

#: Sites the copy-number posterior needs beyond the scalars.
REQUIRED_POSTERIOR_SITES = ("log_rho", "log_a", "log_gamma")

#: Scalars every copy-number fit provides.
_REQUIRED_SCALARS = ("beta_t7", "phi_t7", "phi_cre")

#: Present only for a ``copy_number_dropout`` fit; omitting them where they exist
#: silently evaluates the wrong model, so they are carried through when found.
_OPTIONAL_SCALARS = ("p_drop_t7", "p_drop_cre")

#: Every matrix a :class:`CopyNumberMatrix` can carry.
MATRIX_NAMES = (
    "copies", "sd", "p_infected", "activity", "activity_normalized", "deviation"
)

#: Pooled control T7 a cell type needs before it gets a background reference.
#: 50 is the threshold behind the published ``t7_ge50`` tables.
DEFAULT_CONTROL_T7_THRESHOLD = 50.0


class MissingPosteriorSites(KeyError):
    """A fit does not carry the draws a copy-number reconstruction needs.

    Subclasses ``KeyError`` so existing handlers still catch it, but overrides
    ``__str__``: ``KeyError`` renders its argument with ``repr``, which escapes
    the newlines out of a multi-line message and turns the suggested command into
    one unreadable line.
    """

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class CopyNumberMatrix:
    """Posterior AAV copy number per (cell, cCRE).

    Attributes
    ----------
    copies : (n_cells, n_cre)
        ``E[k_ij | observed counts]``. Continuous, because it is a posterior
        mean over an integer quantity -- 0.02 means "almost certainly zero
        copies", not "a fiftieth of a virus". It is *not* a probability of
        infection, and thresholding it is not an infection call; ``P(k >= 1)``
        would have to be accumulated from the posterior weights instead.
    sd : (n_cells, n_cre) or None
        Posterior sd of the same quantity, combining the spread of ``k`` at fixed
        parameters with the spread induced by parameter uncertainty.
    p_infected : (n_cells, n_cre) or None
        ``P(k >= 1 | obs)`` -- the posterior probability that the cell received
        *any* copy of that construct. This is the probability; ``copies`` is the
        expected count. They answer different questions.
    activity : (n_cells, n_cre) or None
        Posterior mean per-cell enhancer activity, from the Gamma-conjugate
        posterior of the NB2's latent scale (see
        :func:`baystarrfish.inference.posterior_k.posterior_k_moments`). Unlike
        the moment estimator ``cre / E[k]`` this is a posterior of a quantity the
        model contains, shrinks toward the cell-type activity when evidence is
        thin, and stays finite at ``cre = 0``.
    activity_normalized : (n_cells, n_cre) or None
        The same activity divided, inside each posterior draw, by the cell type's
        negative-control reference -- so **1 is background** and the value is a
        per-cell fold change on the scale the published calls use. ``NaN`` for
        cells whose cell type has no eligible control reference; those are not
        missing values to be filled but cell types the data cannot place against
        a background, and every plotting path leaves them grey.
    deviation : (n_cells, n_cre) or None
        The single-cell factor of the activity, ``E[G | cre, k]``, before the
        cell type's ``gamma`` multiplies it. ``activity = deviation * gamma``,
        and ``gamma`` is a constant within a cell type, so this carries all of
        the cell-to-cell variation and none of the cell-type contrast. Use it
        when cell-type identity must not be counted as signal; use ``activity``
        or ``activity_normalized`` when it should. Unlike ``activity_normalized``
        it is defined for every cell, because no control reference enters it.
    """

    copies: np.ndarray
    sd: np.ndarray | None
    p_infected: np.ndarray | None
    activity: np.ndarray | None
    obs_names: np.ndarray | None
    cre_names: list[str]
    kmax: int
    level: str
    infection_model: str
    activity_normalized: np.ndarray | None = None
    deviation: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return int(self.copies.shape[0])

    @property
    def n_cre(self) -> int:
        return int(self.copies.shape[1])

    def matrix(self, which: str = "copies") -> np.ndarray:
        """One of ``"copies"``, ``"sd"``, ``"p_infected"``, ``"activity"``,
        ``"activity_normalized"``, ``"deviation"``."""
        if which not in MATRIX_NAMES:
            raise ValueError(
                f"unknown matrix {which!r}; expected one of {list(MATRIX_NAMES)}"
            )
        value = getattr(self, which)
        if value is None:
            raise ValueError(
                f"{which!r} was not computed; pass return_sd / "
                "return_probability / return_activity / "
                "return_activity_normalized / return_deviation to "
                "infer_copy_number"
            )
        return value

    def to_frame(self, which: str = "copies") -> pd.DataFrame:
        """Dense DataFrame, cells x cCREs.

        Materialises the whole matrix as float64 -- roughly 1.3 GB for the full
        dataset. Prefer :meth:`matrix` and :attr:`cre_names` for anything large.
        """
        return pd.DataFrame(
            self.matrix(which), index=self.obs_names, columns=self.cre_names
        )

    def write_csv(
        self, path: Path | str, which: str = "copies", *, decimals: int = 5
    ) -> Path:
        """Write one matrix as CSV, gzipped when the path ends in ``.gz``.

        Rounded to ``decimals`` because the full matrix is 159 million values:
        at 5 decimals it is ~1 GB of text, and the posterior is nowhere near that
        sharp. Written in row blocks so the text never exists in memory at once.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.round(self.matrix(which), decimals)
        index = (
            self.obs_names if self.obs_names is not None
            else np.arange(values.shape[0]).astype(str)
        )
        chunk = max(1, 2_000_000 // max(1, values.shape[1]))
        mode, header = "w", True
        for start in range(0, values.shape[0], chunk):
            block = slice(start, start + chunk)
            pd.DataFrame(
                values[block], index=index[block], columns=self.cre_names
            ).to_csv(path, mode=mode, header=header, index_label="cell")
            mode, header = "a", False
        return path

    def total_per_cell(self) -> np.ndarray:
        """Expected AAV genomes per cell, summed over cCREs."""
        return self.copies.sum(axis=1)

    def write_npz(self, path: Path | str) -> Path:
        """Save compressed, float32, with the axis labels."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "copies": np.asarray(self.copies, dtype=np.float32),
            "cre_names": np.asarray(self.cre_names, dtype=object),
            "kmax": np.asarray(self.kmax),
            "level": np.asarray(self.level, dtype=object),
            "infection_model": np.asarray(self.infection_model, dtype=object),
        }
        if self.sd is not None:
            payload["sd"] = np.asarray(self.sd, dtype=np.float32)
        if self.p_infected is not None:
            payload["p_infected"] = np.asarray(self.p_infected, dtype=np.float32)
        if self.activity is not None:
            payload["activity"] = np.asarray(self.activity, dtype=np.float32)
        if self.activity_normalized is not None:
            payload["activity_normalized"] = np.asarray(
                self.activity_normalized, dtype=np.float32
            )
        if self.deviation is not None:
            payload["deviation"] = np.asarray(self.deviation, dtype=np.float32)
        if self.obs_names is not None:
            payload["obs_names"] = np.asarray(self.obs_names, dtype=object)
        np.savez_compressed(path, **payload)
        return path


def load_copy_number_draws(
    fit_dir: Path | str, tag: str | None = None
) -> tuple[dict[str, np.ndarray], dict]:
    """Load the posterior draws a copy-number reconstruction needs.

    Returns ``(draws, manifest)``. ``draws`` holds ``rho`` and ``a`` on the
    natural scale (the fit stores them as logs) plus ``log_gamma`` and the
    scalars, ready for :func:`baystarrfish.inference.posterior_k_moments`.

    Raises a ``KeyError`` naming the missing sites, because the common case --
    a fit written with the default ``--posterior-sites log_gamma`` -- is not
    recoverable after the fact: ``log_rho`` and ``log_a`` cannot be rebuilt from
    what was saved.
    """
    from ..io.results import fit_tag, load_posterior_samples

    fit_dir = Path(fit_dir)
    tag = fit_tag(fit_dir, tag)
    manifest = json.loads((fit_dir / "run_manifest.json").read_text())

    with np.load(fit_dir / f"{tag}_posterior_samples.npz", allow_pickle=True) as handle:
        available = set(handle.files)
    missing = [site for site in REQUIRED_POSTERIOR_SITES if site not in available]
    if missing:
        raise MissingPosteriorSites(
            f"{fit_dir} has no posterior site(s) {missing}; it stores "
            f"{sorted(available - {'group_names', 'cre_names'})}. The copy number "
            "depends on the infection rate and the library abundance, and neither "
            "can be reconstructed from what was saved. Refit requesting them:\n\n"
            "    python revision/run_Bayes/run_bayes.py \\\n"
            "        --posterior-sites log_gamma log_rho log_a [other args as before]\n"
        )

    posterior = load_posterior_samples(fit_dir, tag, sites=list(REQUIRED_POSTERIOR_SITES))
    with np.load(fit_dir / f"{tag}_scalar_samples.npz") as handle:
        scalars = {key: np.asarray(handle[key], dtype=np.float64) for key in handle.files}

    absent = [name for name in _REQUIRED_SCALARS if name not in scalars]
    if absent:
        raise KeyError(f"{fit_dir} scalar samples lack {absent}")

    draws = {
        "rho": np.exp(np.asarray(posterior["log_rho"], dtype=np.float64)),
        "a": np.exp(np.asarray(posterior["log_a"], dtype=np.float64)),
        "log_gamma": np.asarray(posterior["log_gamma"], dtype=np.float64),
        **{name: scalars[name] for name in _REQUIRED_SCALARS},
    }
    is_dropout = manifest.get("config", {}).get("infection_model") == "copy_number_dropout"
    for name in _OPTIONAL_SCALARS:
        if name in scalars:
            draws[name] = scalars[name]
        elif is_dropout:
            raise KeyError(
                f"{fit_dir} is a copy_number_dropout fit but its scalar samples "
                f"lack {name}; evaluating it without dropout would be a different model"
            )
    draws["group_names"] = np.asarray(posterior["group_names"])
    draws["cre_names"] = np.asarray(posterior["cre_names"])
    return draws, manifest


def _align_axes(
    data_cre_names: Sequence[str],
    fit_cre_names: Sequence[str],
    group_labels: np.ndarray,
    fit_group_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map the data's cCRE and cell-type axes onto the fit's, or explain why not."""
    fit_cre = list(map(str, fit_cre_names))
    position = {name: index for index, name in enumerate(fit_cre)}
    unknown = [name for name in map(str, data_cre_names) if name not in position]
    if unknown:
        raise ValueError(
            f"{len(unknown)} cCRE(s) in the data are absent from the fit "
            f"(e.g. {unknown[:5]}); the blacklist used for the fit must match"
        )
    cre_index = np.array([position[str(name)] for name in data_cre_names], dtype=np.int64)

    fit_groups = list(map(str, fit_group_names))
    group_position = {name: index for index, name in enumerate(fit_groups)}
    labels = np.asarray(group_labels).astype(str)
    absent = sorted(set(labels) - set(group_position))
    if absent:
        raise ValueError(
            f"{len(absent)} cell type(s) are absent from the fit "
            f"(e.g. {absent[:5]}); was the fit run at a different level?"
        )
    group_index = np.array([group_position[label] for label in labels], dtype=np.int64)
    return group_index, cre_index


def thin_draws(
    draws: Mapping[str, np.ndarray], max_draws: int | None, *, verbose: bool
) -> Mapping[str, np.ndarray]:
    """Keep ``max_draws`` evenly-spaced posterior draws, or all of them.

    Public so that anything computed beside a reconstruction -- a control
    reference, a calibration target -- can be built from exactly the draws the
    reconstruction used: thin once, then pass the result with ``max_draws=None``.
    """
    if max_draws is None:
        return draws
    if max_draws < 1:
        raise ValueError("max_draws must be positive")
    n_draws = len(np.asarray(draws["beta_t7"]))
    if max_draws >= n_draws:
        return draws
    keep = np.linspace(0, n_draws - 1, max_draws).round().astype(np.int64)
    if verbose:
        log(f"[copies] thinning {n_draws} posterior draws to {len(keep)}")
    return {
        key: (value[keep] if key not in ("group_names", "cre_names") else value)
        for key, value in draws.items()
    }


def _warn_if_truncated(
    draws: Mapping[str, np.ndarray], kmax: int, group_index, cre_index
) -> None:
    """Warn when the ``0..kmax`` grid does not cover the infection rates.

    ``E[k | obs]`` is a mean over a truncated grid. Where the Poisson rate
    approaches ``kmax`` the truncation bites, and the reported copy number is
    biased low by an amount the number itself gives no hint of.
    """
    import warnings

    from ..model.collapse import kmax_tail_mass

    lam = (
        draws["rho"].mean(axis=0)[np.unique(group_index)][:, None]
        * draws["a"].mean(axis=0)[np.unique(cre_index)][None, :]
    )
    tail = kmax_tail_mass(lam.ravel(), kmax)
    if tail > 1e-6:
        warnings.warn(
            f"the latent grid 0..{kmax} leaves {tail:.2e} Poisson mass in the tail "
            f"(largest rho*a is {lam.max():.1f}); E[k] is truncated and biased low "
            "for the most-infected pairs. Refit with a larger kmax to resolve them.",
            RuntimeWarning,
            stacklevel=3,
        )


def _control_log_baseline(
    draws: Mapping[str, np.ndarray],
    t7: np.ndarray,
    group_index: np.ndarray,
    cre_index: np.ndarray,
    negative_control_cre: Sequence[str],
    *,
    t7_threshold: float,
    sd_multiplier: float,
    verbose: bool,
) -> np.ndarray:
    """``log b[d, s]`` for the (already thinned) draws, from the control columns.

    Built here rather than taken from the caller so it cannot be paired with a
    different set of draws than the one it was computed from -- thinning the
    posterior without thinning the baseline would silently mismatch the two.
    """
    from ..stats.baseline import negative_control_log_baseline

    fit_cre = [str(name) for name in draws["cre_names"]]
    position = {name: index for index, name in enumerate(fit_cre)}
    unknown = [str(n) for n in negative_control_cre if str(n) not in position]
    if unknown:
        raise ValueError(
            f"negative control(s) {unknown} are not columns of the fit; it "
            f"covers {len(fit_cre)} cCREs"
        )
    control_fit_idx = np.array(
        [position[str(n)] for n in negative_control_cre], dtype=np.int64
    )

    # The controls must also be present in the data, or their T7 totals -- and
    # therefore eligibility -- cannot be evaluated.
    fit_to_data = {int(fit): int(data) for data, fit in enumerate(cre_index)}
    absent = [
        str(n) for n, fit in zip(negative_control_cre, control_fit_idx)
        if int(fit) not in fit_to_data
    ]
    if absent:
        raise ValueError(
            f"negative control(s) {absent} are in the fit but not in the data "
            "passed here; the eligibility filter needs their T7 totals"
        )
    control_data_idx = np.array(
        [fit_to_data[int(fit)] for fit in control_fit_idx], dtype=np.int64
    )

    n_groups = len(draws["group_names"])
    t7_totals = np.zeros((n_groups, len(fit_cre)), dtype=np.float64)
    for column, fit_column in zip(control_data_idx, control_fit_idx):
        t7_totals[:, fit_column] = np.bincount(
            group_index,
            weights=np.asarray(t7[:, column], dtype=np.float64),
            minlength=n_groups,
        )

    baseline = negative_control_log_baseline(
        draws["log_gamma"],
        control_fit_idx,
        t7_totals=t7_totals,
        t7_threshold=t7_threshold,
        control_sd_multiplier=sd_multiplier,
    )
    if verbose:
        present = np.unique(group_index)
        eligible = int(baseline.eligible[present].sum())
        covered = int(np.isin(group_index, present[baseline.eligible[present]]).sum())
        log(
            f"[copies] control reference: {len(negative_control_cre)} controls, "
            f"pooled T7 >= {t7_threshold:g} -> {eligible}/{len(present)} cell types "
            f"eligible, covering {covered:,}/{len(group_index):,} cells "
            "(the rest are NaN)"
        )
    return baseline.log_reference


def infer_copy_number(
    t7: np.ndarray,
    cre: np.ndarray,
    group_labels: np.ndarray,
    draws: Mapping[str, np.ndarray],
    *,
    kmax: int,
    cre_names: Sequence[str],
    obs_names: Sequence[str] | None = None,
    return_sd: bool = False,
    return_probability: bool = False,
    return_activity: bool = False,
    return_activity_normalized: bool = False,
    return_deviation: bool = False,
    negative_control_cre: Sequence[str] | None = None,
    negative_control_t7_threshold: float = DEFAULT_CONTROL_T7_THRESHOLD,
    negative_control_sd_multiplier: float = 0.0,
    chunk: int = 400,
    max_draws: int | None = None,
    dtype=np.float32,
    level: str = "subclass",
    infection_model: str = "copy_number",
    verbose: bool = True,
) -> CopyNumberMatrix:
    """Infer ``E[k | obs]`` for every (cell, cCRE) pair.

    Parameters
    ----------
    t7, cre : (n_cells, n_cre) observed counts, aligned to ``cre_names``.
    group_labels : (n_cells,) cell-type labels matching the fit's granularity.
    draws
        As returned by :func:`load_copy_number_draws`; must carry ``group_names``
        and ``cre_names`` so the axes can be aligned rather than assumed.
    kmax
        The truncation used at fit time. Read it from the run manifest; a
        different grid renormalises the posterior and changes the answer.
    max_draws
        Thin the posterior to this many evenly-spaced draws. Cost is linear in
        the draw count and the all-zero baseline alone is
        ``n_cell_types x n_cre`` patterns -- 128k for the production fit --
        which is paid whatever the cell count. A posterior *mean* over 200 draws
        is within Monte Carlo error of one over 1,000, so this is usually the
        difference between minutes and an hour. Evenly spaced rather than random
        so the result is reproducible without a seed.
    return_probability
        Also return ``P(k >= 1 | obs)`` -- the posterior probability the cell
        received any copy. Free to compute (same posterior weights), but each
        extra matrix is another ``n_cells x n_cre`` array.
    return_activity
        Also return the posterior mean per-cell activity. Same cost and caveat.
    return_activity_normalized
        Also return that activity divided by the cell type's negative-control
        reference, so 1 is background. Requires ``negative_control_cre``.
    return_deviation
        Also return the single-cell factor ``E[G | cre, k]`` alone -- the part of
        the activity that varies within a cell type. No control reference is
        involved, so unlike ``activity_normalized`` it is defined everywhere.
    negative_control_cre
        Names of the negative-control cCREs. These must be the controls the fit
        was run with -- :func:`infer_copy_number_from_fit` reads them from the
        manifest so they cannot disagree.
    negative_control_t7_threshold
        Pooled control T7 a cell type needs before it gets a reference; cell
        types below it come back ``NaN`` rather than being scored against a
        background the data cannot support. Defaults to the published 50.
    negative_control_sd_multiplier
        Raise the reference by this many control standard deviations for a
        stricter background. 0 reproduces the published mean-control reference.
    dtype
        Output dtype. float32 halves a 636 MB matrix and is far finer than the
        posterior is sharp.
    """
    # obsm count matrices arrive as floats; run_model casts them the same way, so
    # the reconstruction must see exactly the integers the fit saw. Without this
    # the pattern stack below promotes to float and the index columns stop being
    # usable as indices.
    t7 = np.asarray(t7).astype(np.int64)
    cre = np.asarray(cre).astype(np.int64)
    if t7.shape != cre.shape:
        raise ValueError(f"t7 {t7.shape} and cre {cre.shape} must have equal shape")
    n_cells, n_cre = t7.shape
    if n_cre != len(cre_names):
        raise ValueError(f"{n_cre} count columns but {len(cre_names)} cCRE names")
    if len(group_labels) != n_cells:
        raise ValueError(f"{len(group_labels)} labels for {n_cells} cells")

    group_index, cre_index = _align_axes(
        cre_names, draws["cre_names"], group_labels, draws["group_names"]
    )
    n_groups = len(draws["group_names"])
    draws = thin_draws(draws, max_draws, verbose=verbose)
    _warn_if_truncated(draws, kmax, group_index, cre_index)

    log_baseline = None
    if return_activity_normalized:
        if negative_control_cre is None:
            raise ValueError(
                "return_activity_normalized needs negative_control_cre -- the "
                "activity is on an arbitrary scale until it is divided by the "
                "controls' geometric mean, so there is no sensible default"
            )
        # After thinning, so the baseline and the draws it divides are the same
        # posterior samples.
        log_baseline = _control_log_baseline(
            draws, t7, group_index, cre_index, negative_control_cre,
            t7_threshold=negative_control_t7_threshold,
            sd_multiplier=negative_control_sd_multiplier,
            verbose=verbose,
        )

    # 1. Baseline: the all-zero observation, once per (cell type, cCRE). Only over
    #    the cell types and cCREs this data actually contains -- a fit covers 328
    #    subclasses and 389 cCREs, and a section or a subsample may use far fewer.
    present_groups, group_slot = np.unique(group_index, return_inverse=True)
    present_cres, cre_slot = np.unique(cre_index, return_inverse=True)
    if verbose:
        log(
            f"[copies] baseline grid: {len(present_groups)} cell types x "
            f"{len(present_cres)} cCREs (fit covers {n_groups} x {len(draws['cre_names'])})"
        )
    grid_group, grid_cre = np.meshgrid(present_groups, present_cres, indexing="ij")
    zeros = np.zeros(grid_group.size, dtype=np.int64)
    baseline = posterior_k_moments(
        zeros, zeros, grid_group.ravel(), grid_cre.ravel(), draws, kmax,
        chunk=chunk, log_baseline=log_baseline,
    )
    shape = (len(present_groups), len(present_cres))
    baseline_mean = baseline.mean.reshape(shape)
    baseline_sd = baseline.sd.reshape(shape)

    # 2. Broadcast it, then correct only where something was actually observed.
    copies = baseline_mean[np.ix_(group_slot, cre_slot)].astype(dtype, copy=True)
    sd = baseline_sd[np.ix_(group_slot, cre_slot)].astype(dtype, copy=True) if return_sd else None
    p_infected = (
        baseline.p_infected.reshape(shape)[np.ix_(group_slot, cre_slot)].astype(dtype, copy=True)
        if return_probability else None
    )
    activity = (
        baseline.activity.reshape(shape)[np.ix_(group_slot, cre_slot)].astype(dtype, copy=True)
        if return_activity else None
    )
    activity_normalized = (
        baseline.activity_normalized.reshape(shape)[
            np.ix_(group_slot, cre_slot)
        ].astype(dtype, copy=True)
        if return_activity_normalized else None
    )
    deviation = (
        baseline.deviation.reshape(shape)[np.ix_(group_slot, cre_slot)].astype(
            dtype, copy=True
        )
        if return_deviation else None
    )

    rows, cols = np.nonzero((t7 > 0) | (cre > 0))
    if len(rows):
        patterns = np.stack(
            [group_index[rows], cre_index[cols], t7[rows, cols], cre[rows, cols]], axis=1
        ).astype(np.int64, copy=False)
        unique, inverse = np.unique(patterns, axis=0, return_inverse=True)
        if verbose:
            log(
                f"[copies] {len(rows):,} nonzero pairs collapse to "
                f"{len(unique):,} unique (cell type, cCRE, t7, cre) patterns"
            )
        moments = posterior_k_moments(
            unique[:, 2], unique[:, 3], unique[:, 0], unique[:, 1], draws, kmax,
            chunk=chunk, log_baseline=log_baseline,
        )
        copies[rows, cols] = moments.mean[inverse].astype(dtype)
        if sd is not None:
            sd[rows, cols] = moments.sd[inverse].astype(dtype)
        if p_infected is not None:
            p_infected[rows, cols] = moments.p_infected[inverse].astype(dtype)
        if activity is not None:
            activity[rows, cols] = moments.activity[inverse].astype(dtype)
        if activity_normalized is not None:
            activity_normalized[rows, cols] = (
                moments.activity_normalized[inverse].astype(dtype)
            )
        if deviation is not None:
            deviation[rows, cols] = moments.deviation[inverse].astype(dtype)
    elif verbose:
        log("[copies] no nonzero observations; every pair is at the baseline")

    if verbose:
        total = copies.sum(axis=1)
        log(
            f"[copies] expected genomes per cell: median {np.median(total):.3g}, "
            f"range {total.min():.3g}-{total.max():.3g}"
        )
    return CopyNumberMatrix(
        copies=copies,
        sd=sd,
        p_infected=p_infected,
        activity=activity,
        activity_normalized=activity_normalized,
        deviation=deviation,
        obs_names=None if obs_names is None else np.asarray(obs_names, dtype=object),
        cre_names=[str(name) for name in cre_names],
        kmax=int(kmax),
        level=level,
        infection_model=infection_model,
    )


def infer_copy_number_from_fit(
    data,
    fit_dir: Path | str,
    *,
    tag: str | None = None,
    return_sd: bool = False,
    return_probability: bool = False,
    return_activity: bool = False,
    return_activity_normalized: bool = False,
    return_deviation: bool = False,
    negative_control_cre: Sequence[str] | None = None,
    negative_control_t7_threshold: float = DEFAULT_CONTROL_T7_THRESHOLD,
    negative_control_sd_multiplier: float = 0.0,
    chunk: int = 400,
    max_draws: int | None = None,
    dtype=np.float32,
    verbose: bool = True,
) -> CopyNumberMatrix:
    """Infer the copy-number matrix for ``data`` under a fit on disk.

    ``kmax``, the cell-type granularity, the infection model and the negative
    controls are read from the fit's manifest, so the reconstruction cannot
    silently disagree with the model that produced the posterior.

    Parameters
    ----------
    data : CountData
        Must be assembled the same way the fit was -- same blacklist, same
        section -- or the axis alignment raises.
    negative_control_cre
        Overrides the manifest's ``annotated_negative_control_cre``. Only pass
        this to answer a deliberate "what if the reference were different"
        question; the default is what the fit and the published tables used.
    """
    draws, manifest = load_copy_number_draws(fit_dir, tag)
    config = manifest.get("config", {})
    kmax = config.get("kmax")
    if kmax is None:
        raise KeyError(
            f"{fit_dir} manifest records no kmax; the truncation is part of the "
            "posterior's normalisation and cannot be guessed"
        )
    level = config.get("level", "subclass")
    labels = data.class_ if level == "class" else data.subclass

    controls = negative_control_cre
    if controls is None:
        controls = config.get("annotated_negative_control_cre")
    if return_activity_normalized and not controls:
        raise KeyError(
            f"{fit_dir} manifest records no 'annotated_negative_control_cre', so "
            "the background reference cannot be reconstructed from the fit. Pass "
            "negative_control_cre= explicitly if you know which cCREs it used."
        )

    return infer_copy_number(
        data.t7,
        data.cre,
        labels,
        draws,
        kmax=int(kmax),
        cre_names=data.cre_names,
        obs_names=data.obs_names,
        return_sd=return_sd,
        return_probability=return_probability,
        return_activity=return_activity,
        return_activity_normalized=return_activity_normalized,
        return_deviation=return_deviation,
        negative_control_cre=controls,
        negative_control_t7_threshold=negative_control_t7_threshold,
        negative_control_sd_multiplier=negative_control_sd_multiplier,
        chunk=chunk,
        max_draws=max_draws,
        dtype=dtype,
        level=level,
        infection_model=config.get("infection_model", "copy_number"),
        verbose=verbose,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m baystarrfish copy-number --fit-dir ... --out ...``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m baystarrfish copy-number",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--fit-dir", type=Path, required=True,
                        help="directory holding run_manifest.json and the posterior draws")
    parser.add_argument("--tag", default=None, help="defaults to the manifest tag")
    parser.add_argument("--out", type=Path, required=True, help="output .npz")
    parser.add_argument("--h5ad", type=Path, default=None,
                        help="input; defaults to the package's configured dataset")
    parser.add_argument("--section", choices=["all", "sec1", "sec2"], default=None,
                        help="defaults to the section recorded in the fit manifest")
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=400,
                        help="patterns per block; peak memory is n_draws x chunk x (kmax+1)")
    parser.add_argument("--max-draws", type=int, default=None,
                        help="thin the posterior to N evenly-spaced draws; cost is "
                             "linear in the draw count and 200 is usually within "
                             "Monte Carlo error of the full 1,000")
    parser.add_argument("--with-sd", action="store_true",
                        help="also compute the posterior sd (one more full matrix)")
    parser.add_argument("--with-probability", action="store_true",
                        help="also compute P(k >= 1 | obs), the infection probability")
    parser.add_argument("--with-activity", action="store_true",
                        help="also compute the posterior mean per-cell activity")
    parser.add_argument("--with-activity-normalized", action="store_true",
                        help="also compute that activity divided by the cell "
                             "type's negative-control reference, so 1 is "
                             "background; NaN where no reference is eligible")
    parser.add_argument("--with-deviation", action="store_true",
                        help="also compute the single-cell factor E[G] alone -- "
                             "the part of the activity that varies within a cell "
                             "type, with no control reference in it")
    parser.add_argument("--negative-control-t7-threshold", type=float,
                        default=DEFAULT_CONTROL_T7_THRESHOLD,
                        help="pooled control T7 a cell type needs before it gets "
                             "a background reference (default: the published 50)")
    parser.add_argument("--negative-control-sd-multiplier", type=float, default=0.0,
                        help="raise the reference by this many control SDs for a "
                             "stricter background (default 0 = the mean control)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write CSVs with this stem, e.g. out/k -> "
                             "out/k_copies.csv.gz")
    parser.add_argument("--csv-which", nargs="+", default=["copies"],
                        choices=list(MATRIX_NAMES),
                        help="which matrices to write as CSV (default: the "
                             "posterior mean only -- each is ~1.3 GB of text)")
    parser.add_argument("--csv-decimals", type=int, default=5)
    args = parser.parse_args(argv)

    from ..data import CountData

    manifest = json.loads((args.fit_dir / "run_manifest.json").read_text())
    section = args.section or manifest.get("section", "all")
    data = CountData.from_h5ad(
        args.h5ad,
        section=section,
        max_cells=args.max_cells,
        max_cres=args.max_cres,
        seed=args.seed,
        negative_control_mode=manifest.get("negative_control_mode", "ordinary"),
    )
    matrix = infer_copy_number_from_fit(
        data, args.fit_dir, tag=args.tag, return_sd=args.with_sd,
        return_probability=args.with_probability,
        return_activity=args.with_activity,
        return_activity_normalized=args.with_activity_normalized,
        return_deviation=args.with_deviation,
        negative_control_t7_threshold=args.negative_control_t7_threshold,
        negative_control_sd_multiplier=args.negative_control_sd_multiplier,
        chunk=args.chunk, max_draws=args.max_draws,
    )
    path = matrix.write_npz(args.out)
    log(f"[copies] wrote {matrix.n_cells:,} x {matrix.n_cre} matrix to {path}")
    if args.csv:
        for which in args.csv_which:
            if getattr(matrix, which) is None:
                log(f"[copies] skipping {which} CSV: not computed")
                continue
            written = matrix.write_csv(
                Path(f"{args.csv}_{which}.csv.gz"), which, decimals=args.csv_decimals
            )
            log(f"[copies] wrote {which} CSV to {written}")
    return 0
