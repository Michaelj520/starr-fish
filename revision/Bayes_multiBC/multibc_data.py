"""Load the SFv6 multi-barcode STARR-FISH input as a baystarrfish ``CountData``.

``CountData.from_h5ad`` is built for the SFv8 in vivo schema: Allen cell-type
labels in ``obs``, ``uns['CRE_info']``, the 400-cCRE nanopore table and the AAV
barcode-mismatch blacklist. The multiBC file has none of those, so this module
fills the same container from what it does carry:

* every barcode is its own model column, so it gets its own ``log_gamma`` and
  its own abundance ``a``; the fitted name is the ``<CRE>_barNNN`` feature from
  ``var_names``, while ``obsm['CRE']`` / ``obsm['T7']`` are keyed by ``barNNN``;
* all cells are one cell type, so subclass and class are one constant label;
* cells flagged in ``obs['saturation_fail']`` are dropped by default;
* the abundance prior is ``log1p`` of the per-barcode plasmid DNA count;
* the ``barcode only`` barcodes are the negative controls, fitted as ordinary
  columns, which is what the ``direct`` activity model requires;
* there is no blacklist: the SFv8 list describes a different barcode library.

Both ``obsm['CRE']`` and ``obsm['T7']`` are transcript counts of the same
construct. Neither is DNA; the plasmid DNA count only sets the abundance prior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from baystarrfish._log import log
from baystarrfish.data import CountData

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
DEFAULT_H5AD = REVISION / "Data" / "scdata_260821_SFv6_multiBC_FINAL.h5ad"
DEFAULT_DNA_CSV = REVISION / "Data" / "STARR_seq_Counts_wReps_DNA.csv"

#: The one cell-type label every cell carries, used as both subclass and class.
GROUP_LABEL = "all"
#: Element name of the barcode-only negative-control constructs.
CONTROL_ELEMENT = "barcode only"
#: Boolean ``obs`` column marking cells whose images saturated.
SATURATION_FLAG = "saturation_fail"

# The feature-name parser and the RNA/DNA join (strict set equality plus the
# barcode_only_ -> "barcode only_" alias) have one definition, in the
# barcode_shuffling analysis; the fold-change and Bayesian activities must
# match barcodes to DNA identically.
BARCODE_SHUFFLING = REVISION / "barcode_shuffling"
if str(BARCODE_SHUFFLING) not in sys.path:
    sys.path.insert(0, str(BARCODE_SHUFFLING))

from compute_activities import feature_metadata, load_dna_counts  # noqa: E402

__all__ = [
    "CONTROL_ELEMENT",
    "DEFAULT_DNA_CSV",
    "DEFAULT_H5AD",
    "GROUP_LABEL",
    "SATURATION_FLAG",
    "load_multibc_counts",
]


def _aligned_counts(
    frame: object, key: str, obs_names: pd.Index, barcodes: pd.Index
) -> np.ndarray:
    """``obsm[key]`` as an int64 ``(n_cells, n_barcodes)`` array in ``barcodes`` order."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"obsm[{key!r}] must be a barcode-keyed DataFrame, got {type(frame).__name__}"
        )
    columns = frame.columns.astype(str)
    if columns.has_duplicates:
        raise ValueError(f"obsm[{key!r}] has duplicate barcode columns")
    if not frame.index.astype(str).equals(obs_names):
        raise ValueError(f"obsm[{key!r}] rows are not aligned to obs_names")
    missing = barcodes.difference(columns)
    extra = columns.difference(barcodes)
    if len(missing) or len(extra):
        raise ValueError(
            f"obsm[{key!r}] barcodes differ from var_names: "
            f"missing={missing[:5].tolist()}, extra={extra[:5].tolist()}"
        )
    frame = frame.set_axis(columns, axis=1)
    values = frame.loc[:, barcodes].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"obsm[{key!r}] counts must be finite and nonnegative")
    counts = np.rint(values)
    if not np.array_equal(values, counts):
        raise ValueError(f"obsm[{key!r}] holds noninteger values; expected raw counts")
    return counts.astype(np.int64)


def _saturation_pass(obs: pd.DataFrame) -> np.ndarray:
    """``(n_cells,)`` bool mask of the cells not flagged in ``obs[SATURATION_FLAG]``."""
    if SATURATION_FLAG not in obs.columns:
        raise KeyError(
            f"obs has no {SATURATION_FLAG!r} column; have {obs.columns.tolist()}"
        )
    flag = obs[SATURATION_FLAG]
    if not pd.api.types.is_bool_dtype(flag) or flag.isna().any():
        raise ValueError(
            f"obs[{SATURATION_FLAG!r}] must be boolean with no missing values, "
            f"got dtype {flag.dtype}"
        )
    return ~flag.to_numpy(dtype=bool)


def _subsample(positions: np.ndarray, max_cells: int | None, seed: int) -> np.ndarray:
    """A sorted random subset of ``positions`` for a smoke run, or all of them."""
    if max_cells is None or max_cells >= len(positions):
        return positions
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(positions, size=max_cells, replace=False))


def load_multibc_counts(
    h5ad: Path | str = DEFAULT_H5AD,
    dna_csv: Path | str = DEFAULT_DNA_CSV,
    *,
    drop_saturation_fail: bool = True,
    max_cells: int | None = None,
    seed: int = 0,
) -> CountData:
    """Read the multiBC input into a ``CountData`` with one column per barcode.

    ``drop_saturation_fail`` removes the cells flagged in ``obs['saturation_fail']``;
    ``max_cells`` then subsamples the remaining cells for a smoke run.

    Only ``obs``, ``var`` and ``obsm['CRE' | 'T7' | 'X_spatial']`` are read;
    ``X``, ``X_raw``, ``layers`` and ``obsp`` stay on disk.
    """
    h5ad = Path(h5ad)
    dna_csv = Path(dna_csv)
    for path in (h5ad, dna_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    log(f"[input] loading {h5ad}")

    with h5py.File(h5ad, "r") as handle:
        obs = read_elem(handle["obs"])
        var_names = pd.Index(read_elem(handle["var"]).index.astype(str))
        obsm = handle["obsm"]
        t7_key = "T7CRE" if "T7CRE" in obsm else "T7"
        for key in ("CRE", t7_key):
            if key not in obsm:
                raise KeyError(f"{h5ad} has no obsm[{key!r}]; have {list(obsm)}")
        cre_frame = read_elem(obsm["CRE"])
        t7_frame = read_elem(obsm[t7_key])
        spatial = (
            np.asarray(read_elem(obsm["X_spatial"]), dtype=np.float64)[:, :2]
            if "X_spatial" in obsm
            else None
        )
    obs_names = pd.Index(obs.index.astype(str))
    if obs_names.has_duplicates:
        raise ValueError("obs_names must be unique")

    features = feature_metadata(var_names)
    barcodes = pd.Index(features["barcode"].to_numpy(), dtype=str)
    cre = _aligned_counts(cre_frame, "CRE", obs_names, barcodes)
    t7 = _aligned_counts(t7_frame, t7_key, obs_names, barcodes)
    del cre_frame, t7_frame

    dna = load_dna_counts(dna_csv, var_names)
    cre_info = pd.DataFrame(
        {
            "element": features["cre"].to_numpy(),
            "barcode": barcodes.to_numpy(),
            "dna_element": dna["element"].to_numpy(),
            "DNA_count": dna["DNA_count"].to_numpy(dtype=np.float64),
            "DNA_CPM": dna["DNA_CPM"].to_numpy(dtype=np.float64),
        },
        index=pd.Index(var_names, name="cre"),
    )
    cre_info["is_control"] = cre_info["element"].eq(CONTROL_ELEMENT)
    negative_controls = cre_info.index[cre_info["is_control"]].tolist()
    if not negative_controls:
        raise ValueError(
            f"no {CONTROL_ELEMENT!r} barcodes found; the direct activity contrast "
            "needs negative controls"
        )

    positions = np.arange(len(obs_names))
    if drop_saturation_fail:
        positions = positions[_saturation_pass(obs)]
        log(
            f"[input] dropped {len(obs_names) - len(positions):,} of "
            f"{len(obs_names):,} cells with {SATURATION_FLAG}=True"
        )
    if positions.size == 0:
        raise ValueError(f"no cells left in {h5ad} after filtering")
    positions = _subsample(positions, max_cells, seed)
    t7, cre = t7[positions], cre[positions]
    obs_names = obs_names[positions]
    spatial = None if spatial is None else spatial[positions]
    n_cells = len(obs_names)
    labels = np.full(n_cells, GROUP_LABEL, dtype=object)

    data = CountData(
        t7=t7,
        cre=cre,
        subclass=labels,
        class_=labels.copy(),
        lib_size_log=np.log1p(cre_info["DNA_count"].to_numpy()),
        cre_names=cre_info.index.tolist(),
        negative_control_mask=None,
        negative_controls=negative_controls,
        negative_control_mode="ordinary",
        blacklist=[],
        cre_info=cre_info,
        subclass_cell_counts=pd.Series(
            [n_cells],
            index=pd.Index([GROUP_LABEL], name="subclass"),
            name="n_cells",
        ),
        pooled_negative_control=None,
        section="all",
        source=str(h5ad),
        obs_names=obs_names.to_numpy(dtype=object),
        spatial=spatial,
    )
    log(
        f"[input] {data.n_cells:,} cells x {data.n_cre} barcodes "
        f"({cre_info['element'].nunique()} elements), one group "
        f"{GROUP_LABEL!r}, {len(negative_controls)} control barcodes"
    )
    return data
