"""Contract tests for the multiBC loader, on a small synthetic h5ad."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from multibc_data import GROUP_LABEL, load_multibc_counts

FEATURES = [
    "CRE010_bar001",
    "CRE010_bar002",
    "CRE020_bar003",
    "barcode only_bar004",
    "barcode only_bar005",
]
DNA = [40, 10, 25, 7, 0]
SATURATION_FAIL = [False, True, False, False, True, False]
PASSING = [i for i, failed in enumerate(SATURATION_FAIL) if not failed]


def _write_inputs(
    tmp_path: Path,
    *,
    cre_scale: float = 1.0,
    drop_barcode: str | None = None,
    with_saturation_flag: bool = True,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    obs_names = [f"Conv_zscan1_000--{i}" for i in range(6)]
    barcodes = [name.rsplit("_", 1)[1] for name in FEATURES]
    cre = pd.DataFrame(
        rng.integers(0, 20, size=(6, 5)).astype(np.float64) * cre_scale,
        index=obs_names,
        columns=barcodes,
    )
    t7 = pd.DataFrame(
        rng.integers(0, 9, size=(6, 5)).astype(np.float32),
        index=obs_names,
        columns=barcodes,
    )
    # Store the columns out of var order: alignment must follow the barcode key.
    shuffled = barcodes[::-1]
    if drop_barcode is not None:
        shuffled = [code for code in shuffled if code != drop_barcode]
    obs = pd.DataFrame(index=obs_names)
    if with_saturation_flag:
        obs["saturation_fail"] = SATURATION_FAIL
    adata = ad.AnnData(
        X=np.zeros((6, 5), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=FEATURES),
    )
    adata.obsm["CRE"] = cre.loc[:, shuffled]
    adata.obsm["T7"] = t7.loc[:, shuffled]
    adata.obsm["X_spatial"] = rng.normal(size=(6, 2))
    h5ad = tmp_path / "multibc.h5ad"
    adata.write_h5ad(h5ad)

    dna = pd.DataFrame(
        {
            "element": [name.replace("barcode only_", "barcode_only_") for name in FEATURES],
            "DNA_count": DNA,
        }
    ).iloc[::-1]
    dna_csv = tmp_path / "dna.csv"
    dna.to_csv(dna_csv, index=False)
    return h5ad, dna_csv, cre, t7


def test_one_column_per_barcode_aligned_by_barcode(tmp_path: Path) -> None:
    h5ad, dna_csv, cre, t7 = _write_inputs(tmp_path)
    data = load_multibc_counts(h5ad, dna_csv, drop_saturation_fail=False)

    assert data.cre_names == FEATURES
    np.testing.assert_array_equal(data.cre, cre.to_numpy().astype(np.int64))
    np.testing.assert_array_equal(data.t7, t7.to_numpy().astype(np.int64))
    assert data.cre.dtype == np.int64 and data.t7.dtype == np.int64
    np.testing.assert_allclose(data.lib_size_log, np.log1p(DNA))
    assert data.negative_controls == ["barcode only_bar004", "barcode only_bar005"]
    assert data.negative_control_mask is None
    assert data.negative_control_mode == "ordinary"
    assert data.blacklist == []
    assert set(data.subclass) == {GROUP_LABEL} and set(data.class_) == {GROUP_LABEL}
    assert data.subclass_cell_counts.to_dict() == {GROUP_LABEL: 6}
    assert data.spatial is not None and data.spatial.shape == (6, 2)
    assert data.cre_info.loc["CRE020_bar003", "element"] == "CRE020"
    assert data.cre_info["is_control"].sum() == 2


def test_drops_saturation_fail_cells_by_default(tmp_path: Path) -> None:
    h5ad, dna_csv, cre, t7 = _write_inputs(tmp_path)
    data = load_multibc_counts(h5ad, dna_csv)

    assert data.obs_names.tolist() == cre.index[PASSING].tolist()
    np.testing.assert_array_equal(data.cre, cre.iloc[PASSING].to_numpy().astype(np.int64))
    np.testing.assert_array_equal(data.t7, t7.iloc[PASSING].to_numpy().astype(np.int64))
    assert data.spatial.shape == (len(PASSING), 2)
    assert data.subclass_cell_counts.to_dict() == {GROUP_LABEL: len(PASSING)}


def test_missing_saturation_flag_is_an_error_only_when_filtering(tmp_path: Path) -> None:
    h5ad, dna_csv, _, _ = _write_inputs(tmp_path, with_saturation_flag=False)
    with pytest.raises(KeyError, match="saturation_fail"):
        load_multibc_counts(h5ad, dna_csv)
    assert load_multibc_counts(h5ad, dna_csv, drop_saturation_fail=False).n_cells == 6


def test_run_kwargs_describe_a_single_group(tmp_path: Path) -> None:
    h5ad, dna_csv, _, _ = _write_inputs(tmp_path)
    kwargs = load_multibc_counts(h5ad, dna_csv).to_run_kwargs()
    assert kwargs["t7"].shape == kwargs["cre"].shape == (len(PASSING), 5)
    assert np.unique(kwargs["subclass_labels"]).size == 1
    assert np.unique(kwargs["class_labels"]).size == 1
    assert kwargs["negative_control_mask"] is None


def test_max_cells_keeps_rows_aligned(tmp_path: Path) -> None:
    h5ad, dna_csv, cre, t7 = _write_inputs(tmp_path)
    full = load_multibc_counts(h5ad, dna_csv)
    sub = load_multibc_counts(h5ad, dna_csv, max_cells=3, seed=1)
    # Subsampling draws from the cells that passed the saturation filter.
    positions = pd.Index(full.obs_names).get_indexer(sub.obs_names)
    assert full.n_cells == len(PASSING)
    assert sub.n_cells == 3 and (positions >= 0).all()
    np.testing.assert_array_equal(sub.cre, full.cre[positions])
    np.testing.assert_array_equal(sub.t7, full.t7[positions])
    np.testing.assert_array_equal(sub.spatial, full.spatial[positions])
    assert sub.subclass_cell_counts.to_dict() == {GROUP_LABEL: 3}


def test_rejects_noninteger_counts(tmp_path: Path) -> None:
    h5ad, dna_csv, _, _ = _write_inputs(tmp_path, cre_scale=0.5)
    with pytest.raises(ValueError, match="noninteger"):
        load_multibc_counts(h5ad, dna_csv)


def test_rejects_missing_barcode_column(tmp_path: Path) -> None:
    h5ad, dna_csv, _, _ = _write_inputs(tmp_path, drop_barcode="bar003")
    with pytest.raises(ValueError, match="barcodes differ"):
        load_multibc_counts(h5ad, dna_csv)
