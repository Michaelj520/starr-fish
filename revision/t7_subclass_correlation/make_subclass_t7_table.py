#!/usr/bin/env python3
"""Per-subclass total T7 counts (blacklisted cCREs removed), split by section.

Writes ``revision/Data/subclass_total_t7_counts.csv`` with, per subclass, the
total T7 barcode counts summed over all cells in physical section 1, section 2,
and both combined. The 11 blacklisted cCREs (``revision/Data/cre_blacklist.csv``)
and all blank barcodes are dropped before summing, matching the ``total_t7``
used to rank the correlation heatmaps.

Sections follow the 5/28 z-scan convention (``Conv_zscan2_`` -> sec1,
``Conv_zscan1_`` -> sec2), the same mapping used elsewhere in the revision.
"""

from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REVISION_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(REVISION_DIR, "Data")
DEFAULT_H5AD = os.path.join(
    DATA_DIR, "scdata_5_28_2025_BRBB500gn_final_CRE_T7CRE_NEW.h5ad"
)
BLACKLIST_CSV = os.path.join(DATA_DIR, "cre_blacklist.csv")
OUT_CSV = os.path.join(DATA_DIR, "subclass_total_t7_counts.csv")


def section_labels(obs_names: pd.Index) -> pd.Series:
    """Map the 5/28 z-scan identifiers to the two physical sections."""
    names = pd.Index(obs_names).astype(str)
    zscan = names.to_series(index=names).str.extract(r"^Conv_zscan([12])_", expand=False)
    labels = zscan.map({"2": "sec1", "1": "sec2"})
    if labels.isna().any():
        examples = names[labels.isna().to_numpy()][:5].tolist()
        raise ValueError(f"cannot assign section for obs names: {examples}")
    return labels


def main() -> None:
    A = ad.read_h5ad(DEFAULT_H5AD, backed="r")
    try:
        t7 = A.obsm["T7CRE"]
        if not isinstance(t7, pd.DataFrame):
            raise TypeError("expected obsm['T7CRE'] to be a DataFrame")
        subclass = A.obs["subclass_name"].astype(str)
        cls = A.obs["class_name"].astype(str)
        section = section_labels(A.obs_names)
    finally:
        A.file.close()

    blacklist = pd.read_csv(BLACKLIST_CSV)["cre"].astype(str).tolist()
    bl = [c for c in blacklist if c in t7.columns]
    blanks = [c for c in t7.columns if str(c).lower().startswith("blank")]
    drop = sorted(set(bl) | set(blanks))
    per_cell = t7.drop(columns=drop).sum(axis=1)  # total T7 per cell
    print(f"dropped {len(bl)} blacklisted cCREs and {len(blanks)} blank barcodes; "
          f"summing over {t7.shape[1] - len(drop)} columns")

    grp = per_cell.groupby(subclass.to_numpy())
    sec1 = per_cell[section.to_numpy() == "sec1"].groupby(
        subclass[section.to_numpy() == "sec1"].to_numpy()).sum()
    sec2 = per_cell[section.to_numpy() == "sec2"].groupby(
        subclass[section.to_numpy() == "sec2"].to_numpy()).sum()

    table = pd.DataFrame({
        "subclass": grp.sum().index,
        "class": cls.groupby(subclass.to_numpy()).agg(lambda s: s.mode().iat[0]).reindex(grp.sum().index).to_numpy(),
        "n_cells": subclass.value_counts().reindex(grp.sum().index).astype(int).to_numpy(),
        "sec1_total_t7": sec1.reindex(grp.sum().index).fillna(0).round().astype(np.int64).to_numpy(),
        "sec2_total_t7": sec2.reindex(grp.sum().index).fillna(0).round().astype(np.int64).to_numpy(),
        "total_t7": grp.sum().round().astype(np.int64).to_numpy(),
    })
    table = table.sort_values("total_t7", ascending=False).reset_index(drop=True)

    # sanity: section columns sum to the total
    mismatch = (table["sec1_total_t7"] + table["sec2_total_t7"] - table["total_t7"]).abs()
    if mismatch.max() > 1:
        raise AssertionError(f"sec1+sec2 != total (max diff {mismatch.max()})")

    table.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(table)} subclasses)")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
