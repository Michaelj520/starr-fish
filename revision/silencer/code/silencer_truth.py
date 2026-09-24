"""The annotated-silencer truth matrix, aligned to model cell-type labels.

The ChromHMM annotation produced by ``annotate_chromstates.py`` is indexed by
cCRE x ChromStates *filename* (``001_CLA_EPd_CTX_Car3_Glut``). The Bayesian
posterior is indexed by Allen subclass *name* (``CLA-EPd-CTX Car3 Glut``). The
filename's underscores are lossy -- they stand for both spaces and hyphens -- so
the join goes through the three-digit subclass number in the filename and the
Allen ``cluster_annotation_term.csv`` table, exactly as
``STARRFISH_in_vivo/Data/preprocess_chromstate.py`` did for the Chr-A assay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from baystarrfish.data.paths import data_root, revision_data_root

#: ``001_CLA_EPd_CTX_Car3_Glut`` -> subclass number 1.
_FILENAME_PREFIX: Final[re.Pattern[str]] = re.compile(r"^(\d{3})_(.+)$")

SILENCER_STATES: Final[tuple[str, str]] = ("Chr-R", "Hc-P")


@dataclass(frozen=True)
class SilencerTruth:
    """Cell-type x cCRE annotation frames, both on Allen subclass names."""

    repressive_fraction: pd.DataFrame
    nd_fraction: pd.DataFrame
    unmapped_columns: tuple[str, ...]

    def positives(self, cutoff: float) -> pd.DataFrame:
        return self.repressive_fraction.ge(cutoff)


def subclass_number_to_name(annotation_csv: Path | None = None) -> pd.Series:
    """Map Allen subclass number -> subclass name with ``/`` replaced by ``-``."""
    path = (
        Path(annotation_csv)
        if annotation_csv is not None
        else data_root() / "abc_atlas" / "cluster_annotation_term.csv"
    )
    table = pd.read_csv(path, usecols=["subclass_number", "subclass"])
    table = table.dropna(subset=["subclass_number", "subclass"])
    table["subclass_number"] = table["subclass_number"].astype(int)
    table["subclass"] = table["subclass"].astype(str).str.replace("/", "-", regex=False)
    return table.groupby("subclass_number")["subclass"].first()


def rename_to_subclass_names(
    matrix: pd.DataFrame, mapping: pd.Series
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Rename cCRE x filename columns to subclass names, dropping unmapped ones."""
    renamed: dict[str, str] = {}
    unmapped: list[str] = []
    for column in matrix.columns.astype(str):
        match = _FILENAME_PREFIX.match(column)
        number = int(match.group(1)) if match else None
        if number is None or number not in mapping.index:
            unmapped.append(column)
            continue
        renamed[column] = mapping.loc[number]
    kept = matrix.loc[:, list(renamed)].rename(columns=renamed)
    if kept.columns.has_duplicates:
        duplicated = sorted(kept.columns[kept.columns.duplicated()].unique())
        raise ValueError(f"subclass names are not unique after mapping: {duplicated}")
    return kept, tuple(unmapped)


#: Kept for callers written against the private name.
_rename_to_subclass_names = rename_to_subclass_names


def load_silencer_truth(
    results_dir: Path | None = None, annotation_csv: Path | None = None
) -> SilencerTruth:
    """Load the repressive and ND fraction matrices as cell type x cCRE."""
    root = (
        Path(results_dir)
        if results_dir is not None
        else revision_data_root().parent / "silencer" / "results"
    )
    repressive_path = root / "cre_by_celltype_repressive_fraction.csv"
    nd_path = root / "cre_by_celltype_ND_fraction.csv"
    for path in (repressive_path, nd_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run annotate_chromstates.py first")
    mapping = subclass_number_to_name(annotation_csv)
    repressive, unmapped = rename_to_subclass_names(
        pd.read_csv(repressive_path, index_col=0), mapping
    )
    nd, _ = rename_to_subclass_names(pd.read_csv(nd_path, index_col=0), mapping)
    return SilencerTruth(
        repressive_fraction=repressive.T,
        nd_fraction=nd.reindex(columns=repressive.columns).T,
        unmapped_columns=unmapped,
    )
