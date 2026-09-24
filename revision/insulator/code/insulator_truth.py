"""Annotated-insulator matrix, aligned to model cell-type labels.

An insulator is a cCRE x cell-type pair whose ChromHMM coverage is almost
entirely ``Chr-O`` and carries essentially no ``Chr-A``:

    Chr-O fraction > --open-cutoff (0.9)   and   Chr-A fraction < --active-cutoff (0.1)

Both matrices come from ``revision/silencer/code/annotate_chromstates.py``; the
filename -> Allen subclass mapping is shared with ``silencer_truth`` so the two
annotations are joinable pair for pair. chrX is unsegmented, so its cCREs are
NaN in both matrices and are never counted either way.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from baystarrfish.data.paths import revision_data_root

_SILENCER_CODE: Final[Path] = Path(__file__).resolve().parents[2] / "silencer" / "code"
if str(_SILENCER_CODE) not in sys.path:
    sys.path.insert(0, str(_SILENCER_CODE))

from silencer_truth import rename_to_subclass_names, subclass_number_to_name  # noqa: E402

OPEN_STATE: Final[str] = "Chr_O"
ACTIVE_STATE: Final[str] = "Chr_A"
DEFAULT_OPEN_CUTOFF: Final[float] = 0.9
DEFAULT_ACTIVE_CUTOFF: Final[float] = 0.1


@dataclass(frozen=True)
class InsulatorTruth:
    """Cell type x cCRE state fractions and the derived insulator mask."""

    open_fraction: pd.DataFrame
    active_fraction: pd.DataFrame
    nd_fraction: pd.DataFrame
    unmapped_columns: tuple[str, ...]

    def mask(self, open_cutoff: float = DEFAULT_OPEN_CUTOFF,
             active_cutoff: float = DEFAULT_ACTIVE_CUTOFF) -> pd.DataFrame:
        """True where the pair is an annotated insulator; NaN coverage is False."""
        return (self.open_fraction > open_cutoff) & (self.active_fraction < active_cutoff)

    @property
    def measured(self) -> pd.DataFrame:
        """True where the pair has a segmentation at all (chrX is NaN)."""
        return self.open_fraction.notna() & self.active_fraction.notna()


def _load_state(root: Path, state: str,
                mapping: pd.Series) -> tuple[pd.DataFrame, tuple[str, ...]]:
    path = root / f"cre_by_celltype_{state}_fraction.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run annotate_chromstates.py first")
    renamed, unmapped = rename_to_subclass_names(pd.read_csv(path, index_col=0), mapping)
    return renamed.T, unmapped


def load_insulator_truth(results_dir: Path | None = None,
                         annotation_csv: Path | None = None) -> InsulatorTruth:
    """Load Chr-O / Chr-A / ND fractions as cell type x cCRE frames."""
    root = (
        Path(results_dir)
        if results_dir is not None
        else revision_data_root().parent / "silencer" / "results"
    )
    mapping = subclass_number_to_name(annotation_csv)
    open_fraction, unmapped = _load_state(root, OPEN_STATE, mapping)
    active_fraction, _ = _load_state(root, ACTIVE_STATE, mapping)
    nd_fraction, _ = _load_state(root, "ND", mapping)
    return InsulatorTruth(
        open_fraction=open_fraction,
        active_fraction=active_fraction.reindex(index=open_fraction.index,
                                                columns=open_fraction.columns),
        nd_fraction=nd_fraction.reindex(index=open_fraction.index,
                                        columns=open_fraction.columns),
        unmapped_columns=unmapped,
    )
