#!/usr/bin/env python
"""Annotate STARR-FISH cCREs with per-cell-type ChromHMM state coverage.

For every cCRE in a BED file and every ``*_8states_dense.bed`` ChromHMM
segmentation in a directory, compute the fraction of the cCRE's bp covered by
each state.  The headline output is the repressive fraction, i.e. the coverage
by the union of ``Chr-R`` (repressed chromatin) and ``Hc-P`` (poised
heterochromatin), written as a cCRE x cell-type matrix.

The segmentations are non-overlapping, gap-free dense BEDs, so the union
fraction is the sum of the two per-state fractions.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Sequence, Tuple

import numpy as np
import pandas as pd

STATES: Final[Tuple[str, ...]] = (
    "Chr-A",
    "Chr-Pr",
    "Chr-Po",
    "Chr-O",
    "Chr-R",
    "Hc-P",
    "Hc-H",
    "ND",
)
REPRESSIVE_STATES: Final[Tuple[str, ...]] = ("Chr-R", "Hc-P")
STATE_INDEX: Final[Dict[str, int]] = {s: i for i, s in enumerate(STATES)}
SEG_SUFFIX: Final[str] = "_8states_dense.bed"


@dataclass(frozen=True)
class CREPanel:
    """Immutable cCRE panel indexed by chromosome for interval lookup."""

    names: Tuple[str, ...]
    categories: Tuple[str, ...]
    chroms: Tuple[str, ...]
    starts: np.ndarray  # int64 [n_cre]
    ends: np.ndarray  # int64 [n_cre]
    # per chromosome: (sorted starts, ends, global row index), all aligned
    by_chrom: Dict[str, Tuple[List[int], List[int], List[int]]]

    @property
    def n_cre(self) -> int:
        return len(self.names)

    @property
    def lengths(self) -> np.ndarray:
        return (self.ends - self.starts).astype(np.float64)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def load_cre_panel(bed_path: Path) -> CREPanel:
    """Load a cCRE BED (chrom, start, end, category, name)."""
    names: List[str] = []
    categories: List[str] = []
    chroms: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    with _open_text(bed_path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip() or line.startswith(("track", "browser", "#")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise ValueError(f"{bed_path}:{lineno}: expected >=5 columns, got {len(fields)}")
            chroms.append(fields[0])
            starts.append(int(fields[1]))
            ends.append(int(fields[2]))
            categories.append(fields[3])
            names.append(fields[4])
    if not names:
        raise ValueError(f"{bed_path}: no cCRE records found")
    if len(set(names)) != len(names):
        raise ValueError(f"{bed_path}: cCRE names in column 5 are not unique")

    by_chrom: Dict[str, Tuple[List[int], List[int], List[int]]] = {}
    grouped: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for row, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
        grouped[chrom].append((start, end, row))
    for chrom, records in grouped.items():
        records.sort()
        by_chrom[chrom] = (
            [r[0] for r in records],
            [r[1] for r in records],
            [r[2] for r in records],
        )
    return CREPanel(
        names=tuple(names),
        categories=tuple(categories),
        chroms=tuple(chroms),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        by_chrom=by_chrom,
    )


def cell_type_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(SEG_SUFFIX):
        raise ValueError(f"{path}: does not end with {SEG_SUFFIX!r}")
    return name[: -len(SEG_SUFFIX)]


def _overlap_bp_one_file(args: Tuple[str, CREPanel]) -> Tuple[str, np.ndarray]:
    """Stream one segmentation BED and accumulate overlap bp [n_cre, n_state]."""
    seg_path_str, panel = args
    seg_path = Path(seg_path_str)
    counts = np.zeros((panel.n_cre, len(STATES)), dtype=np.int64)
    max_cre_len = int(panel.lengths.max()) if panel.n_cre else 0

    with _open_text(seg_path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.startswith(("track", "browser", "#")) or not line.strip():
                continue
            fields = line.split("\t", 4)
            if len(fields) < 4:
                raise ValueError(f"{seg_path}:{lineno}: expected >=4 columns")
            chrom = fields[0]
            chrom_data = panel.by_chrom.get(chrom)
            if chrom_data is None:
                continue
            state_col = STATE_INDEX.get(fields[3])
            if state_col is None:
                raise ValueError(f"{seg_path}:{lineno}: unknown state {fields[3]!r}")
            seg_start = int(fields[1])
            seg_end = int(fields[2])

            cre_starts, cre_ends, cre_rows = chrom_data
            # candidates: cCREs whose start is before seg_end and that are long
            # enough to possibly reach seg_start
            hi = bisect.bisect_left(cre_starts, seg_end)
            lo = bisect.bisect_left(cre_starts, seg_start - max_cre_len)
            for idx in range(lo, hi):
                ov = min(cre_ends[idx], seg_end) - max(cre_starts[idx], seg_start)
                if ov > 0:
                    counts[cre_rows[idx], state_col] += ov
    return cell_type_from_path(seg_path), counts


def annotate(
    panel: CREPanel, seg_paths: Sequence[Path], n_workers: int
) -> Tuple[List[str], np.ndarray]:
    """Return (cell_types, fractions [n_cre, n_cell_type, n_state])."""
    tasks = [(str(p), panel) for p in seg_paths]
    results: List[Tuple[str, np.ndarray]] = []
    if n_workers <= 1:
        results = [_overlap_bp_one_file(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, res in enumerate(pool.map(_overlap_bp_one_file, tasks, chunksize=1), 1):
                results.append(res)
                print(f"[{i}/{len(tasks)}] {res[0]}", flush=True)

    results.sort(key=lambda kv: kv[0])
    cell_types = [ct for ct, _ in results]
    bp = np.stack([c for _, c in results], axis=1)  # [n_cre, n_ct, n_state]
    fractions = bp / panel.lengths[:, None, None]
    # A segmentation is gap-free over the chromosomes it contains, so zero total
    # coverage means the cCRE's chromosome is absent from that file (e.g. chrX is
    # not segmented).  Those entries are unmeasured, not zero-repressive.
    uncovered = bp.sum(axis=2) == 0
    if uncovered.any():
        rows = np.flatnonzero(uncovered.any(axis=1))
        print(f"warning: {uncovered.sum()} cCRE x cell-type pairs have no "
              f"segmentation coverage -> NaN "
              f"(chroms: {sorted({panel.chroms[r] for r in rows})})", flush=True)
        fractions[uncovered] = np.nan
    return cell_types, fractions


def write_outputs(
    panel: CREPanel,
    cell_types: Sequence[str],
    fractions: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.Index(panel.names, name="cCRE")
    columns = pd.Index(list(cell_types), name="cell_type")

    rep_cols = [STATE_INDEX[s] for s in REPRESSIVE_STATES]
    repressive = np.nansum(fractions[:, :, rep_cols], axis=2)
    repressive[~np.isfinite(fractions[:, :, rep_cols]).all(axis=2)] = np.nan
    pd.DataFrame(repressive, index=index, columns=columns).to_csv(
        out_dir / "cre_by_celltype_repressive_fraction.csv", float_format="%.6g"
    )
    for state in STATES:
        safe = state.replace("-", "_")
        pd.DataFrame(
            fractions[:, :, STATE_INDEX[state]], index=index, columns=columns
        ).to_csv(out_dir / f"cre_by_celltype_{safe}_fraction.csv", float_format="%.6g")

    long_df = pd.DataFrame(
        {
            "cCRE": np.repeat(panel.names, len(cell_types) * len(STATES)),
            "cell_type": np.tile(np.repeat(cell_types, len(STATES)), panel.n_cre),
            "state": np.tile(STATES, panel.n_cre * len(cell_types)),
            "fraction": fractions.reshape(-1),
        }
    )
    long_df.to_csv(out_dir / "cre_by_celltype_state_fraction_long.csv.gz", index=False,
                   float_format="%.6g")

    with warnings.catch_warnings():  # all-NaN cCREs (unsegmented chromosomes)
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = pd.DataFrame(
            {
                "cCRE": panel.names,
                "category": panel.categories,
                "chrom": panel.chroms,
                "start": panel.starts,
                "end": panel.ends,
                "length_bp": panel.lengths.astype(np.int64),
                "n_celltype_measured": np.isfinite(repressive).sum(axis=1),
                "mean_repressive_fraction": np.nanmean(repressive, axis=1),
                "median_repressive_fraction": np.nanmedian(repressive, axis=1),
                "max_repressive_fraction": np.nanmax(repressive, axis=1),
                "n_celltype_repressive_gt50pct": (repressive > 0.5).sum(axis=1),
                "frac_celltype_repressive_gt50pct": np.where(
                    np.isfinite(repressive).any(axis=1),
                    (repressive > 0.5).sum(axis=1)
                    / np.maximum(np.isfinite(repressive).sum(axis=1), 1),
                    np.nan,
                ),
                "mean_ND_fraction": np.nanmean(fractions[:, :, STATE_INDEX["ND"]], axis=1),
            }
        )
    summary.to_csv(out_dir / "cre_repressive_summary.csv", index=False, float_format="%.6g")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_ct = pd.DataFrame(
            {
                "cell_type": list(cell_types),
                "n_cre_measured": np.isfinite(repressive).sum(axis=0),
                "mean_repressive_fraction": np.nanmean(repressive, axis=0),
                "n_cre_repressive_gt50pct": (repressive > 0.5).sum(axis=0),
                "mean_ND_fraction": np.nanmean(fractions[:, :, STATE_INDEX["ND"]], axis=0),
            }
        )
    per_ct.to_csv(out_dir / "celltype_repressive_summary.csv", index=False,
                  float_format="%.6g")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cre-bed", type=Path,
                        default=root / "STARRFISH_in_vivo/Data/CRE.bed")
    parser.add_argument("--chromstate-dir", type=Path,
                        default=root / "revision/Data/ChromStates")
    parser.add_argument("--out-dir", type=Path,
                        default=root / "revision/silencer/results")
    parser.add_argument("--workers", type=int,
                        default=min(16, (os.cpu_count() or 2) - 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_cre_panel(args.cre_bed)
    seg_paths = sorted(args.chromstate_dir.glob(f"*{SEG_SUFFIX}"))
    if not seg_paths:
        raise FileNotFoundError(f"no *{SEG_SUFFIX} files in {args.chromstate_dir}")
    print(f"{panel.n_cre} cCREs x {len(seg_paths)} cell types, "
          f"{args.workers} workers", flush=True)
    cell_types, fractions = annotate(panel, seg_paths, args.workers)
    write_outputs(panel, cell_types, fractions, args.out_dir)
    rep = np.nansum(fractions[:, :, [STATE_INDEX[s] for s in REPRESSIVE_STATES]], axis=2)
    rep[~np.isfinite(fractions[:, :, [STATE_INDEX[s] for s in REPRESSIVE_STATES]]).all(axis=2)] = np.nan
    print(f"done -> {args.out_dir}\n"
          f"mean repressive fraction = {np.nanmean(rep):.4f}; "
          f"cCRE x cell-type pairs >50% repressive = {(rep > 0.5).sum()}", flush=True)


if __name__ == "__main__":
    main()
