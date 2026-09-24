#!/usr/bin/env python
"""Silencer and insulator totals aggregated over cell types.

Collapses the two cell type x cCRE annotations into panel-level counts: how many
pairs carry each label, how many distinct cCREs carry it in at least one cell
type, and how many cell types carry it for at least one cCRE. Reported twice --
over every scored pair, and over the pairs the assay tested at ``--t7-threshold``.

    silencer   Chr-R + Hc-P fraction > --silencer-cutoff
    insulator  Chr-O > --open-cutoff  and  Chr-A < --active-cutoff

Also writes the per-cCRE breadth (how many cell types give a cCRE each label),
which is what separates the two labels: silencer cCREs are shared across cell
types, insulator cCREs are not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baystarrfish.data.paths import repo_root
from insulator_truth import (DEFAULT_ACTIVE_CUTOFF, DEFAULT_OPEN_CUTOFF,
                             load_insulator_truth)
from silencer_truth import load_silencer_truth

SILENCER: str = "silencer"
INSULATOR: str = "insulator"


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotation-dir", type=Path,
                        default=root / "revision/silencer/results")
    parser.add_argument("--tests", type=Path,
                        default=root / "revision/silencer/results"
                                     / "silencer_left_tail_tests.csv.gz")
    parser.add_argument("--out-dir", type=Path, default=root / "revision/insulator/results")
    parser.add_argument("--t7-threshold", type=float, default=50.0)
    parser.add_argument("--silencer-cutoff", type=float, default=0.1)
    parser.add_argument("--open-cutoff", type=float, default=DEFAULT_OPEN_CUTOFF)
    parser.add_argument("--active-cutoff", type=float, default=DEFAULT_ACTIVE_CUTOFF)
    parser.add_argument("--stem", type=str, default="label_totals_across_celltypes")
    return parser.parse_args()


def tested_mask(reference: pd.DataFrame, tests: Path,
                t7_threshold: float) -> pd.DataFrame:
    """Cell type x cCRE mask of the pairs the assay tested at the T7 filter."""
    frame = pd.read_csv(tests, usecols=["t7_threshold", "group", "cre"])
    frame = frame[frame["t7_threshold"] == t7_threshold]
    tested = pd.crosstab(frame["group"], frame["cre"]) > 0
    return tested.reindex(index=reference.index, columns=reference.columns,
                          fill_value=False)


def totals(scope: str, label: str, mask: pd.DataFrame,
           denominator: pd.DataFrame) -> dict[str, object]:
    return {
        "scope": scope,
        "label": label,
        "n_pairs": int(mask.values.sum()),
        "n_pairs_scored": int(denominator.values.sum()),
        "pair_fraction": float(mask.values.sum() / denominator.values.sum()),
        "n_cre_any_celltype": int(mask.any(axis=0).sum()),
        "n_cre_scored": int(denominator.any(axis=0).sum()),
        "n_celltype_any_cre": int(mask.any(axis=1).sum()),
        "n_celltype_scored": int(denominator.any(axis=1).sum()),
    }


def overlap(scope: str, silencer: pd.DataFrame,
            insulator: pd.DataFrame) -> dict[str, object]:
    """How the two label sets share pairs and cCREs."""
    sil_cre = set(silencer.columns[silencer.any(axis=0)])
    ins_cre = set(insulator.columns[insulator.any(axis=0)])
    return {
        "scope": scope,
        "n_pairs_both_labels": int((silencer & insulator).values.sum()),
        "n_cre_silencer_only": len(sil_cre - ins_cre),
        "n_cre_insulator_only": len(ins_cre - sil_cre),
        "n_cre_both_labels": len(sil_cre & ins_cre),
        "n_cre_either_label": len(sil_cre | ins_cre),
    }


def breadth(silencer: pd.DataFrame, insulator: pd.DataFrame) -> pd.DataFrame:
    """Per cCRE: in how many cell types it carries each label."""
    table = pd.DataFrame({
        "n_celltype_silencer": silencer.sum(axis=0).astype(int),
        "n_celltype_insulator": insulator.sum(axis=0).astype(int),
    }).rename_axis("cCRE").reset_index()
    return table.sort_values(["n_celltype_silencer", "n_celltype_insulator"],
                             ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    insulator_truth = load_insulator_truth(args.annotation_dir)
    silencer_truth = load_silencer_truth(args.annotation_dir)

    measured = insulator_truth.measured
    insulator = insulator_truth.mask(args.open_cutoff, args.active_cutoff) & measured
    repressive = silencer_truth.repressive_fraction
    silencer = (repressive > args.silencer_cutoff) & repressive.notna()
    silencer = silencer.reindex(index=measured.index, columns=measured.columns,
                                fill_value=False)

    tested = tested_mask(measured, args.tests, args.t7_threshold) & measured
    scopes = {"panel": (silencer, insulator, measured),
              "tested": (silencer & tested, insulator & tested, tested)}

    summary = pd.DataFrame([
        totals(scope, label, mask, denominator)
        for scope, (sil, ins, denominator) in scopes.items()
        for label, mask in ((SILENCER, sil), (INSULATOR, ins))
    ])
    overlaps = pd.DataFrame([overlap(scope, sil, ins)
                             for scope, (sil, ins, _) in scopes.items()])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        f"{args.stem}.csv": summary,
        f"{args.stem}_overlap.csv": overlaps,
        f"{args.stem}_per_cre_panel.csv": breadth(silencer, insulator),
        f"{args.stem}_per_cre_tested.csv": breadth(silencer & tested, insulator & tested),
    }
    for name, frame in paths.items():
        frame.to_csv(args.out_dir / name, index=False)

    print(summary.to_string(index=False))
    print()
    print(overlaps.to_string(index=False))
    for name in paths:
        print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
