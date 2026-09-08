#!/usr/bin/env python3
"""Top-T7 subclass selection and within-subclass partitioning.

The heterogeneity experiment splits the highest-T7 subclasses into ``N_GROUPS``
synthetic subclasses (``<subclass>_group_<i>``) drawn from the same cells, then
compares each split subclass against its intact estimate. Splitting only the
``subclass`` label leaves the parent ``class`` intact, so the five synthetic
subclasses stay nested under the original class in the hierarchical model.

The parallel annotated analysis instead promotes an existing per-cell label
(currently ``supertype_name``) to the model's subclass grouping for the same
target subclasses.  The validation in this module keeps that relabelling
traceable and prevents standardized names from being silently merged.

The size-matched random null reuses that validated annotation only for its
group sizes: each target subclass is shuffled and cut into random blocks whose
cell counts equal the annotated supertype counts, so cell support is held fixed
while biological structure is removed.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
REVISION_DATA = ANALYSIS_DIR.parent / "Data"
SUBCLASS_T7_CSV = REVISION_DATA / "subclass_total_t7_counts.csv"

TOP_N = 10
N_GROUPS = 5
SPLIT_SEED = 0


def standardize_name(names: pd.Series) -> pd.Series:
    """Match ``analysis_utils.standardize_obs``: drop Allen numeric prefix, /->-."""
    return (
        names.astype(str)
        .str.replace(r"^\d+\s+", "", regex=True)
        .str.replace("/", "-", regex=False)
    )


def top_subclasses(
    csv: Path = SUBCLASS_T7_CSV, n: int = TOP_N
) -> list[str]:
    """Return the ``n`` standardized subclass names with the largest total T7."""
    table = pd.read_csv(csv)
    if "total_t7" not in table.columns or "subclass" not in table.columns:
        raise KeyError(f"{csv} must contain 'subclass' and 'total_t7' columns")
    table = table.sort_values("total_t7", ascending=False)
    selected = standardize_name(table["subclass"].head(n))
    if selected.duplicated().any():
        raise ValueError("standardized top subclass names are not unique")
    return selected.tolist()


def group_label(subclass: str, group: int) -> str:
    return f"{subclass}_group_{group}"


def natural_key(value: str) -> tuple:
    """Return a deterministic human/numeric sorting key for subgroup labels."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def relabel_subclasses(
    subclasses: np.ndarray,
    targets: list[str],
    n_groups: int = N_GROUPS,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Partition each target subclass into ``n_groups`` labelled subgroups.

    Cells outside ``targets`` keep their original label. The partition is
    reproducible: an independent RNG stream is spawned per target subclass from
    ``seed`` so results are stable regardless of target ordering or how many
    subclasses are split. Each subgroup gets contiguous shares of a shuffled
    index list (near-equal sizes via ``np.array_split``).

    Returns the relabelled array and a per-cell assignment frame keyed by the
    original positional index.
    """
    subclasses = np.asarray(subclasses).astype(str)
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    # object dtype so longer "<subclass>_group_<i>" labels are not truncated to
    # the fixed width of the incoming string array.
    new_labels = subclasses.astype(object)
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = {
        target: child
        for target, child in zip(sorted(targets), seed_seq.spawn(len(targets)))
    }

    records = []
    for target in sorted(targets):
        positions = np.flatnonzero(subclasses == target)
        if positions.size < n_groups:
            raise ValueError(
                f"subclass {target!r} has {positions.size} cells; "
                f"cannot split into {n_groups} groups"
            )
        rng = np.random.default_rng(child_seeds[target])
        shuffled = rng.permutation(positions)
        for group, block in enumerate(np.array_split(shuffled, n_groups), start=1):
            label = group_label(target, group)
            new_labels[block] = label
            records.append(
                pd.DataFrame(
                    {
                        "position": block,
                        "original_subclass": target,
                        "group": group,
                        "new_subclass": label,
                    }
                )
            )
    assignment = (
        pd.concat(records, ignore_index=True).sort_values("position")
        if records
        else pd.DataFrame(
            columns=["position", "original_subclass", "group", "new_subclass"]
        )
    )
    return new_labels, assignment


def relabel_subclasses_from_obs(
    subclasses: np.ndarray,
    subgroup_labels: np.ndarray,
    targets: list[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, list[str]]]:
    """Relabel target subclasses using an existing per-cell annotation.

    The source labels are standardized with :func:`standardize_name`, matching
    the class/subclass preprocessing used by the Bayesian fit. Cells outside
    ``targets`` retain their subclass. The returned assignment covers only
    target cells and records both the raw annotation and the model label.

    Raises rather than silently merging when source labels are missing, when
    multiple raw labels collapse to the same standardized label, when one
    subgroup spans multiple target subclasses, or when a subgroup label
    collides with an untouched subclass.
    """
    subclasses = np.asarray(subclasses).astype(str)
    raw_subgroups = np.asarray(subgroup_labels, dtype=object)
    if subclasses.ndim != 1 or raw_subgroups.ndim != 1:
        raise ValueError("subclasses and subgroup_labels must be one-dimensional")
    if subclasses.shape != raw_subgroups.shape:
        raise ValueError(
            "subclasses and subgroup_labels must have the same number of cells"
        )

    target_mask = np.isin(subclasses, targets)
    missing_targets = sorted(set(targets) - set(subclasses[target_mask]))
    if missing_targets:
        raise ValueError(f"target subclasses absent from cells: {missing_targets}")
    missing_source = pd.isna(raw_subgroups) | pd.Series(raw_subgroups).astype(
        "string"
    ).str.strip().eq("").fillna(True).to_numpy()
    if (target_mask & missing_source).any():
        positions = np.flatnonzero(target_mask & missing_source)[:5].tolist()
        raise ValueError(f"target cells have missing subgroup labels at {positions}")

    raw_strings = pd.Series(raw_subgroups).astype(str)
    standardized = standardize_name(raw_strings).to_numpy(dtype=object)
    selected = pd.DataFrame(
        {
            "position": np.flatnonzero(target_mask),
            "original_subclass": subclasses[target_mask],
            "source_subgroup": raw_strings.to_numpy()[target_mask],
            "new_subclass": standardized[target_mask],
        }
    )

    collapsed = selected[["source_subgroup", "new_subclass"]].drop_duplicates()
    collapsed_counts = collapsed.groupby("new_subclass")["source_subgroup"].nunique()
    duplicated = collapsed_counts[collapsed_counts > 1].index.tolist()
    if duplicated:
        raise ValueError(
            "distinct source subgroup labels collapse after standardization: "
            f"{sorted(duplicated, key=natural_key)}"
        )

    parent_counts = selected.groupby("new_subclass")["original_subclass"].nunique()
    cross_parent = parent_counts[parent_counts > 1].index.tolist()
    if cross_parent:
        raise ValueError(
            "subgroup labels map to multiple target subclasses: "
            f"{sorted(cross_parent, key=natural_key)}"
        )

    untouched = set(subclasses[~target_mask])
    collisions = sorted(set(selected["new_subclass"]) & untouched, key=natural_key)
    if collisions:
        raise ValueError(
            f"subgroup labels collide with untouched subclasses: {collisions}"
        )

    members_by_subclass = {
        target: sorted(
            selected.loc[
                selected["original_subclass"] == target, "new_subclass"
            ].unique(),
            key=natural_key,
        )
        for target in targets
    }
    empty = [target for target, members in members_by_subclass.items() if not members]
    if empty:
        raise ValueError(f"target subclasses have no subgroup labels: {empty}")

    relabelled = subclasses.astype(object)
    relabelled[target_mask] = standardized[target_mask]
    return relabelled, selected.sort_values("position"), members_by_subclass


def matched_group_label(subclass: str, index: int) -> str:
    return f"{subclass}_random_{index}"


def relabel_subclasses_size_matched(
    subclasses: np.ndarray,
    subgroup_labels: np.ndarray,
    targets: list[str],
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, list[str]]]:
    """Partition each target subclass into random groups matching supertype sizes.

    This is the size-matched random null for the annotated-supertype analysis:
    within every target subclass the cells are shuffled once and cut, without
    replacement, into blocks whose sizes are exactly the annotated supertype
    sizes of that subclass. Group ``i`` therefore has the same cell support as
    the ``i``-th annotated supertype but no biological structure, so any
    residual between-group divergence measures estimation noise at that cell
    count rather than heterogeneity.

    The annotated labels are validated by :func:`relabel_subclasses_from_obs`
    before being reduced to sizes, so the same standardization, collision, and
    cross-parent guarantees apply. An independent RNG stream is spawned per
    target subclass from ``seed`` so the partition is stable regardless of
    target ordering.
    """
    _, reference, members_by_subclass = relabel_subclasses_from_obs(
        subclasses, subgroup_labels, targets
    )
    subclasses = np.asarray(subclasses).astype(str)
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = {
        target: child
        for target, child in zip(sorted(targets), seed_seq.spawn(len(targets)))
    }

    relabelled = subclasses.astype(object)
    matched_members: dict[str, list[str]] = {}
    records = []
    for target in sorted(targets):
        source_members = members_by_subclass[target]
        sizes = (
            reference.loc[
                reference["original_subclass"] == target, "new_subclass"
            ]
            .value_counts()
            .reindex(source_members)
            .astype(int)
            .to_numpy()
        )
        positions = np.flatnonzero(subclasses == target)
        if positions.size != int(sizes.sum()):
            raise ValueError(
                f"{target!r} has {positions.size} cells but annotated supertype "
                f"sizes sum to {int(sizes.sum())}"
            )
        rng = np.random.default_rng(child_seeds[target])
        shuffled = rng.permutation(positions)
        blocks = np.split(shuffled, np.cumsum(sizes)[:-1])
        labels = []
        for index, (source, block, size) in enumerate(
            zip(source_members, blocks, sizes), start=1
        ):
            if block.size != int(size):
                raise ValueError(
                    f"size-matched block for {target!r} group {index} has "
                    f"{block.size} cells; expected {int(size)}"
                )
            label = matched_group_label(target, index)
            labels.append(label)
            relabelled[block] = label
            records.append(
                pd.DataFrame(
                    {
                        "position": block,
                        "original_subclass": target,
                        "matched_supertype": source,
                        "group": index,
                        "new_subclass": label,
                    }
                )
            )
        matched_members[target] = labels

    all_labels = [label for labels in matched_members.values() for label in labels]
    if len(set(all_labels)) != len(all_labels):
        raise ValueError("size-matched group labels are not unique")
    collisions = sorted(set(all_labels) & set(subclasses), key=natural_key)
    if collisions:
        raise ValueError(
            f"size-matched group labels collide with existing subclasses: {collisions}"
        )

    assignment = pd.concat(records, ignore_index=True).sort_values("position")
    return relabelled, assignment, {target: matched_members[target] for target in targets}
