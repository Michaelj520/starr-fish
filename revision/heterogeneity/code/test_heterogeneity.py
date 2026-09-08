from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.collections as mcollections
import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import make_heterogeneity_plots as plots
from relabel import relabel_subclasses_from_obs, relabel_subclasses_size_matched


def test_relabel_subclasses_from_obs_preserves_untargeted_cells() -> None:
    subclasses = np.array(["A", "A", "A", "B", "B", "C"])
    supertypes = np.array(
        ["001 A_1", "002 A_2", "002 A_2", "003 B_1", "003 B_1", "004 C_1"]
    )

    labels, assignment, members = relabel_subclasses_from_obs(
        subclasses, supertypes, ["A", "B"]
    )

    assert labels.tolist() == ["A_1", "A_2", "A_2", "B_1", "B_1", "C"]
    assert members == {"A": ["A_1", "A_2"], "B": ["B_1"]}
    assert assignment.columns.tolist() == [
        "position",
        "original_subclass",
        "source_subgroup",
        "new_subclass",
    ]
    assert assignment["position"].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("subclasses", "supertypes", "message"),
    [
        (
            np.array(["A", "A"], dtype=object),
            np.array(["001 A_1", None], dtype=object),
            "missing subgroup labels",
        ),
        (
            np.array(["A", "A"], dtype=object),
            np.array(["001 A_1", "999 A_1"], dtype=object),
            "collapse after standardization",
        ),
        (
            np.array(["A", "B"], dtype=object),
            np.array(["001 Shared", "001 Shared"], dtype=object),
            "multiple target subclasses",
        ),
        (
            np.array(["A", "C"], dtype=object),
            np.array(["001 C", "002 C_1"], dtype=object),
            "collide with untouched subclasses",
        ),
    ],
)
def test_relabel_subclasses_from_obs_rejects_invalid_annotations(
    subclasses: np.ndarray, supertypes: np.ndarray, message: str
) -> None:
    targets = sorted(set(subclasses) - {"C"})
    with pytest.raises(ValueError, match=message):
        relabel_subclasses_from_obs(subclasses, supertypes, targets)


def agreement_fixture() -> tuple[
    pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]
]:
    combined = pd.DataFrame(
        [[1.0, 3.0], [2.0, 4.0]],
        index=["A", "B"],
        columns=["c1", "c2"],
    )
    split = pd.DataFrame(
        [[0.0, 2.0], [2.0, 4.0], [2.5, 4.5]],
        index=["A_1", "A_2", "B_1"],
        columns=["c1", "c2"],
    )
    targets = ["A", "B"]
    members = {"A": ["A_1", "A_2"], "B": ["B_1"]}
    return combined, split, targets, members


def test_random_agreement_remains_unweighted_and_backward_compatible() -> None:
    combined, split, targets, members = agreement_fixture()
    table = plots.bayesian_subset_agreement(
        combined, split, targets, members, excluded_cres=set()
    )

    a = table[table["cell_type"] == "A"].set_index("cre")
    assert a.loc["c1", "mean_subgroup_activity"] == pytest.approx(1.0)
    assert a.loc["c1", "subgroup_sd"] == pytest.approx(np.sqrt(2.0))
    assert a.loc["c1", "n_subgroups"] == 2

    b = table[table["cell_type"] == "B"].set_index("cre")
    assert b.loc["c1", "mean_subgroup_activity"] == pytest.approx(2.5)
    assert np.isnan(b.loc["c1", "subgroup_sd"])
    assert b.loc["c1", "n_subgroups"] == 1

    exported = plots.agreement_for_export(table, "random")
    assert exported.columns.tolist() == [
        "cell_type",
        "cre",
        "whole_activity",
        "mean_subset_activity",
        "subset_sd",
        "n_subsets",
        "difference",
        "absolute_difference",
    ]


def test_supertype_agreement_uses_cell_counts_but_retains_heterogeneity() -> None:
    combined, split, targets, members = agreement_fixture()
    weights = {"A": {"A_1": 1, "A_2": 3}, "B": {"B_1": 2}}
    table = plots.bayesian_subset_agreement(
        combined,
        split,
        targets,
        members,
        excluded_cres=set(),
        subgroup_weights=weights,
    )

    a = table[table["cell_type"] == "A"].set_index("cre")
    assert a.loc["c1", "mean_subgroup_activity"] == pytest.approx(1.5)
    assert a.loc["c1", "unweighted_mean_subgroup_activity"] == pytest.approx(1.0)
    assert a.loc["c1", "subgroup_sd"] == pytest.approx(np.sqrt(2.0))
    assert a.loc["c1", "difference"] == pytest.approx(0.5)
    assert a.loc["c1", "unweighted_difference"] == pytest.approx(0.0)

    b = table[table["cell_type"] == "B"].set_index("cre")
    assert b.loc["c1", "mean_subgroup_activity"] == pytest.approx(2.5)
    assert np.isnan(b.loc["c1", "subgroup_sd"])

    exported = plots.agreement_for_export(table, "supertype")
    assert {
        "cell_weighted_mean_supertype_activity",
        "unweighted_mean_supertype_activity",
        "supertype_sd",
        "n_supertypes",
        "cell_weighted_difference",
        "unweighted_difference",
    } <= set(exported.columns)

    summary = plots.summarize_agreement(
        table, include_supertype_heterogeneity=True
    ).set_index("cell_type")
    assert summary.loc["A", "median_supertype_sd"] == pytest.approx(np.sqrt(2.0))
    assert np.isnan(summary.loc["B", "median_supertype_sd"])


def test_pairwise_supertype_ccc_is_computed_within_each_parent() -> None:
    _, split, targets, members = agreement_fixture()
    weights = {"A": {"A_1": 1, "A_2": 3}, "B": {"B_1": 2}}

    pairwise = plots.pairwise_supertype_agreement(
        split, targets, members, weights, excluded_cres=set()
    )

    assert len(pairwise) == 1
    pair = pairwise.iloc[0]
    assert pair["cell_type"] == "A"
    assert {pair["supertype_1"], pair["supertype_2"]} == {"A_1", "A_2"}
    assert pair["n_cres"] == 2
    assert pair["concordance_correlation"] == pytest.approx(1.0 / 3.0)
    assert pair["pearson_r"] == pytest.approx(1.0)
    assert pair["minimum_pair_cells"] == 1

    summary = plots.summarize_pairwise_supertype_agreement(
        pairwise, targets, members
    ).set_index("cell_type")
    assert summary.loc["A", "n_pairs"] == 1
    assert summary.loc["A", "median_pairwise_ccc"] == pytest.approx(1.0 / 3.0)
    assert summary.loc["B", "n_pairs"] == 0
    assert np.isnan(summary.loc["B", "median_pairwise_ccc"])


def test_pairwise_ccc_plot_includes_mean_vs_whole_baseline(monkeypatch) -> None:
    _, split, targets, members = agreement_fixture()
    weights = {"A": {"A_1": 1, "A_2": 3}, "B": {"B_1": 2}}
    pairwise = plots.pairwise_supertype_agreement(
        split, targets, members, weights, excluded_cres=set()
    )
    pairwise_summary = plots.summarize_pairwise_supertype_agreement(
        pairwise, targets, members
    )
    whole_summary = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "concordance_correlation": [0.9, 0.95],
        }
    )
    captured: dict[str, object] = {}

    def capture(fig, stem: Path) -> None:
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_pairwise_supertype_ccc(
        pairwise,
        pairwise_summary,
        whole_summary,
        targets,
        Path("unused"),
    )

    fig = captured["fig"]
    labels = {text.get_text() for text in fig.axes[0].texts}
    assert {"0.90", "0.95"} <= labels
    assert any("Blue diamonds" in text.get_text() for text in fig.texts)
    assert captured["stem"] == Path("unused/bayesian_supertype_pairwise_ccc")


def test_pairwise_ccc_cell_support_plot_uses_log_scale(monkeypatch) -> None:
    pairwise = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B"],
            "minimum_pair_cells": [10, 100, 1000],
            "concordance_correlation": [0.1, 0.3, 0.5],
        }
    )
    captured: dict[str, object] = {}

    def capture(fig, stem: Path) -> None:
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_pairwise_ccc_vs_minimum_cells(
        pairwise, ["A", "B"], Path("unused")
    )

    fig = captured["fig"]
    assert fig.axes[0].get_xscale() == "log"
    assert "Spearman" in fig.axes[0].texts[0].get_text()
    assert captured["stem"] == Path(
        "unused/bayesian_supertype_pairwise_ccc_vs_min_cells"
    )


def test_supertype_plot_keeps_points_when_sd_is_undefined(monkeypatch) -> None:
    combined, split, targets, members = agreement_fixture()
    table = plots.bayesian_subset_agreement(
        combined,
        split,
        targets,
        members,
        excluded_cres=set(),
        subgroup_weights={"A": {"A_1": 1, "A_2": 2}, "B": {"B_1": 2}},
    )
    summary = plots.summarize_agreement(
        table, include_supertype_heterogeneity=True
    )
    captured: dict[str, object] = {}

    def capture(fig, stem: Path) -> None:
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_bayesian_subset_agreement(
        table,
        summary,
        targets,
        {"A": 3, "B": 2},
        members,
        "supertype",
        Path("unused"),
        ncols=2,
    )

    fig = captured["fig"]
    b_axis = fig.axes[1]
    scatters = [
        collection
        for collection in b_axis.collections
        if isinstance(collection, mcollections.PathCollection)
    ]
    error_bars = [
        collection
        for collection in b_axis.collections
        if isinstance(collection, mcollections.LineCollection)
    ]
    assert sum(len(scatter.get_offsets()) for scatter in scatters) == 2
    assert error_bars == []
    assert captured["stem"] == Path("unused/bayesian_supertype_mean_vs_whole")
    assert "cell-count-weighted mean" in fig._suptitle.get_text()
    assert any("between-supertype heterogeneity" in text.get_text() for text in fig.texts)


def test_split_targets_supports_legacy_and_explicit_manifests(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_manifest.json").write_text(
        json.dumps({"split_subclasses": ["A"], "n_groups": 2})
    )
    assert plots.split_targets(legacy) == (
        ["A"],
        {"A": ["A_group_1", "A_group_2"]},
        "random",
    )

    annotated = tmp_path / "annotated"
    annotated.mkdir()
    (annotated / "run_manifest.json").write_text(
        json.dumps(
            {
                "split_subclasses": ["A"],
                "grouping": "supertype",
                "subgroups_by_subclass": {"A": ["A_1", "A_2"]},
            }
        )
    )
    assert plots.split_targets(annotated) == (
        ["A"],
        {"A": ["A_1", "A_2"]},
        "supertype",
    )


def test_bootstrap_negctrl_only_effect_matrix(tmp_path: Path) -> None:
    axes = {"subclasses": ["A_1", "A_2"], "cres": ["c1", "c2", "neg"]}
    (tmp_path / "bootstrap_axes.json").write_text(json.dumps(axes))
    pd.Series(["neg"], name="cre").to_csv(
        tmp_path / "negative_controls.csv", index=False
    )
    pd.DataFrame(False, index=axes["subclasses"], columns=axes["cres"]).to_csv(
        tmp_path / "qvalue_filter_mask.csv"
    )
    activity = np.array(
        [
            [[2.0, 4.0, 1.0], [3.0, 6.0, 1.5]],
            [[4.0, 8.0, 2.0], [6.0, 12.0, 3.0]],
        ]
    )
    np.save(tmp_path / "celltype_activity_array.npy", activity)

    observed = plots.bootstrap_effect_matrix(tmp_path, self_cre=False)
    mean_log = np.log(activity).mean(axis=0)
    expected = mean_log - mean_log[:, [2]]

    np.testing.assert_allclose(observed.to_numpy(), expected)


def test_bootstrap_artifact_names_are_separate_from_bayesian() -> None:
    bootstrap = plots.artifact_names("supertype", "bootstrap")
    bayesian = plots.artifact_names("supertype", "bayesian")
    assert bootstrap["figure"] == "bootstrap_supertype_mean_vs_whole"
    assert bootstrap["table"] == "bootstrap_supertype_vs_whole.csv"
    assert bootstrap["manifest"] != bayesian["manifest"]
    assert bootstrap["pairwise_figure"] == "bootstrap_supertype_pairwise_ccc"
    assert bayesian["pairwise_table"] == "bayesian_supertype_pairwise_ccc.csv"
    assert bayesian["pairwise_support_figure"].endswith("_vs_min_cells")


def test_size_matched_random_groups_reproduce_supertype_sizes() -> None:
    subclasses = np.array(["A"] * 7 + ["B"] * 3 + ["C"] * 2)
    supertypes = np.array(
        ["001 A_1"] * 4 + ["002 A_2"] * 2 + ["003 A_3"]
        + ["004 B_1"] * 3
        + ["005 C_1"] * 2
    )
    targets = ["A", "B"]

    labels, assignment, members = relabel_subclasses_size_matched(
        subclasses, supertypes, targets, seed=7
    )

    assert members == {"A": ["A_random_1", "A_random_2", "A_random_3"], "B": ["B_random_1"]}
    # Untargeted cells keep their subclass; every target cell is reassigned once.
    assert labels[-2:].tolist() == ["C", "C"]
    assert set(labels[:10]) == set(members["A"]) | set(members["B"])
    assert len(assignment) == 10
    assert assignment["position"].is_unique

    sizes = assignment["new_subclass"].value_counts()
    assert sizes["A_random_1"] == 4
    assert sizes["A_random_2"] == 2
    assert sizes["A_random_3"] == 1
    assert sizes["B_random_1"] == 3
    # Group i is matched to the i-th annotated supertype of its parent.
    matched = assignment.drop_duplicates("new_subclass").set_index("new_subclass")
    assert matched.loc["A_random_2", "matched_supertype"] == "A_2"

    repeat_labels, _, _ = relabel_subclasses_size_matched(
        subclasses, supertypes, targets, seed=7
    )
    assert repeat_labels.tolist() == labels.tolist()
    shifted_labels, _, _ = relabel_subclasses_size_matched(
        subclasses, supertypes, targets, seed=8
    )
    assert shifted_labels.tolist() != labels.tolist()


def test_size_matched_grouping_reuses_supertype_artifact_names() -> None:
    # The random null must land on the same filenames as the annotated run so
    # the two result folders can be compared figure-for-figure.
    for model in ("bayesian", "bootstrap"):
        assert plots.artifact_names("supertype_like_random", model) == (
            plots.artifact_names("supertype", model)
        )
    assert plots.subgroup_noun("supertype_like_random", plural=True) == (
        "size-matched random subsets"
    )


def test_pairwise_ccc_plot_labels_the_size_matched_null(monkeypatch) -> None:
    saved: dict[str, object] = {}

    def capture(fig, stem):
        saved["title"] = fig.axes[0].get_title()
        saved["stem"] = stem
        plots.plt.close(fig)

    monkeypatch.setattr(plots, "save_figure", capture)
    pairwise = pd.DataFrame(
        {
            "cell_type": ["A", "A", "A"],
            "concordance_correlation": [0.8, 0.7, 0.6],
            "minimum_pair_cells": [10, 20, 30],
        }
    )
    summary = pd.DataFrame({"cell_type": ["A"], "median_pairwise_ccc": [0.7], "n_pairs": [3]})
    whole = pd.DataFrame({"cell_type": ["A"], "concordance_correlation": [0.9]})

    plots.plot_pairwise_supertype_ccc(
        pairwise,
        summary,
        whole,
        ["A"],
        Path("figures"),
        grouping="supertype_like_random",
    )

    assert "size-matched random subsets" in saved["title"]
    assert saved["stem"].name == "bayesian_supertype_pairwise_ccc"


def _pair_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    annotated = pd.DataFrame(
        {
            "cell_type": ["A", "A", "A"],
            "supertype_1": ["A_1", "A_1", "A_2"],
            "supertype_2": ["A_2", "A_3", "A_3"],
            "n_cells_1": [100, 100, 50],
            "n_cells_2": [50, 10, 10],
            "minimum_pair_cells": [50, 10, 10],
            "concordance_correlation": [0.40, 0.20, 0.10],
        }
    )
    null = pd.DataFrame(
        {
            "cell_type": ["A", "A", "A"],
            "supertype_1": ["A_random_1", "A_random_1", "A_random_2"],
            "supertype_2": ["A_random_2", "A_random_3", "A_random_3"],
            "n_cells_1": [100, 100, 50],
            "n_cells_2": [50, 10, 10],
            "minimum_pair_cells": [50, 10, 10],
            "concordance_correlation": [0.55, 0.35, 0.30],
        }
    )
    members = {"A": ["A_1", "A_2", "A_3"]}
    null_members = {"A": ["A_random_1", "A_random_2", "A_random_3"]}
    return annotated, null, members, null_members


def test_paired_null_comparison_matches_pairs_by_ordinal() -> None:
    annotated, null, members, null_members = _pair_frames()

    merged = plots.paired_null_comparison(annotated, null, members, null_members)

    assert len(merged) == 3
    # Ordinal (0, 1) is the 100/50-cell pair in both runs.
    first = merged[(merged["ordinal_1"] == 0) & (merged["ordinal_2"] == 1)].iloc[0]
    assert first["supertype_2_annotated"] == "A_2"
    assert first["supertype_2_null"] == "A_random_2"
    assert first["ccc_difference"] == pytest.approx(0.40 - 0.55)

    tests = plots.summarize_paired_null_comparison(merged, ["A"])
    overall = tests[tests["cell_type"] == "ALL"].iloc[0]
    assert overall["n_pairs"] == 3
    assert overall["hodges_lehmann"] < 0
    assert 0 < overall["wilcoxon_p_value"] <= 1


def test_paired_null_comparison_rejects_unmatched_cell_counts() -> None:
    annotated, null, members, null_members = _pair_frames()
    null.loc[0, "n_cells_2"] = 51

    with pytest.raises(ValueError, match="not size-matched"):
        plots.paired_null_comparison(annotated, null, members, null_members)


def test_paired_ccc_plot_draws_two_boxes_per_cell_type(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture(fig, stem):
        axis = fig.axes[0]
        captured["boxes"] = len(
            [
                patch
                for patch in axis.patches
                if isinstance(patch, plots.plt.matplotlib.patches.PathPatch)
            ]
        )
        captured["legend"] = [
            text.get_text() for text in axis.get_legend().get_texts()
        ]
        captured["title"] = axis.get_title()
        plots.plt.close(fig)

    monkeypatch.setattr(plots, "save_figure", capture)
    annotated, null, members, null_members = _pair_frames()
    merged = plots.paired_null_comparison(annotated, null, members, null_members)
    tests = plots.summarize_paired_null_comparison(merged, ["A"])
    summary = pd.DataFrame(
        {"cell_type": ["A"], "median_pairwise_ccc": [0.2], "n_pairs": [3]}
    )
    whole = pd.DataFrame({"cell_type": ["A"], "concordance_correlation": [0.9]})

    plots.plot_pairwise_supertype_ccc(
        annotated,
        summary,
        whole,
        ["A"],
        Path("figures"),
        null_comparison=merged,
        null_tests=tests,
    )

    assert captured["boxes"] == 2
    assert captured["legend"] == [
        "Annotated supertype pairs",
        "Size-matched random pairs (null)",
    ]
    assert "size-matched random cell subsets" in captured["title"]


def test_sign_flip_permutation_is_exact_for_small_samples() -> None:
    differences = np.array([-0.2, -0.1, -0.3])

    p_value, exact = plots.sign_flip_permutation_p(differences, seed=0)

    # All three differences share a sign, so only the all-negative and
    # all-positive assignments reach the observed |mean|: 2 of 2**3.
    assert exact is True
    assert p_value == pytest.approx(2 / 8)

    large = np.concatenate([differences] * 10)
    sampled_p, sampled_exact = plots.sign_flip_permutation_p(large, seed=0)
    assert sampled_exact is False
    assert 0 < sampled_p <= 1


def test_hodges_lehmann_is_the_median_walsh_average() -> None:
    # Walsh averages of (-0.4, -0.2, 0.6) over i <= j sort to
    # -0.4, -0.3, -0.2, 0.1, 0.2, 0.6, whose median is -0.05.
    assert plots.hodges_lehmann(np.array([-0.4, -0.2, 0.6])) == pytest.approx(-0.05)


def test_paired_ccc_test_reports_rank_based_and_parametric_results() -> None:
    rng = np.random.default_rng(0)
    differences = rng.normal(-0.05, 0.1, size=40)

    result = plots.paired_ccc_test(differences)

    assert result["n_pairs"] == 40
    assert result["n_negative"] + result["n_positive"] == 40
    for key in (
        "wilcoxon_p_value",
        "sign_test_p_value",
        "permutation_p_value",
        "t_p_value",
        "shapiro_p_value",
    ):
        assert 0 < result[key] <= 1
    assert -1 <= result["rank_biserial"] <= 1
    assert result["hodges_lehmann"] < 0


def test_cell_type_level_test_uses_one_median_per_cell_type() -> None:
    merged = pd.DataFrame(
        {
            "cell_type": ["A"] * 4 + ["B"] * 2 + ["C"] * 2,
            "ccc_difference": [-0.1, -0.2, -0.3, -0.4, -0.05, -0.15, 0.2, 0.3],
        }
    )

    result = plots.cell_type_level_test(merged)

    # Three cell types, not eight pairs: A and B negative, C positive.
    assert result["n_cell_types"] == 3
    assert result["n_negative"] == 2
    assert result["n_positive"] == 1
    assert result["permutation_exact"] is True
    assert 0 < result["permutation_p_value"] <= 1
