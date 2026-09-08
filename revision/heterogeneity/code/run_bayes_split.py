#!/usr/bin/env python3
"""Fit the joint CRE/T7 model on within-subclass heterogeneity labels.

The top-T7 subclasses are either partitioned into ``N_GROUPS`` reproducible
random subsets (``<subclass>_group_<i>``) or relabelled with their annotated
``supertype_name`` values. All other subclasses stay intact. The fit is
otherwise identical to ``revision/run_Bayes/run_bayes.py`` so the resulting
activities are directly comparable to the intact-subclass fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
REF_CODE = ANALYSIS_DIR.parent / "bayesian_vs_fold_change" / "code"
# run_bayes now lives with the other fit drivers, not beside analysis_utils.
RUN_BAYES_DIR = ANALYSIS_DIR.parent / "run_Bayes"
sys.path.insert(0, str(RUN_BAYES_DIR))
sys.path.insert(0, str(REF_CODE))
sys.path.insert(0, str(CODE_DIR))

from analysis_utils import (  # noqa: E402  (path injected above)
    DEFAULT_H5AD,
    LIBSIZE_CSV,
    STARRFISH_ROOT,
    cre_blacklist,
    input_fingerprint,
    jsonable,
    log,
    negative_control_names,
    read_and_prepare_adata,
    write_json,
)
from run_bayes import dropout_prior_config  # noqa: E402

from relabel import (  # noqa: E402
    N_GROUPS,
    SPLIT_SEED,
    TOP_N,
    relabel_subclasses,
    relabel_subclasses_from_obs,
    relabel_subclasses_size_matched,
    top_subclasses,
)

DEFAULT_OUTDIR = ANALYSIS_DIR / "results" / "split" / "bayesian"
DEFAULT_SUPERTYPE_OUTDIR = ANALYSIS_DIR / "results" / "supertype" / "bayesian"
DEFAULT_SUPERTYPE_LIKE_RANDOM_OUTDIR = (
    ANALYSIS_DIR / "results" / "supertype_like_random" / "bayesian"
)
OUTDIR_BY_GROUPING = {
    "random": DEFAULT_OUTDIR,
    "supertype": DEFAULT_SUPERTYPE_OUTDIR,
    "supertype_like_random": DEFAULT_SUPERTYPE_LIKE_RANDOM_OUTDIR,
}
SUBGROUP_OBS_COLUMN = "supertype_name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--grouping",
        choices=["random", "supertype", "supertype_like_random"],
        default="random",
        help=(
            "Within-target grouping strategy. 'random' preserves the existing "
            "five-way split; 'supertype' uses h5ad obs['supertype_name']; "
            "'supertype_like_random' draws random groups without replacement "
            "whose sizes match the annotated supertype sizes of each target."
        ),
    )
    parser.add_argument("--level", choices=["class", "subclass"], default="subclass")
    parser.add_argument("--channel", choices=["t7", "joint"], default="joint")
    parser.add_argument("--method", choices=["svi", "nuts"], default="svi")
    parser.add_argument(
        "--infection-model",
        choices=["copy_number", "copy_number_dropout", "binary"],
        default="copy_number_dropout",
    )
    parser.add_argument("--dropout-prior-label", default="default_beta_1_9")
    parser.add_argument("--p-drop-t7-alpha", type=float, default=1.0)
    parser.add_argument("--p-drop-t7-beta", type=float, default=9.0)
    parser.add_argument("--p-drop-cre-alpha", type=float, default=1.0)
    parser.add_argument("--p-drop-cre-beta", type=float, default=9.0)
    parser.add_argument(
        "--negative-control-mode",
        choices=["pooled", "ordinary"],
        default="ordinary",
    )
    parser.add_argument(
        "--activity-model",
        choices=["hierarchical", "direct"],
        default="direct",
    )
    parser.add_argument("--kmax", type=int, default=None)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--guide",
        choices=["AutoNormal", "AutoLowRankMultivariateNormal"],
        default="AutoNormal",
    )
    parser.add_argument("--num-warmup", type=int, default=1_000)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--num-posterior", type=int, default=1_000)
    parser.add_argument("--posterior-sites", nargs="+", default=["log_gamma"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--n-groups", type=int, default=N_GROUPS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--max-cres", type=int, default=None, help="Smoke testing only.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = OUTDIR_BY_GROUPING[args.grouping]
    if args.activity_model == "direct" and args.negative_control_mode != "ordinary":
        raise ValueError(
            "--activity-model direct requires --negative-control-mode ordinary"
        )
    if args.cpu:
        import os

        os.environ["JAX_PLATFORMS"] = "cpu"

    sys.path.insert(0, str(STARRFISH_ROOT / "STARRFISH"))
    import bayesian_hierarchical as bh  # noqa: E402

    priors = dataclasses.replace(
        bh.ModelPriors(),
        p_drop_t7_alpha=args.p_drop_t7_alpha,
        p_drop_t7_beta=args.p_drop_t7_beta,
        p_drop_cre_alpha=args.p_drop_cre_alpha,
        p_drop_cre_beta=args.p_drop_cre_beta,
    )

    adata = read_and_prepare_adata(
        args.h5ad, section="all", max_cells=args.max_cells,
        max_cres=args.max_cres, seed=args.seed,
    )

    targets = top_subclasses(n=args.top_n)
    present = set(adata.obs["subclass"].astype(str).unique())
    missing = [t for t in targets if t not in present]
    if missing:
        raise ValueError(f"top subclasses absent from adata: {missing}")
    original = adata.obs["subclass"].astype(str).to_numpy()
    if args.grouping == "random":
        relabelled, assignment = relabel_subclasses(
            original, targets, n_groups=args.n_groups, seed=args.split_seed
        )
        subgroups_by_subclass = {
            target: [
                f"{target}_group_{group}"
                for group in range(1, args.n_groups + 1)
            ]
            for target in targets
        }
        subgroup_obs_column = None
    else:
        subgroup_obs_column = SUBGROUP_OBS_COLUMN
        if subgroup_obs_column not in adata.obs.columns:
            raise KeyError(
                f"h5ad obs is missing required column {subgroup_obs_column!r}"
            )
        source_labels = adata.obs[subgroup_obs_column].to_numpy()
        if args.grouping == "supertype":
            relabelled, assignment, subgroups_by_subclass = (
                relabel_subclasses_from_obs(original, source_labels, targets)
            )
        else:
            relabelled, assignment, subgroups_by_subclass = (
                relabel_subclasses_size_matched(
                    original, source_labels, targets, seed=args.split_seed
                )
            )
    adata.obs["subclass"] = relabelled
    n_subgroups_by_subclass = {
        target: len(members) for target, members in subgroups_by_subclass.items()
    }
    subgroup_cell_counts = (
        assignment["new_subclass"]
        .value_counts()
        .reindex(
            [
                member
                for target in targets
                for member in subgroups_by_subclass[target]
            ]
        )
        .astype(int)
        .to_dict()
    )
    log(
        f"[bayesian-split] grouping={args.grouping}; relabelled {len(targets)} "
        f"subclasses into {sum(n_subgroups_by_subclass.values())} groups; "
        f"{adata.obs['subclass'].nunique()} total subclasses"
    )

    cre_info = adata.uns["CRE_info"].copy()
    blacklist = cre_blacklist(cre_info.index)
    cre_names = [n for n in cre_info.index.astype(str) if n not in blacklist]
    negative_controls = negative_control_names(cre_info, blacklist)
    negative_control_mask = np.isin(cre_names, negative_controls)
    model_negative_control_mask = (
        negative_control_mask if args.negative_control_mode == "pooled" else None
    )

    t7 = adata.obsm["T7CRE"].loc[:, cre_names].to_numpy()
    cre = adata.obsm["CRE"].loc[:, cre_names].to_numpy()
    subclasses = adata.obs["subclass"].astype(str).to_numpy()
    classes = adata.obs["class"].astype(str).to_numpy()

    raw_libsize = pd.read_csv(LIBSIZE_CSV, index_col=0)
    raw_libsize.index = raw_libsize.index.astype(str)
    raw_libsize = raw_libsize.reindex(cre_names, fill_value=0)
    lib_size_log = np.log1p(raw_libsize["counts"].to_numpy(dtype=np.float64))

    args.outdir.mkdir(parents=True, exist_ok=True)
    cre_info.to_csv(args.outdir / "cre_info.csv")
    pd.Series(blacklist, name="cre").to_csv(args.outdir / "cre_blacklist.csv", index=False)
    pd.Series(negative_controls, name="cre").to_csv(
        args.outdir / "negative_controls.csv", index=False
    )
    pd.Series(adata.obs["subclass"].value_counts(), name="n_cells").to_csv(
        args.outdir / "subclass_cell_counts.csv"
    )
    assignment.to_csv(args.outdir / "cell_group_assignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "original_subclass": target,
                "new_subclass": member,
                "n_cells": subgroup_cell_counts[member],
            }
            for target in targets
            for member in subgroups_by_subclass[target]
        ]
    ).to_csv(args.outdir / "subgroup_cell_counts.csv", index=False)
    pd.Series(targets, name="subclass").to_csv(
        args.outdir / "split_subclasses.csv", index=False
    )

    tag = f"{args.level}_{args.channel}_{args.infection_model}_{args.method}"
    posterior_sites = (
        ["all"] if any(s.lower() == "all" for s in args.posterior_sites)
        else args.posterior_sites
    )
    result = bh.run_model(
        t7, cre, subclasses, classes, lib_size_log, cre_names,
        level=args.level, channel=args.channel, method=args.method, kmax=args.kmax,
        num_steps=args.steps, lr=args.lr, guide=args.guide,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, num_posterior=args.num_posterior, seed=args.seed,
        negative_control_mask=model_negative_control_mask,
        infection_model=args.infection_model,
        activity_model=args.activity_model,
        priors=priors,
        posterior_sites_to_return=posterior_sites,
    )
    grouping_config = {
        "grouping": args.grouping,
        "subgroup_obs_column": subgroup_obs_column,
        "subgroups_by_subclass": subgroups_by_subclass,
        "n_subgroups_by_subclass": n_subgroups_by_subclass,
        "subgroup_cell_counts": subgroup_cell_counts,
    }
    if args.grouping == "random":
        grouping_config.update(
            {
                "n_groups": args.n_groups,
                "split_seed": args.split_seed,
            }
        )
    elif args.grouping == "supertype_like_random":
        grouping_config.update(
            {
                "split_seed": args.split_seed,
                "size_matched_source_column": subgroup_obs_column,
            }
        )
    result["config"].update(
        {
            "input": input_fingerprint(args.h5ad),
            "blacklist_cre": blacklist,
            "section": "all",
            "posterior_sites_to_return": posterior_sites,
            "split_subclasses": targets,
            "dropout_model": (
                "zero_inflated"
                if args.infection_model == "copy_number_dropout"
                else "none"
            ),
            "dropout_prior": dropout_prior_config(args),
            "negative_control_mode": args.negative_control_mode,
            "activity_model": args.activity_model,
            "annotated_negative_control_cre": negative_controls,
            **grouping_config,
        }
    )

    prefix = args.outdir / tag
    summary = result["summary"]
    for key in ("rho", "infection", "gamma"):
        if key in summary:
            summary[key].to_csv(f"{prefix}_{key}.csv", index=False)
    if "delta_mean" in summary:
        summary["delta_mean"].to_csv(f"{prefix}_delta_mean.csv")
    result["evidence"]["per_pair"].to_csv(f"{prefix}_evidence_per_pair.csv", index=False)
    write_json(Path(f"{prefix}_evidence_totals.json"), jsonable(result["evidence"]["totals"]))
    write_json(Path(f"{prefix}_ppc.json"), jsonable(result["ppc"]))

    diagnostics = {k: v for k, v in result["diagnostics"].items() if k != "losses"}
    if "losses" in result["diagnostics"]:
        losses = np.asarray(result["diagnostics"]["losses"])
        diagnostics.update(
            {
                "loss_start": float(losses[0]),
                "loss_end": float(losses[-1]),
                "loss_all_finite": bool(np.isfinite(losses).all()),
            }
        )
        np.save(f"{prefix}_losses.npy", losses)
    write_json(Path(f"{prefix}_diagnostics.json"), jsonable(diagnostics))
    np.savez(f"{prefix}_scalar_samples.npz", **result["scalar_samples"])

    posterior = result.pop("posterior_samples")
    posterior = {
        k: np.asarray(v, dtype=np.float32)
        if np.issubdtype(np.asarray(v).dtype, np.floating)
        else np.asarray(v)
        for k, v in posterior.items()
    }
    posterior["group_names"] = np.asarray(result["group_names"], dtype=object)
    posterior["cre_names"] = np.asarray(result["cre_names"], dtype=object)
    np.savez_compressed(f"{prefix}_posterior_samples.npz", **posterior)

    with Path(f"{prefix}_result.pkl").open("wb") as handle:
        pickle.dump(result, handle)
    write_json(
        args.outdir / "run_manifest.json",
        {
            "tag": tag,
            "input": input_fingerprint(args.h5ad),
            "n_cells": int(adata.n_obs),
            "n_cres_fitted": int(len(cre_names)),
            "n_subclasses": int(adata.obs["subclass"].nunique()),
            "n_classes": int(adata.obs["class"].nunique()),
            "split_subclasses": targets,
            "blacklist": blacklist,
            "negative_controls": negative_controls,
            "negative_control_mode": args.negative_control_mode,
            "activity_model": args.activity_model,
            "dropout_prior": dropout_prior_config(args),
            "config": result["config"],
            **grouping_config,
        },
    )
    log(f"[bayesian-split] wrote intermediates to {args.outdir}")


if __name__ == "__main__":
    main()
