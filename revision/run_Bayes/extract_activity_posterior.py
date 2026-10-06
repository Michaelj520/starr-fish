#!/usr/bin/env python3
"""Write the posterior per-cell activity alone, from a copy-number npz that has it.

    python revision/run_Bayes/extract_activity_posterior.py \
        --source revision/Bayes_NewData/copy_number/activity_normalized.npz \
        --out revision/Bayes_NewData/copy_number/activity_posterior.npz

``activity_normalized.npz`` already carries ``activity`` from the same posterior
pass, so this copies that array with its axis labels instead of running the
reconstruction a second time. The result holds ``activity``, ``obs_names`` and
``cre_names`` -- about a third of the source's size for readers that need only
the un-normalised activity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from baystarrfish._log import log

#: Arrays copied from the source, in the order they are written.
KEYS = ("activity", "obs_names", "cre_names")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="npz written by `python -m baystarrfish copy-number "
                             "--with-activity`")
    parser.add_argument("--out", type=Path, required=True, help="output .npz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.suffix != ".npz":
        raise SystemExit(f"--out must end in .npz, got {args.out}")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out} (pass --overwrite)")

    # allow_pickle: obs_names and cre_names are object arrays this package wrote.
    with np.load(args.source, allow_pickle=True) as handle:
        missing = [key for key in KEYS if key not in handle.files]
        if missing:
            raise SystemExit(
                f"{args.source} lacks {missing}; rerun the copy-number CLI with "
                "--with-activity"
            )
        payload = {key: handle[key] for key in KEYS}

    activity = np.asarray(payload["activity"], dtype=np.float32)
    if activity.shape != (len(payload["obs_names"]), len(payload["cre_names"])):
        raise SystemExit(
            f"activity {activity.shape} does not match {len(payload['obs_names'])} "
            f"cells x {len(payload['cre_names'])} cCREs"
        )
    payload["activity"] = activity

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    log(f"[activity] wrote {activity.shape[0]:,} x {activity.shape[1]} activity to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
