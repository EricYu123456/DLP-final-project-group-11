"""Calibrate the expected gain of replacing an incumbent using paired measurements from one probe panel."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from coldstart.switch_calibration import fit_switches, switch_scores, tail_panel
from train_conditional_router import development_rosters, robustness
from run_experiment import atomic_csv, atomic_json


def run(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        positions = {int(v): j for j, v in enumerate(tr)}
        ordered = {k: np.array([positions[int(v)] for v in f[f"balanced_{k}"]]) for k in (11, 23, 37)}
    with np.load(args.out / args.readout / "predictions.npz") as f:
        oof, pred, ds = f["historical"], f["dev"], f["dev_source"]
    dt, dd = d.domains[tr], d.domains[dev]
    h = d.labels[np.ix_(s.train_models, tr)]
    y, truth = d.labels[np.ix_(s.val_models, tr)], d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(ordered))[0], truth, dd)
    rows = []
    for n, policy in product((120, 200), ("balanced", "opportunity")):
        panels = [
            a[:n] if policy == "balanced" else tail_panel(h, y, a, dt, oof, k, n) for k, a in ordered.items()
        ]
        for nearest, prior in product((0.15, 0.78), (4.0, 14.0)):
            profiles = [fit_switches(h, y, idx, dt, oof, nearest, prior) for idx in panels]
            for optimistic, cal, sig in product((0, 0.45), (0.55, 1.0), (0.9, 1.6, 2.6)):
                cfg = dict(
                    name=args.name,
                    algorithm="switch",
                    readout=args.readout,
                    policy=policy,
                    probe_count=n,
                    nearest=nearest,
                    prior=prior,
                    optimism=optimistic,
                    calibration=cal,
                    signal=sig,
                )
                scores = np.stack([switch_scores(p, pred, ds, cfg) for p in profiles])
                row = dict(**cfg, **metrics(scores.argmax(1), truth, dd))
                row["subpool_gain"], row["within_domain_gain"] = robustness(scores, truth, dd, rosters)
                row["robust_objective"] = row["objective"] + 0.5 * (
                    row["subpool_gain"] + row["within_domain_gain"]
                )
                rows.append(row)

    def key(r):
        return r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92, r[
            "robust_objective"
        ]

    best = max(rows, key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in rows if r["probe_count"] == 120), key=key), native=native),
    )
    print(best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        run(p.parse_args())
