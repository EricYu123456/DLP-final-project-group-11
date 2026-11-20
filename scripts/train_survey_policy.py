"""Choose a cross-fitted switching policy using the candidates' own graded probes and historical controls."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from coldstart.survey_policy import fit_policy, route_policy
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, robustness


def run(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {int(v): j for j, v in enumerate(tr)}
        panels = {seed: np.array([pos[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    with np.load(args.out / args.readout / "predictions.npz") as f:
        oof, p, ds = f["historical"], f["dev"], f["dev_source"]
    dt, dd = d.domains[tr], d.domains[dev]
    h, y, truth = (
        d.labels[np.ix_(s.train_models, tr)],
        d.labels[np.ix_(s.val_models, tr)],
        d.labels[np.ix_(s.val_models, dev)],
    )
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []
    for n, nearest, ridge in product((120, 160, 200), (0.15, 0.75), (65.0, 290.0)):
        profiles = [fit_policy(h, y, idx[:n], dt, oof, nearest, ridge, seed) for seed, idx in panels.items()]
        for allow, calibrate in product((-0.55, 0.0, 0.4, 0.85), (0.45, 0.85, 1.0)):
            cfg = dict(
                name=args.name,
                algorithm="survey",
                readout=args.readout,
                policy="balanced",
                probe_count=n,
                nearest=nearest,
                ridge=ridge,
                allow=allow,
                calibrate=calibrate,
            )
            scores = np.stack([route_policy(z, p, ds, cfg) for z in profiles])
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
