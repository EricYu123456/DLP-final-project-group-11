"""Select region-specific model fingerprints and calibration based on the 200-answer panel."""

import argparse
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from train_conditional_router import ROOT, development_rosters, input_context, metrics, robustness
from coldstart.conditional_capability import (
    assign_regions,
    calibrated_scores,
    calibrate_switches,
    choose_panel,
    local_fingerprints,
    score_local,
)
from run_experiment import atomic_csv, atomic_json


def select(args):
    d, s, *_ = input_context(args.run)
    path = args.out / args.name
    with np.load(args.out / "questions.npz") as split:
        tr, dev = split["training"], split["development"]
        pos = {int(x): i for i, x in enumerate(tr)}
        standard = {seed: np.array([pos[int(x)] for x in split[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    with np.load(path / "predictions.npz") as scores:
        oof, pred, source = scores["historical"], scores["dev"], scores["dev_source"]
    h, y = d.labels[np.ix_(s.train_models, tr)], d.labels[np.ix_(s.val_models, tr)]
    truth, dom = d.labels[np.ix_(s.val_models, dev)], d.domains[dev]
    native = json.loads((args.report / f"{args.name}_selection.json").read_text())["native"]
    rosters = development_rosters(truth, dom)
    rows, finalists = [], []

    def key(row):
        win = row["macro"] > native["macro"] and row["micro"] > native["micro"] and row["top_share"] < 0.92
        return win, row.get("robust_objective", row["objective"])

    for n in (5, 10):
        reg = joblib.load(path / f"regions_{n}.joblib")
        gr, gd, means, pop = [reg[k] for k in ("groups", "group_domain", "means", "pop")]
        local = assign_regions(pred, source, reg["models"])
        for policy, budget in product(("balanced", "regional", "adaptive"), (120, 200)):
            panels = {
                seed: standard[seed][:budget]
                if policy == "balanced"
                else choose_panel(h, y, gr, gd, means, policy, seed, budget)
                for seed in standard
            }
            for focus in (3, 15):
                common = dict(
                    name=args.name,
                    algorithm="local",
                    n_regions=n,
                    policy=policy,
                    probe_count=budget,
                    prior=12,
                    focus=focus,
                )
                z = {
                    seed: local_fingerprints(h[:, ix], y[:, ix], gr[ix], means, pop, gd, focus, 12)
                    for seed, ix in panels.items()
                }
                options = []
                for sig, scale, discount in product((0.65, 1.4, 2.5), (0.5, 1.3), (0.0, 0.25)):
                    cfg = dict(**common, signal=sig, local_scale=scale, global_discount=discount)
                    raw = np.stack([score_local(v, pred, local, means, pop, gd, cfg) for v in z.values()])
                    row = dict(**cfg, calibration=0.0, **metrics(raw.argmax(1), truth, dom))
                    rows.append(row)
                    options.append((row, raw))
                row, raw = max(options, key=lambda item: key(item[0]))
                cal = [
                    calibrate_switches(h, y, ix, oof, gr, means, pop, gd, z[seed], row, seed)
                    for seed, ix in panels.items()
                ]
                for amount in (0.0, 0.5, 1.0):
                    adjusted = np.stack([calibrated_scores(r, c, amount) for r, c in zip(raw, cal)])
                    result = dict(row, calibration=amount)
                    result.update(metrics(adjusted.argmax(1), truth, dom))
                    result["subpool_gain"], result["within_domain_gain"] = robustness(
                        adjusted, truth, dom, rosters
                    )
                    result["robust_objective"] = (
                        result["objective"]
                        + 0.5 * result["subpool_gain"]
                        + 0.5 * result["within_domain_gain"]
                    )
                    finalists.append(result)
                    rows.append(result)
    best = max(finalists, key=key)
    atomic_csv(args.report / f"{args.name}_local_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_local_selection.json",
        dict(best=best, short=max((r for r in finalists if r["probe_count"] == 120), key=key), native=native),
    )
    print(args.name, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="text")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        select(p.parse_args())
