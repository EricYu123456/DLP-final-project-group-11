"""Select a forest of question partitions optimized for routing across historical rosters."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from coldstart.policy_forest import PolicyForest, fit_forest_profile, forest_scores, leaf_similarity
from train_conditional_router import development_rosters, robustness
from train_skill_embeddings import skill_measurements
from run_experiment import atomic_csv, atomic_json


def run(args):
    d, s, *_ = input_context(args.run)
    path = args.out / args.name
    path.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {int(v): i for i, v in enumerate(tr)}
        panels = {seed: np.array([pos[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    dt, dd = d.domains[tr], d.domains[dev]
    ref = np.r_[s.train_models, s.val_models] if args.refit else s.train_models
    h = d.labels[np.ix_(ref, tr)]
    head = joblib.load(args.out / args.readout / "readout.joblib")
    x = skill_measurements(args, d, np.r_[tr, dev], head, "development")
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds = f["dev_source"]
    model = PolicyForest.fit(x[: len(tr)], h, dt)
    prefix = "refit" if args.refit else "development"
    joblib.dump(model, path / f"{prefix}.joblib", compress=2)
    _, leaves = model.predict(x[: len(tr)], dt)
    np.savez_compressed(path / f"{prefix}_leaves.npz", leaves=leaves)
    print(args.name, len(h), round(perf_counter() - start, 1), flush=True)
    if args.refit:
        return
    pred, xx = model.predict(x[len(tr) :], ds)
    truth = d.labels[np.ix_(s.val_models, dev)]
    y = d.labels[np.ix_(s.val_models, tr)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []
    for budget in (120, 200):
        sims = [leaf_similarity(xx, leaves[ix[:budget]]) for ix in panels.values()]
        for penalty, nearest in product((3.0, 15.0), (0.0, 0.65)):
            z = [
                fit_forest_profile(h, y, ix[:budget], dt, leaves, penalty, nearest) for ix in panels.values()
            ]
            for signal, residual in product((0.75, 1.6, 3.0), (0.1, 0.6, 1.3)):
                cfg = dict(
                    name=args.name,
                    algorithm="forest",
                    readout=args.readout,
                    axes=args.axes,
                    policy="balanced",
                    probe_count=budget,
                    penalty=penalty,
                    nearest=nearest,
                    signal=signal,
                    residual=residual,
                )
                score = np.stack(
                    [forest_scores(v, model.means, pred, ds, sim, cfg) for v, sim in zip(z, sims)]
                )
                row = dict(**cfg, **metrics(score.argmax(1), truth, dd))
                row["subpool_gain"], row["within_domain_gain"] = robustness(score, truth, dd, rosters)
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
    p.add_argument("--axes")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        run(p.parse_args())
