"""Fit latent response types on history and select their text gate on distinct development LLMs."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from coldstart.latent_skills import gate_scores, item_posteriors, measure_types, route_types, train_gate
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, robustness


def train(args):
    data, split, *_ = input_context(args.run)
    path = args.out / args.name / ("refit" if args.refit else "dev")
    path.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "questions.npz") as bank:
        tr, dev = bank["training"], bank["development"]
        position = {int(v): i for i, v in enumerate(tr)}
        panels = {s: np.array([position[int(i)] for i in bank[f"balanced_{s}"]]) for s in (11, 23, 37)}
    with np.load(args.out / args.coordinates / "development_features.npz") as f:
        data_x = f["x"]
    with np.load(args.out / args.readout / "predictions.npz") as f:
        predicted, ds = f["dev"], f["dev_source"]
    refs = np.r_[split.train_models, split.val_models] if args.refit else split.train_models
    h = data.labels[np.ix_(refs, tr)]
    dt, dd = data.domains[tr], data.domains[dev]
    means = np.stack([h[:, dt == j].mean(1) for j in np.unique(dt)])
    valid = data.labels[np.ix_(split.val_models, dev)]
    y = data.labels[np.ix_(split.val_models, tr)]
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(
            known_choices(f, data, data.model_ids[split.val_models], dev, tuple(panels))[0], valid, dd
        )
    rosters = development_rosters(valid, dd)
    rows = []
    for classes in (6, 12):
        rates, posterior = item_posteriors(h, dt, classes)
        np.savez_compressed(path / f"types_{classes}.npz", rates=rates, posterior=posterior, means=means)
        stages = (160, 470, 1100)
        dev_probs = {}

        def collect(step, gate):
            torch.save(gate.state_dict(), path / f"gate_{classes}_{step}.pt")
            if not args.refit:
                dev_probs[step] = gate_scores(gate, data_x[len(tr) :], ds)

        train_gate(data_x[: len(tr)], dt, posterior, stages, collect, args.device)
        if args.refit:
            continue
        for budget, shrink, neighbor in product((120, 200), (5, 22), (0, 0.65)):
            embs = [
                measure_types(h, y, order[:budget], dt, rates, posterior, shrink, neighbor)
                for order in panels.values()
            ]
            for step, mix, sig in product(stages, (0.5, 1.0), (0.8, 1.7, 3.0)):
                cfg = dict(
                    name=args.name,
                    algorithm="types",
                    readout=args.readout,
                    coordinates=args.coordinates,
                    policy="balanced",
                    probe_count=budget,
                    classes=classes,
                    steps=step,
                    shrink=shrink,
                    neighbor=neighbor,
                    type_mix=mix,
                    signal=sig,
                )
                scores = np.stack([route_types(z, predicted, dev_probs[step], ds, means, cfg) for z in embs])
                row = dict(**cfg, **metrics(scores.argmax(1), valid, dd))
                row["subpool_gain"], row["within_domain_gain"] = robustness(scores, valid, dd, rosters)
                row["robust_objective"] = row["objective"] + 0.5 * (
                    row["subpool_gain"] + row["within_domain_gain"]
                )
                rows.append(row)
        print(f"Trained {classes} latent item types per source with {len(h)} reference models", flush=True)
    if rows:

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="latent-general")
    parser.add_argument("--readout", default="general")
    parser.add_argument("--coordinates", default="joint-general")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):
        train(parser.parse_args())
