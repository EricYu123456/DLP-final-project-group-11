"""Validate compact model embeddings learned from agreement with pairs of reference LLMs."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from coldstart.conditional_text import consensus_features
from coldstart.paired_moments import MomentReadout, measure, score_moments
from train_conditional_router import development_rosters, proxy_data, robustness
from run_experiment import atomic_csv, atomic_json


def extra_features(args, prompt_ids, proxies, phase):
    if not proxies:
        return None
    return consensus_features(proxy_data(args.out, prompt_ids, proxies, phase))


def run(args):
    data, split, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        lookup = {int(v): j for j, v in enumerate(tr)}
        panels = {s: np.array([lookup[int(v)] for v in f[f"balanced_{s}"]]) for s in (11, 23, 37)}
    with np.load(args.out / args.coordinates / "development_features.npz") as f:
        x = f["x"]
    extra = extra_features(args, data.prompt_ids[np.r_[tr, dev]], args.proxies, "development")
    if extra is not None:
        x = np.column_stack((x, extra))
    dt, dd = data.domains[tr], data.domains[dev]
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds = f["dev_source"]
    ref = np.r_[split.train_models, split.val_models] if args.refit else split.train_models
    h = data.labels[np.ix_(ref, tr)]
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    filename = dest / ("refit.joblib" if args.refit else "development.joblib")
    start = perf_counter()
    model = MomentReadout.fit(x[: len(tr)], dt, h)
    joblib.dump(model, filename, compress=2)
    print(
        f"{len(ref)} historical models and their pairwise response moments fitted in {perf_counter() - start:.1f}s",
        flush=True,
    )
    if args.refit:
        return
    y, valid = data.labels[np.ix_(split.val_models, tr)], data.labels[np.ix_(split.val_models, dev)]
    rosters = development_rosters(valid, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(
            known_choices(f, data, data.model_ids[split.val_models], dev, tuple(panels))[0], valid, dd
        )
    coords = model.predict(x[len(tr) :], ds)
    rows = []
    for budget, pseudo, sharp in product((120, 200), (4.0, 15.0), (30.0, 95.0)):
        embedded = [measure(h, y, ix[:budget], dt, model, pseudo, sharp) for ix in panels.values()]
        for signal in (0.65, 1.4, 2.7, 4.5):
            scores = np.stack([score_moments(z, coords, ds, signal) for z in embedded])
            cfg = dict(
                name=args.name,
                algorithm="moments",
                readout=args.readout,
                coordinates=args.coordinates,
                proxies=" ".join(args.proxies),
                policy="balanced",
                probe_count=budget,
                pseudo=pseudo,
                sharpness=sharp,
                signal=signal,
            )
            row = dict(**cfg, **metrics(scores.argmax(1), valid, dd))
            row["subpool_gain"], row["within_domain_gain"] = robustness(scores, valid, dd, rosters)
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
    p.add_argument("--name", default="moments-general")
    p.add_argument("--readout", default="general")
    p.add_argument("--coordinates", default="joint-general")
    p.add_argument("--proxies", nargs="*", default=[])
    p.add_argument("--refit", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        run(p.parse_args())
