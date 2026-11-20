"""Measure model ability by its response to independently verified solver mistakes on training questions."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from coldstart.latent_answers import answer_arrays
from coldstart.proxy_confusion import coordinates, embed, historical_design, judge_probs, references, score
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, proxy_data, robustness
from train_latent_answers import historical_keys


def run(args):
    d, s, *_ = input_context(args.run)
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        ix = {int(v): i for i, v in enumerate(tr)}
        panels = {seed: np.array([ix[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    dt, dd = d.domains[tr], d.domains[dev]
    data = answer_arrays(proxy_data(args.out, d.prompt_ids[np.r_[tr, dev]], args.proxies, "development"))
    t, q = ({k: v[: len(tr)] for k, v in data.items()}, {k: v[len(tr) :] for k, v in data.items()})
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds, peer = f["dev_source"], f["dev"]
    keys = historical_keys(args.out, d.prompt_ids[tr])
    start = perf_counter()
    pack, x, xp = historical_design(t, dt, keys)
    joblib.dump(pack, dest / "question.joblib", compress=2)
    np.savez_compressed(dest / "training.npz", x=x, xp=xp)
    xx = coordinates(pack, q, judge_probs(pack["judge"], q, ds, pack["domains"]), ds)
    refs = [s.train_models, np.r_[s.train_models, s.val_models]]
    for pop, prefix in zip(refs, ("development", "refit")):
        fitted = references(d.labels[np.ix_(pop, tr)], dt, x, xp)
        np.savez_compressed(dest / f"{prefix}.npz", **fitted)
    atomic_json(
        dest / "cost.json",
        dict(
            shared_seconds=perf_counter() - start,
            verified_historical_keys=int((keys >= 0).sum()),
            reference_models=[len(v) for v in refs],
            dimensions=x.shape[1] + pack["domains"],
        ),
    )
    with np.load(dest / "development.npz") as f:
        ref = dict(f)
    h, y, truth = (
        d.labels[np.ix_(s.train_models, tr)],
        d.labels[np.ix_(s.val_models, tr)],
        d.labels[np.ix_(s.val_models, dev)],
    )
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []
    for n, nearest, ridge in product((120, 200), (0.18, 0.70), (25.0, 105.0, 450.0)):
        fits = [embed(ref, h, y, dt, idx[:n], x, xp, nearest, ridge, seed) for seed, idx in panels.items()]
        for signal, mix in product((0.6, 1.15, 2.1, 3.0), (0.0, 0.4)):
            cfg = dict(
                name=args.name,
                algorithm="proxy-confusion",
                readout=args.readout,
                policy="balanced",
                probe_count=n,
                nearest=nearest,
                ridge=ridge,
                proxies=" ".join(args.proxies),
                signal=signal,
                peer=mix,
            )
            out = np.stack([score(p, xx, peer, ds, ref["means"], cfg) for p in fits])
            row = dict(**cfg, **metrics(out.argmax(1), truth, dd))
            row["subpool_gain"], row["within_domain_gain"] = robustness(out, truth, dd, rosters)
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
    p.add_argument("--proxies", nargs="+", required=True)
    p.add_argument("--readout", default="text")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        run(p.parse_args())
