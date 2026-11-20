"""Fit an answer-mixture embedding on historical binary outcomes without access to correct option labels."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from coldstart.latent_answers import answer_arrays, fit_history, fit_newcomers, predict_answers
from coldstart.switch_calibration import peer_vectors
from train_conditional_router import development_rosters, proxy_data, robustness
from run_experiment import atomic_csv, atomic_json


def historical_keys(folder, ids):
    with np.load(folder / "historical_answer_keys.npz") as f:
        if not np.array_equal(f["prompt_ids"], ids):
            raise ValueError("Historical answer keys must match the allowed training split.")
        return f["option"]


def fit_answers(net, hz, raw, dt, h, y, ix, prior, nearest):
    z = fit_newcomers(net, hz, raw, dt, h, y, ix, prior, nearest)
    train = predict_answers(net, z, raw, dt)
    center = np.stack([train[:, dt == j].mean(1) for j in range(net.d)], axis=1)
    hist = np.stack([h[:, dt == j].mean(1) for j in range(net.d)], axis=1)
    w, adj = peer_vectors(h, y, ix, dt, nearest)
    return dict(z=z, center=center, observed=w @ hist + adj, w=w, hist=hist)


def assembled_scores(pf, values, ds, peer_pred, cfg):
    center = pf["center"][:, ds]
    mean = cfg["source_mix"] * pf["observed"][:, ds] + (1 - cfg["source_mix"]) * center
    peer = pf["w"] @ (peer_pred - pf["hist"][:, ds].T).T
    return mean + cfg["signal"] * ((1 - cfg["peer_mix"]) * (values - center) + cfg["peer_mix"] * peer)


def run(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {int(i): j for j, i in enumerate(tr)}
        panels = {seed: np.array([pos[int(i)] for i in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    dt, dd = d.domains[tr], d.domains[dev]
    ref = np.r_[s.train_models, s.val_models] if args.refit else s.train_models
    h = d.labels[np.ix_(ref, tr)]
    raw = answer_arrays(proxy_data(args.out, d.prompt_ids[np.r_[tr, dev]], args.proxies, "development"))
    rt = {k: v[: len(tr)] for k, v in raw.items()}
    rd = {k: v[len(tr) :] for k, v in raw.items()}
    if args.answer_keys:
        rt["option_key"] = historical_keys(args.out, d.prompt_ids[tr])
    path = args.out / args.name
    path.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    net, hz = fit_history(rt, dt, h)
    prefix = "refit" if args.refit else "development"
    torch.save(
        dict(experts=net.e, domains=net.d, state=net.state_dict(), historical=hz), path / f"{prefix}.pt"
    )
    print(args.name, len(ref), "reference models", round(perf_counter() - start, 1), "s", flush=True)
    if args.refit:
        return
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds, peer = f["dev_source"], f["dev"]
    y, truth = d.labels[np.ix_(s.val_models, tr)], d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []
    for n, prior, nearest in product((120, 200), (0.45, 2.8), (0.20, 0.76)):
        pfs = [fit_answers(net, hz, rt, dt, h, y, idx[:n], prior, nearest) for idx in panels.values()]
        values = [predict_answers(net, pf["z"], rd, ds) for pf in pfs]
        for signal, source, peers in product((0.7, 1.45, 2.65), (0.40, 1.0), (0.0, 0.45)):
            cfg = dict(
                name=args.name,
                algorithm="answers",
                answer_keys=args.answer_keys,
                readout=args.readout,
                proxies=" ".join(args.proxies),
                policy="balanced",
                probe_count=n,
                prior=prior,
                nearest=nearest,
                signal=signal,
                source_mix=source,
                peer_mix=peers,
            )
            out = np.stack([assembled_scores(pf, v, ds, peer, cfg) for pf, v in zip(pfs, values)])
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
    p.add_argument("--readout", required=True)
    p.add_argument("--proxies", nargs="+", required=True)
    p.add_argument("--refit", action="store_true")
    p.add_argument("--answer-keys", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):
        run(p.parse_args())
