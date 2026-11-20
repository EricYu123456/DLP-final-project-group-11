"""Learn transferable model answer preferences from verified historical questions and each candidate's panel."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from coldstart.latent_answers import answer_arrays
from coldstart.nominal_response import fit_new, outputs, query, scores, train_reference
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, proxy_data, robustness
from training_answer_targets import resolve


def train(args):
    d, s, *_ = input_context(args.run)
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        index = {int(v): i for i, v in enumerate(tr)}
        panels = {seed: np.array([index[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    dt, dd = d.domains[tr], d.domains[dev]
    ids = d.prompt_ids[tr]
    support = proxy_data(args.out, ids, args.proxies, "development")
    raw = answer_arrays(support)
    keys = resolve(args.out, ids, support, raw)
    np.savez_compressed(dest / "keys.npz", prompt_ids=ids, keys=keys)
    test = answer_arrays(proxy_data(args.out, d.prompt_ids[dev], args.proxies, "development"))
    with np.load(args.out / args.readout / "predictions.npz") as f:
        peer, ds = f["dev"], f["dev_source"]
    y = d.labels[np.ix_(s.val_models, tr)]
    truth = d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    costs = []
    rows = []
    for cohort, prefix in ((s.train_models, "dev"), (np.r_[s.train_models, s.val_models], "refit")):
        begin = perf_counter()
        h = d.labels[np.ix_(cohort, tr)]
        means = np.stack([h[:, dt == j].mean(1) for j in np.unique(dt)])
        net, hist = train_reference(raw, dt, keys, h, args.steps)
        torch.save(
            dict(weights=net.state_dict(), experts=raw["votes"].shape[1], domains=net.domains),
            dest / f"{prefix}.pt",
        )
        np.savez_compressed(dest / f"{prefix}.npz", history=hist, means=means)
        costs.append(
            dict(
                reference_models=len(cohort),
                seconds=perf_counter() - begin,
                verified=int((keys >= 0).sum()),
                model_dimensions=net.width,
            )
        )
        if prefix == "refit":
            continue
        train_q = query(net, raw, dt)
        test_q = query(net, test, ds)
        for n, near, prior in product((120, 200), (0.25, 0.75), (1.0, 4.0)):
            fitted = [
                fit_new(net, hist, means, train_q, dt, keys, h, y, idx[:n], near, prior)
                for idx in panels.values()
            ]
            nominal = [outputs(net, p["z"], test_q, ds) for p in fitted]
            for signal, mix in product((0.7, 1.4, 2.4), (0.0, 0.4)):
                cfg = dict(
                    name=args.name,
                    algorithm="nominal",
                    readout=args.readout,
                    policy="balanced",
                    probe_count=n,
                    proxies=" ".join(args.proxies),
                    nearest=near,
                    prior=prior,
                    signal=signal,
                    peer=mix,
                )
                out = np.stack([scores(p, v, peer, ds, means, cfg) for p, v in zip(fitted, nominal)])
                r = dict(**cfg, **metrics(out.argmax(1), truth, dd))
                r["subpool_gain"], r["within_domain_gain"] = robustness(out, truth, dd, rosters)
                r["robust_objective"] = r["objective"] + 0.5 * (r["subpool_gain"] + r["within_domain_gain"])
                rows.append(r)

    def rank(r):
        return r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92, r[
            "robust_objective"
        ]

    best = max(rows, key=rank)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in rows if r["probe_count"] == 120), key=rank), native=native),
    )
    atomic_json(dest / "cost.json", dict(training=costs))
    print(args.name, costs, best, flush=True)


def load_net(path):
    from coldstart.nominal_response import NominalResponse

    params = torch.load(path, map_location="cpu", weights_only=True)
    net = NominalResponse(params["experts"], params["domains"])
    net.load_state_dict(params["weights"])
    return net.eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--proxies", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=850)
    p.add_argument("--readout", default="text")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        train(p.parse_args())
