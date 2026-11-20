"""Learn incumbent–challenger log odds from historical disagreements and validate new-model embeddings."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from coldstart.contrastive_routing import (
    ContrastiveDecoder,
    contrast_scores,
    embed_challengers,
    features,
    train_contrasts,
)
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, robustness
from train_response_moments import extra_features


def load_decoder(filename):
    state = torch.load(filename, map_location="cpu", weights_only=True)
    net = ContrastiveDecoder(state["width"], state["history"], state["domains"])
    net.load_state_dict(state["state"])
    return net.eval()


def run(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        lookup = {int(v): i for i, v in enumerate(tr)}
        panels = {seed: np.array([lookup[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    with np.load(args.out / args.coordinates / "development_features.npz") as f:
        x = f["x"]
    extra = extra_features(args, d.prompt_ids[np.r_[tr, dev]], args.proxies, "development")
    if extra is not None:
        x = np.column_stack((x, extra))
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds = f["dev_source"]
    dt, dd = d.domains[tr], d.domains[dev]
    ref = np.r_[s.train_models, s.val_models] if args.refit else s.train_models
    h = d.labels[np.ix_(ref, tr)]
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    prefix = "refit" if args.refit else "development"
    start = perf_counter()

    def snapshot(step, net):
        torch.save(
            dict(state=net.cpu().state_dict(), width=x.shape[1], history=len(h), domains=net.d),
            dest / f"{prefix}_{step}.pt",
        )
        net.to(args.device)
        print(f"{prefix}: {step} paired-odds training steps in {perf_counter() - start:.1f}s", flush=True)

    if not all((dest / f"{prefix}_{step}.pt").exists() for step in args.stages):
        train_contrasts(x[: len(tr)], dt, h, args.stages, snapshot, args.device)
    if args.refit:
        return
    y, truth = d.labels[np.ix_(s.val_models, tr)], d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []

    def key(r):
        return r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92, r[
            "robust_objective"
        ]

    def evaluate(step, priors, budgets):
        net = load_decoder(dest / f"{prefix}_{step}.pt")
        coords = features(net, x[: len(tr)], dt)
        target = features(net, x[len(tr) :], ds)
        for n, prior, nearest in product(budgets, priors, (0.2, 0.8)):
            fits = [
                embed_challengers(net, coords, h, y, idx[:n], dt, prior, nearest) for idx in panels.values()
            ]
            for signal, anchors in product((0.55, 1.0, 1.75, 3.0), (1, 3)):
                cfg = dict(
                    name=args.name,
                    algorithm="contrast",
                    readout=args.readout,
                    coordinates=args.coordinates,
                    proxies=" ".join(args.proxies),
                    policy="balanced",
                    probe_count=n,
                    steps=step,
                    prior=prior,
                    nearest=nearest,
                    signal=signal,
                    anchors=anchors,
                )
                out = np.stack([contrast_scores(p, target, ds, cfg) for p in fits])
                row = dict(**cfg, **metrics(out.argmax(1), truth, dd))
                row["subpool_gain"], row["within_domain_gain"] = robustness(out, truth, dd, rosters)
                row["robust_objective"] = row["objective"] + 0.5 * (
                    row["subpool_gain"] + row["within_domain_gain"]
                )
                rows.append(row)

    for step in args.stages:
        evaluate(step, (0.0,), (200,))
    step = max(rows, key=key)["steps"]
    evaluate(step, (0.0,), (120,))
    evaluate(step, (3.0, 12.0), (120, 200))
    best = max(rows, key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in rows if r["probe_count"] == 120), key=key), native=native),
    )
    print(best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="contrast-general")
    p.add_argument("--readout", default="general")
    p.add_argument("--coordinates", default="joint-general")
    p.add_argument("--proxies", nargs="*", default=[])
    p.add_argument("--device", default="cpu")
    p.add_argument("--stages", type=int, nargs="+", default=[400, 1400, 4200])
    p.add_argument("--refit", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):
        run(p.parse_args())
