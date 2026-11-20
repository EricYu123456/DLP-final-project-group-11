"""Train a bidirectional routing encoder on reference labels and independent solver uncertainty."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.special import logit
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from coldstart.conditional_text import consensus_features
from coldstart.distilled_text import (
    DistilledText,
    batch_ids,
    distilled_scores,
    encode_tokens,
    fit_distilled,
    historical_embeddings,
    tokenize,
    training_loss,
)
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_capability_router import VramMeter, prompt_texts
from train_conditional_router import development_rosters, proxy_data, robustness


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = DistilledText(state["history"], state["domains"], state["auxiliary"])
    model.load_state_dict(state["state"])
    return model.eval(), state["limit"]


def fit(args, d, s, tr, dev, dest):
    seq, pad = tokenize(
        prompt_texts(args.raw / "question_order.csv", d.prompt_ids[np.r_[tr, dev]]), args.limit
    )
    h = d.labels[np.ix_(s.train_models, tr)]
    dt = d.domains[tr]
    aux = consensus_features(proxy_data(args.out, d.prompt_ids[tr], args.proxies, "development"))
    aux = (aux - aux.mean(0)) / aux.std(0).clip(0.07)
    _, _, axes = randomized_svd(aux, min(24, aux.shape[1]), n_iter=4, random_state=8813)
    aux = aux @ axes.T
    aux = (aux / aux.std(0).clip(0.3)).clip(-4.0, 4.0)
    rng = np.random.default_rng(8813)
    torch.manual_seed(8813)
    n_domains = int(dt.max()) + 1
    meter = VramMeter(16) if args.device.startswith("cuda") else None
    net = DistilledText(len(h), n_domains, aux.shape[1]).to(args.device)
    other = [p for name, p in net.named_parameters() if not name.startswith("text.")]
    opt = torch.optim.AdamW(
        [dict(params=net.text.parameters(), lr=1.35e-5), dict(params=other, lr=0.0007)], weight_decay=0.035
    )
    final = max(args.stages)

    def rate(n):
        return min(1, (n + 1) / 90) * (0.25 + 0.75 * (1 + np.cos(np.pi * min(n, final) / final)) / 2)

    schedule = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    hist = torch.as_tensor(h.T.copy(), device=args.device)
    domain = torch.as_tensor(dt, device=args.device, dtype=torch.long)
    target = torch.as_tensor(aux, device=args.device, dtype=torch.float32)
    raw = np.stack([h[:, dt == j].mean(1) for j in range(n_domains)])
    priors = torch.as_tensor(logit(raw.clip(0.025, 0.975)), device=args.device, dtype=torch.float32)
    mass = 0.55 / len(dt) + 0.45 / (n_domains * np.bincount(dt)[dt])
    pa, pb = np.triu_indices(len(h), 1)
    # Similar-ability pairs carry source-specific information that broad ability alone misses.
    strength = raw.mean(0)
    weight = 0.25 + np.exp(-np.abs(strength[pa] - strength[pb]) / 0.18) * (
        1 + 0.75 * np.maximum(strength[pa], strength[pb])
    )
    weight /= weight.sum()
    start = perf_counter()
    time_rows = []
    try:
        for step in range(1, final + 1):
            ix = rng.choice(len(dt), args.batch, p=mass)
            pairs = rng.choice(len(pa), min(256, len(pa)), p=weight, replace=False)
            a, b = (torch.as_tensor(v[pairs], device=args.device) for v in (pa, pb))
            data = batch_ids(seq, ix, pad, args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                loss = training_loss(net, data, hist[ix], domain[ix], target[ix], priors, a, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            schedule.step()
            if step % 200 == 0:
                print(
                    "distilled text", step, round(perf_counter() - start, 1), float(loss.detach()), flush=True
                )
            if step in args.stages:
                state = {
                    k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.cpu())
                    for k, v in net.state_dict().items()
                }
                torch.save(
                    dict(
                        state=state,
                        history=len(h),
                        domains=n_domains,
                        auxiliary=aux.shape[1],
                        limit=args.limit,
                    ),
                    dest / f"encoder_{step}.pt",
                )
                # Decode the rounded checkpoint for the published feature cache and raw-prompt serving.
                restored, _ = load_model(dest / f"encoder_{step}.pt")
                restored.to(args.device)
                features = encode_tokens(restored, seq, pad, args.device)
                del restored, state
                np.savez_compressed(dest / f"development_{step}.npz", features=features)
                time_rows.append(
                    dict(step=step, seconds=perf_counter() - start, board_peak_mib=meter.peak if meter else 0)
                )
                atomic_csv(dest / "training_cost.csv", time_rows)
    finally:
        if meter:
            meter.close()


def validate(args, d, s, tr, dev, dest, panels):
    dt, dd = d.domains[tr], d.domains[dev]
    h, y, truth = (
        d.labels[np.ix_(s.train_models, tr)],
        d.labels[np.ix_(s.val_models, tr)],
        d.labels[np.ix_(s.val_models, dev)],
    )
    rosters = development_rosters(truth, dd)
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds = f["dev_source"]
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []
    for step in args.stages:
        with np.load(dest / f"development_{step}.npz") as f:
            x = f["features"]
        shared = historical_embeddings(x[: len(tr)], dt, h)
        for n, reg, nearest in product((120, 200), (85.0, 420.0, 1800.0), (0.12, 0.73)):
            profiles = [
                fit_distilled(shared, x[: len(tr)], dt, h, y, idx[:n], nearest, reg)
                for idx in panels.values()
            ]
            for signal in (0.45, 0.90, 1.6, 2.8):
                cfg = dict(
                    name=args.name,
                    algorithm="distilled",
                    readout=args.readout,
                    policy="balanced",
                    probe_count=n,
                    steps=step,
                    ridge=reg,
                    nearest=nearest,
                    signal=signal,
                )
                pred = np.stack([distilled_scores(p, x[len(tr) :], ds, shared, signal) for p in profiles])
                row = dict(**cfg, **metrics(pred.argmax(1), truth, dd))
                row["subpool_gain"], row["within_domain_gain"] = robustness(pred, truth, dd, rosters)
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


def run(args):
    d, s, *_ = input_context(args.run)
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        target = np.r_[f["confirmation"], f["previous_test"]]
        index = {int(v): i for i, v in enumerate(tr)}
        panels = {seed: np.array([index[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    if args.phase == "train":
        fit(args, d, s, tr, dev, dest)
    elif args.phase == "validate":
        validate(args, d, s, tr, dev, dest, panels)
    else:
        texts = prompt_texts(args.raw / "question_order.csv", d.prompt_ids[target])
        for step in args.stages:
            model, limit = load_model(dest / f"encoder_{step}.pt")
            seq, pad = tokenize(texts, limit)
            z = encode_tokens(model.to(args.device), seq, pad, args.device)
            np.savez_compressed(dest / f"evaluation_{step}.npz", ids=d.prompt_ids[target], features=z)
            del model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("train", "validate", "encode"), required=True)
    p.add_argument("--name", default="distilled-text")
    p.add_argument("--readout", default="text")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stages", nargs="+", type=int, default=[260, 820, 1900])
    p.add_argument("--proxies", nargs="+", default=["smol", "qwen", "math", "phi"])
    p.add_argument("--batch", type=int, default=36)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):
        run(p.parse_args())
