"""Measure the 112-model panel's cost–accuracy curve on the original release validation split."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, benchmark_means, paired_gain
from panel_capabilities import pack_panel, panel_text, public_routes, select_panel
from run_embedllm_release import SEEDS, chunks
from run_experiment import atomic_csv, atomic_json


BUDGETS = (80, 120, 160, 200)


def nested_order(domains, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(domains == j)) for j in np.unique(domains)]
    fractions = np.array([len(g) for g in groups]) / len(domains)
    counts = np.zeros(len(groups), dtype=int)
    order = []
    for n in range(1, len(domains) + 1):
        group = int(np.argmax(n * fractions - counts))
        order.append(groups[group][counts[group]])
        counts[group] += 1
    return np.array(order)


def freeze(args):
    names = pd.read_csv(args.raw / "model_order.csv").sort_values("model_id").model_name.to_numpy(str)
    details = {}
    for seed in SEEDS:
        with np.load(args.weights / f"panel_{seed}.npz") as source:
            vectors, dom, responses, ids = [source[key] for key in ("x", "domains", "y", "prompt_ids")]
        texts = panel_text(ids, args.raw)
        order = nested_order(dom, seed)
        for n in BUDGETS:
            key = f"{seed}_{n}"
            if n == 200:
                filename = f"router_{seed}.npz"
                with np.load(args.weights / filename) as original:
                    if str(original["encoder"]) == "mpnet" or not float(original["signal"]):
                        details[key] = dict(
                            questions=n, router=filename, selection="existing locked release router"
                        )
                        continue
            pos = order[:n] if n < 200 else np.arange(n)
            x, d, y, t = vectors[pos], dom[pos], responses[:, pos], texts[pos]
            start = perf_counter()
            cfg = select_panel(x, d, y, t, seed, retentions=(0.5, 0.75, 1.0))
            router = pack_panel(x, d, y, t, ids[pos], names, cfg)
            dest = f"budget_{key}.npz"
            np.savez_compressed(args.weights / dest, **router)
            details[key] = dict(questions=n, router=dest, seconds=perf_counter() - start, config=cfg)
    atomic_json(args.report / "release112_budget_selection.json", details)


def score(args):
    details = json.loads((args.report / "release112_budget_selection.json").read_text(encoding="utf-8"))
    with np.load(args.prepared) as data:
        pos = np.flatnonzero(data["original_split"] == 1)
        x, dom, ids = [data[key][pos] for key in ("embeddings", "benchmark_ids", "prompt_ids")]
    texts = panel_text(ids, args.raw)
    truth = np.full((112, len(ids)), np.nan, dtype=np.float32)
    for part in chunks(args.raw / "val.csv"):
        last = part.drop_duplicates(["model_id", "prompt_id"], keep="last")
        truth[last.model_id, np.searchsorted(ids, last.prompt_id)] = last.label
    if not np.isfinite(truth).all():
        raise ValueError("The published validation response matrix is incomplete.")
    with np.load(args.weights / "original.npz") as original:
        weights = original["weight"]
        logits = weights @ x.T + original["bias"][:, None]
    deterministic = logits.argmax(0)
    noise_std = 0.05 * np.linalg.norm(weights, axis=1, keepdims=True)
    records, comparisons, results = [], [], {}
    for seed in SEEDS:
        noisy = (logits + noise_std * np.random.default_rng(seed).standard_normal(logits.shape)).argmax(0)
        choices = {"original-deterministic": deterministic, "original-noisy": noisy}
        for n in BUDGETS:
            path = args.weights / details[f"{seed}_{n}"]["router"]
            with np.load(path) as router:
                choices[f"panel-{n}"] = public_routes(router, x, texts)
        for name, choice in choices.items():
            hit = truth[choice, np.arange(len(dom))]
            counts = np.bincount(choice, minlength=len(truth))
            results.setdefault(name, []).append(hit)
            records.append(
                dict(
                    method=name,
                    seed=seed,
                    queries=len(dom),
                    macro=benchmark_means(hit, dom).mean(),
                    micro=hit.mean(),
                    top_share=counts.max() / len(dom),
                    models_used=(counts > 0).sum(),
                )
            )
    for n in BUDGETS:
        name = f"panel-{n}"
        reference = ["original-deterministic", "original-noisy"]
        if n < 200:
            reference.append("panel-200")
        for target in reference:
            comparisons.append(
                dict(
                    method=name,
                    reference=target,
                    **paired_gain(np.stack(results[name]), np.stack(results[target]), dom),
                )
            )
    atomic_csv(args.report / "release112_budget_validation.csv", records)
    atomic_csv(args.report / "release112_budget_paired.csv", comparisons)
    print(pd.DataFrame(records).groupby("method")[["macro", "micro", "top_share"]].mean(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "score"), required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/processed/embedllm.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/capability_routing/release112")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability")
    opts = parser.parse_args()
    with threadpool_limits(limits=2):
        (freeze if opts.phase == "freeze" else score)(opts)
