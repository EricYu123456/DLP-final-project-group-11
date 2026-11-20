"""Fit and select routing embeddings with targeted measurements of conditional strengths."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, benchmark_means, input_context, known_choices, metrics
from panel_capabilities import panel_text
from run_experiment import atomic_csv, atomic_json
from coldstart.conditional_capability import choose_panel, fit_profile, make_regions, regional_prediction
from coldstart.conditional_text import ConditionalReadout, text_features, text_pack


def proxy_data(source, ids, proxies, phase):
    data = []
    for name in proxies:
        recorded = "una-gated" if name == "una-mc-gated" else name
        with np.load(source / f"evidence_{recorded}_{phase}.npz") as f:
            index = np.argsort(f["prompt_ids"])
            pos = index[np.searchsorted(f["prompt_ids"][index], ids)]
            if not np.array_equal(f["prompt_ids"][pos], ids):
                raise ValueError(f"Missing question in {name} answer evidence")
            probabilities, answers = f["probabilities"][pos], f["answers"][pos]
        if name == "una-mc-gated":
            from coldstart.answer_evidence import GATE_PROXIES, expert_gate

            gate = expert_gate(dict(zip(GATE_PROXIES, proxy_data(source, ids, GATE_PROXIES, phase))), multiple_only=True)
            probabilities[~gate], answers[~gate] = 0, ""
        data.append((probabilities, answers))
    return data


def development_rosters(truth, domains):
    rng = np.random.default_rng(1349)
    mean = benchmark_means(truth, domains).mean(1)
    rosters = []
    for size in (4, 7, 10):
        for _ in range(4):
            chosen = rng.choice(len(truth), size, replace=False)
            rosters.append(chosen)
        # Include dominant-incumbent pools and competitive pools when selecting a routing policy.
        for anchor in np.argsort(-mean)[:3]:
            weaker = np.flatnonzero(mean < mean[anchor] - 0.065)
            if len(weaker) >= size - 1:
                rosters.append(np.r_[anchor, rng.choice(weaker, size - 1, replace=False)])
    return rosters


def robustness(scores, truth, domains, rosters):
    gains, conditional = [], []
    by_domain = benchmark_means(truth, domains)
    for pool in rosters:
        choices = pool[scores[:, pool].argmax(1)]
        result = truth[choices, np.arange(len(domains))]
        best = (0.55 * benchmark_means(truth[pool], domains).mean(1) + 0.45 * truth[pool].mean(1)).max()
        gains.append(0.55 * benchmark_means(result, domains).mean() + 0.45 * result.mean() - best)
        chance = by_domain[choices, domains]
        conditional.append(benchmark_means(result - chance, domains).mean())
    return float(np.mean(gains)), float(np.mean(conditional))


def panel_and_vectors(
    h, answers, train_groups, group_domain, hist_means, policy, seed, baseline_ids, budget, prior
):
    pop = np.bincount(train_groups)
    idx = (
        baseline_ids[:budget]
        if policy == "balanced"
        else choose_panel(h, answers, train_groups, group_domain, hist_means, policy, seed, budget)
    )
    z = fit_profile(h[:, idx], answers[:, idx], train_groups[idx], pop, group_domain, prior)
    return idx, z


def score_vectors(z, pred, region, hist_means, group_domain, pop, cfg):
    out = regional_prediction(z, pred, region, hist_means, group_domain, cfg["signal"], cfg["local_scale"])
    if cfg["global_discount"]:
        mass = 0.5 * pop / pop.sum() + 0.5 * pop / np.bincount(group_domain, weights=pop)[group_domain] / (
            group_domain.max() + 1
        )
        average = (
            regional_prediction(
                z, hist_means, np.arange(len(pop)), hist_means, group_domain, 0, cfg["local_scale"]
            )
            @ mass
        )
        out -= cfg["global_discount"] * average[:, None]
    return out


def features_for(args, d, positions, pack, proxies, phase):
    ids = d.prompt_ids[positions]
    text = panel_text(ids, ROOT / "data/raw")
    evidence = proxy_data(args.out, ids, proxies, phase) if proxies else None
    return text_features(pack, d.embeddings[positions], text, evidence)


def train(args, d, s, split, path):
    start = perf_counter()
    tr, dev = split["training"], split["development"]
    h = d.labels[np.ix_(s.train_models, tr)]
    ids = d.prompt_ids[tr]
    x, dom = d.embeddings[tr], d.domains[tr]
    pack = text_pack(x, panel_text(ids, ROOT / "data/raw"))
    feats = features_for(args, d, np.r_[tr, dev], pack, args.proxies, "development")
    tr_x, dev_x = feats[: len(tr)], feats[len(tr) :]
    rng = np.random.default_rng(1349)
    fold = np.empty(len(tr), dtype=int)
    for j in np.unique(dom):
        order = rng.permutation(np.flatnonzero(dom == j))
        fold[order] = np.arange(len(order)) % 3
    oof = np.zeros((len(tr), len(h)), dtype=np.float32)
    for i in range(3):
        a, b = fold != i, fold == i
        fit = ConditionalReadout.fit(x[a], tr_x[a], h[:, a], dom[a])
        oof[b], _ = fit.predict(x[b], tr_x[b])
    fit = ConditionalReadout.fit(x, tr_x, h, dom)
    pred, mem = fit.predict(d.embeddings[dev], dev_x)
    joblib.dump(dict(pack=pack, readout=fit, proxies=args.proxies), path / "readout.joblib", compress=2)
    np.savez_compressed(path / "predictions.npz", historical=oof, dev=pred, dev_source=mem.argmax(1))
    print(f"{args.name} cross-fitted {len(h)} model readouts in {perf_counter() - start:.1f}s", flush=True)
    atomic_json(
        path / "training.json", dict(seconds=perf_counter() - start, proxies=args.proxies, training=len(tr))
    )


def validate(args, d, s, split, path):
    tr, dev = split["training"], split["development"]
    h = d.labels[np.ix_(s.train_models, tr)]
    truth = d.labels[np.ix_(s.val_models, dev)]
    answers = d.labels[np.ix_(s.val_models, tr)]
    with np.load(path / "predictions.npz") as inp:
        oof, pred, source = inp["historical"], inp["dev"], inp["dev_source"]
    with np.load(args.out / "embedllm_known_ids.npz") as ref:
        native = known_choices(ref, d, d.model_ids[s.val_models], dev, (11, 23, 37))[0]
    dom = d.domains[dev]
    native_metrics = metrics(native, truth, dom)
    rosters = development_rosters(truth, dom)
    lookup = {int(iv): i for i, iv in enumerate(tr)}
    ids = {seed: np.array([lookup[int(i)] for i in split[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    records, finalists = [], []
    for n_regions in (5, 10):
        (groups, local), group_domain, models = make_regions(oof, d.domains[tr], n_regions, [(pred, source)])
        pop = np.bincount(groups)
        hist_means = np.stack([h[:, groups == j].mean(1) for j in range(len(pop))])
        joblib.dump(
            dict(models=models, group_domain=group_domain, groups=groups, means=hist_means, pop=pop),
            path / f"regions_{n_regions}.joblib",
            compress=2,
        )
        for policy in ("balanced", "regional", "adaptive"):
            for budget in (120, 200):
                panels = [
                    panel_and_vectors(
                        h, answers, groups, group_domain, hist_means, policy, seed, ids[seed], budget, 12
                    )[0]
                    for seed in (11, 23, 37)
                ]
                for prior in (6, 20):
                    embeddings = [
                        fit_profile(h[:, ix], answers[:, ix], groups[ix], pop, group_domain, prior)
                        for ix in panels
                    ]
                    local_rows = []
                    for sig, scale, discount in product((0.6, 1.3, 2.2), (0.5, 1.2), (0.0, 0.25)):
                        cfg = dict(
                            name=args.name,
                            n_regions=n_regions,
                            policy=policy,
                            probe_count=budget,
                            prior=prior,
                            signal=sig,
                            local_scale=scale,
                            global_discount=discount,
                        )
                        scores = np.stack(
                            [
                                score_vectors(z, pred, local, hist_means, group_domain, pop, cfg)
                                for z in embeddings
                            ]
                        )
                        val = metrics(scores.argmax(1), truth, dom)
                        row = dict(**cfg, **val)
                        records.append(row)
                        local_rows.append((row, scores))
                    ranked = sorted(
                        local_rows,
                        key=lambda pair: (
                            pair[0]["macro"] > native_metrics["macro"]
                            and pair[0]["micro"] > native_metrics["micro"],
                            pair[0]["objective"],
                        ),
                        reverse=True,
                    )
                    for row, scores in ranked[:3]:
                        row["subpool_gain"], row["within_domain_gain"] = robustness(
                            scores, truth, dom, rosters
                        )
                        row["robust_objective"] = (
                            row["objective"] + 0.5 * row["subpool_gain"] + 0.5 * row["within_domain_gain"]
                        )
                        finalists.append(row)

    def key(row):
        return (
            row["macro"] > native_metrics["macro"]
            and row["micro"] > native_metrics["micro"]
            and row["top_share"] < 0.92,
            row["robust_objective"],
        )

    best = max(finalists, key=key)
    cheap = max((r for r in finalists if r["probe_count"] == 120), key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", records)
    atomic_json(
        args.report / f"{args.name}_selection.json", dict(best=best, short=cheap, native=native_metrics)
    )
    print(args.name, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("train", "validate"), required=True)
    p.add_argument("--proxies", nargs="*", default=[])
    p.add_argument("--name", default="text")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    opts = p.parse_args()
    target = opts.out / opts.name
    target.mkdir(parents=True, exist_ok=True)
    dataset, split, *_ = input_context(opts.run)
    with np.load(opts.out / "questions.npz") as f, threadpool_limits(limits=3):
        (train if opts.phase == "train" else validate)(opts, dataset, split, f, target)
