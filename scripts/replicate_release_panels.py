"""Measure accuracy and panel-to-panel uncertainty for the frozen 140- and 200-question CPU routers."""

import argparse
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coldstart.model_covariates import model_descriptions
from compare_release_equal_budget import final_rows
from evaluate_capability_router import ROOT, benchmark_means, input_context
from evaluate_panel_budget import nested_order
from panel_capabilities import RELEASE_SEEDS, panel_text
from panel_text_kernel import fit_text_kernel, text_kernel_scores
from run_embedllm_release import panel
from run_experiment import atomic_csv, atomic_json

BASE = ROOT / "artifacts/capability_routing"
RELEASE = BASE / "release_confirmation"
OUT = RELEASE / "replication"
REPORT = ROOT / "reports/routing/capability/release_confirmation/replication"
DATA = ROOT / "data"
SEEDS = tuple(range(7100, 7115))


def prepare():
    with np.load(DATA / "processed/embedllm.npz") as f:
        group, dom, split, ids, vectors = [
            f[k]
            for k in (
                "group_ids",
                "benchmark_ids",
                "original_split",
                "prompt_ids",
                "embeddings",
            )
        ]
    with np.load(RELEASE / "reserve.npz") as f:
        excluded = set(group[split != 0]) | set(f["groups"])
        reviewed = excluded | set(group[f["bank"]])
    d, *_ = input_context(ROOT / "artifacts/main_v2")
    with np.load(BASE / "confirmation/questions.npz") as f:
        excluded.update(d.group_ids[f["confirmation"]])
        reviewed.update(
            d.group_ids[np.r_[f["confirmation"], f["development"], f["previous_test"]]]
        )
    for folder in (
        BASE / "release112",
        BASE / "release112-stratified",
        BASE / "release112_stratified",
        RELEASE,
    ):
        for file in folder.glob("*panel_*.npz"):
            with np.load(file) as data:
                if "y" in data and "prompt_ids" in data:
                    reviewed.update(group[data["prompt_ids"]])
    rng = np.random.default_rng(731942)
    eligible = (split == 0) & ~np.isin(group, list(reviewed))
    sources = np.unique(dom[eligible])
    counts = np.bincount(dom[split == 1])[sources].astype(float)
    counts = np.round(3000 * counts / counts.sum()).astype(int)
    held = []
    for source, request in zip(sources, counts):
        groups = np.unique(group[eligible & (dom == source)])
        held.extend(
            rng.choice(groups, min(request, int(0.35 * len(groups))), replace=False)
        )
    fresh = np.flatnonzero((split == 0) & np.isin(group, held))
    if not len(fresh) or set(group[fresh]) & reviewed:
        raise ValueError(
            "The new confirmation sample must use previously unexamined groups."
        )
    np.savez_compressed(
        OUT / "reserve.npz", groups=np.unique(group[fresh]), prompt_ids=ids[fresh]
    )
    excluded.update(group[fresh])
    available = np.flatnonzero((split == 0) & ~np.isin(group, list(excluded)))
    draws = []
    for seed in SEEDS:
        order = np.random.default_rng(seed + 501).permutation(available)
        _, first = np.unique(group[order], return_index=True)
        chosen = panel(vectors, dom, order[first], seed)
        draws.append(ids[chosen])
    np.savez_compressed(OUT / "panels.npz", seeds=SEEDS, panels=draws)
    atomic_json(
        REPORT / "sample.json",
        dict(
            seeds=SEEDS,
            fresh_questions=len(fresh),
            fresh_groups=len(np.unique(group[fresh])),
            fresh_counts=np.bincount(dom[fresh], minlength=10).tolist(),
            training_questions=len(available),
            training_groups=len(np.unique(group[available])),
            probes=200,
            reduced_probes=140,
            selected_seed=None,
        ),
    )
    print(
        "New confirmation",
        len(fresh),
        "questions on",
        len(sources),
        "sources;",
        len(draws),
        "panels",
        flush=True,
    )


def splits():
    with (
        np.load(RELEASE / "reserve.npz") as data,
        np.load(OUT / "reserve.npz") as fresh,
    ):
        return dict(
            reserve=data["prompt_ids"], official=data["test"], fresh=fresh["prompt_ids"]
        )


def grades(file, ids):
    df = final_rows(DATA / "raw" / file, ids)
    matrix = df.pivot(index="model_id", columns="prompt_id", values="label").reindex(
        columns=ids
    )
    if len(matrix) != 112 or matrix.isna().any().any():
        raise ValueError(
            "An official comparison requires a complete set of published grades."
        )
    return matrix.to_numpy(dtype=np.float32)


def fit():
    parts = splits()
    query = np.concatenate(list(parts.values()))
    with np.load(OUT / "panels.npz") as f:
        panels = f["panels"]
    bank = np.unique(panels)
    bank_y = grades("train.csv", bank)
    texts = panel_text(np.r_[bank, query], DATA / "raw")
    train_text, query_text = texts[: len(bank)], texts[len(bank) :]
    names = (
        pd.read_csv(DATA / "raw/model_order.csv")
        .sort_values("model_id")
        .model_name.to_numpy(str)
    )
    meta, size = model_descriptions(names, DATA / "metadata/model_families.csv")
    with np.load(DATA / "processed/embedllm.npz") as f:
        dom = f["benchmark_ids"]
    choices = {140: [], 200: []}
    costs = []
    for seed, ids in zip(SEEDS, panels):
        for n in choices:
            subset = (
                np.searchsorted(bank, ids[nested_order(dom[ids], seed)[:n]])
                if n < 200
                else np.searchsorted(bank, ids)
            )
            start = perf_counter()
            pack, _ = fit_text_kernel(
                dom[bank[subset]],
                bank_y[:, subset],
                train_text[subset],
                meta,
                size,
                seed,
            )
            seconds = perf_counter() - start
            choices[n].append(
                text_kernel_scores(pack, query_text).argmax(0).astype(np.int16)
            )
            costs.append(dict(seed=seed, questions=n, fit_seconds=seconds))
        print("Two fixed budgets", seed, flush=True)
    # The original three fitted models are a distinct, untouched comparator on the new sample.
    released = [
        text_kernel_scores(
            joblib.load(RELEASE / f"text-kernel_{seed}.joblib"),
            query_text[-len(parts["fresh"]) :],
        ).argmax(0)
        for seed in RELEASE_SEEDS
    ]
    np.savez_compressed(
        OUT / "routes.npz",
        query=query,
        released_fresh=released,
        **{f"panel_{n}": values for n, values in choices.items()},
    )
    atomic_csv(REPORT / "costs.csv", costs)


def two_way_interval(gap, domain, groups):
    rng = np.random.default_rng(7219)
    draws, n = 2000, len(gap)
    panels = rng.multinomial(n, np.full(n, 1 / n), size=draws) / n
    macro, micro = np.zeros(draws), np.zeros(draws)
    sources = np.unique(domain)
    for source in sources:
        use = domain == source
        _, index = np.unique(groups[use], return_inverse=True)
        lengths = np.bincount(index)
        m = len(lengths)
        total = np.stack([np.bincount(index, weights=v[use]) for v in gap])
        weights = rng.multinomial(m, np.full(m, 1 / m), size=draws)
        mean = ((panels @ total) * weights).sum(1) / (weights @ lengths)
        macro += mean / len(sources)
        micro += mean * use.mean()
    a, b = np.quantile(macro, [0.025, 0.975])
    c, d = np.quantile(micro, [0.025, 0.975])
    return dict(macro_low=a, macro_high=b, micro_low=c, micro_high=d)


def score():
    parts = splits()
    rows, pairs = [], []
    with (
        np.load(OUT / "routes.npz") as data,
        np.load(DATA / "processed/embedllm.npz") as prep,
    ):
        query, domain, groups = data["query"], prep["benchmark_ids"], prep["group_ids"]
        for stage, ids in parts.items():
            ix = np.flatnonzero(np.isin(query, ids))
            if not np.array_equal(query[ix], ids):
                raise ValueError("The frozen routes and confirmation prompts differ.")
            truth = grades("test.csv" if stage == "official" else "train.csv", ids)
            baseline = (
                dict(reserve=RELEASE, official=BASE / "release112", fresh=OUT)[stage]
                / "original.npz"
            )
            with np.load(baseline) as f:
                if not np.array_equal(f["prompt_ids"], ids):
                    raise ValueError(
                        "Full-data TextMF and panel evaluation prompts differ."
                    )
                controls = {
                    "deterministic": truth[f["deterministic"], np.arange(len(ids))],
                    "noisy": truth[f["noisy"], np.arange(len(ids))].mean(0),
                }
            for n in (140, 200, "released"):
                if n == "released" and stage != "fresh":
                    continue
                routes = (
                    data["released_fresh"]
                    if n == "released"
                    else data[f"panel_{n}"][:, ix]
                )
                hits = truth[routes, np.arange(len(ids))]
                for seed, picks, hit in zip(
                    RELEASE_SEEDS if n == "released" else SEEDS, routes, hits
                ):
                    rows.append(
                        dict(
                            split=stage,
                            panel=n,
                            seed=seed,
                            questions=len(ids),
                            micro=hit.mean(),
                            macro=benchmark_means(hit, domain[ids]).mean(),
                            top_share=np.bincount(picks).max() / len(ids),
                        )
                    )
                for name, control in controls.items():
                    gap = hits - control
                    pairs.append(
                        dict(
                            split=stage,
                            panel=n,
                            reference=name,
                            reference_micro=control.mean(),
                            accuracy=hits.mean(),
                            micro_gain=gap.mean(),
                            macro_gain=benchmark_means(gap, domain[ids]).mean(),
                            panel_wins=int((gap.mean(1) > 0).sum()),
                            panel_count=len(hits),
                            **two_way_interval(gap, domain[ids], groups[ids]),
                        )
                    )
    atomic_csv(REPORT / "seeds.csv", rows)
    atomic_csv(REPORT / "paired.csv", pairs)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "score"), required=True)
    args = parser.parse_args()
    for directory in (OUT, REPORT):
        directory.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        {"prepare": prepare, "fit": fit, "score": score}[args.phase]()
