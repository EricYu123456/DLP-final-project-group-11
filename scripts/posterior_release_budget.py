"""Average ability estimates within one measured panel and stop when the next questions have little expected value."""

import argparse
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coldstart.model_covariates import model_descriptions
from cover_release_panels import BASE, DATA, OUT, REPORT, SEEDS, disjoint_bank
from evaluate_capability_router import benchmark_means
from evaluate_panel_budget import nested_order
from panel_capabilities import panel_text, template_membership
from panel_tabular_skills import ability_factors
from panel_text_kernel import fit_text_kernel, kernel, text_kernel_scores
from replicate_release_panels import grades, two_way_interval
from run_experiment import atomic_csv, atomic_json

LIMITS = {"posterior-stop": 0.005, "posterior-economical": 0.01}
SIZES = (100, 140, 170, 200)


def fit_posterior(dom, y, texts, meta, size, seed, ungraded):
    pack, tune = fit_text_kernel(dom, y, texts, meta, size, seed, return_mix=True)
    support = pack["text_kernel"]
    gram = kernel(support, texts, np.eye(int(dom.max()) + 1)[dom])
    membership = template_membership(texts, ungraded, dom)
    check = kernel(support, ungraded, membership)
    rng = np.random.default_rng(seed + 1893 + len(dom))
    source, residual, risks = [], [], []
    for _ in range(12):
        w = rng.exponential(1, len(dom))
        for j in np.unique(dom):
            inside = dom == j
            w[inside] /= w[inside].mean()
        z, axes, targets = ability_factors(dom, y, meta, size, w)
        weight = sum(
            c * np.linalg.solve(gram + np.diag(p / w), targets)
            for c, p in zip(tune["coefficients"], tune["regularizers"])
        )
        coefficients = axes @ weight.T
        source.append(z)
        residual.append(coefficients)
        risks.append(z @ membership.T + coefficients @ check.T)
    pack["z"] = 0.5 * (pack["z"] + np.mean(source, axis=0))
    average = 0.5 * (pack["axes"] @ pack["weights"].T + np.mean(residual, axis=0))
    left, sv, right = np.linalg.svd(average, full_matrices=False)
    rank = min(12, len(sv))
    pack["axes"] = left[:, :rank]
    pack["weights"] = (sv[:rank, None] * right[:rank]).T
    expected = pack["z"] @ membership.T + (pack["axes"] @ pack["weights"].T) @ check.T
    draws = np.stack(risks)
    draws = expected + np.sqrt(1 - len(dom) / 200) * (draws - draws.mean(0))
    use = expected.argmax(0)
    regret = (draws.max(1) - draws[:, use, np.arange(len(use))]).mean()
    return pack, float(regret)


def fit():
    with (
        np.load(OUT / "panels.npz") as f,
        np.load(OUT / "routes.npz") as ref,
        np.load(DATA / "processed/embedllm.npz") as raw,
    ):
        panels, q = f["coverage"], ref["prompts"]
        dom, g, sp = raw["benchmark_ids"], raw["group_ids"], raw["original_split"]
    held, _ = disjoint_bank(g, sp)
    forbidden = held | set(g[q]) | set(g[panels].ravel())
    available = np.flatnonzero((sp == 0) & ~np.isin(g, list(forbidden)))
    rng = np.random.default_rng(61527)
    _, first = np.unique(g[available], return_index=True)
    ungraded = rng.choice(available[first], 300, replace=False)
    bank = np.unique(panels)
    y = grades("train.csv", bank)
    text = panel_text(np.r_[bank, ungraded, q], DATA / "raw")
    sample, query = (
        text[len(bank) : len(bank) + len(ungraded)],
        text[len(bank) + len(ungraded) :],
    )
    names = (
        pd.read_csv(DATA / "raw/model_order.csv")
        .sort_values("model_id")
        .model_name.to_numpy(str)
    )
    meta, size = model_descriptions(names, DATA / "metadata/model_families.csv")
    routes = {key: [] for key in (*LIMITS, "posterior-200")}
    trace = []
    for seed, panel in zip(SEEDS, panels):
        order = nested_order(dom[panel], seed)
        pending = set(LIMITS)
        cumulative = 0
        for n in SIZES:
            # Each stopping decision gets only the responses in this prefix. Later prefixes belong to longer arms.
            ids = panel[np.sort(order[:n])]
            loc = np.searchsorted(bank, ids)
            start = perf_counter()
            pack, regret = fit_posterior(
                dom[ids], y[:, loc], text[loc], meta, size, seed, sample
            )
            seconds = perf_counter() - start
            cumulative += seconds
            release = [
                key for key in sorted(pending) if regret < LIMITS[key] or n == 200
            ]
            pending.difference_update(release)
            if n == 200:
                release.append("posterior-200")
            trace.append(
                dict(
                    seed=seed,
                    questions=n,
                    estimated_gain=regret,
                    checkpoint_seconds=seconds,
                    sequential_seconds=cumulative,
                    finished=" ".join(release),
                )
            )
            if release:
                v = text_kernel_scores(pack, query).argmax(0).astype(np.int16)
                for key in release:
                    routes[key].append(v)
                    if seed == SEEDS[0]:
                        pack.update(names=names, prompt_ids=ids)
                        joblib.dump(pack, OUT / f"{key}_{seed}.joblib", compress=2)
                        np.savez_compressed(
                            OUT / f"{key}_embeddings.npz",
                            embeddings=np.column_stack((pack["z"], pack["axes"])),
                            model_names=names,
                            probe_ids=ids,
                        )
                        atomic_json(
                            REPORT / f"{key}_selection.json",
                            {
                                str(seed): dict(
                                    questions=n, proxies=[], estimated_gain=regret
                                )
                            },
                        )
        print(
            "Posterior stopping",
            seed,
            [
                (v["questions"], round(v["estimated_gain"], 4), v["finished"])
                for v in trace[-4:]
            ],
            flush=True,
        )
    np.savez_compressed(
        OUT / "posterior_routes.npz", prompts=q, ungraded=ungraded, **routes
    )
    atomic_csv(REPORT / "posterior_stopping.csv", trace)


def score():
    rows, pairs = [], []
    trace = pd.read_csv(REPORT / "posterior_stopping.csv").fillna({"finished": ""})
    with (
        np.load(OUT / "posterior_routes.npz") as data,
        np.load(OUT / "routes.npz") as base,
        np.load(DATA / "processed/embedllm.npz") as raw,
    ):
        q, cut, dom, g = (
            data["prompts"],
            int(base["fresh_count"]),
            raw["benchmark_ids"],
            raw["group_ids"],
        )
        if not np.array_equal(q, base["prompts"]):
            raise ValueError(
                "The posterior and unchanged routers must cover the same questions."
            )
        for part, idx in (
            ("fresh", np.arange(cut)),
            ("official", np.arange(cut, len(q))),
        ):
            ids = q[idx]
            truth = grades("train.csv" if part == "fresh" else "test.csv", ids)
            controls = {
                "same-200": truth[base["coverage"][:, idx], np.arange(len(ids))],
                "posterior-200": truth[
                    data["posterior-200"][:, idx], np.arange(len(ids))
                ],
            }
            with np.load(
                (OUT if part == "fresh" else BASE / "release112") / "original.npz"
            ) as f:
                if not np.array_equal(f["prompt_ids"], ids):
                    raise ValueError(
                        "The original comparisons must use the same prompt IDs."
                    )
                for name in ("noisy", "deterministic"):
                    controls[name] = truth[
                        np.atleast_2d(f[name]), np.arange(len(ids))
                    ].mean(0)
            for key in (*LIMITS, "posterior-200"):
                routes = data[key][:, idx]
                hits = truth[routes, np.arange(len(ids))]
                for seed, v, h in zip(SEEDS, routes, hits):
                    step = trace[
                        (trace.seed == seed)
                        & trace.finished.str.split().map(lambda names: key in names)
                    ].iloc[0]
                    rows.append(
                        dict(
                            split=part,
                            method=key,
                            seed=seed,
                            queries=len(ids),
                            probes=int(step.questions),
                            micro=h.mean(),
                            macro=benchmark_means(h, dom[ids]).mean(),
                            top_share=np.bincount(v).max() / len(ids),
                            seconds=step.checkpoint_seconds
                            if key == "posterior-200"
                            else step.sequential_seconds,
                        )
                    )
                for name, h in controls.items():
                    if name == key:
                        continue
                    gap = hits - h
                    pairs.append(
                        dict(
                            split=part,
                            method=key,
                            reference=name,
                            accuracy=hits.mean(),
                            reference_accuracy=h.mean(),
                            gain=gap.mean(),
                            wins=(gap.mean(1) > 0).sum(),
                            **two_way_interval(gap, dom[ids], g[ids]),
                        )
                    )
    atomic_csv(REPORT / "posterior_seeds.csv", rows)
    atomic_csv(REPORT / "posterior_paired.csv", pairs)
    print(
        pd.DataFrame(rows)
        .groupby(["split", "method"])[["probes", "micro", "top_share", "seconds"]]
        .mean(),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("fit", "score"), required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        {"fit": fit, "score": score}[args.phase]()
