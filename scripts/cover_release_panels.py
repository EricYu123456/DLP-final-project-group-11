"""Choose 200 graded prompts by partitioning the unlabeled question bank across topics and formats."""

import argparse
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from coldstart.conditional_text import lexical
from coldstart.model_covariates import model_descriptions
from evaluate_capability_router import benchmark_means, input_context
from panel_capabilities import panel_text
from panel_rasch import subtype_matrix
from panel_text_kernel import fit_text_kernel, text_kernel_scores
from replicate_release_panels import BASE, DATA, RELEASE, ROOT, grades, two_way_interval
from run_embedllm_release import panel
from run_experiment import atomic_csv, atomic_json

OUT = RELEASE / "coverage"
REPORT = ROOT / "reports/routing/capability/release_confirmation/coverage"
SEEDS = tuple(range(8800, 8815))


def disjoint_bank(group, split):
    d, *_ = input_context(ROOT / "artifacts/main_v2")
    with np.load(BASE / "confirmation/questions.npz") as f:
        held = set(
            d.group_ids[np.r_[f["confirmation"], f["development"], f["previous_test"]]]
        )
        panels = set(
            d.group_ids[
                np.concatenate([f[f"balanced_{s}"][:200] for s in (11, 23, 37)])
            ]
        )
    held.update(group[split != 0])
    for filename in (
        RELEASE / "reserve.npz",
        RELEASE / "replication/reserve.npz",
        BASE / "confirmation/active-peer-states/reserve.npz",
    ):
        with np.load(filename) as f:
            held.update(f["groups"])
            if "bank" in f:
                panels.update(group[f["bank"]])
    for folder in (
        BASE / "release112",
        BASE / "release112-stratified",
        BASE / "release112_stratified",
        RELEASE,
    ):
        for filename in folder.glob("*panel_*.npz"):
            with np.load(filename) as f:
                if "y" in f and "prompt_ids" in f:
                    panels.update(group[f["prompt_ids"]])
    with np.load(RELEASE / "replication/panels.npz") as f:
        panels.update(group[f["panels"]].ravel())
    active = json.loads(
        (
            ROOT
            / "reports/routing/capability/confirmation/active_peer_acquisition.json"
        ).read_text()
    )
    for run in active:
        panels.update(group[run["stages"][-1]["prompt_ids"]])
    return held, panels


def representative_panel(bank, features, dom, quotas, rng):
    def sample(rows, count):
        local = features[rows]
        centered = local - local.mean(0)
        if count == 1:
            group = rng.choice(len(rows), size=min(4, len(rows)), replace=False)
            return [rows[group[np.square(centered[group]).sum(1).argmin()]]]
        # Split the highest-variance direction in proportion to the number of questions each branch will supply.
        _, directions = np.linalg.eigh(centered.T @ centered)
        rank = np.argsort(centered @ directions[:, -1], kind="stable")
        left = count // 2
        cut = int(
            np.clip(round(len(rows) * left / count), left, len(rows) - (count - left))
        )
        return sample(rows[rank[:cut]], left) + sample(rows[rank[cut:]], count - left)

    return np.concatenate(
        [sample(bank[dom[bank] == j], int(n)) for j, n in enumerate(quotas) if n]
    )


def prepare():
    with np.load(DATA / "processed/embedllm.npz") as f:
        group, dom, sp, ids, vectors = [
            f[k]
            for k in (
                "group_ids",
                "benchmark_ids",
                "original_split",
                "prompt_ids",
                "embeddings",
            )
        ]
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("The upstream release indexes its embeddings by prompt ID.")
    held, panels = disjoint_bank(group, sp)
    eligible = (sp == 0) & ~np.isin(group, list(held | panels))
    rng = np.random.default_rng(78419)
    mass = np.bincount(dom[sp == 1], minlength=10)
    capacity = np.array(
        [len(np.unique(group[eligible & (dom == j)])) for j in range(10)]
    )
    # Leave most remaining groups available for independent studies on rare sources.
    limit = np.floor(0.4 * capacity).astype(int)
    counts = np.zeros(10, int)
    for n in range(1, min(1500, limit.sum()) + 1):
        source = np.argmax(
            np.where(counts < limit, n * mass / mass.sum() - counts, -np.inf)
        )
        counts[source] += 1
    fresh_groups = np.unique(
        np.concatenate(
            [
                rng.choice(np.unique(group[eligible & (dom == j)]), n, replace=False)
                for j, n in enumerate(counts)
                if n
            ]
        )
    )
    fresh = np.flatnonzero((sp == 0) & np.isin(group, fresh_groups))
    available = np.flatnonzero(
        (sp == 0) & ~np.isin(group, list(held | set(fresh_groups)))
    )
    if set(group[fresh]) & (held | panels) or set(group[available]) & set(fresh_groups):
        raise ValueError(
            "A new group also occurs in a prior evaluation or onboarding panel."
        )
    np.savez_compressed(OUT / "reserve.npz", groups=fresh_groups, prompt_ids=fresh)
    text = panel_text(available, DATA / "raw")
    sample = np.random.default_rng(78420).choice(
        available, min(5000, len(available)), replace=False
    )
    center = vectors[sample].mean(0)
    _, _, axes = randomized_svd(vectors[sample] - center, 32, random_state=78420)
    semantic = (vectors[available] - center) @ axes.T
    fmt = np.column_stack((lexical(text, []), subtype_matrix(text))).astype(np.float32)
    fmt = np.clip((fmt - fmt.mean(0)) / fmt.std(0).clip(0.25), -4, 4)
    features = np.zeros((len(dom), semantic.shape[1] + fmt.shape[1]), np.float32)
    features[available] = np.column_stack(
        (semantic / semantic.std(0).mean().clip(0.001), 0.55 * fmt)
    )
    designs = {"control": [], "coverage": []}
    for seed in SEEDS:
        gen = np.random.default_rng(seed + 501)
        shuffled = gen.permutation(available)
        _, first = np.unique(group[shuffled], return_index=True)
        bank = shuffled[first]
        control = panel(vectors, dom, bank, seed)
        adjusted = representative_panel(
            bank, features, dom, np.bincount(dom[control], minlength=10), gen
        )
        if len(np.unique(group[adjusted])) != 200:
            raise ValueError(
                "Each deployment must use exactly 200 distinct question groups."
            )
        designs["control"].append(control)
        designs["coverage"].append(adjusted)
    np.savez_compressed(OUT / "panels.npz", **designs)
    atomic_json(
        REPORT / "sample.json",
        dict(
            groups=len(fresh_groups),
            queries=len(fresh),
            counts=np.bincount(dom[fresh], minlength=10).tolist(),
            eligible_groups=len(np.unique(group[available])),
            seeds=SEEDS,
            source_quotas=np.bincount(dom[control], minlength=10).tolist(),
        ),
    )


def fit():
    with (
        np.load(OUT / "panels.npz") as f,
        np.load(OUT / "reserve.npz") as held,
        np.load(RELEASE / "reserve.npz") as orig,
    ):
        designs = dict(f)
        fresh, official = held["prompt_ids"], orig["test"]
    q = np.r_[fresh, official]
    bank = np.unique(np.concatenate(list(designs.values())))
    y = grades("train.csv", bank)
    texts = panel_text(np.r_[bank, q], DATA / "raw")
    names = (
        pd.read_csv(DATA / "raw/model_order.csv")
        .sort_values("model_id")
        .model_name.to_numpy(str)
    )
    meta, size = model_descriptions(names, DATA / "metadata/model_families.csv")
    with np.load(DATA / "processed/embedllm.npz") as f:
        dom = f["benchmark_ids"]
    results, costs = {key: [] for key in designs}, []
    for j, seed in enumerate(SEEDS):
        for name, panels in designs.items():
            ids = panels[j]
            loc = np.searchsorted(bank, ids)
            start = perf_counter()
            pack, _ = fit_text_kernel(dom[ids], y[:, loc], texts[loc], meta, size, seed)
            costs.append(dict(method=name, seed=seed, seconds=perf_counter() - start))
            scores = text_kernel_scores(pack, texts[len(bank) :])
            results[name].append(scores.argmax(0).astype(np.int16))
            if name == "coverage" and j == 0:
                pack.update(names=names, prompt_ids=ids)
                joblib.dump(pack, OUT / "text-kernel_8800.joblib", compress=2)
                atomic_json(
                    REPORT / "text-kernel_selection.json",
                    {str(seed): dict(questions=len(ids), proxies=[])},
                )
                np.savez_compressed(
                    OUT / "embeddings_8800.npz",
                    embeddings=np.column_stack((pack["z"], pack["axes"])),
                    model_names=names,
                    probe_ids=ids,
                )
        print("Paired 200-question deployments", seed, flush=True)
    np.savez_compressed(
        OUT / "routes.npz", prompts=q, fresh_count=len(fresh), **results
    )
    atomic_csv(REPORT / "costs.csv", costs)


def score():
    rows, pairs = [], []
    with (
        np.load(OUT / "routes.npz") as f,
        np.load(DATA / "processed/embedllm.npz") as corpus,
    ):
        q, cut, dom, g = (
            f["prompts"],
            int(f["fresh_count"]),
            corpus["benchmark_ids"],
            corpus["group_ids"],
        )
        for part, subset in (
            ("fresh", np.arange(cut)),
            ("official", np.arange(cut, len(q))),
        ):
            ids = q[subset]
            truth = grades("train.csv" if part == "fresh" else "test.csv", ids)
            hits = {}
            with np.load(
                (OUT if part == "fresh" else BASE / "release112") / "original.npz"
            ) as ref:
                if not np.array_equal(ref["prompt_ids"], ids):
                    raise ValueError(
                        "The original EmbedLLM predictions must match the same test IDs."
                    )
                for name in ("noisy", "deterministic"):
                    picked = np.atleast_2d(ref[name])
                    hits[name] = truth[picked, np.arange(len(ids))].mean(0)
            for method in ("control", "coverage"):
                picked = f[method][:, subset]
                h = truth[picked, np.arange(len(ids))]
                hits[method] = h
                for seed, v, r in zip(SEEDS, picked, h):
                    rows.append(
                        dict(
                            split=part,
                            method=method,
                            seed=seed,
                            questions=len(ids),
                            micro=r.mean(),
                            macro=benchmark_means(r, dom[ids]).mean(),
                            top_share=np.bincount(v).max() / len(ids),
                        )
                    )
            for name in ("control", "noisy", "deterministic"):
                gap = hits["coverage"] - hits[name]
                pairs.append(
                    dict(
                        split=part,
                        reference=name,
                        questions=len(ids),
                        accuracy=hits["coverage"].mean(),
                        reference_accuracy=hits[name].mean(),
                        gain=gap.mean(),
                        wins=int((gap.mean(1) > 0).sum()),
                        **two_way_interval(gap, dom[ids], g[ids]),
                    )
                )
    atomic_csv(REPORT / "seeds.csv", rows)
    atomic_csv(REPORT / "paired.csv", pairs)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "fit", "score"), required=True)
    args = p.parse_args()
    for directory in (OUT, REPORT):
        directory.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        {"prepare": prepare, "fit": fit, "score": score}[args.phase]()
