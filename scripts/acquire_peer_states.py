"""Target common probe batches at decision-relevant disagreement between two historical peers."""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from coldstart.conditional_capability import paired_profile
from confirm_release_panel import paired_groups
from evaluate_capability_router import ROOT, input_context, metrics, test_rosters
from fit_peer_states import context, donor_profile, fit_pool, predict, state_probability
from run_experiment import atomic_csv, atomic_json
from select_confirmation_router import SEEDS


def prepare(args):
    d, *_ = input_context(args.run)
    with (
        np.load(args.out / "questions.npz") as f,
        np.load(args.out.parent / "release_confirmation/replication/reserve.npz") as r,
    ):
        tr = f["training"]
        probe = np.concatenate([f[f"balanced_{seed}"][:200] for seed in SEEDS])
        available = tr[
            ~np.isin(d.group_ids[tr], np.union1d(r["groups"], d.group_ids[probe]))
        ]
        mass = np.bincount(d.domains[f["confirmation"]])
    with (
        np.load(ROOT / "data/processed/embedllm.npz") as raw,
        np.load(args.out.parent / "release_confirmation/reserve.npz") as previous,
    ):
        global_groups = raw["group_ids"]
        reviewed = (
            set(global_groups[raw["original_split"] != 0])
            | set(previous["groups"])
            | set(global_groups[previous["bank"]])
        )
        for folder in (
            args.out.parent / "release112",
            args.out.parent / "release112-stratified",
            args.out.parent / "release112_stratified",
        ):
            for file in folder.glob("panel_*.npz"):
                with np.load(file) as data:
                    reviewed.update(global_groups[data["prompt_ids"]])
    available = available[~np.isin(d.group_ids[available], list(reviewed))]
    capacity = np.array(
        [
            len(np.unique(d.group_ids[available[d.domains[available] == j]]))
            for j in range(len(mass))
        ]
    )
    count = np.zeros_like(capacity)
    for n in range(1, 1601):
        j = np.argmax(
            np.where(count < capacity, n * mass / mass.sum() - count, -np.inf)
        )
        count[j] += 1
    rng = np.random.default_rng(52419)
    groups = np.concatenate(
        [
            rng.choice(
                np.unique(d.group_ids[available[d.domains[available] == j]]),
                n,
                replace=False,
            )
            for j, n in enumerate(count)
        ]
    )
    with np.load(ROOT / "data/processed/embedllm.npz") as f:
        rows = (f["original_split"] == 0) & np.isin(f["group_ids"], groups)
        ids = f["prompt_ids"][rows]
        if (
            len(np.unique(f["group_ids"][rows])) != len(groups)
            or set(groups) & reviewed
        ):
            raise ValueError(
                "Every fresh group must come exclusively from previously unexamined training prompts."
            )
    folder = args.out / "active-peer-states"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / "reserve.npz", groups=groups, prompt_ids=ids)
    atomic_json(
        args.report / "active_peer_sample.json",
        dict(
            groups=len(groups),
            upstream_questions=len(ids),
            questions_by_source=count.tolist(),
            seed=52419,
            warm_up=80,
            batch=40,
            maximum_questions=200,
            stopping_gain=0.002,
        ),
    )


def active_context(args, d, s, split):
    with np.load(args.out / "active-peer-states/reserve.npz") as f:
        ctx = context(args, d, s, split, "modern", extra_reserve=f["groups"])
    rng = np.random.default_rng(52420)
    dev = split["development"]
    samples = np.concatenate(
        [
            rng.choice(
                dev[d.domains[dev] == j],
                min(210, np.sum(d.domains[dev] == j)),
                replace=False,
            )
            for j in np.unique(d.domains[dev])
        ]
    )
    ctx["decision_features"], ctx["decision_domains"] = ctx["inputs"](
        samples, "development"
    )
    return ctx


def weighted_profile(ctx, hp, y, dom, peers):
    h, source = ctx["history"], ctx["domains"]
    cell = 4 * dom + (2 * hp[peers[0]] + hp[peers[1]]).astype(int)
    hist_cells = 4 * source + (2 * h[peers[0]] + h[peers[1]]).astype(int)
    pop = np.bincount(hist_cells, minlength=4 * (source.max() + 1))
    weights, survey = paired_profile(hp, y, cell, pop, np.arange(len(pop)) // 4)
    near = softmax(-(np.abs(y[:, None] - hp[None]) @ survey) / 0.029, axis=1)
    weights = 0.55 * weights + 0.45 * near
    residual = y - weights @ hp
    offset = []
    for j in np.unique(source):
        keep = dom == j
        w = survey[keep] / survey[keep].sum()
        n_eff = 1 / (w @ w)
        offset.append(
            (n_eff * (residual[:, keep] @ w) + 9 * (residual @ survey)) / (n_eff + 9)
        )
    return (weights, np.stack(offset, axis=1), peers), cell, hist_cells


def allocation(ctx, y, cell, prob, score, seed):
    dom = ctx["decision_domains"]
    source = ctx["domains"]
    winner = score.argmax(0)
    top = np.argsort(-score, axis=0)[:2]
    if len(top) == 1:
        return np.ones(4 * (source.max() + 1)), 0.0
    a, b = top
    column = np.arange(len(dom))
    close = 0.12 + 0.88 * np.exp(
        -np.maximum(0, score[a, column] - score[b, column]) / 0.065
    )
    mass = 0.45 / (source.max() + 1) + 0.55 * np.bincount(source) / len(source)
    utility = np.empty(len(mass) * 4)
    rng = np.random.default_rng(seed + len(cell) * 501)
    simulation = np.zeros((36, *score.shape), np.float32)
    for j in range(len(mass)):
        queries = np.flatnonzero(dom == j)
        for k in range(4):
            c = 4 * j + k
            answered = y[:, cell == c]
            n = answered.shape[1]
            difference = answered[a[queries]] - answered[b[queries]]
            var = (np.square(difference).sum(1) + 3.2) / (n + 8)
            utility[c] = (
                mass[j] * np.mean(close[queries] * prob[queries, k] ** 2 * var)
                + 0.00012
            )
            centered = answered - answered.mean(1, keepdims=True) if n else answered
            perturb = (
                rng.normal(size=(36, n)) @ centered.T
                + 2 * rng.normal(size=(36, len(y)))
            ) / (n + 24)
            simulation[:, :, queries] += (
                perturb[:, :, None] * prob[queries, k][None, None, :]
            )
    draw = score[None] + simulation
    possible = draw.max(1) - draw[:, winner, column]
    return utility, float(
        sum(mass[j] * possible[:, dom == j].mean() for j in range(len(mass)))
    )


def acquire(ctx, d, names, seed, observe, stop_early=False):
    warm = ctx["panels"][seed][:80]
    rows = list(map(int, warm))
    hist_rows = ctx["train"]
    hp = d.labels[np.ix_(ctx["ref"], warm)]
    initial = observe(warm)
    if not np.isfinite(initial).all():
        return np.asarray(rows), None, [], None
    _, _, peers = donor_profile(ctx, hp, initial, d.domains[warm])
    rng = np.random.default_rng(seed + 7521)
    shuffled = rng.permutation(len(hist_rows))
    _, first = np.unique(d.group_ids[hist_rows[shuffled]], return_index=True)
    available = shuffled[np.sort(first)]
    remaining = None
    trace = []
    prob = None
    for size in (80, 120, 160, 200):
        indices = np.asarray(rows)
        y = observe(indices)
        if not np.isfinite(y).all():
            return indices, None, trace, None
        hp = d.labels[np.ix_(ctx["ref"], indices)]
        donors, cell, history_cells = weighted_profile(
            ctx, hp, y, d.domains[indices], peers
        )
        pool = fit_pool(
            ctx,
            hp,
            y,
            d.domains[indices],
            names,
            d.prompt_ids[indices],
            seed,
            "modern",
            donors,
        )
        if prob is None:
            prob = state_probability(pool["readout"], ctx["decision_features"])
            remaining = [
                list(available[history_cells[available] == c])
                for c in range(4 * (ctx["domains"].max() + 1))
            ]
        scores = predict(pool, ctx["decision_features"], ctx["decision_domains"])
        priority, bound = allocation(ctx, y, cell, prob, scores, seed)
        trace.append(
            dict(
                probes=size,
                estimated_remaining_gain=bound,
                peer_rows=peers,
                prompt_ids=d.prompt_ids[indices].tolist(),
            )
        )
        reason = "estimated-gain" if size >= 120 and bound < 0.002 else None
        if size == 200 or (stop_early and reason):
            return indices, pool, trace, reason or "budget"
        count = np.bincount(cell, minlength=len(priority))
        for _ in range(40):
            benefit = priority / ((count + 18) * (count + 19))
            c = int(np.argmax(np.where([bool(v) for v in remaining], benefit, -np.inf)))
            rows.append(int(hist_rows[remaining[c].pop()]))
            count[c] += 1


def run(args):
    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    ctx = active_context(args, d, s, split)
    q = ctx["fresh"]
    x, source = ctx["inputs"](q, "development")
    old = split["confirmation"]
    xo, so = ctx["inputs"](old, "evaluation")
    X, ds = np.vstack((x, xo)), np.r_[source, so]
    all_q = np.r_[q, old]
    names, rosters = test_rosters(d, s, prepared, family)
    cases = {}
    logs = []
    for roster, unordered, truth in rosters:
        order = np.argsort(unordered)
        ids = unordered[order]
        pool_names = [names[v] for v in ids]
        cases[roster] = {}
        for seed in SEEDS:
            observed = {}

            def read(rows):
                # Only current requested cells are revealed to the active policy.
                missing = [int(i) for i in rows if int(i) not in observed]
                if missing:
                    values = truth[np.ix_(order, missing)]
                    observed.update(zip(missing, values.T))
                return np.column_stack([observed[int(i)] for i in rows])

            begin = perf_counter()
            selected, active, trace, _ = acquire(ctx, d, pool_names, seed, read)
            if len(observed) != 200 or len(np.unique(d.group_ids[selected])) != 200:
                raise ValueError(
                    "Active acquisition exceeded or duplicated its 200-question common panel."
                )
            # The stopping arm receives only the prefix it asks for, independently of the fixed-200 arm.
            allowed = next(
                (
                    row["probes"]
                    for row in trace
                    if row["probes"] >= 120 and row["estimated_remaining_gain"] < 0.002
                ),
                200,
            )
            if allowed == 200:
                stopped = active
            else:
                prefix = set(selected[:allowed])

                def read_prefix(rows):
                    if not set(rows) <= prefix:
                        raise ValueError(
                            "An early-stopped profile requested an unmeasured response."
                        )
                    return np.column_stack([observed[int(i)] for i in rows])

                _, stopped, _, _ = acquire(ctx, d, pool_names, seed, read_prefix, True)
            base = ctx["panels"][seed]
            balanced = fit_pool(
                ctx,
                d.labels[np.ix_(ctx["ref"], base)],
                truth[np.ix_(order, base)],
                d.domains[base],
                pool_names,
                d.prompt_ids[base],
                seed,
                "modern",
            )
            bank = {}
            for name, pool in (
                ("active-peer", active),
                ("active-peer-stop", stopped),
                ("balanced-200", balanced),
            ):
                score = predict(pool, X, ds)
                bank[name] = score.argmax(0)
                if seed == SEEDS[0] and name != "balanced-200":
                    pool["method"] = name
                    joblib.dump(pool, args.out / f"{roster}_{name}.joblib", compress=2)
                    np.savez_compressed(
                        args.out / f"{roster}_{name}_scores.npz",
                        scores=score.astype(np.float32),
                        prompt_ids=d.prompt_ids[all_q],
                    )
                if name == "active-peer":
                    typical = np.einsum(
                        "msk,sk->ms", pool["abilities"], pool["readout"]["mass"]
                    )
                    leader = (
                        typical @ (np.bincount(ctx["domains"]) / len(ctx["domains"]))
                    ).argmax()
                    bank["panel-fixed"] = np.full(len(all_q), leader)
            cases[roster][seed] = bank
            logs.append(
                dict(
                    roster=roster,
                    seed=seed,
                    active_probes=200,
                    stopped_probes=allowed,
                    history=len(ctx["train"]),
                    fit_and_comparison_seconds=perf_counter() - begin,
                    stages=trace,
                )
            )
            print("Active profile", roster, seed, allowed, flush=True)
    atomic_json(args.report / "active_peer_acquisition.json", logs)
    np.savez_compressed(
        ctx["folder"] / "routes.npz",
        prompt_ids=d.prompt_ids[all_q],
        fresh=len(q),
        **{
            f"{roster}_{method}": np.stack([bank[seed][method] for seed in SEEDS])
            for roster, bank in cases.items()
            for method in bank[SEEDS[0]]
        },
    )


def score(args):
    d, s, _, _, prepared, family = input_context(args.run)
    _, rosters = test_rosters(d, s, prepared, family)
    dest = args.out / "active-peer-states"
    rows, pairs = [], []
    with (
        np.load(dest / "routes.npz") as saved,
        np.load(dest / "original.npz") as native,
    ):
        where = {int(v): i for i, v in enumerate(d.prompt_ids)}
        q = np.array([where[int(v)] for v in saved["prompt_ids"]])
        cut = int(saved["fresh"])
        for roster, unordered, truth in rosters:
            order = np.argsort(unordered)
            ids = unordered[order]
            with np.load(args.out / f"choices_75_{roster}.npz") as f:
                refs = {
                    k: f[k][:, : len(q) - cut]
                    for k in (
                        "EmbedLLM-known",
                        "EmbedLLM-noisy",
                        "previous-8D-refit",
                        "conditional",
                    )
                }
            w = native["weight"][ids]
            native_scores = w @ d.embeddings[q[:cut]].T + native["bias"][ids, None]
            for stage, raw in (
                ("fresh", np.arange(cut)),
                ("confirmation", np.arange(cut, len(q))),
            ):
                ix = raw[np.isfinite(truth[np.ix_(order, q[raw])]).all(0)]
                labels = truth[np.ix_(order, q[ix])]
                dom, groups = d.domains[q[ix]], d.group_ids[q[ix]]
                choices = {
                    method: saved[f"{roster}_{method}"][:, ix]
                    for method in (
                        "active-peer",
                        "active-peer-stop",
                        "balanced-200",
                        "panel-fixed",
                    )
                }
                if stage == "fresh":
                    z = native_scores[:, ix]
                    choices["EmbedLLM-known"] = np.tile(z.argmax(0), (len(SEEDS), 1))
                    sd = 0.05 * np.linalg.norm(w, axis=1, keepdims=True)
                    choices["EmbedLLM-noisy"] = np.stack(
                        [
                            (
                                z
                                + sd * np.random.default_rng(seed).normal(size=z.shape)
                            ).argmax(0)
                            for seed in SEEDS
                        ]
                    )
                else:
                    choices.update({k: v[:, ix - cut] for k, v in refs.items()})
                hits = {k: labels[v, np.arange(len(ix))] for k, v in choices.items()}
                for k, v in choices.items():
                    rows.append(
                        dict(
                            split=stage,
                            roster=roster,
                            method=k,
                            questions=len(ix),
                            **metrics(v, labels, dom),
                        )
                    )
                for method in ("active-peer", "active-peer-stop"):
                    for k, h in hits.items():
                        if k != method:
                            pairs.append(
                                dict(
                                    split=stage,
                                    roster=roster,
                                    method=method,
                                    reference=k,
                                    **paired_groups(hits[method], h, dom, groups),
                                )
                            )
    atomic_csv(args.report / "active_peer_results.csv", rows)
    atomic_csv(args.report / "active_peer_paired.csv", pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "fit", "score"), required=True)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation"
    )
    p.add_argument(
        "--report", type=Path, default=ROOT / "reports/routing/capability/confirmation"
    )
    args = p.parse_args()
    with threadpool_limits(limits=3):
        {"prepare": prepare, "fit": run, "score": score}[args.phase](args)
