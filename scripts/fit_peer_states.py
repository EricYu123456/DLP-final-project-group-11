"""Embed candidates by their complementary strengths across two historical peers' joint outcomes."""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from coldstart.conditional_text import lexical
from coldstart.reference_outcomes import features as answer_features
from coldstart.switch_calibration import peer_vectors
from evaluate_capability_router import (
    ROOT,
    input_context,
    metrics,
    test_rosters,
)
from panel_capabilities import panel_text, template_membership
from run_experiment import atomic_csv
from select_confirmation_router import SEEDS
from train_conditional_router import proxy_data

HELPERS = ("smol", "qwen", "math", "phi", "yi", "zephyr", "moe-gated", "bagel-gated")


def question_features(basis, x, texts, evidence):
    membership = template_membership(basis["examples"], texts, basis["domains"])
    source = membership.argmax(1)
    phi = np.column_stack(
        ((x - basis["center"]) @ basis["axes"].T, lexical(texts, []), membership)
    )
    if evidence:
        phi = np.column_stack(
            (
                answer_features(evidence, phi, source, membership.shape[1]),
                *(part[:, 0, :4] for part, _ in evidence),
            )
        )
    return phi.astype(np.float32), source


def context(args, d, s, split, mode, extra_reserve=None):
    tr = split["training"]
    with np.load(args.out.parent / "release_confirmation/replication/reserve.npz") as f:
        reserve = f["groups"]
    if mode == "mc" and extra_reserve is None:
        with np.load(args.out / "active-peer-states/reserve.npz") as f:
            extra_reserve = f["groups"]
    option_view = mode in ("option", "option-control")
    if option_view:
        with (
            np.load(args.out / "active-peer-states/reserve.npz") as active,
            np.load(
                args.out.parent / "release_confirmation/coverage/reserve.npz"
            ) as coverage,
        ):
            reserve = np.union1d(reserve, active["groups"])
            extra_reserve = coverage["groups"]
    fresh_groups = reserve if extra_reserve is None else np.asarray(extra_reserve)
    if extra_reserve is not None:
        reserve = np.union1d(reserve, fresh_groups)
    panels = {seed: split[f"balanced_{seed}"][:200] for seed in SEEDS}
    probe_groups = np.unique(d.group_ids[np.concatenate(list(panels.values()))])
    eligible = ~np.isin(d.group_ids[tr], np.union1d(reserve, probe_groups))
    train = tr[eligible]
    fresh = tr[
        np.isin(d.group_ids[tr], fresh_groups) & ~np.isin(d.group_ids[tr], probe_groups)
    ]
    ref = np.r_[s.train_models, s.val_models]
    folder = args.out / (
        "peer-states" if extra_reserve is None else "active-peer-states"
    )
    if option_view:
        folder = args.out / "option-peer-states"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(9301)
    basis_file = folder / "question_basis.joblib"
    if basis_file.is_file():
        basis = joblib.load(basis_file)
    else:
        sampled = np.concatenate(
            [
                rng.choice(
                    train[d.domains[train] == j],
                    min(80, np.sum(d.domains[train] == j)),
                    replace=False,
                )
                for j in np.unique(d.domains[train])
            ]
        )
        x = d.embeddings[train]
        center = x.mean(0)
        _, _, axes = randomized_svd(x - center, 36, random_state=9301)
        basis = dict(
            center=center,
            axes=axes,
            examples=panel_text(d.prompt_ids[sampled], ROOT / "data/raw"),
            domains=d.domains[sampled],
        )
        joblib.dump(basis, basis_file, compress=2)
    probes = [] if mode == "text" else list(HELPERS)
    if mode == "live":
        probes.append("una-gated")
    elif mode == "mc":
        probes.append("una-mc-gated")
    elif mode == "option":
        probes.append("option-math-gated")

    def inputs(rows, phase):
        ids = d.prompt_ids[rows]
        texts = panel_text(ids, ROOT / "data/raw")
        ev = proxy_data(args.out, ids, probes, phase) if probes else []
        return question_features(basis, d.embeddings[rows], texts, ev)

    # Features contain only prompt-side evidence. Each supervised shared fit indexes eligible training rows.
    x, _ = inputs(train, "development")
    return dict(
        basis=basis,
        features=x,
        domains=d.domains[train],
        history=d.labels[np.ix_(ref, train)],
        train=train,
        fresh=fresh,
        ref=ref,
        panels=panels,
        helpers=probes,
        inputs=inputs,
        folder=folder,
    )


def donor_profile(ctx, hp, y, dp):
    h, source = ctx["history"], ctx["domains"]
    all_dom = np.r_[source, dp]
    all_h = np.column_stack((h, hp))
    observed = np.arange(h.shape[1], all_h.shape[1])
    target = np.full((len(y), len(all_dom)), np.nan)
    target[:, observed] = y
    weights, offset = peer_vectors(all_h, target, observed, all_dom, 0.45)
    typical = np.stack([h[:, source == j].mean(1) for j in np.unique(source)], axis=1)
    pop = np.bincount(source) / len(source)
    leader = int(
        ((weights @ typical + offset) @ (0.65 * pop + 0.35 / len(pop))).argmax()
    )
    # The first peer matches the leader. The second matches an alternative on the leader's missed probes.
    mass = (0.65 * pop + 0.35 / len(pop))[dp] / np.bincount(dp)[dp]
    lead = int((np.abs(hp - y[leader]) @ mass).argmin())
    fail = 1.0 - hp[lead]
    prior_rescue = weights @ (h * (1 - h[lead])[None]).mean(1)
    rescue = 0.55 * ((y * fail) @ mass) + 0.45 * prior_rescue
    if len(y) > 1:
        rescue[leader] = -np.inf
    challenger = int(rescue.argmax())
    importance = mass * (0.5 + fail)
    errors = (np.abs(hp - y[challenger]) @ importance) / importance.sum()
    distinct = np.sum(hp != hp[lead], axis=1) >= max(5, len(dp) // 20)
    if not distinct.any():
        distinct = np.arange(len(h)) != lead
    other = int(np.where(distinct, errors, np.inf).argmin())
    return weights, offset, (lead, other)


def train_readout(ctx, peers):
    x, dom, h = ctx["features"], ctx["domains"], ctx["history"]
    state = (2 * h[peers[0]] + h[peers[1]]).astype(int)
    rng = np.random.default_rng(913)
    calibration = np.zeros(len(x), bool)
    for value in np.unique(4 * dom + state):
        ids = rng.permutation(np.flatnonzero(4 * dom + state == value))
        calibration[ids[::6]] = True
    model = CatBoostClassifier(
        iterations=390,
        depth=5,
        learning_rate=0.055,
        l2_leaf_reg=8,
        loss_function="MultiClass",
        thread_count=3,
        random_seed=913,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x[~calibration], state[~calibration])
    cal = model.predict(x[calibration], prediction_type="RawFormulaVal", thread_count=3)
    classes = np.asarray(model.classes_, int)
    label = np.searchsorted(classes, state[calibration])

    def loss(temperature):
        return -np.log(
            softmax(cal / temperature, axis=1)[np.arange(len(cal)), label].clip(1e-8)
        ).mean()

    temperature = minimize_scalar(loss, bounds=(0.7, 1.7), method="bounded").x
    domains = np.unique(dom)
    average = np.stack([h[:, dom == j].mean(1) for j in domains])
    means = np.zeros((len(domains), 4, len(h)))
    mass = np.zeros((len(domains), 4))
    trees = []
    for k in range(4):
        inside = state == k
        for j in domains:
            use = inside & (dom == j)
            means[j, k] = (h[:, use].sum(1) + 6 * average[j]) / (use.sum() + 6)
            mass[j, k] = (use.sum() + 0.5) / (np.sum(dom == j) + 2)
        if inside.sum() < 50:
            trees.append(None)
            continue
        residual = h[:, inside].T - means[dom[inside], k]
        residual -= residual.mean(1, keepdims=True)
        _, _, axes = randomized_svd(residual, min(12, len(h) - 1), random_state=927 + k)
        target = residual @ axes.T
        scale = target.std(0).clip(0.1)
        tree = ExtraTreesRegressor(
            n_estimators=72,
            max_features=0.62,
            min_samples_leaf=9,
            max_depth=14,
            n_jobs=3,
            random_state=927 + k,
        )
        trees.append((tree.fit(x[inside], target / scale), scale[:, None] * axes))
    return dict(
        model=model,
        temperature=temperature,
        means=means,
        mass=mass,
        average=average,
        trees=trees,
    )


def fit_pool(ctx, hp, y, dp, names, ids, seed, mode, donors=None):
    weights, offset, peers = donor_profile(ctx, hp, y, dp) if donors is None else donors
    file = ctx["folder"] / f"{mode}_{peers[0]}_{peers[1]}.joblib"
    if not file.is_file():
        joblib.dump(train_readout(ctx, peers), file, compress=2)
    pack = joblib.load(file)
    state = (2 * hp[peers[0]] + hp[peers[1]]).astype(int)
    prior = np.einsum("mh,skh->msk", weights, pack["means"]) + offset[:, :, None]
    ability = prior.copy()
    for j in range(ability.shape[1]):
        for k in range(4):
            use = (dp == j) & (state == k)
            ability[:, j, k] = (y[:, use].sum(1) + 24 * prior[:, j, k]) / (
                use.sum() + 24
            )
    target = weights @ pack["average"].T + offset
    ability += (target - np.einsum("msk,sk->ms", ability, pack["mass"]))[:, :, None]
    return dict(
        algorithm="peer-states",
        method="peer-states" + ("" if mode == "modern" else "-" + mode),
        model_names=np.asarray(names),
        seed=seed,
        probe_ids=ids,
        peers=peers,
        basis=ctx["basis"],
        helpers=ctx["helpers"],
        readout=pack,
        abilities=ability,
        weights=weights,
    )


def state_probability(pack, x):
    logits = pack["model"].predict(x, prediction_type="RawFormulaVal", thread_count=3)
    prob = np.zeros((len(x), 4))
    prob[:, np.asarray(pack["model"].classes_, int)] = softmax(
        logits / pack["temperature"], axis=1
    )
    return prob


def predict(pool, x, dom, ablate=False):
    pack = pool["readout"]
    prob = state_probability(pack, x)
    mean = np.einsum("mqk,qk->mq", pool["abilities"][:, dom], prob)
    full = mean.copy()
    for k, bank in enumerate(pack["trees"]):
        if bank is not None:
            model, axes = bank
            full += 0.85 * (pool["weights"] @ axes.T @ model.predict(x).T) * prob[:, k]
    return (full, mean) if ablate else full


def run(args):
    from confirm_release_panel import paired_groups

    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    ctx = context(args, d, s, split, args.mode)
    fresh, old = ctx["fresh"], np.r_[split["confirmation"], split["previous_test"]]
    x, ds = ctx["inputs"](fresh, "development")
    xo, do = ctx["inputs"](old, "evaluation")
    queries = np.r_[fresh, old]
    x, ds = np.vstack((x, xo)), np.r_[ds, do]
    names, rosters = test_rosters(d, s, prepared, family)
    native_file = args.out.parent / "release_confirmation/replication/original.npz"
    if args.mode == "mc":
        native_file = ctx["folder"] / "original.npz"
    if args.mode in ("option", "option-control"):
        native_file = args.out.parent / "release_confirmation/coverage/original.npz"
    with np.load(native_file) as f:
        native_w, native_b = f["weight"], f["bias"]
    records, pairs, costs, seeds = [], [], [], []
    for roster, unordered, truth in rosters:
        order = np.argsort(unordered)
        candidates = unordered[order]
        roster_names = [names[int(i)] for i in candidates]
        if args.mode == "option":
            with np.load(ctx["folder"] / f"{roster}_control.npz") as f:
                eight_helper = f["routes"]
                if not np.array_equal(f["prompt_ids"], d.prompt_ids[queries]):
                    raise ValueError(
                        "The option likelihood and control must use the same questions."
                    )
        if args.mode == "mc":
            with np.load(ctx["folder"] / "routes.npz") as f:
                eight_helper = f[f"{roster}_balanced-200"]
                if not np.array_equal(
                    f["prompt_ids"], d.prompt_ids[queries[: eight_helper.shape[1]]]
                ):
                    raise ValueError(
                        "The two peer routers must score the same questions in the same order."
                    )
        routes, ablated, fixed = [], [], []
        for seed, probe in ctx["panels"].items():
            mask = np.isfinite(truth[np.ix_(order, probe)])
            if not mask.all():
                raise ValueError(
                    "Joint-state onboarding needs one observed grade per candidate and panel prompt."
                )
            hp = d.labels[np.ix_(ctx["ref"], probe)]
            y = truth[np.ix_(order, probe)]
            start = perf_counter()
            pool = fit_pool(
                ctx,
                hp,
                y,
                d.domains[probe],
                roster_names,
                d.prompt_ids[probe],
                seed,
                args.mode,
            )
            fit_seconds = perf_counter() - start
            full, state = predict(pool, x, ds, ablate=True)
            routes.append(full.argmax(0))
            ablated.append(state.argmax(0))
            prior = np.einsum("msk,sk->ms", pool["abilities"], pool["readout"]["mass"])
            fixed.append(
                np.full(
                    len(queries),
                    (
                        prior @ (np.bincount(ctx["domains"]) / len(ctx["domains"]))
                    ).argmax(),
                )
            )
            costs.append(
                dict(
                    roster=roster,
                    seed=seed,
                    probes=len(probe),
                    historical_questions=len(ctx["train"]),
                    reference_models=len(ctx["ref"]),
                    fresh_questions=len(fresh),
                    fit_or_load_seconds=fit_seconds,
                    first_peer=names[int(d.model_ids[ctx["ref"][pool["peers"][0]]])],
                    second_peer=names[int(d.model_ids[ctx["ref"][pool["peers"][1]]])],
                )
            )
            if seed == SEEDS[0]:
                name = pool["method"]
                joblib.dump(pool, args.out / f"{roster}_{name}.joblib", compress=2)
                np.savez_compressed(
                    args.out / f"{roster}_{name}_embeddings.npz",
                    embeddings=np.column_stack(
                        (pool["abilities"].reshape(len(y), -1), pool["weights"])
                    ),
                    model_names=roster_names,
                    model_ids=candidates,
                    domain_names=d.benchmark_names,
                    reference_model_ids=d.model_ids[ctx["ref"]],
                    probe_ids=pool["probe_ids"],
                    state_labels=[
                        "both wrong",
                        "only second correct",
                        "only first correct",
                        "both correct",
                    ],
                )
                np.savez_compressed(
                    args.out / f"{roster}_{name}_scores.npz",
                    scores=full.astype(np.float32),
                    prompt_ids=d.prompt_ids[queries],
                )
            print("Fitted", roster, args.mode, seed, costs[-1], flush=True)
        route, state = np.stack(routes), np.stack(ablated)
        if args.mode == "option-control":
            np.savez_compressed(
                ctx["folder"] / f"{roster}_control.npz",
                routes=route,
                prompt_ids=d.prompt_ids[queries],
            )
        known = (
            native_w[candidates] @ d.embeddings[fresh].T + native_b[candidates, None]
        )
        native_sd = 0.05 * np.linalg.norm(native_w[candidates], axis=1)[:, None]
        det = np.broadcast_to(known.argmax(0), (len(SEEDS), len(fresh)))
        noisy = np.stack(
            [
                (
                    known
                    + native_sd * np.random.default_rng(seed).normal(size=known.shape)
                ).argmax(0)
                for seed in SEEDS
            ]
        )
        with np.load(args.out / f"choices_75_{roster}.npz") as f:
            refs = {
                key: f[key]
                for key in (
                    "EmbedLLM-known",
                    "EmbedLLM-noisy",
                    "conditional",
                    "previous-8D-refit",
                )
            }
        for stage, ix in (
            ("fresh", np.arange(len(fresh))),
            (
                "confirmation",
                np.arange(len(fresh), len(fresh) + len(split["confirmation"])),
            ),
            (
                "previous-test",
                np.arange(len(fresh) + len(split["confirmation"]), len(queries)),
            ),
        ):
            valid = np.isfinite(truth[np.ix_(order, queries[ix])]).all(0)
            ix = ix[valid]
            labels, domains, groups = (
                truth[np.ix_(order, queries[ix])],
                d.domains[queries[ix]],
                d.group_ids[queries[ix]],
            )
            choices = {
                "peer-states": route[:, ix],
                "state-means": state[:, ix],
                "panel-fixed": np.stack(fixed)[:, ix],
            }
            if args.mode == "option" or (
                args.mode == "mc" and stage != "previous-test"
            ):
                choices["balanced-eight-helpers"] = eight_helper[:, ix]
            if stage == "fresh":
                choices.update(
                    {"EmbedLLM-known": det[:, ix], "EmbedLLM-noisy": noisy[:, ix]}
                )
            else:
                choices.update({k: v[:, ix - len(fresh)] for k, v in refs.items()})
            hit = {k: labels[v, np.arange(len(ix))] for k, v in choices.items()}
            for k, val in choices.items():
                records.append(
                    dict(
                        split=stage,
                        roster=roster,
                        method=k,
                        queries=len(ix),
                        **metrics(val, labels, domains),
                    )
                )
                for seed, v in zip(SEEDS, val):
                    seeds.append(
                        dict(
                            split=stage,
                            roster=roster,
                            method=k,
                            seed=seed,
                            **metrics(v[None], labels, domains),
                        )
                    )
                if k != "peer-states":
                    difference = hit["peer-states"] - hit[k]
                    pairs.append(
                        dict(
                            split=stage,
                            roster=roster,
                            reference=k,
                            **paired_groups(
                                hit["peer-states"], hit[k], domains, groups
                            ),
                            micro_positive_seeds=int((difference.mean(1) > 0).sum()),
                            rescues=(difference > 0).sum(1).mean(),
                            harms=(difference < 0).sum(1).mean(),
                        )
                    )
    for k, rows in (
        ("results", records),
        ("paired", pairs),
        ("costs", costs),
        ("seeds", seeds),
    ):
        atomic_csv(args.report / f"peer_states_{args.mode}_{k}.csv", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("modern", "text", "live", "mc", "option-control", "option"),
        default="modern",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation"
    )
    p.add_argument(
        "--report", type=Path, default=ROOT / "reports/routing/capability/confirmation"
    )
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    with threadpool_limits(limits=3):
        run(p.parse_args())
