"""Compare a 200-question, paired-covariance model embedding with the original 112-ID release."""

import argparse
import json
import sys
from itertools import islice, product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

from route_query import ROOT
from coldstart.conditional_text import consensus_features, lexical
from coldstart.model_covariates import architecture_kernel, model_descriptions
from panel_capabilities import RELEASE_SEEDS as SEEDS, panel_text, template_membership
from panel_posterior import design, train


def prepare(args):
    from evaluate_capability_router import input_context
    from run_experiment import atomic_json

    d, *_ = input_context(ROOT / "artifacts/main_v2")
    with np.load(args.root / "confirmation/questions.npz") as f:
        unavailable = set(d.group_ids[f["confirmation"]].tolist())
    with np.load(args.prepared) as f:
        ids, sp, group, ds = [f[k] for k in ("prompt_ids", "original_split", "group_ids", "benchmark_ids")]
    unavailable.update(group[sp != 0].tolist())
    prior_holds = set(unavailable)
    for folder in ("release112", "release112-stratified", "release112_stratified"):
        for file in (args.root / folder).glob("panel_*.npz"):
            with np.load(file) as panel:
                unavailable.update(group[panel["prompt_ids"]].tolist())
    bank = np.flatnonzero((sp == 0) & ~np.isin(group, list(unavailable)))
    rng = np.random.default_rng(111805)
    mass = np.bincount(ds[sp == 2]) / np.sum(sp == 2)
    counts = np.round(4200 * mass).astype(int)
    selected = []
    for dom, n in enumerate(counts):
        pool = bank[ds[bank] == dom]
        values, weight = np.unique(group[pool], return_counts=True)
        order = rng.permutation(len(values))
        requested = int(np.searchsorted(np.cumsum(weight[order]), n)) + 1
        # GPQA has few underlying stems after excluding overlapping official evaluation groups.
        # Retain enough unseen stems to support all ten benchmarks during adaptive acquisition.
        limit = max(1, len(values) - min(30, int(np.ceil(len(values) * 0.58))))
        selected.extend(values[order[: min(requested, limit)]])
    q = np.flatnonzero((sp == 0) & np.isin(group, selected))
    remaining = np.flatnonzero((sp == 0) & ~np.isin(group, list(prior_holds.union(selected))))
    candidate = []
    for dom in np.unique(ds):
        pool = remaining[ds[remaining] == dom]
        _, first = np.unique(group[pool], return_index=True)
        pool = pool[first]
        candidate.extend(rng.choice(pool, min(len(pool), max(220, int(2200 * mass[dom]))), replace=False))
    reserve = dict(
        groups=np.array(selected),
        prompt_ids=ids[q],
        bank=np.array(candidate),
        valid=ids[sp == 1],
        test=ids[sp == 2],
    )
    np.savez_compressed(args.out / "reserve.npz", **reserve)
    atomic_json(
        args.report / "design.json",
        dict(
            seed=111805,
            reserved_questions=len(q),
            refit_excludes_groups=len(selected),
            counts=np.bincount(ds[q]).tolist(),
            candidate_bank=len(candidate),
            probe_candidates_by_benchmark=np.bincount(ds[candidate], minlength=len(mass)).tolist(),
            training="Each estimate uses one common panel of 200 prompts with 112 graded model responses per prompt. Public names and architectural parent information contain no benchmark scores.",
            controls="Canonical upstream TextMF is refit after excluding every reserved group. The two original release splits retain their official-reference fit.",
            selection="Cross-validation within each individual 200-question panel; development reporting on the previously examined published validation split.",
        ),
    )
    print("Fresh public reserve", len(q), "questions and", len(selected), "groups.", flush=True)


def question_ids(args, target):
    with np.load(args.out / "reserve.npz") as f:
        if target == "support":
            ids = [f["bank"]]
            for seed in SEEDS:
                with np.load(args.root / "release112" / f"panel_{seed}.npz") as panel:
                    ids.append(panel["prompt_ids"])
            return np.unique(np.concatenate(ids))
        return np.unique(np.r_[f["prompt_ids"], f["valid"], f["test"]])


def extract(args):
    import torch
    from coldstart.answer_evidence import answer_reader, expert_gate, selective_answers, GATE_PROXIES

    torch.set_num_threads(1)
    ids = question_ids(args, args.part)
    texts = panel_text(ids, args.raw)
    gated = args.proxy.endswith("-gated")
    if gated:
        evidence = {}
        for name in GATE_PROXIES:
            with np.load(args.out / f"{name}_{args.part}.npz") as f:
                if not np.array_equal(f["ids"], ids):
                    raise ValueError("The proxy gate and extraction prompts must agree.")
                evidence[name] = (f["probabilities"], f["answers"])
        chosen = expert_gate(evidence)
    encoder = answer_reader(args.proxy.removesuffix("-gated"), published=True)
    start = perf_counter()
    prob, ans = [], []
    batch = 48 if args.proxy == "smol" else 24
    try:
        for i in range(0, len(ids), 384):
            if gated:
                p, a = selective_answers(encoder, texts[i : i + 384], chosen[i : i + 384], batch)
            else:
                p, a = encoder.encode(texts[i : i + 384], batch)
            prob.append(p)
            ans.extend(a)
            if i % 1536 == 0:
                print(args.part, args.proxy, i, len(ids), flush=True)
    finally:
        if hasattr(encoder, "close"):
            encoder.close()
    np.savez_compressed(
        args.out / f"{args.proxy}_{args.part}.npz",
        ids=ids,
        probabilities=np.vstack(prob),
        answers=np.asarray(ans),
        seconds=perf_counter() - start,
    )


def solver_evidence(args, ids, part):
    evidence = []
    for name in args.proxies:
        with np.load(args.out / f"{name}_{part}.npz") as f:
            idx = np.searchsorted(f["ids"], ids)
            if not np.array_equal(f["ids"][idx], ids):
                raise ValueError(f"Missing question-only evidence for {name}")
            evidence.append((f["probabilities"][idx], f["answers"][idx]))
    return evidence


def auxiliary(args, ids, text, part):
    basic = lexical(text, [])
    if not args.proxies:
        return basic
    evidence = solver_evidence(args, ids, part)
    return np.column_stack((basic, consensus_features(evidence)))


def select(x, dom, y, text, aux, meta, size, seed, weights=None):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(x), int)
    for j in np.unique(dom):
        pos = rng.permutation(np.flatnonzero(dom == j))
        fold[pos] = np.arange(len(pos)) % 4
    options = list(product((0.15, 0.8), (0.10, 0.70), (0.55, 1.4), (1.5, 6.0), (0.0, 0.65, 1.4)))
    unseen = {cfg: [] for cfg in options}
    for i in range(4):
        tr, va = fold != i, fold == i
        membership = template_membership(text[tr], text[va], dom[tr])
        configurations = {}
        for corr, md, noise, ridge, signal in options:
            key = corr, md, noise, ridge
            if key not in configurations:
                cfg = dict(noise_corr=corr, metadata=md, noise=noise, ridge=ridge)
                pack = train(
                    x[tr],
                    dom[tr],
                    y[:, tr],
                    text[tr],
                    aux[tr],
                    meta,
                    size,
                    cfg,
                    weights[tr] if weights is not None else None,
                )
                query = design(pack, x[va], text[va], aux[va], 1, membership)
                n = membership.shape[1]
                configurations[key] = (
                    pack["vectors"][:, :n] @ query[:, :n].T,
                    pack["vectors"][:, n:] @ query[:, n:].T,
                )
            base, deviation = configurations[key]
            pred = base + signal * deviation
            # Within the training fold, identify practical contenders and estimate pairwise risk out of fold.
            order = np.argsort(-base, axis=0)[:16]
            a, b = np.triu_indices(len(order), 1)
            scores = np.take_along_axis(pred, order, axis=0)
            observed = np.take_along_axis(y[:, va], order, axis=0)
            residual = (scores[a] - scores[b]) - (observed[a] - observed[b])
            unseen[corr, md, noise, ridge, signal].extend(np.mean(residual**2, axis=0))
    best = min(options, key=lambda k: np.mean(unseen[k]))
    return dict(zip(("noise_corr", "metadata", "noise", "ridge", "signal"), best))


def fit(args):
    from run_experiment import atomic_json

    names = pd.read_csv(args.raw / "model_order.csv").sort_values("model_id").model_name.to_numpy(str)
    meta, size = model_descriptions(names, ROOT / "data/metadata/model_families.csv")
    if args.anchor:
        with np.load(args.prepared) as f:
            population = np.bincount(f["benchmark_ids"][f["original_split"] == 0])
        mass = population / population.sum()
    if args.architecture:
        meta = 0.5 * meta + 0.5 * architecture_kernel(names, args.out / "architectures.csv")
    if args.active_panel or args.population:
        from acquire_release_panel import acquire, acquire_precision, quadrature_panel

        with np.load(args.out / "reserve.npz") as f:
            candidates = f["bank"]
        with np.load(args.prepared) as f:
            all_dom = f["benchmark_ids"]
            bank_x = f["embeddings"][candidates]
            source_mass = np.bincount(all_dom[f["original_split"] == 0])
            source_mass = source_mass / source_mass.sum()
        bank_text = panel_text(candidates, args.raw)
        bank_aux = auxiliary(args, candidates, bank_text, "support")
        if args.quadrature:
            from types import SimpleNamespace

            design_args = SimpleNamespace(**{**vars(args), "proxies": args.design_proxies})
            bank_design = auxiliary(design_args, candidates, bank_text, "support")
    cfgs = {}
    for seed in SEEDS:
        start = perf_counter()
        weights = None
        if args.active_panel:
            panel_file = args.out / f"{args.panel_from or args.name}_panel_{seed}.npz"
            if panel_file.exists():
                with np.load(panel_file) as f:
                    idx, y, weights = f["positions"], f["y"], f["weights"]
            else:
                acquire_panel = acquire_precision if args.precision_panel else acquire
                if args.quadrature:
                    acquire_panel = quadrature_panel
                idx, y, weights = acquire_panel(
                    args,
                    seed,
                    candidates,
                    bank_x,
                    all_dom[candidates],
                    bank_text,
                    bank_design if args.quadrature else bank_aux,
                    meta,
                    size,
                    source_mass,
                    len(all_dom),
                )
                np.savez_compressed(
                    panel_file, positions=idx, prompt_ids=candidates[idx], y=y, weights=weights
                )
            ids = candidates[idx]
            x, dom, text, aux = bank_x[idx], all_dom[ids], bank_text[idx], bank_aux[idx]
        else:
            with np.load(args.root / "release112" / f"panel_{seed}.npz") as f:
                x, dom, y, ids = [f[k] for k in ("x", "domains", "y", "prompt_ids")]
            text = panel_text(ids, args.raw)
            aux = auxiliary(args, ids, text, "support")
        if args.probe_count < 200 and not args.quadrature:
            from evaluate_panel_budget import nested_order

            take = nested_order(dom, seed)[: args.probe_count]
            if args.active_panel:
                raise ValueError("Smaller active panels must stop acquisition at their own budget.")
            x, dom, y, text, aux, ids = x[take], dom[take], y[:, take], text[take], aux[take], ids[take]
        if args.population:
            from population_panel import calibrated_weights

            weights = calibrated_weights(
                bank_x, x, all_dom[candidates], dom, bank_aux, aux, bank_text, text, weights
            )
        if args.answers:
            from answer_panel_abilities import fit_answers
            from prepare_historical_answer_keys import verified_options

            keys = verified_options(ids, dict(zip(map(int, ids), text)))
            evidence = solver_evidence(args, ids, "support")
            pack, config = fit_answers(x, dom, y, text, aux, evidence, keys, meta, size, seed, weights)
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        elif args.cross_model:
            from cross_model_panel import fit_cross_model

            pack, config = fit_cross_model(x, dom, y, text, aux, meta, size, seed, weights)
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        elif args.anchor:
            from anchored_panel_abilities import fit_anchor

            pack, config = fit_anchor(x, dom, y, text, aux, meta, size, seed, mass, weights)
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        elif args.stack:
            from stack_panel_abilities import fit_stack

            pack, config = fit_stack(x, dom, y, text, aux, meta, size, seed, weights)
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        elif args.text_kernel:
            from panel_text_kernel import fit_text_kernel

            pack, config = fit_text_kernel(dom, y, text, meta, size, seed)
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        elif args.tabular:
            from panel_tabular_skills import select_skills

            pack, config = select_skills(
                x, dom, y, text, aux, meta, size, seed, weights, device=args.panel_device
            )
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        elif args.ensemble:
            from ensemble_panel_abilities import fit_ensemble

            pack, config = fit_ensemble(x, dom, y, text, aux, meta, size, seed, weights)
            joblib.dump(
                dict(**pack, names=names, prompt_ids=ids), args.out / f"{args.name}_{seed}.joblib", compress=2
            )
        else:
            config = select(x, dom, y, text, aux, meta, size, seed, weights)
            pack = train(x, dom, y, text, aux, meta, size, config, weights)
            np.savez_compressed(
                args.out / f"{args.name}_{seed}.npz",
                **pack,
                names=names,
                prompt_ids=ids,
                signal=config["signal"],
            )
        cfgs[seed] = dict(
            **config,
            questions=len(ids),
            proxies=args.proxies,
            seconds=perf_counter() - start,
            adaptive=args.active_panel and not args.quadrature,
            quadrature=args.quadrature,
            acquisition_proxies=args.design_proxies if args.quadrature else [],
            architecture=args.architecture,
            population_calibration=args.population,
            precision_allocation=args.precision_panel,
        )
        print(args.name, seed, cfgs[seed], flush=True)
    atomic_json(args.report / f"{args.name}_selection.json", cfgs)


def score(args):
    from panel_capabilities import public_routes
    from run_embedllm_release import chunks
    from run_experiment import atomic_csv
    from run_routing import benchmark_means

    with np.load(args.out / "reserve.npz") as res:
        ids = res[{"valid": "valid", "test": "test", "reserve": "prompt_ids"}[args.part]]
    with np.load(args.prepared) as f:
        x, dom, groups = f["embeddings"][ids], f["benchmark_ids"][ids], f["group_ids"][ids]
    texts = panel_text(ids, args.raw)
    aux = auxiliary(args, ids, texts, "evaluation")
    labels = np.full((112, len(ids)), np.nan, dtype=np.float32)
    with np.load(args.prepared) as f:
        lut = np.full(int(f["prompt_ids"].max()) + 1, -1, dtype=int)
    lut[ids] = np.arange(len(ids))
    source = {"valid": "val", "test": "test", "reserve": "train"}[args.part]
    for rows in chunks(args.raw / f"{source}.csv"):
        valid = lut[rows.prompt_id.to_numpy()] >= 0
        last = rows[valid].drop_duplicates(["model_id", "prompt_id"], keep="last")
        labels[last.model_id, lut[last.prompt_id]] = last.label
    if not np.isfinite(labels).all():
        raise ValueError("Incomplete original response matrix for the reserve.")
    orig = (args.out if args.part == "reserve" else args.root / "release112") / "original.npz"
    with np.load(orig) as f:
        if args.part in ("test", "reserve"):
            assert np.array_equal(f["prompt_ids"], ids)
            det, noisy = f["deterministic"], f["noisy"]
        else:
            logits = f["weight"] @ x.T + f["bias"][:, None]
            sd = 0.05 * np.linalg.norm(f["weight"], axis=1, keepdims=True)
            det = logits.argmax(0)
            noisy = [
                (logits + sd * np.random.default_rng(seed).normal(size=logits.shape)).argmax(0)
                for seed in SEEDS
            ]
    rows, paired, outcomes, details = [], [], {}, []
    best_fixed = int(labels.mean(1).argmax())
    if args.answers:
        from coldstart.latent_answers import answer_arrays

        raw_answers = answer_arrays(solver_evidence(args, ids, "evaluation"))
    for j, seed in enumerate(SEEDS):
        if args.answers:
            from answer_panel_abilities import answer_scores

            scores = answer_scores(
                joblib.load(args.out / f"{args.name}_{seed}.joblib"), x, texts, aux, raw_answers
            )
        elif args.cross_model:
            from cross_model_panel import cross_model_scores

            scores = cross_model_scores(joblib.load(args.out / f"{args.name}_{seed}.joblib"), x, texts, aux)
        elif args.anchor:
            from anchored_panel_abilities import anchor_scores

            scores = anchor_scores(joblib.load(args.out / f"{args.name}_{seed}.joblib"), x, texts, aux)
        elif args.stack:
            from stack_panel_abilities import stack_scores

            scores = stack_scores(joblib.load(args.out / f"{args.name}_{seed}.joblib"), x, texts, aux)
        elif args.text_kernel:
            from panel_text_kernel import text_kernel_scores

            scores = text_kernel_scores(joblib.load(args.out / f"{args.name}_{seed}.joblib"), texts)
        elif args.tabular:
            from panel_tabular_skills import tabular_scores

            scores = tabular_scores(joblib.load(args.out / f"{args.name}_{seed}.joblib"), x, texts, aux)
        elif args.ensemble:
            from ensemble_panel_abilities import ensemble_scores

            scores = ensemble_scores(joblib.load(args.out / f"{args.name}_{seed}.joblib"), x, texts, aux)
        else:
            with np.load(args.out / f"{args.name}_{seed}.npz") as pack:
                scores = pack["vectors"] @ design(pack, x, texts, aux, float(pack["signal"])).T
        ours = scores.argmax(0)
        if seed == SEEDS[0]:
            np.savez_compressed(
                args.out / f"{args.name}_{args.part}_scores.npz",
                scores=scores.astype(np.float32),
                prompt_ids=ids,
            )
        with np.load(args.root / "release112" / f"router_{seed}.npz") as pack:
            earlier = public_routes(pack, x, texts)
        for method, choice in (
            (args.name, ours),
            ("panel-22D", earlier),
            ("original-deterministic", det),
            ("original-noisy", noisy[j]),
            ("best-fixed-hindsight", np.full(len(ids), best_fixed)),
        ):
            quality = labels[choice, np.arange(len(ids))]
            outcomes.setdefault(method, []).append(quality)
            freq = np.bincount(choice, minlength=112)
            rows.append(
                dict(
                    method=method,
                    seed=seed,
                    questions=len(ids),
                    macro=benchmark_means(quality, dom).mean(),
                    micro=quality.mean(),
                    top_share=freq.max() / len(ids),
                    members_used=int(np.sum(freq > 0)),
                )
            )
            for domain in np.unique(dom):
                selected = dom == domain
                details.append(
                    dict(
                        method=method,
                        seed=seed,
                        benchmark_id=int(domain),
                        queries=selected.sum(),
                        accuracy=quality[selected].mean(),
                        top_share=np.bincount(choice[selected]).max() / selected.sum(),
                    )
                )
    compared = [args.name] if args.part == "valid" else [args.name, "panel-22D"]
    for method in compared:
        for ref in ("panel-22D", "original-deterministic", "original-noisy", "best-fixed-hindsight"):
            if method == ref:
                continue
            left, right = np.stack(outcomes[method]), np.stack(outcomes[ref])
            gap = left - right
            paired.append(
                dict(
                    method=method,
                    reference=ref,
                    **paired_groups(left, right, dom, groups),
                    micro_positive_seeds=int((gap.mean(1) > 0).sum()),
                    rescues=(gap > 0).sum(1).mean(),
                    harms=(gap < 0).sum(1).mean(),
                    recoverable=(labels.max(0) > right).sum(1).mean(),
                )
            )
    atomic_csv(args.report / f"{args.name}_{args.part}.csv", rows)
    atomic_csv(args.report / f"{args.name}_{args.part}_paired.csv", paired)
    atomic_csv(args.report / f"{args.name}_{args.part}_by_benchmark.csv", details)
    print(pd.DataFrame(rows).groupby("method")[["macro", "micro", "top_share"]].mean(), flush=True)


def paired_groups(left, right, domain, groups):
    from run_routing import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, benchmark_means

    gap = left - right
    by_seed = benchmark_means(gap, domain).mean(-1)
    difference = gap.mean(0, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro = np.zeros(BOOTSTRAP_DRAWS)
    micro = np.zeros(BOOTSTRAP_DRAWS)
    for j in np.unique(domain):
        chosen = domain == j
        _, idx = np.unique(groups[chosen], return_inverse=True)
        count = np.bincount(idx)
        totals = np.bincount(idx, weights=difference[chosen])
        draws = rng.integers(0, len(count), size=(BOOTSTRAP_DRAWS, len(count)))
        mean = totals[draws].sum(1) / count[draws].sum(1)
        macro += mean / len(np.unique(domain))
        micro += mean * chosen.mean()
    return dict(
        macro_gain=by_seed.mean(),
        micro_gain=difference.mean(),
        positive_seed_count=int((by_seed > 0).sum()),
        macro_ci_low=np.quantile(macro, 0.025),
        macro_ci_high=np.quantile(macro, 0.975),
        micro_ci_low=np.quantile(micro, 0.025),
        micro_ci_high=np.quantile(micro, 0.975),
        question_groups=len(np.unique(groups)),
    )


def release_type(args, name):
    path = args.out / f"{name}_11.joblib"
    if not path.exists():
        return "covariance"
    pack = joblib.load(path)
    if "text_kernel" in pack:
        return "text_kernel"
    if "answer_map" in pack:
        return "answers"
    if "cross_model" in pack:
        return "cross_model"
    if "anchor_configs" in pack:
        return "anchor"
    return "stack" if "mixture" in pack else ("tabular" if "target" in pack else "ensemble")


def lock_methods(args):
    from run_experiment import atomic_json

    records = []
    for file in sorted(args.report.glob("*_valid.csv")):
        name = file.name.removesuffix("_valid.csv")
        # This later serving-cost experiment was outside the original validation contest.
        if name == "text-kernel":
            continue
        data = pd.read_csv(file)
        rows = data[data.method == name]
        if len(rows) != len(SEEDS):
            continue
        specs = json.loads((args.report / f"{name}_selection.json").read_text(encoding="utf-8"))
        cfg = specs[str(SEEDS[0])]
        ref = data.groupby("method")[["micro", "macro"]].mean().loc["original-noisy"]
        record = dict(
            name=name,
            technique=release_type(args, name),
            questions=cfg["questions"],
            proxies=cfg["proxies"],
            micro=rows.micro.mean(),
            macro=rows.macro.mean(),
            top_share=rows.top_share.mean(),
            mean_fit_seconds=np.mean([v["seconds"] for v in specs.values()]),
            public_default_micro=ref.micro,
            public_default_macro=ref.macro,
        )
        records.append(record)

    def rank(row):
        return (
            row["micro"] > row["public_default_micro"]
            and row["macro"] > row["public_default_macro"]
            and row["top_share"] < 0.97,
            row["micro"],
            row["macro"],
        )

    best = max(records, key=rank)
    eligible = [
        r for r in records if r["micro"] >= best["micro"] - 0.005 and r["macro"] >= best["macro"] - 0.0075
    ]
    economical = min(
        eligible, key=lambda r: (len(r["proxies"]), r["questions"], r["mean_fit_seconds"], -r["micro"])
    )
    preset = next(r for r in records if r["name"] == "crossfit-stack")
    atomic_json(
        args.report / "selection.json",
        dict(
            primary=best,
            economical=economical,
            panel_preset=preset,
            selection_data="Previously studied public validation questions. Reserve and original test outcomes play no role.",
            panel_preset_rule="Fixed 200-question panel with an internal four-fold stack. This preset was specified before inspecting its own public validation results.",
            economic_rule="First minimize auxiliary query models, then probe count and fitting time, within 0.50 points micro and 0.75 points macro of the public-validation leader.",
        ),
    )
    print("Frozen release choices", best["name"], economical["name"], preset["name"], flush=True)


def score_locked(args):
    from types import SimpleNamespace

    lock = json.loads((args.report / "selection.json").read_text(encoding="utf-8"))
    seen = set()
    for key in ("primary", "economical", "panel_preset"):
        record = lock[key]
        if record["name"] in seen:
            continue
        seen.add(record["name"])
        opt = SimpleNamespace(**vars(args))
        opt.name, opt.proxies = record["name"], record["proxies"]
        opt.stack, opt.tabular, opt.ensemble, opt.anchor, opt.cross_model, opt.answers = (
            record["technique"] == v
            for v in ("stack", "tabular", "ensemble", "anchor", "cross_model", "answers")
        )
        score(opt)


def serving_method(report, name, out=None, seed=11):
    if name is None:
        fitted = out is not None and (out / f"text-kernel_{seed}.joblib").is_file()
        name = "text-kernel" if fitted and (report / "text-kernel_selection.json").is_file() else "primary"
    if name in ("primary", "economical", "panel_preset"):
        lock = json.loads((report / "selection.json").read_text(encoding="utf-8"))
        return report, lock[name]["name"]
    if name in ("shortest", "near-200", "text-shortest", "text-near-200"):
        budget = report / ("text_kernel_probe_cost" if name.startswith("text-") else "probe_cost")
        lock = json.loads((budget / "selection.json").read_text(encoding="utf-8"))
        chosen = lock[name.removeprefix("text-").replace("-", "_")]["name"]
        return (report if chosen in ("tabular-panel", "text-kernel") else budget), chosen
    return report, name


def route(args):
    if args.name == "clean-200":
        args.name, args.seed = "text-kernel", 8800
        args.out, args.report = args.out / "coverage", args.report / "coverage"
    args.report, args.name = serving_method(args.report, args.name, args.out, args.seed)
    available = json.loads((args.report / f"{args.name}_selection.json").read_text(encoding="utf-8"))
    if str(args.seed) not in available:
        raise ValueError(f"Available deployment seeds for {args.name}: {', '.join(available)}")
    spec = available[str(args.seed)]
    filename = args.out / f"{args.name}_{args.seed}.joblib"
    options = {}
    if filename.exists():
        pack = joblib.load(filename)
        if "text_kernel" in pack:
            from panel_text_kernel import text_kernel_scores

            score_fn = text_kernel_scores
        elif "answer_map" in pack:
            from answer_panel_abilities import answer_scores

            score_fn = answer_scores
        elif "cross_model" in pack:
            from cross_model_panel import cross_model_scores

            score_fn = cross_model_scores
        elif "anchor_configs" in pack:
            from anchored_panel_abilities import anchor_scores

            score_fn = anchor_scores
        elif "mixture" in pack:
            from stack_panel_abilities import stack_scores

            score_fn = stack_scores
        elif "target" in pack:
            from panel_tabular_skills import tabular_scores

            score_fn = tabular_scores
        else:
            from ensemble_panel_abilities import ensemble_scores

            score_fn = ensemble_scores
        skills = pack.get("tabular", pack)
        if "target" in skills and (args.prompt is None or args.panel_device == "cuda"):
            from panel_tabular_skills import prepare_estimators

            options["estimators"] = prepare_estimators(skills, args.panel_device)
    else:
        with np.load(args.out / f"{args.name}_{args.seed}.npz") as f:
            pack = dict(f)
        score_fn = None
    names = [str(name).replace("__", "/") for name in pack["names"]]
    while batch := list(islice(args.query_source, args.batch_size)):
        texts = np.asarray(batch)
        if "text_kernel" in pack:
            mat = score_fn(pack, texts)
        else:
            from serve_confirmation_router import encode_prompts

            x = encode_prompts(batch, args.panel_device, args.encoder_precision)
            aux = lexical(texts, [])
            if spec["proxies"]:
                from coldstart.answer_evidence import encode_evidence

                values = encode_evidence(spec["proxies"], texts, min(len(batch), 12), published=True)
                evidence = [values[name] for name in spec["proxies"]]
                aux = np.column_stack((aux, consensus_features(evidence)))
                if "answer_map" in pack:
                    from coldstart.latent_answers import answer_arrays

                    options["raw"] = answer_arrays(evidence)
            if score_fn is None:
                mat = pack["vectors"] @ design(pack, x, texts, aux, float(pack["signal"])).T
            else:
                mat = score_fn(pack, x, texts, aux, **options)
        for scores in mat.T:
            i = int(scores.argmax())
            output = dict(model_id=i, selected=names[i], questions=len(pack["prompt_ids"]), method=args.name)
            if args.all_scores:
                output["scores"] = dict(zip(names, map(float, scores)))
            print(json.dumps(output, ensure_ascii=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("prepare", "extract", "fit", "score", "route", "lock", "score-locked"),
        required=True,
    )
    p.add_argument("--part", choices=("support", "evaluation", "valid", "test", "reserve"), default="valid")
    p.add_argument("--name")
    p.add_argument("--active-panel", action="store_true")
    p.add_argument("--precision-panel", action="store_true")
    p.add_argument("--quadrature", action="store_true")
    p.add_argument("--design-proxies", nargs="*", default=["smol", "qwen", "math", "yi"])
    p.add_argument("--architecture", action="store_true")
    p.add_argument("--population", action="store_true")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--tabular", action="store_true")
    p.add_argument("--text-kernel", action="store_true")
    p.add_argument("--stack", action="store_true")
    p.add_argument("--anchor", action="store_true")
    p.add_argument("--cross-model", action="store_true")
    p.add_argument("--answers", action="store_true")
    p.add_argument("--probe-count", type=int, choices=range(40, 201, 20), default=200)
    p.add_argument("--panel-from", help="Reuse the same observed adaptive panel when comparing estimators.")
    p.add_argument("--proxy")
    p.add_argument("--proxies", nargs="*", default=[])
    query = p.add_mutually_exclusive_group()
    query.add_argument("--prompt")
    query.add_argument("--stdin", action="store_true")
    query.add_argument("--stdin-json", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--panel-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--encoder-precision", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--all-scores", action="store_true")
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    p.add_argument("--prepared", type=Path, default=ROOT / "data/processed/embedllm.npz")
    p.add_argument("--root", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/release_confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/release_confirmation")
    args = p.parse_args(argv)
    if args.phase == "route":
        from route_query import json_prompts

        if args.batch_size < 1:
            p.error("--batch-size must be positive.")
        if args.stdin_json:
            args.query_source = json_prompts(sys.stdin, p)
        elif args.stdin:
            args.query_source = (line.rstrip("\r\n") for line in sys.stdin if line.strip())
        elif args.prompt:
            args.query_source = iter([args.prompt])
        else:
            p.error("Supply questions with --prompt, --stdin or --stdin-json.")
    if not args.name and args.phase != "route":
        args.name = "covariance"
    if args.precision_panel or args.quadrature:
        args.active_panel = True
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=3):
        {
            "prepare": prepare,
            "extract": extract,
            "fit": fit,
            "score": score,
            "route": route,
            "lock": lock_methods,
            "score-locked": score_locked,
        }[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
