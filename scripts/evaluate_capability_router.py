"""Select ability-based routing on validation, then replay frozen selections on the two test rosters."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from pilot_routing_learning import B, ROOT, input_context, latent_prior, model_latents
from coldstart.capability import (
    CapabilityBasis,
    KernelReadout,
    capability_query,
    fingerprint_profiles,
    fit_capabilities,
    informative_order,
    representative_order,
    stratified_order,
)
from coldstart.adaptation import adapt_many
from coldstart.probes import select_probe_order
from evaluate_metadata_excluded import load_oracle
from coldstart.models import load_decoder
from run_experiment import atomic_csv, atomic_json
from run_routing import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, benchmark_means


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ProbeBank:
    def __init__(self, dataset, split, seeds):
        self.items = np.asarray(split.train_items)
        self.domains = dataset.benchmark_ids[self.items]
        self.history = dataset.labels[np.ix_(split.train_models, self.items)]
        self.basis = CapabilityBasis.fit(self.history, self.domains)
        self.domain_means = benchmark_means(self.history, self.domains).T
        self.seeds = seeds
        self.orders = {}
        local = np.arange(len(self.items))
        for seed in seeds:
            # Use the same bank ordering as the established balanced-random protocol.
            self.orders["balanced", seed] = select_probe_order(
                dataset.embeddings[self.items], self.domains, local, 200, "balanced_random", seed
            )
            self.orders["informative", seed] = informative_order(
                self.basis, self.history, self.domains, 200, seed
            )
            self.orders["representative", seed] = representative_order(self.history, self.domains, 200, seed)
            self.orders["stratified", seed] = stratified_order(
                self.basis, self.history, self.domains, dataset.embeddings[self.items], 200, seed
            )


def metrics(choices, truth, domains):
    correct = truth[choices, np.arange(len(domains))]
    macro = float(benchmark_means(correct, domains).mean())
    micro = float(correct.mean(dtype=np.float64))
    shares = [np.bincount(row, minlength=len(truth)).max() / len(domains) for row in choices]
    return {
        "macro": macro,
        "micro": micro,
        "top_share": float(np.mean(shares)),
        "objective": 0.65 * macro + 0.35 * micro,
    }


def selection_key(row, targets):
    meets_target = row["macro"] > targets[0] and row["micro"] > targets[1] and row["top_share"] < targets[2]
    return meets_target, row["objective"]


def paired_gain(left, right, domains):
    difference = left - right
    query = difference.mean(axis=0, dtype=np.float64)
    strata = np.unique(domains)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro, micro = np.zeros(BOOTSTRAP_DRAWS), np.zeros(BOOTSTRAP_DRAWS)
    for domain in strata:
        values = query[domains == domain]
        draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values))).mean(axis=1)
        macro += draws / len(strata)
        micro += draws * len(values) / len(query)
    by_seed = benchmark_means(difference, domains).mean(-1)
    return dict(
        macro_gain=by_seed.mean(),
        micro_gain=query.mean(),
        positive_seed_count=int((by_seed > 0).sum()),
        macro_ci_low=np.quantile(macro, 0.025),
        macro_ci_high=np.quantile(macro, 0.975),
        micro_ci_low=np.quantile(micro, 0.025),
        micro_ci_high=np.quantile(micro, 0.975),
    )


def get_kernel(args, d, s):
    file = args.weights / "kernel.npz"
    if file.exists():
        with np.load(file) as f:
            return KernelReadout(f["anchors"], f["coefficients"], int(f["n_models"]))
    if args.phase == "test":
        raise FileNotFoundError(file)
    start = perf_counter()
    k = KernelReadout.fit(
        d.embeddings[s.train_items],
        d.labels[np.ix_(s.train_models, s.train_items)],
        d.benchmark_ids[s.train_items],
    )
    np.savez(
        file,
        anchors=k.anchors,
        coefficients=k.coefficients,
        n_models=k.n_models,
        seconds=perf_counter() - start,
    )
    return k


def load_features(names, phase, args, d, items, kernel, weights=None):
    values = []
    for name in names:
        if name == "kernel":
            values.append(kernel.predict(d.embeddings[items]))
        else:
            cache = args.weights / f"{name}_{phase}.npz"
            if phase == "test" and not cache.exists():
                from embed_new_model import read_prompts
                import torch

                lookup = read_prompts(d.prompt_ids[items], ROOT / "data/raw/question_order.csv")
                texts = [lookup[i] for i in d.prompt_ids[items]]
                if name.startswith("topic/"):
                    from coldstart.topic_readout import TopicReadout

                    net = TopicReadout(cache.parent, name.split("/")[1])
                    _, mem = kernel.predict(d.embeddings[items])
                    p, memberships = net.predict(d.embeddings[items], texts, mem)
                    np.savez(cache, historical_scores=p, domains=memberships, prompt_ids=d.prompt_ids[items])
                elif name.startswith("views_"):
                    from coldstart.question_views import FrozenQuestionViews
                    from train_question_views import load_readouts

                    folder = cache.parent
                    net = FrozenQuestionViews(folder.name.removeprefix("views_"))
                    raw = np.vstack(
                        [net.encode(texts[i : i + 512], batch=24) for i in range(0, len(texts), 512)]
                    )
                    heads = ["ridge", "tree"] if (folder / "response.cbm").exists() else ["ridge"]
                    for head, (p, memberships) in load_readouts(folder, heads)(raw).items():
                        np.savez(
                            folder / f"{head}_test.npz",
                            historical_scores=p,
                            domains=memberships,
                            prompt_ids=d.prompt_ids[items],
                        )
                else:
                    from coldstart.capability_text import ResponseEncoder, tokenize_prompts, tokenizer_for

                    net, spec = ResponseEncoder.restore(args.weights / f"{name}.pt")
                    tokens = tokenize_prompts(tokenizer_for(spec["size"]), texts, spec["limit"])
                    p, memberships = net.predict(*tokens, batch=24)
                    np.savez(cache, historical_scores=p, domains=memberships, prompt_ids=d.prompt_ids[items])
                del net
                torch.cuda.empty_cache()
            with np.load(cache) as f:
                if not np.array_equal(f["prompt_ids"], d.prompt_ids[items]):
                    raise ValueError(f"Question order differs for {name}.")
                values.append((f["historical_scores"], f["domains"]))
    return tuple(np.average([v[j] for v in values], axis=0, weights=weights) for j in range(2))


def profile_grid(bank, labels):
    grids = {}
    for policy, rank, scale in product(
        ("balanced", "informative", "representative"), (8, 24, 61), (0.0, 0.08)
    ):
        vectors, settings = [], []
        for ridge, temperature in product((0.05, 0.2, 0.8), (0.0, 0.035)):
            cfg = dict(policy=policy, rank=rank, ridge=ridge, domain_scale=scale, temperature=temperature)
            vectors.append(np.stack([newcomer_vectors(bank, labels, cfg, seed) for seed in bank.seeds]))
            settings.append(cfg)
        grids[policy, rank, scale] = (np.stack(vectors), settings)
    return grids


def newcomer_vectors(bank, labels, cfg, seed):
    selected = bank.orders[cfg["policy"], seed][: cfg.get("probe_count", 200)]
    responses = labels[:, bank.items[selected]]
    return fit_capabilities(bank.basis, bank.history[:, selected].T, bank.domains[selected], responses, cfg)


def fingerprint_grid(bank, labels):
    profiles, settings = [], []
    for policy, bandwidth, ridge in product(
        ("balanced", "representative", "informative"), (0.05, 0.18), (0.05, 0.5, 2.0)
    ):
        fitted = []
        for seed in bank.seeds:
            ix = bank.orders[policy, seed]
            fitted.append(
                fingerprint_profiles(
                    bank.history[:, ix].T,
                    bank.domains[ix],
                    labels[:, bank.items[ix]],
                    bank.basis.domain_count,
                    bandwidth,
                    ridge,
                    1.0,
                )
            )
        fitted = np.stack(fitted)
        for correction in (0.4, 0.8):
            z = fitted.copy()
            z[:, :, -bank.basis.domain_count :] *= correction
            profiles.append(z)
            settings.append(
                dict(
                    algorithm="fingerprint",
                    policy=policy,
                    bandwidth=bandwidth,
                    ridge=ridge,
                    correction=correction,
                )
            )
    return np.stack(profiles), settings


def fingerprint_search(bank, grid, scores, memberships, truth, domains, targets=(-1, -1, 2)):
    profiles, settings = grid
    best = None
    for signal in (0.3, 0.75, 1.2):
        cfg = dict(algorithm="fingerprint", signal=signal)
        query = capability_query(bank.basis, bank.domain_means, scores, memberships, cfg)
        choices = (profiles @ query.T).argmax(axis=-2)
        for c, setting in zip(choices, settings):
            row = {**setting, "signal": signal, **metrics(c, truth, domains)}
            if best is None or selection_key(row, targets) > selection_key(best, targets):
                best = row
    return best


def question_design(bank, scores, memberships, rank, scale, signal):
    domain_means = memberships @ bank.domain_means
    return bank.basis.design(domain_means + signal * (scores - domain_means), memberships, rank, scale)


def validation_search(bank, grids, scores, memberships, truth, domains, targets=(-1, -1, 2)):
    best = None
    query_designs = {}
    for (_, rank, scale), (profiles, settings) in grids.items():
        for signal in (0.25, 0.65, 1.0):
            key = rank, scale, signal
            if key not in query_designs:
                query_designs[key] = question_design(bank, scores, memberships, *key)
            choices = (profiles @ query_designs[key].T).argmax(axis=-2)
            for c, cfg in zip(choices, settings):
                row = {**cfg, "signal": signal, **metrics(c, truth, domains)}
                if best is None or selection_key(row, targets) > selection_key(best, targets):
                    best = row
    return best


def original_cold(f, d, bank, labels, items, policy, strength, budget=200):
    weights, bias, offset, profiles = (f[x] for x in ("projection", "bias", "offset", "profiles"))
    mean = profiles.mean(axis=0, dtype=np.float64)
    covariance = np.cov(profiles.T)
    covariance = 0.8 * covariance + 0.2 * np.diag(np.diag(covariance))
    precision = strength * np.linalg.inv(covariance)
    query = d.embeddings[items] @ weights.T + bias
    choices = []
    for seed in bank.seeds:
        p = bank.items[bank.orders[policy, seed][:budget]]
        design = d.embeddings[p] @ weights.T + bias
        fits = adapt_many(design, np.full(len(p), offset), labels[:, p], mean, precision)
        choices.append((np.stack([x.latent for x in fits]) @ query.T).argmax(axis=0))
    return np.stack(choices)


def known_choices(f, d, ids, items, seeds):
    profiles = f["profiles"][np.searchsorted(f["model_ids"], ids)]
    query = d.embeddings[items] @ f["projection"].T + f["bias"]
    logits = profiles @ query.T
    choices = np.broadcast_to(logits.argmax(0), (len(seeds), len(items)))
    # Upstream evaluator_router calls forward() with its default noisy path. For a linear
    # projection the induced score noise is independent Gaussian with this exact variance.
    std = 0.05 * np.linalg.norm(profiles @ f["projection"], axis=1, keepdims=True)
    noisy = np.stack(
        [
            (logits + std * np.random.default_rng(seed).standard_normal(logits.shape)).argmax(0)
            for seed in seeds
        ]
    )
    return choices, noisy


def old_choices(args, d, bank, config, prior_strength, labels, items):
    choices = []
    for seed in bank.seeds:
        dec, _ = load_decoder(args.old / f"{B}_{seed}.pt", "cpu")
        p = bank.items[bank.orders["balanced", seed]]
        support = dec.export_features(d.embeddings[p], d.benchmark_ids[p])
        prior = latent_prior(dec, config, prior_strength)
        fitted = model_latents(support, labels[:, p], d.benchmark_ids[p], prior, config)
        _, query = dec.export_features(d.embeddings[items], d.benchmark_ids[items])
        choices.append((fitted @ query.T).argmax(0))
    return np.stack(choices)


def baselines(args, d, bank, config, prior_strength, ids, labels, items, originals):
    choices = {}
    for policy, val in originals.items():
        with np.load(args.weights / "embedllm_historical.npz") as f:
            choices["EmbedLLM-200-" + policy] = original_cold(
                f, d, bank, labels, items, policy, val["strength"]
            )
    with np.load(args.weights / "embedllm_known_ids.npz") as f:
        choices["EmbedLLM-known"], choices["EmbedLLM-known-noisy"] = known_choices(
            f, d, ids, items, bank.seeds
        )
    choices["previous-8D"] = old_choices(args, d, bank, config, prior_strength, labels, items)
    fixed = []
    for seed in bank.seeds:
        p = bank.items[bank.orders["balanced", seed]]
        means = np.stack(
            [
                np.nanmean(labels[:, p[d.benchmark_ids[p] == j]], axis=1)
                for j in range(bank.basis.domain_count)
            ]
        )
        fixed.append(np.full(len(items), means.mean(0).argmax()))
    choices["probe-best-fixed"] = np.stack(fixed)
    return choices


def stage_options(args):
    yield "kernel", 0, [["kernel"]]
    topic_times = args.weights / "topic/times.json"
    if topic_times.exists():
        for stage, seconds in read_json(topic_times).items():
            name = f"topic/{stage}"
            yield name, seconds, [[name, "kernel"]]
    views = {}
    for size in ("small", "expert"):
        folder = args.weights / f"views_{size}"
        if not (folder / "fit_times.json").exists():
            continue
        extraction = read_json(folder / "extraction.json")
        fits = read_json(folder / "fit_times.json")
        for head, key in (("ridge", "ridge_seconds"), ("tree", "all_seconds")):
            name = f"views_{size}/{head}"
            views[name] = extraction["seconds"] + fits[key]
            yield name, views[name], [[name], [name, "kernel"]]
    for cap in (8, 16):
        path = args.weights / f"gpu{cap}" / "training.json"
        if not path.exists():
            continue
        history = read_json(path)
        completed_members = []
        previous = None
        for entry in history:
            name = f"gpu{cap}/{entry['stage']}"
            if previous and previous["member"] != entry["member"]:
                completed_members.append(f"gpu{cap}/{previous['stage']}")
            groups = [[name], [name, "kernel"]]
            if completed_members:
                groups.extend([completed_members + [name], completed_members + [name, "kernel"]])
            yield name, entry["elapsed_seconds"], groups
            for view, view_seconds in views.items():
                if cap == 8 and view.startswith("views_expert"):
                    continue
                total = entry["elapsed_seconds"] + view_seconds
                # The middle budget starts at 30 minutes, leaving no eligible tier from 10 to 30.
                if 600 < total < 1800:
                    continue
                yield f"{name}+{view}", total, [[name, view, "kernel"]]
            previous = entry


def short_panel(args, d, s, bank, kernel, full, targets):
    """Choose a smaller deployment panel on validation with the 200-probe kernel decoder fixed."""
    items = np.asarray(s.val_items)
    labels = d.labels[s.val_models]
    truth, domains = labels[:, items], d.domains[items]
    hist, membership = kernel.predict(d.embeddings[items])
    query = capability_query(bank.basis, bank.domain_means, hist, membership, full)
    start = perf_counter()
    candidates = []
    for n, policy in product((80, 120, 160), ("balanced", "stratified")):
        cfg = dict(full, probe_count=n, policy=policy)
        vectors = np.stack([newcomer_vectors(bank, labels, cfg, seed) for seed in bank.seeds])
        row = dict(cfg, **metrics((vectors @ query.T).argmax(1), truth, domains))
        row["validation_win"] = bool(selection_key(row, targets)[0])
        candidates.append(row)
    qualifying = [c for c in candidates if c["validation_win"]]
    if not qualifying:
        return None
    best = min(qualifying, key=lambda r: (r["probe_count"], -r["objective"]))
    best["selection_seconds"] += perf_counter() - start
    with np.load(args.weights / "embedllm_historical.npz") as f:
        matches = []
        for strength in (0.1, 1.0, 10.0):
            choices = original_cold(f, d, bank, labels, items, best["policy"], strength, best["probe_count"])
            matches.append(dict(strength=strength, **metrics(choices, truth, domains)))
    best["matched_reference"] = max(matches, key=lambda row: row["objective"])
    return best


def validate(args, d, s, config, prior_strength, bank, kernel):
    items = np.asarray(s.val_items)
    labels, domains = d.labels[s.val_models], d.benchmark_ids[items]
    truth = labels[:, items]
    originals_path = args.out / "baseline_selection.json"
    originals = read_json(originals_path) if originals_path.exists() else {}
    missing = [p for p in ("balanced", "informative", "representative") if p not in originals]
    if missing:
        with np.load(args.weights / "embedllm_historical.npz") as f:
            for policy in missing:
                candidates = []
                for strength in (0.1, 1.0, 10.0):
                    choices = original_cold(f, d, bank, labels, items, policy, strength)
                    candidates.append(dict(strength=strength, **metrics(choices, truth, domains)))
                originals[policy] = max(candidates, key=lambda row: row["objective"])
        atomic_json(originals_path, originals)
        base = baselines(
            args, d, bank, config, prior_strength, d.model_ids[s.val_models], labels, items, originals
        )
        atomic_csv(
            args.out / "validation_baselines.csv",
            [{"method": k, **metrics(v, truth, domains)} for k, v in base.items()],
        )
    with (args.out / "validation_baselines.csv").open(newline="") as f:
        refs = {r["method"]: r for r in csv.DictReader(f)}
    if "EmbedLLM-known-noisy" not in refs:
        with np.load(args.weights / "embedllm_known_ids.npz") as f:
            _, noisy = known_choices(f, d, d.model_ids[s.val_models], items, bank.seeds)
        refs["EmbedLLM-known-noisy"] = dict(method="EmbedLLM-known-noisy", **metrics(noisy, truth, domains))
        atomic_csv(args.out / "validation_baselines.csv", list(refs.values()))
    cold = max(originals.values(), key=lambda r: r["objective"])
    targets = (
        max(cold["macro"], float(refs["EmbedLLM-known"]["macro"])),
        max(cold["micro"], float(refs["EmbedLLM-known"]["micro"])),
        float(refs["previous-8D"]["top_share"]),
    )
    grids = profile_grid(bank, labels)
    fingerprints = fingerprint_grid(bank, labels)
    path = args.out / "validation_profiles.json"
    stages = read_json(path) if path.exists() else {}
    for stage, seconds, combinations in stage_options(args):
        if stage in stages and "validation_win" in stages[stage]:
            continue
        start = perf_counter()
        variants = []
        for names in combinations:
            weights = [None]
            if "kernel" in names and len(names) > 1:
                weights.append([1.0] * (len(names) - 1) + [3.0 * (len(names) - 1)])
                if names[0].startswith("topic/"):
                    weights.extend(([0.4, 0.6], [0.7, 0.3]))
                if any(n.startswith("views_") for n in names) and any(n.startswith("gpu") for n in names):
                    weights.append([1.0, 2.0, 2.0])
            for mix in weights:
                scores, classes = load_features(names, "validation", args, d, items, kernel, mix)
                lowrank = validation_search(bank, grids, scores, classes, truth, domains, targets)
                variants.append(dict(encoders=names, encoder_weights=mix, algorithm="pca", **lowrank))
                fingerprint = fingerprint_search(bank, fingerprints, scores, classes, truth, domains, targets)
                variants.append(dict(encoders=names, encoder_weights=mix, **fingerprint))
        best = max(variants, key=lambda row: selection_key(row, targets))
        if stage == "kernel":
            with np.load(args.weights / "kernel.npz") as f:
                seconds = float(f["seconds"])
        stages[stage] = dict(
            **best,
            validation_win=bool(selection_key(best, targets)[0]),
            training_seconds=seconds,
            selection_seconds=perf_counter() - start,
        )
        atomic_json(path, stages)
        print(stage, stages[stage], flush=True)
    if "short-panel" not in stages:
        shorter = short_panel(args, d, s, bank, kernel, stages["kernel"], targets)
        if shorter:
            stages["short-panel"] = shorter
            atomic_json(path, stages)
            print("short-panel", shorter, flush=True)


def freeze(args, d, bank):
    stages = read_json(args.out / "validation_profiles.json")
    with (args.out / "validation_baselines.csv").open(newline="") as f:
        base = {r["method"]: r for r in csv.DictReader(f)}
    native = base["EmbedLLM-known"]
    cold = max(
        (v for k, v in base.items() if k.startswith("EmbedLLM-200-")), key=lambda v: float(v["objective"])
    )
    cost = lambda x: x["training_seconds"] + x["selection_seconds"]
    tiers = {}
    for cap, name, lo, hi in (
        (8, "8g-10m", 0, 600),
        (8, "8g-30to60m", 1800, 3600),
        (16, "16g-30to60m", 1800, 3600),
        (8, "8g-over60m", 3600, np.inf),
        (16, "16g-over60m", 3600, np.inf),
    ):
        ledger = {r["stage"]: r for r in read_json(args.weights / f"gpu{cap}/training.json")}
        eligible = []
        for stage, cfg in stages.items():
            components = stage.split("+")
            if "/" not in stage or not lo <= cost(cfg) <= hi:
                continue
            family, head = components[0].split("/")
            if family != f"gpu{cap}" and family != ("views_small" if cap == 8 else "views_expert"):
                continue
            hardware = (
                ledger[head].copy()
                if family.startswith("gpu")
                else read_json(args.weights / family / "extraction.json")
            )
            if len(components) > 1:
                view_folder = components[1].split("/")[0]
                other = read_json(args.weights / view_folder / "extraction.json")
                hardware["sequential_components"] = components
                hardware["component_seconds"] = [
                    ledger[head]["elapsed_seconds"],
                    cfg["training_seconds"] - ledger[head]["elapsed_seconds"],
                ]
                hardware["elapsed_seconds"] = cfg["training_seconds"]
                hardware["board_peak_mib"] = max(hardware["board_peak_mib"], other["board_peak_mib"])
            if hardware["board_peak_mib"] > cap * 1024 or (
                cap == 16 and hardware["board_peak_mib"] < 15 * 1024
            ):
                continue
            eligible.append(dict(stage=stage, **cfg, hardware=hardware))
        if not eligible:
            raise ValueError(f"No measured checkpoint satisfies {name}.")
        tiers[name] = max(eligible, key=lambda v: (v["validation_win"], v["objective"], -cost(v)))
    tiers["cpu-kernel"] = dict(stage="kernel", **stages["kernel"])
    topics = [dict(stage=k, **v) for k, v in stages.items() if k.startswith("topic/")]
    if topics:
        tiers["cpu-topic"] = max(topics, key=lambda v: (v["validation_win"], v["objective"], -cost(v)))
    if "short-panel" in stages:
        tiers["cpu-short"] = dict(stage="short-panel", **stages["short-panel"])
    accepted = [
        k
        for k, v in tiers.items()
        if all(v[m] > max(float(cold[m]), float(native[m])) for m in ("macro", "micro"))
        and v["top_share"] < float(base["previous-8D"]["top_share"])
    ]
    best = max(
        (k for k in tiers if not k.startswith("cpu-")),
        key=lambda k: (tiers[k]["validation_win"], tiers[k]["objective"]),
    )
    best_requested = best
    expert = [dict(stage=k, **v) for k, v in stages.items() if k.startswith("views_expert/")]
    if expert:
        frozen = max(expert, key=lambda v: (v["validation_win"], v["objective"], -cost(v)))
        if frozen["validation_win"] and frozen["objective"] > tiers[best]["objective"]:
            best = "gpu-frozen"
            tiers[best] = dict(**frozen, hardware=read_json(args.weights / "views_expert/extraction.json"))
            accepted.append(best)
    smallest = (
        min(
            accepted,
            key=lambda k: (tiers[k]["training_seconds"], tiers[k].get("probe_count", 200), cost(tiers[k])),
        )
        if accepted
        else best
    )
    lock = {
        "primary_roster": "audited_13",
        "metric": "routed correctness; macro equal over five benchmarks",
        "selection_objective": "Prefer joint validation wins on macro and micro over cold and known-ID EmbedLLM with lower top share than previous 8D. Within that tier maximize 0.65 * macro + 0.35 * micro.",
        "probe_cap_per_candidate": 200,
        "seeds": bank.seeds,
        "tiers": tiers,
        "best_hardware_on_validation": best,
        "best_requested_tier": best_requested,
        "economical_on_validation": smallest,
        "economical_rule": "Among joint validation winners, minimize shared training time, then questions per candidate, then selection time. Fall back to the best GPU validation objective.",
        "original_cold_policy": cold["method"],
        "original_fits": read_json(args.out / "baseline_selection.json"),
        "validation_targets": {"native": native, "cold": cold, "old": base["previous-8D"]},
    }
    atomic_json(args.out / "selection.json", lock)
    packed = {
        "axes": bank.basis.axes,
        "domain_means": bank.domain_means,
        "domain_names": np.asarray(d.benchmark_names),
    }
    for policy, seed in bank.orders:
        ix = bank.orders[policy, seed]
        key = f"{policy}_{seed}"
        packed[key + "_ids"] = d.prompt_ids[bank.items[ix]]
        packed[key + "_domains"] = bank.domains[ix]
        packed[key + "_history"] = bank.history[:, ix].T
    np.savez(args.weights / "probe_bank.npz", **packed)
    print("Validation selections:", best, smallest, flush=True)


def test_rosters(d, s, prepared, family):
    ids = d.model_ids[s.test_models]
    labels = d.labels[s.test_models]
    extra = load_oracle(d, prepared / "embedllm.npz", family)
    names = dict(zip(d.model_ids, d.model_names))
    names.update(zip(extra.model_ids, extra.model_names))
    rosters = [
        ("audited_13", ids, labels),
        ("expanded_37", np.r_[ids, extra.model_ids], np.vstack([labels, extra.labels])),
    ]
    return names, rosters


def fit_shared_panels(args, d, s, prepared, family, seeds):
    from panel_capabilities import pack_panel, panel_text, public_routes, select_panel

    names, rosters = test_rosters(d, s, prepared, family)
    rosters = [("validation_13", d.model_ids[s.val_models], d.labels[s.val_models]), *rosters]
    dest = args.weights / "panels"
    dest.mkdir(parents=True, exist_ok=True)
    decisions = {}
    raw = ROOT / "data/raw"
    train = np.asarray(s.train_items)
    orders = {
        seed: train[
            select_probe_order(
                d.embeddings[train], d.domains[train], np.arange(len(train)), 200, "balanced_random", seed
            )
        ]
        for seed in seeds
    }
    for roster, mids, labels in rosters:
        order = mids.argsort()
        mids, labels = mids[order], labels[order]
        val_choices = []
        for seed in seeds:
            # This bank order depends on source and seed only. Retain items observed for all competitors.
            indices = orders[seed]
            indices = indices[np.isfinite(labels[:, indices]).all(0)]
            ids = d.prompt_ids[indices]
            x, domains, y = d.embeddings[indices], d.domains[indices], labels[:, indices]
            texts = panel_text(ids, raw)
            cfg = select_panel(x, domains, y, texts, seed)
            name = f"{roster}_{seed}"
            decisions[name] = cfg
            pack = pack_panel(x, domains, y, texts, ids, np.asarray([names[mid] for mid in mids]), cfg)
            np.savez_compressed(dest / f"{name}.npz", **pack, model_ids=mids)
            if roster == "validation_13":
                val = np.asarray(s.val_items)
                val_choices.append(public_routes(pack, d.embeddings[val], panel_text(d.prompt_ids[val], raw)))
        if val_choices:
            decisions["validation_metrics"] = metrics(np.stack(val_choices), labels[:, val], d.domains[val])
    atomic_json(args.out / "panel_selection.json", decisions)
    print("Panel-only validation", decisions["validation_metrics"], flush=True)
    calibrate_shared_panels(args, d, s, seeds)


def calibrate_shared_panels(args, d, s, seeds):
    from panel_capabilities import calibrated_vectors, panel_design, panel_text

    dest = args.weights / "panels"
    val = np.asarray(s.val_items)
    order = d.model_ids[s.val_models].argsort()
    truth = d.labels[np.ix_(np.asarray(s.val_models)[order], val)]
    texts = panel_text(d.prompt_ids[val], ROOT / "data/raw")
    choices = {}
    for seed in seeds:
        with np.load(dest / f"validation_13_{seed}.npz") as panel:
            design = panel_design(panel, d.embeddings[val], texts)
            for r, scale in product((0.15, 0.45, 0.75, 1.0), (0.25, 0.65, 1.0)):
                vectors = calibrated_vectors(panel, r, scale)
                choices.setdefault((r, scale), []).append((vectors @ design.T).argmax(0))
    with (args.out / "validation_baselines.csv").open(newline="") as stream:
        refs = {row["method"]: row for row in csv.DictReader(stream)}
    cold = max(
        (row for name, row in refs.items() if name.startswith("EmbedLLM-200-")),
        key=lambda row: float(row["objective"]),
    )
    target = (
        max(float(cold["macro"]), float(refs["EmbedLLM-known"]["macro"])),
        max(float(cold["micro"]), float(refs["EmbedLLM-known"]["micro"])),
        float(refs["previous-8D"]["top_share"]),
    )
    rows = [
        dict(retention=r, residual_scale=scale, **metrics(np.stack(v), truth, d.domains[val]))
        for (r, scale), v in choices.items()
    ]
    choice = max(rows, key=lambda row: selection_key(row, target))
    for roster, seed in product(("audited_13", "expanded_37"), seeds):
        with np.load(dest / f"{roster}_{seed}.npz") as f:
            panel = dict(f)
        panel["vectors"] = calibrated_vectors(panel, choice["retention"], choice["residual_scale"])
        np.savez_compressed(dest / f"{roster}_{seed}_calibrated.npz", **panel)
    atomic_json(args.out / "panel_calibration.json", choice)
    print("Validation-calibrated panels", choice, flush=True)


def score_test(args, d, s, config, strength, prepared, family, bank, kernel):
    from panel_capabilities import panel_design, panel_text

    lock = read_json(args.out / "selection.json")
    names, rosters = test_rosters(d, s, prepared, family)
    rows, trials, paired, breakdown, usage = [], [], [], [], []
    for roster, mids, labels in rosters:
        order = mids.argsort()
        mids, labels = mids[order], labels[order]
        all_items = np.asarray(s.test_items)
        common = np.isfinite(labels[:, all_items]).all(axis=0)
        items = all_items[common]
        dom, truth = d.domains[items], labels[:, items]
        routes = baselines(args, d, bank, config, strength, mids, labels, items, lock["original_fits"])
        incoming = panel_text(d.prompt_ids[items], ROOT / "data/raw")
        panel_choices, calibrated = [], []
        for seed in bank.seeds:
            with np.load(args.weights / f"panels/{roster}_{seed}.npz") as panel:
                if not np.array_equal(panel["model_ids"], mids):
                    raise ValueError("Candidate IDs differ from the locked panel.")
                design = panel_design(panel, d.embeddings[items], incoming)
                panel_choices.append((panel["vectors"] @ design.T).argmax(0))
            with np.load(args.weights / f"panels/{roster}_{seed}_calibrated.npz") as panel:
                calibrated.append((panel["vectors"] @ design.T).argmax(0))
        routes["panel-only-200"] = np.stack(panel_choices)
        routes["panel-calibrated-200"] = np.stack(calibrated)
        best_fixed = int(benchmark_means(truth, dom).mean(1).argmax())
        routes["best-fixed-hindsight"] = np.full((len(bank.seeds), len(items)), best_fixed)
        for tier, cfg in lock["tiers"].items():
            if "matched_reference" in cfg:
                n = cfg["probe_count"]
                with np.load(args.weights / "embedllm_historical.npz") as f:
                    routes[f"EmbedLLM-{n}-{cfg['policy']}"] = original_cold(
                        f, d, bank, labels, items, cfg["policy"], cfg["matched_reference"]["strength"], n
                    )
            hist, mem = load_features(
                cfg["encoders"], "test", args, d, all_items, kernel, cfg.get("encoder_weights")
            )
            design = capability_query(bank.basis, bank.domain_means, hist[common], mem[common], cfg)
            z = np.stack([newcomer_vectors(bank, labels, cfg, seed) for seed in bank.seeds])
            routes[tier] = (z @ design.T).argmax(1)
            # Each row is a deployment-ready profile fitted with only the first probe seed.
            np.savez(
                args.weights / f"{roster}_{tier}.npz",
                model_names=np.asarray([names[mid] for mid in mids]),
                model_ids=mids,
                vectors=z[0],
                tier=tier,
                stage=cfg["stage"],
                seed=bank.seeds[0],
            )
        correctness = {k: truth[route, np.arange(len(items))] for k, route in routes.items()}
        for name, choices in routes.items():
            acc = correctness[name]
            stats = metrics(choices, truth, dom)
            stats.pop("objective")
            shuffled = []
            for i, seed in enumerate(bank.seeds):
                use = np.bincount(choices[i], minlength=len(mids))
                trials.append(
                    dict(
                        roster=roster,
                        method=name,
                        seed=seed,
                        n_queries=len(items),
                        macro=benchmark_means(acc[i], dom).mean(),
                        micro=acc[i].mean(),
                        top_id=int(mids[use.argmax()]),
                        top_share=use.max() / len(items),
                    )
                )
                for mid, count in zip(mids, use):
                    usage.append(
                        dict(roster=roster, method=name, seed=seed, model_id=int(mid), assigned=int(count))
                    )
                shuffle = []
                for j, bname in enumerate(d.benchmark_names):
                    ix = dom == j
                    freq = np.bincount(choices[i, ix], minlength=len(mids)) / ix.sum()
                    expected = freq @ truth[:, ix].mean(axis=1, dtype=np.float64)
                    shuffle.append(expected)
                    breakdown.append(
                        dict(
                            roster=roster,
                            method=name,
                            seed=seed,
                            benchmark=bname,
                            queries=int(ix.sum()),
                            accuracy=acc[i, ix].mean(),
                            same_usage_expected=expected,
                            top_model_id=int(mids[freq.argmax()]),
                            top_share=float(freq.max()),
                            models_used=int((freq > 0).sum()),
                            rescued_vs_previous=int((acc[i, ix] > correctness["previous-8D"][i, ix]).sum()),
                            harmed_vs_previous=int((acc[i, ix] < correctness["previous-8D"][i, ix]).sum()),
                        )
                    )
                shuffled.append(np.mean(shuffle))
            rows.append(
                dict(
                    roster=roster,
                    method=name,
                    n_queries=len(items),
                    n_seeds=len(bank.seeds),
                    **stats,
                    macro_oracle=benchmark_means(truth.max(0), dom).mean(),
                    micro_oracle=truth.max(0).mean(dtype=np.float64),
                    macro_gain_over_within_benchmark_shuffle=stats["macro"] - np.mean(shuffled),
                )
            )
            if name not in lock["tiers"] and not name.startswith("panel-"):
                continue
            selected = lock["tiers"].get(name, {"policy": "balanced"})
            matched = f"EmbedLLM-{selected.get('probe_count', 200)}-{selected['policy']}"
            for ref in dict.fromkeys(
                [
                    lock["original_cold_policy"],
                    matched,
                    "EmbedLLM-known",
                    "EmbedLLM-known-noisy",
                    "previous-8D",
                    "probe-best-fixed",
                ]
            ):
                diff = acc - correctness[ref]
                paired.append(
                    dict(
                        roster=roster,
                        method=name,
                        reference=ref,
                        **paired_gain(acc, correctness[ref], dom),
                        mean_rescues=(diff > 0).sum(1).mean(),
                        mean_harms=(diff < 0).sum(1).mean(),
                        mean_switches=(choices != routes[ref]).sum(1).mean(),
                    )
                )
        np.savez_compressed(
            args.weights / f"{roster}_routes.npz",
            prompt_ids=d.prompt_ids[items],
            benchmark_ids=dom,
            model_ids=mids,
            **{k: mids[v] for k, v in routes.items()},
        )
    for suffix, data in [
        ("summary", rows),
        ("by_seed", trials),
        ("paired", paired),
        ("by_benchmark", breakdown),
        ("assignments", usage),
    ]:
        atomic_csv(args.out / f"test_{suffix}.csv", data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=("validation", "freeze", "test", "cpu-cost", "panel-fit"), required=True
    )
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--weights", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--out", type=Path, default=ROOT / "reports/routing/capability")
    p.add_argument("--old", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        start = perf_counter()
        d, s, config, strength, prepared, family = input_context(args.run)
        loaded = perf_counter()
        if args.phase == "panel-fit":
            fit_shared_panels(args, d, s, prepared, family, config["seeds"])
            return
        bank = ProbeBank(d, s, config["seeds"])
        if args.phase == "cpu-cost":
            planned = perf_counter()
            fit = KernelReadout.fit(d.embeddings[bank.items], bank.history, bank.domains)
            # Materialize exactly the shared coefficients used by an incoming query.
            fit.predict(d.embeddings[np.asarray(s.val_items)[:1]])
            trained = perf_counter()
            config = read_json(args.out / "validation_profiles.json")
            profiles = []
            for name in ("kernel", "short-panel"):
                if name not in config:
                    continue
                clock = perf_counter()
                for i in s.val_models:
                    newcomer_vectors(bank, d.labels[[i]], config[name], bank.seeds[0])
                profiles.append(
                    dict(
                        stage=name,
                        questions=config[name].get("probe_count", 200),
                        mean_ms_per_model=1000 * (perf_counter() - clock) / len(s.val_models),
                    )
                )
            costs = dict(
                dataset_seconds=loaded - start,
                build_bank_and_all_probe_orders_seconds=planned - loaded,
                fit_and_first_vector_prediction_seconds=trained - planned,
                total_seconds=trained - start,
                includes="Prepared dataset through all three seed plans and CPU MPNet readout; cached published text vectors.",
                profiles=profiles,
            )
            atomic_json(args.out / "cpu_cost.json", costs)
            print(costs, flush=True)
            return
        kernel = get_kernel(args, d, s)
        if args.phase == "validation":
            validate(args, d, s, config, strength, bank, kernel)
        elif args.phase == "freeze":
            freeze(args, d, bank)
        else:
            score_test(args, d, s, config, strength, prepared, family, bank, kernel)


if __name__ == "__main__":
    main()
