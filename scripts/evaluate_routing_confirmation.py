"""Apply the frozen portfolio once to fresh question groups and replay the earlier test set."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from confirmation_models import ConfirmationModels
from evaluate_capability_router import (
    ROOT,
    benchmark_means,
    input_context,
    known_choices,
    metrics,
    original_cold,
    old_choices,
    paired_gain,
    test_rosters,
)
from run_experiment import atomic_csv
from select_confirmation_router import SEEDS, orders
from coldstart.switch_calibration import paired_uncertainty, rate_challengers


def predictions(args, runner, lock, bank, targets, roster, ids, names):
    runner.target_names = np.array([names[i] for i in ids])
    methods, exports, cache, fit_cost, measured, frontier = {}, {}, {}, [], {}, []
    for method in (
        v
        for v in (
            "primary",
            "conditional",
            "economical",
            "adaptive",
            "cascade",
            "cascade-cpu",
            "cascade-budget",
            "reference-modern",
            "reference-live",
            "reference-panel",
            "reference-joint-modern",
            "reference-joint-live",
        )
        if v in lock
    ):
        choice = lock[method]
        by_seed = []
        gate_teacher = []
        for seed in SEEDS:
            matrices, payloads, observed = [], [], None
            for cfg in choice["configs"]:
                cfg_key = json.dumps(cfg, sort_keys=True), seed
                if cfg_key not in cache:
                    start = perf_counter()
                    idx = runner.panel(cfg, targets, seed, bank.orders["balanced", seed])
                    masked = np.full(targets.shape, np.nan, dtype=np.float32)
                    masked[:, idx] = targets[:, idx]
                    mat, payload = runner.score(cfg, masked, idx, seed)
                    deferred = runner.last_deferred.copy() if cfg["algorithm"] == "cascade" else None
                    cache[cfg_key] = idx, mat, payload, deferred
                    if seed == SEEDS[0] and runner.refit:
                        from serve_confirmation_router import readout_file

                        location = readout_file(args.out, cfg)
                        location.parent.mkdir(exist_ok=True)
                        if not location.exists():
                            joblib.dump(runner.export_readout(cfg), location, compress=2)
                    fit_cost.append(
                        dict(
                            reference_models=len(runner.ref),
                            roster=roster,
                            algorithm=cfg["algorithm"],
                            component=cfg["name"],
                            seed=seed,
                            probes=len(idx),
                            minimum_observed=int(np.isfinite(masked[:, idx]).sum(1).min()),
                            seconds=perf_counter() - start,
                        )
                    )
                    if cfg["algorithm"] == "cascade":
                        for phase in ("confirmation", "previous_test"):
                            present = np.isin(runner.query, runner.manifest[phase])
                            if present.any():
                                fit_cost[-1][f"{phase}_large_query_fraction"] = runner.last_deferred[
                                    present
                                ].mean()
                idx, mat, payload, deferred = cache[cfg_key]
                if method == "cascade" and args.split == "both":
                    from coldstart.cascade import decision, teacher_config
                    from coldstart.latent_skills import route_types

                    block = runner.component(cfg)
                    quick, risk = decision(
                        payload["z"],
                        block["peer"],
                        block["posterior"],
                        block["head"]["dom"],
                        block["pack"]["means"],
                        block["student"],
                        cfg,
                    )
                    full = runner.component(teacher_config(cfg))
                    heavy = route_types(
                        payload["z"],
                        full["head"]["p"],
                        full["posterior"],
                        full["head"]["dom"],
                        block["pack"]["means"],
                        cfg,
                    )
                    frontier.append((quick.argmax(0), heavy.argmax(0), risk))
                if observed is not None and not np.array_equal(idx, observed):
                    raise ValueError("A portfolio must share one common probe panel.")
                observed = idx
                matrices.append(mat)
                payloads.append(payload)
            score = np.average(matrices, axis=0, weights=choice["weights"])
            measured.setdefault(method, {})[seed] = observed
            uncertainty = paired_uncertainty(runner.h, targets, observed, runner.dt)
            score = rate_challengers(
                score,
                uncertainty,
                choice.get("allowance", 0.0),
                specialization=choice.get("specialization", 0.0),
            )
            by_seed.append(score.argmax(0))
            if method == lock.get("default_method") and runner.refit and not args.prefix:
                gate_teacher.append(score.astype(np.float32))
            if seed == SEEDS[0] and runner.refit:
                exports[method] = dict(
                    configs=choice["configs"],
                    weights=choice["weights"],
                    profiles=payloads,
                    model_names=np.array([names[i] for i in ids]),
                    model_ids=ids,
                    probe_ids=runner.d.prompt_ids[runner.train[observed]],
                    seed=seed,
                    method=method,
                    refit=True,
                    allowance=choice.get("allowance", 0.0),
                    specialization=choice.get("specialization", 0.0),
                    uncertainty=uncertainty,
                )
                np.savez_compressed(
                    args.out / f"{args.prefix}{roster}_{method}_scores.npz",
                    scores=score,
                    prompt_ids=runner.d.prompt_ids[runner.query],
                    **(
                        {"large_decoder_used": deferred}
                        if len(choice["configs"]) == 1 and choice["configs"][0]["algorithm"] == "cascade"
                        else {}
                    ),
                )
        methods[method] = np.stack(by_seed)
        if gate_teacher:
            np.savez_compressed(
                args.out / f"{roster}_{method}_trials.npz",
                scores=np.stack(gate_teacher),
                model_ids=ids,
                prompt_ids=runner.d.prompt_ids[runner.query],
            )
    for method, pack in exports.items():
        joblib.dump(pack, args.out / f"{args.prefix}{roster}_{method}.joblib", compress=2)
        if method == lock.get("default_method") and not args.prefix:
            parts, layout = {}, {}
            for config, profile in zip(pack["configs"], pack["profiles"]):
                kind = config["algorithm"]
                for field, values in profile.items():
                    if isinstance(values, np.ndarray) and values.ndim >= 2 and len(values) == len(ids):
                        parts[f"{kind}_{field}"] = values
                if kind == "local":
                    layout["local_region_domains"] = runner.component(config)["region"]["group_domain"]
            np.savez_compressed(
                args.out / f"{roster}_{method}_embeddings.npz",
                embeddings=np.column_stack([v.reshape(len(ids), -1) for v in parts.values()]),
                part_names=list(parts),
                part_widths=[v[0].size for v in parts.values()],
                historical_model_names=[str(runner.d.model_names[i]).replace("__", "/") for i in runner.ref],
                benchmark_names=runner.d.benchmark_names,
                model_names=np.char.replace(pack["model_names"].astype(str), "__", "/"),
                component_weights=pack["weights"],
                probe_ids=pack["probe_ids"],
                seed=pack["seed"],
                **layout,
                **parts,
            )
        if method == "cascade-budget" and not args.prefix:
            domains = runner.d.benchmark_names
            peers = [str(runner.d.model_names[i]).replace("__", "/") for i in runner.ref]
            classes = int(pack["configs"][0]["classes"])
            columns = [*(f"peer/{v}" for v in peers), *(f"ability/{v}" for v in domains)]
            columns.extend(f"skill/{v}/{i + 1}" for v in domains for i in range(classes))
            np.savez_compressed(
                args.out / f"{roster}_{method}_embeddings.npz",
                embeddings=pack["profiles"][0]["z"],
                model_names=np.char.replace(pack["model_names"].astype(str), "__", "/"),
                coordinate_names=columns,
                probe_ids=pack["probe_ids"],
                seed=pack["seed"],
            )
    return methods, fit_cost, measured, frontier


def summarize(roster, cohort, source, stages, ids, truth, dom, d, methods, storage, lock):
    rows, trials, paired, details = storage
    for phase, take in stages:
        selected = take & np.isfinite(truth).all(0)
        target, domain = truth[:, selected], dom[selected]
        choices = {name: route[:, selected] for name, route in methods.items()}
        fixed = int(target.mean(1).argmax())
        choices["best-fixed-hindsight"] = np.full((len(SEEDS), len(domain)), fixed)
        reachable = target.max(0)
        outcomes = {k: target[v, np.arange(len(domain))] for k, v in choices.items()}
        for method, route in choices.items():
            common = dict(split=phase, roster=roster, reference_models=cohort, method=method)
            counts = np.stack([np.bincount(row, minlength=len(ids)) for row in route])
            quality = metrics(route, target, domain)
            rows.append(
                dict(
                    **common,
                    queries=len(domain),
                    macro=quality["macro"],
                    micro=quality["micro"],
                    top_share=quality["top_share"],
                    mean_models_used=(counts > 0).sum(1).mean(),
                    most_routed_model_id=int(ids[counts.mean(0).argmax()]),
                )
            )
            for j, seed in enumerate(SEEDS):
                trials.append(
                    dict(
                        **common,
                        seed=seed,
                        macro=benchmark_means(outcomes[method][j], domain).mean(),
                        micro=outcomes[method][j].mean(),
                        top_id=int(ids[counts[j].argmax()]),
                        top_share=counts[j].max() / len(domain),
                    )
                )
            if method not in (
                "primary",
                "conditional",
                "economical",
                "adaptive",
                "cascade",
                "cascade-cpu",
                "cascade-budget",
                "reference-modern",
                "reference-live",
                "reference-panel",
                "reference-joint-modern",
                "reference-joint-live",
                "panel-only-200",
                "panel-calibrated-200",
                "panel-120",
                "panel-200",
                "panel-text-120",
                "panel-text-200",
            ):
                continue
            reference_names = [
                "EmbedLLM-known",
                "EmbedLLM-noisy",
                source,
                "probe-best-fixed",
                "best-fixed-hindsight",
            ]
            if "previous-8D-refit" in choices:
                reference_names.append("previous-8D-refit")
            exact = f"EmbedLLM-same-{method}"
            if exact in choices:
                reference_names.append(exact)
            config = (
                lock[method]["configs"][0]
                if method in lock
                else {"policy": "balanced", "probe_count": 120 if method.endswith("-120") else 200}
            )
            matched = (
                f"EmbedLLM-balanced-{int(config['probe_count'])}"
                if config["policy"] in ("balanced", "stop-evpi") and exact not in choices
                else None
            )
            if matched in choices and matched not in reference_names:
                reference_names.append(matched)
            for ref in reference_names:
                left, right = outcomes[method], outcomes[ref]
                delta = left - right
                recoverable = (reachable[None, :] > right).sum(1)
                paired.append(
                    dict(
                        **common,
                        reference=ref,
                        **paired_gain(left, right, domain),
                        micro_positive_seeds=int((delta.mean(1) > 0).sum()),
                        rescues=(delta > 0).sum(1).mean(),
                        harms=(delta < 0).sum(1).mean(),
                        switches=(route != choices[ref]).sum(1).mean(),
                        recoverable=recoverable.mean(),
                        fraction_recovered=np.mean((delta > 0).sum(1) / recoverable.clip(1)),
                        oracle_accuracy=reachable.mean(),
                    )
                )
            for j, benchmark in enumerate(d.benchmark_names):
                mask = domain == j
                assigned = np.stack([np.bincount(row[mask], minlength=len(ids)) for row in route])
                delta = outcomes[method][:, mask] - outcomes["EmbedLLM-known"][:, mask]
                details.append(
                    dict(
                        **common,
                        benchmark=benchmark,
                        queries=int(mask.sum()),
                        accuracy=outcomes[method][:, mask].mean(),
                        native=outcomes["EmbedLLM-known"][:, mask].mean(),
                        rescues=(delta > 0).sum(1).mean(),
                        harms=(delta < 0).sum(1).mean(),
                        top_share=(assigned.max(1) / mask.sum()).mean(),
                        top_id=int(ids[assigned.mean(0).argmax()]),
                    )
                )


def run(args):
    lock = json.loads((args.selection or args.report / "selection.json").read_text(encoding="utf-8"))
    refs = json.loads((args.report / "references.json").read_text(encoding="utf-8"))
    d, s, config, prior_strength, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        manifest = dict(f)
    if args.split == "previous-test":
        q = manifest["previous_test"]
        phases = [("previous-test", np.ones(len(q), dtype=bool))]
    else:
        q = np.r_[manifest["confirmation"], manifest["previous_test"]]
        cut = np.arange(len(q)) < len(manifest["confirmation"])
        phases = [("confirmation", cut), ("previous-test", ~cut)]
    names, rosters = test_rosters(d, s, prepared, family)
    out = ([], [], [], [])
    costs, curves = [], []
    pure_panels = {}
    from panel_capabilities import panel_design, panel_text

    raw_questions = panel_text(d.prompt_ids[q], ROOT / "data/raw") if args.split == "both" else None
    for cohort in args.cohorts:
        refit = cohort == 75
        runner = ConfirmationModels(args, d, s, manifest, q, "evaluation", refit)
        bank = orders(d, s, manifest, runner.ref)
        for roster, unordered, all_labels in rosters:
            order = unordered.argsort()
            ids, labels = unordered[order], all_labels[order]
            probe_responses = labels[:, manifest["training"]]
            routes, fit_cost, measured, frontier = predictions(
                args, runner, lock, bank, probe_responses, roster, ids, names
            )
            if raw_questions is not None:
                if roster not in pure_panels:
                    pure, calibrated = [], []
                    for seed in SEEDS:
                        directory = args.out.parent / "panels"
                        with np.load(directory / f"{roster}_{seed}.npz") as panel:
                            probes = d.prompt_ids[manifest[f"balanced_{seed}"][:200]]
                            if not np.array_equal(panel["model_ids"], ids) or set(panel["prompt_ids"]) != set(
                                probes
                            ):
                                raise ValueError(
                                    "The panel-only control must use the same 200 candidate prompts."
                                )
                            features = panel_design(panel, d.embeddings[q], raw_questions)
                            pure.append((panel["vectors"] @ features.T).argmax(0))
                        with np.load(directory / f"{roster}_{seed}_calibrated.npz") as panel:
                            calibrated.append((panel["vectors"] @ features.T).argmax(0))
                    pure_panels[roster] = {
                        "panel-only-200": np.stack(pure),
                        "panel-calibrated-200": np.stack(calibrated),
                    }
                    for name in ("panel-120", "panel-200", "panel-text-120", "panel-text-200"):
                        rows = []
                        for seed in SEEDS:
                            with np.load(args.out / "strict_panels" / f"{roster}_{name}_{seed}.npz") as f:
                                if not np.array_equal(f["prompt_ids"], d.prompt_ids[q]) or not np.array_equal(
                                    f["model_ids"], ids
                                ):
                                    raise ValueError(
                                        "The strict-panel experiment and the evaluation roster or prompts differ."
                                    )
                                rows.append(f["scores"].argmax(0))
                        pure_panels[roster][name] = np.stack(rows)
                routes.update(pure_panels[roster])
            costs.extend(fit_cost)
            with np.load(args.out / "embedllm_known_ids.npz") as f:
                routes["EmbedLLM-known"], routes["EmbedLLM-noisy"] = known_choices(f, d, ids, q, SEEDS)
            best = lock["strong_cold_policy"]
            cold_name = f"EmbedLLM-{best['policy']}-{best['budget']}"
            cold_file = "embedllm_historical75.npz" if refit else "embedllm_historical.npz"
            with np.load(args.out / cold_file) as f:
                for policy, n in {(best["policy"], best["budget"]), ("balanced", 120), ("balanced", 200)}:
                    ctrl = refs[f"cold-{policy}-{n}"]
                    routes[f"EmbedLLM-{policy}-{n}"] = original_cold(
                        f, d, bank, labels, q, policy, ctrl["strength"], n
                    )
                for method, panels in measured.items():
                    if any(
                        all(np.array_equal(panels[seed], bank.orders["balanced", seed][:n]) for seed in SEEDS)
                        for n in (120, 200)
                    ):
                        continue
                    exact = SimpleNamespace(
                        seeds=SEEDS, items=bank.items, orders={("exact", s): ix for s, ix in panels.items()}
                    )
                    routes[f"EmbedLLM-same-{method}"] = original_cold(
                        f, d, exact, labels, q, "exact", refs["cold-balanced-200"]["strength"]
                    )
            old_root = args.out / f"previous8_{cohort}"
            if args.split == "both" or len(list(old_root.glob("*.pt"))) == len(SEEDS):
                routes["previous-8D-refit"] = old_choices(
                    SimpleNamespace(old=old_root), d, bank, config, prior_strength, labels, q
                )
            choices = []
            source_mass = np.bincount(runner.dt) / len(runner.dt)
            for seed in SEEDS:
                idx = bank.orders["balanced", seed][:200]
                prior = np.stack(
                    [
                        np.nanmean(probe_responses[:, idx[runner.dt[idx] == j]], axis=1)
                        for j in range(len(source_mass))
                    ]
                )
                choices.append(np.full(len(q), (source_mass @ prior).argmax()))
            routes["probe-best-fixed"] = np.stack(choices)
            if refit and roster == "audited_13" and args.split == "both":
                from coldstart.answer_evidence import question_parts

                method = lock["default_method"]
                new, old = routes[method][0], routes["previous-8D-refit"][0]
                fresh = np.isin(q, manifest["confirmation"]) & np.isfinite(labels[:, q]).all(0)
                delta = labels[new, q] - labels[old, q]
                examples = []
                for j, benchmark in enumerate(d.benchmark_names):
                    for tag, difference in (("rescued", 1), ("harmed", -1)):
                        ix = np.flatnonzero(fresh & (d.domains[q] == j) & (delta == difference))
                        if len(ix):
                            k = ix[d.prompt_ids[q[ix]].argmin()]
                            stem, _ = question_parts(raw_questions[k])
                            examples.append(
                                dict(
                                    method=method,
                                    seed=SEEDS[0],
                                    benchmark=benchmark,
                                    outcome=tag,
                                    prompt_id=int(d.prompt_ids[q[k]]),
                                    question=" ".join(stem.split())[:320],
                                    previous_model=str(names[ids[old[k]]]).replace("__", "/"),
                                    previous_correct=int(labels[old[k], q[k]]),
                                    selected_model=str(names[ids[new[k]]]).replace("__", "/"),
                                    selected_correct=int(labels[new[k], q[k]]),
                                )
                            )
                if examples:
                    atomic_csv(args.report / f"{args.prefix}switch_examples.csv", examples)
            np.savez_compressed(
                args.out / f"{args.prefix}choices_{cohort}_{roster}.npz",
                prompt_ids=d.prompt_ids[q],
                model_ids=ids,
                **routes,
            )
            summarize(
                roster, cohort, cold_name, phases, ids, labels[:, q], d.domains[q], d, routes, out, lock
            )
            if frontier:
                for phase, take in phases:
                    keep = take & np.isfinite(labels[:, q]).all(0)
                    for threshold in (-1.0, 0.04, 0.08, 0.13, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 1.0):
                        trigger = np.stack([risk[keep] > threshold for _, _, risk in frontier])
                        chosen = np.stack(
                            [
                                np.where(use, heavy[keep], light[keep])
                                for (light, heavy, _), use in zip(frontier, trigger)
                            ]
                        )
                        curves.append(
                            dict(
                                split=phase,
                                roster=roster,
                                reference_models=cohort,
                                threshold=threshold,
                                heavy_query_fraction=trigger.mean(),
                                **metrics(chosen, labels[:, q[keep]], d.domains[q[keep]]),
                            )
                        )
            print(f"Scored {roster} with {cohort} historical embeddings", flush=True)
    for name, records in zip(("summary", "seeds", "paired", "by_benchmark", "costs"), (*out, costs)):
        atomic_csv(args.report / f"{args.prefix}{name}.csv", records)
    if curves:
        atomic_csv(args.report / f"{args.prefix}hardware_curve.csv", curves)
    if not args.prefix and args.split == "both" and 75 in args.cohorts:
        configs = lock[lock["default_method"]]["configs"]
        helpers = set()
        for cfg in configs:
            head = args.out / cfg.get("readout", cfg["name"]) / "training.json"
            helpers.update(json.loads(head.read_text(encoding="utf-8"))["proxies"])
            helpers.update((cfg.get("proxies") or "").split())
        extraction = []
        for helper in sorted(helpers):
            for phase in ("development", "evaluation"):
                with np.load(args.out / f"evidence_{helper}_{phase}.npz") as f:
                    count = len(f["prompt_ids"])
                    active = int(f["probabilities"][:, 0, 6].sum()) if helper.endswith("-gated") else count
                    extraction.append(
                        dict(
                            helper=helper,
                            phase=phase,
                            available_prompts=count,
                            active_prompts=active,
                            recorded_seconds=float(f["seconds"]),
                            board_peak_MiB=int(f["board_peak_mib"]),
                        )
                    )
        if extraction:
            atomic_csv(args.report / "selected_feature_cost.csv", extraction)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts", type=int, nargs="+", choices=(75, 62), default=[75, 62])
    p.add_argument("--selection", type=Path)
    p.add_argument("--split", choices=("both", "previous-test"), default="both")
    p.add_argument("--prefix", default="")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        run(p.parse_args())
