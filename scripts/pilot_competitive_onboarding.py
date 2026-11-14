"""Compare probe selection and stopping across fixed incumbent strength levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pilot_routing_acquisition import acquisition_banks
from run_sequential_routing import (B, CAP, WARMUP, fit_random, input_context, probabilities, read_json,
                                    seed_episodes, stop_indices, trajectory)
from route_query import ROOT
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries
from run_routing import add_one_outcomes, benchmark_means


ANCHOR_QUANTILES = (0.50, 0.75, 0.90, 1.0)
FIXED_BUDGETS = (10, 20, 50, 100, CAP)
EXACT_STOP = "exact_own_stop"
RANDOM_STOP = "random_own_stop"
EXACT_CAP = "exact_cap"
RANDOM_CAP = "random_cap"
INCUMBENT_ONLY = "incumbent_only"
NEWCOMER_ONLY = "newcomer_only"
SAME_K = "retrospective_random_at_exact_k"
PROBE_FIXED = "exact_k_support_selected_fixed_identity"
VALIDATED_K = "validated_random_fixed_count"
RANDOM_PREFIX = "random_fixed_"


def anchors(dataset, split):
    support = split.train_items
    rows = split.train_models
    ids = dataset.model_ids[rows]
    strength = benchmark_means(dataset.labels[np.ix_(rows, support)],
                               dataset.benchmark_ids[support]).mean(axis=1)
    ranking = np.lexsort((ids, strength))
    selected = []
    for quantile in ANCHOR_QUANTILES:
        rank = int(np.floor(quantile * (len(rows) - 1) + 0.5))
        position = ranking[rank]
        row = rows[position]
        selected.append((row, {"quantile": quantile, "rank_of_training_models": rank + 1,
                               "model_id": int(ids[position]), "model_name": dataset.model_names[row],
                               "support_macro_accuracy": float(strength[position]),
                               "used_in_descriptive_competitive_average": quantile < 1.0}))
    return selected


def run_episode(episode, margin, tolerance, options, mode, phase):
    def execute(stopping_tolerance):
        observed = []

        def observe(probe):
            response = episode["observe"](probe)
            observed.append((probe, response))
            return response

        seed = np.random.SeedSequence([episode["seed"], episode["model_id"]])
        result = trajectory(episode["random_order"], episode["probe_domains"], episode["probe_design"],
                            episode["probe_offsets"], observe, episode["reference"], episode["reference_incumbent"],
                            episode["prior"], margin, options, seed, stopping_tolerance, selection=mode)
        if [probe for probe, _ in observed] != result[2]:
            raise RuntimeError("Observed response history differs from the selected path.")
        return (*result, np.asarray([response for _, response in observed], dtype=np.float64))

    if phase == "validation":
        capped = execute(0.0)
        latents, trace, chosen, _, responses = capped
        prefix = int(stop_indices(np.asarray(trace)[:, 0], tolerance))
        budget = prefix + WARMUP
        stop_reason = ("cap" if budget == CAP else
                       "sufficient_reference_evidence" if trace[prefix][1] > 0 else "weak_on_reference")
        stopped = (latents[:prefix + 1], trace[:prefix + 1], chosen[:budget], stop_reason, responses[:budget])
    else:
        stopped = execute(tolerance)
        capped = stopped if len(stopped[2]) == CAP else execute(0.0)
        if stopped[2] != capped[2][:len(stopped[2])]:
            raise RuntimeError("Online stopping did not reproduce the capped path prefix.")
    return stopped, capped


def assessments(episode, incumbent_answers, options, margin, tolerance, phase):
    exact, exact_cap = run_episode(episode, margin, tolerance, options, "exact", phase)
    random, random_cap = run_episode(episode, margin, tolerance, options, "random", phase)

    def selection(path, budget):
        index = budget - WARMUP
        predicted = probabilities(path[0][index], episode)
        return ((predicted - episode["incumbent"]) > margin, budget, path[1][index][2],
                path[3] if budget == len(path[2]) else "", path[1][index][0])

    used = len(exact[2])
    pairs = np.stack([incumbent_answers[exact[2]], exact[-1]])
    probes_domains = episode["probe_domains"][exact[2]]
    fixed_improvement = np.diff(benchmark_means(pairs, probes_domains).mean(axis=1))[0]
    target_length = len(episode["incumbent"])
    results = {
        EXACT_STOP: selection(exact, used), RANDOM_STOP: selection(random, len(random[2])),
        EXACT_CAP: selection(exact_cap, CAP), RANDOM_CAP: selection(random_cap, CAP),
        PROBE_FIXED: (np.full(target_length, fixed_improvement > 0), used, None, "", None),
        INCUMBENT_ONLY: (np.zeros(target_length, dtype=bool), 0, 0.0, "", None),
        NEWCOMER_ONLY: (np.ones(target_length, dtype=bool), 0, 0.0, "", None),
    }
    # Fixed-count baselines need one fit after their own selected prefix.
    # Other arms update after each answer, including the random EVPI control.
    random_fits = {budget: fit_random(episode, budget, options) for budget in set(FIXED_BUDGETS) | {used}}
    for method, budget in [(SAME_K, used), *[(f"{RANDOM_PREFIX}{k}", k) for k in FIXED_BUDGETS]]:
        latent, runtime = random_fits[budget]
        predicted = probabilities(latent, episode)
        results[method] = ((predicted - episode["incumbent"]) > margin, budget, runtime, "", None)
    return results, exact, random


def evaluate_anchor(dataset, split, config, learning, anchor, identity, rows, target,
                    reference, candidates, options, margin, tolerance, args):
    domains = dataset.benchmark_ids[target]
    seed_list = config["seeds"]
    shape = len(seed_list), len(rows), len(target)
    newcomer_choices, by_arrival, traces = {}, [], []
    incumbent_answers = dataset.labels[anchor, candidates]
    truth = dataset.labels[np.ix_(np.r_[anchor, rows], target)]
    for si, seed in enumerate(seed_list):
        episodes = seed_episodes(dataset, split, config, learning, anchor, rows, target,
                                 reference, candidates, seed, args)
        for mi, episode in enumerate(episodes):
            results, exact, random = assessments(episode, incumbent_answers, options, margin, tolerance, args.phase)
            for method, (chosen, budget, runtime, reason, evpi) in results.items():
                newcomer_choices.setdefault(method, np.zeros(shape, dtype=bool))[si, mi] = chosen
                by_arrival.append({**identity, "seed": seed, "newcomer_id": episode["model_id"], "method": method,
                                   "n_queries": len(target), "probe_attempts": budget,
                                   "local_cpu_elapsed_seconds": runtime, "stop_reason": reason if method in (EXACT_STOP, RANDOM_STOP) else "",
                                   "evpi_at_stop": evpi if method in (EXACT_STOP, RANDOM_STOP) else None})
            for mode, path in (("exact", exact), ("random", random)):
                traces.append({"anchor_id": identity["model_id"], "seed": seed, "newcomer_id": episode["model_id"],
                               "selector": mode, "callback_count": len(path[-1]), "stop_reason": path[3],
                               "prompt_ids": dataset.prompt_ids[candidates[path[2]]].tolist(),
                               "evpi": [step[0] for step in path[1]]})
        print(f"{args.phase}: anchor {identity['model_id']} seed {seed} completed", flush=True)
    outcomes = {method: add_one_outcomes(chosen, truth, domains)
                for method, chosen in newcomer_choices.items()}
    for entry in by_arrival:
        si = seed_list.index(entry["seed"])
        mi = np.flatnonzero(dataset.model_ids[rows] == entry["newcomer_id"])[0]
        result = outcomes[entry["method"]]
        actual = result["correct"][si, mi]
        rescue = int(result["rescue"][si, mi].sum())
        harm = int(result["harm"][si, mi].sum())
        opportunities = int(result["opportunity"][mi].sum())
        entry.update(macro_accuracy=float(benchmark_means(actual, domains).mean()),
                     micro_accuracy=float(actual.mean(dtype=np.float64)),
                     newcomer_selected=int(result["selected"][si, mi].sum()), rescues=rescue, harms=harm,
                     opportunities=opportunities, missed_opportunities=opportunities - rescue)
    return outcomes, by_arrival, traces


def select_fixed_count(results, domains, identities):
    rows = []
    competitive = [result for result, entry in zip(results, identities)
                   if entry["used_in_descriptive_competitive_average"]]
    for budget in FIXED_BUDGETS:
        gains = [benchmark_means(result[f"{RANDOM_PREFIX}{budget}"]["correct"].mean(axis=(0, 1)), domains).mean()
                 for result in competitive]
        rows.append({"budget": budget, "competitive_mean_validation_macro_accuracy": float(np.mean(gains))})
    chosen = max(rows, key=lambda row: (row["competitive_mean_validation_macro_accuracy"], -row["budget"]))
    return chosen["budget"], rows


def summarize(outcomes, arrivals, domains, names, identity):
    summary, by_benchmark = [], []
    truth_incumbent = outcomes[INCUMBENT_ONLY]["correct"][0]
    truth_newcomer = outcomes[NEWCOMER_ONLY]["correct"][0]
    incumbent_macro = benchmark_means(truth_incumbent, domains).mean(axis=1)
    newcomer_macro = benchmark_means(truth_newcomer, domains).mean(axis=1)
    hindsight_fixed_macro = np.maximum(incumbent_macro, newcomer_macro).mean()
    hindsight_fixed_micro = np.maximum(truth_incumbent.mean(axis=1), truth_newcomer.mean(axis=1)).mean()
    for method, result in outcomes.items():
        values = result["correct"]
        accuracy = benchmark_means(values.mean(axis=(0, 1), dtype=np.float64), domains)
        oracle = benchmark_means(result["oracle"].mean(axis=0, dtype=np.float64), domains)
        measured = [row for row in arrivals if row["method"] == method]
        counts = [row["probe_attempts"] for row in measured]
        runtime = [row["local_cpu_elapsed_seconds"] for row in measured if row["local_cpu_elapsed_seconds"] is not None]
        rescues, harms = int(result["rescue"].sum()), int(result["harm"].sum())
        opportunities = int(result["opportunity"].sum()) * values.shape[0]
        summary.append({**identity, "method": method, "n_seeds": values.shape[0], "arrivals": len(measured),
                        "n_queries": len(domains), "macro_accuracy": accuracy.mean(),
                        "micro_accuracy": values.mean(dtype=np.float64), "macro_query_oracle": oracle.mean(),
                        "micro_query_oracle": result["oracle"].mean(dtype=np.float64),
                        "macro_hindsight_best_fixed_identity": hindsight_fixed_macro,
                        "micro_hindsight_best_fixed_identity": hindsight_fixed_micro,
                        "macro_gain_over_incumbent": accuracy.mean() - incumbent_macro.mean(),
                        "macro_gain_over_always_newcomer": accuracy.mean() - newcomer_macro.mean(),
                        "macro_gain_over_same_usage_shuffle": benchmark_means(
                            (values - result["shuffled"]).mean(axis=(0, 1)), domains).mean(),
                        "newcomer_selection_share": result["selected"].mean(),
                        "rescues": rescues, "harms": harms, "net_correct": rescues - harms,
                        "opportunities": opportunities, "opportunity_capture": rescues / opportunities if opportunities else None,
                        "macro_oracle_gain_over_incumbent": oracle.mean() - incumbent_macro.mean(),
                        "macro_oracle_gain_over_always_newcomer": oracle.mean() - newcomer_macro.mean(),
                        "macro_oracle_gain_over_hindsight_best_fixed_identity": oracle.mean() - hindsight_fixed_macro,
                        "total_probe_attempts": sum(counts), "mean_probe_attempts": float(np.mean(counts)),
                        "median_probe_attempts": float(np.median(counts)), "min_probe_attempts": min(counts),
                        "max_probe_attempts": max(counts),
                        "mean_local_cpu_elapsed_seconds": float(np.mean(runtime)) if runtime else None,
                        "stops_cap": sum(row["stop_reason"] == "cap" for row in measured),
                        "stops_weak": sum(row["stop_reason"] == "weak_on_reference" for row in measured),
                        "stops_sufficient": sum(row["stop_reason"] == "sufficient_reference_evidence" for row in measured)})
        for i, name in enumerate(names):
            by_benchmark.append({**identity, "method": method, "benchmark": name,
                                 "n_queries": int((domains == i).sum()), "accuracy": accuracy[i], "oracle": oracle[i]})
    return summary, by_benchmark


def paired(outcomes, domains, identity):
    comparisons = [(EXACT_STOP, method) for method in (RANDOM_STOP, EXACT_CAP, RANDOM_CAP, SAME_K, VALIDATED_K,
                                                        PROBE_FIXED, INCUMBENT_ONLY, NEWCOMER_ONLY)]
    comparisons.extend([(EXACT_CAP, RANDOM_CAP), (RANDOM_STOP, RANDOM_CAP),
                        (RANDOM_STOP, VALIDATED_K), (RANDOM_STOP, INCUMBENT_ONLY)])
    rows = []
    for method, comparator in comparisons:
        left = outcomes[method]["correct"].mean(axis=1, dtype=np.float64)
        right = outcomes[comparator]["correct"].mean(axis=1, dtype=np.float64)
        result = paired_queries(left, right, domains)
        rows.append({**identity, "method": method, "comparator": comparator, **result})
    return rows


def run(args):
    dataset, split, config, _, _, _ = input_context(args.run)
    selection = read_json(args.sequential_selection)
    learning = read_json(args.learning_selection)
    margin, tolerance = selection["admission_margin"], selection["evpi_tolerance"]
    if (margin, tolerance) != (0.025, 0.0003):
        raise ValueError("This comparison uses the original frozen margin 0.025 and EVPI 0.0003.")
    reference, candidates = acquisition_banks(split.train_items, dataset.benchmark_ids)
    identified = anchors(dataset, split)
    identities = [identity for _, identity in identified]
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    if args.phase == "validation":
        rows, target = split.val_models, split.val_items
    else:
        rows, target = split.test_models, split.test_items
    domains = dataset.benchmark_ids[target]
    protocol = {"anchors": identities, "anchor_rule": "equal-benchmark macro scores from all training support for all 62 established models",
                "margin": margin, "evpi_tolerance": tolerance, "seed_order": config["seeds"], "checkpoint_arm": B,
                "benchmark_names": list(dataset.benchmark_names),
                "validation_newcomer_ids": dataset.model_ids[split.val_models].tolist(),
                "test_newcomer_ids": dataset.model_ids[split.test_models].tolist(),
                "validation_queries": len(split.val_items), "test_queries": len(split.test_items),
                "warmup": WARMUP, "cap": CAP, "fixed_counts": list(FIXED_BUDGETS),
                "reference_prompt_ids": dataset.prompt_ids[reference].tolist(),
                "candidate_prompt_ids": dataset.prompt_ids[candidates].tolist(),
                "reference_and_probe_policy": "existing disjoint support-only banks, all outcomes on held-out query prompts",
                "fixed_count_selection": "max equally anchor-averaged macro in validation (q50/q75/q90), ties fewer",
                "compatibility": "same legacy B prior and full-support incumbent refit for all arms",
                "matched_count_random": "retrospective comparison at the realized count of the exact stopping arm",
                "timing": "local CPU elapsed selection and fit wall time; no LLM calls",
                "interval": "paired queries stratified by benchmark, conditional on fixed rosters, seeds and support plans"}
    saved = args.output / "validation_selection.json"
    if args.phase == "test":
        frozen = read_json(saved)
        if frozen["protocol"] != protocol:
            raise ValueError("Test protocol differs from the validation run.")
        selected_count = frozen["fixed_count"]
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes, arrivals, traces = [], [], []
    for anchor, identity in identified:
        result, trial_rows, paths = evaluate_anchor(dataset, split, config, learning, anchor, identity, rows, target,
                                                   reference, candidates, options, margin, tolerance, args)
        outcomes.append(result)
        arrivals.append(trial_rows)
        traces.extend(paths)
    if args.phase == "validation":
        selected_count, tuning = select_fixed_count(outcomes, domains, identities)
        atomic_csv(args.output / "validation_random_fixed_count_grid.csv", tuning)
        atomic_json(saved, {"protocol": protocol, "fixed_count": selected_count})
    summary, benchmark_rows, paired_rows = [], [], []
    chosen = f"{RANDOM_PREFIX}{selected_count}"
    for result, trials, identity in zip(outcomes, arrivals, identities):
        result[VALIDATED_K] = result[chosen]
        trials.extend({**row, "method": VALIDATED_K} for row in trials.copy() if row["method"] == chosen)
        metrics, benchmarks = summarize(result, trials, domains, dataset.benchmark_names, identity)
        summary.extend(metrics)
        benchmark_rows.extend(benchmarks)
        paired_rows.extend(paired(result, domains, identity))
    selected = [result for result, identity in zip(outcomes, identities)
                if identity["used_in_descriptive_competitive_average"]]
    average = {key: {"correct": np.concatenate([result[key]["correct"] for result in selected], axis=1)}
               for key in selected[0]}
    paired_rows.extend(paired(average, domains, {"model_name": "competitive_anchor_average"}))
    for suffix, records in (("summary", summary), ("by_benchmark", benchmark_rows), ("paired", paired_rows),
                            ("by_arrival", [row for rows in arrivals for row in rows])):
        atomic_csv(args.output / f"{args.phase}_{suffix}.csv", records)
    args.trace_output.mkdir(parents=True, exist_ok=True)
    path = args.trace_output / f"{args.phase}_probe_trace.jsonl"
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in traces), encoding="utf-8")
    print(f"Saved {args.phase}; validation-selected random fixed count = {selected_count}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("validation", "test"))
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--learning-selection", type=Path, default=ROOT / "reports/routing/learning/selection.json")
    parser.add_argument("--sequential-selection", type=Path, default=ROOT / "reports/routing/sequential/validation_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/competitive_onboarding")
    parser.add_argument("--trace-output", type=Path, default=ROOT / "artifacts/competitive_onboarding")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
