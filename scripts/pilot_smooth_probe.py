"""Compare smooth decision-boundary acquisition with frozen validation controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from route_query import ROOT
from pilot_competitive_onboarding import EXACT_CAP, EXACT_STOP, RANDOM_STOP, VALIDATED_K, run_episode
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries
from run_sequential_routing import (CAP, SCREENED_PER_STEP, WARMUP, input_context, probabilities,
                                    read_json, seed_episodes, stop_indices, trajectory)


BUDGETS = (20, 50, 100, CAP)
TOLERANCES = (0.0003, 0.0006, 0.001, 0.002, 0.004)
DETAIL_TOLERANCE = 0.002
DETAIL_CONDITION = "evpi_0.002"


def smooth_path(episode, margin, options, *, tolerance=0.0, cap=CAP):
    seed = np.random.SeedSequence([episode["seed"], episode["model_id"]])
    latents, trace, _, _ = trajectory(episode["random_order"], episode["probe_domains"],
                                      episode["probe_design"], episode["probe_offsets"], episode["observe"],
                                      episode["reference"], episode["reference_incumbent"], episode["prior"],
                                      margin, options, seed, tolerance, selection="smooth", cap=cap)
    return latents, np.asarray(trace)


def metrics(episode, path, budget, margin, truth, weights):
    latents, trace = path
    index = budget - WARMUP
    use_newcomer = probabilities(latents[index], episode) - episode["incumbent"] > margin
    incumbent, newcomer = truth
    outcomes = np.where(use_newcomer, newcomer, incumbent)
    opportunity = newcomer > incumbent
    return {"macro": float(outcomes @ weights), "micro": outcomes.mean(dtype=np.float64),
            "probe_attempts": budget, "local_cpu_elapsed_seconds": trace[index, 2],
            "newcomer_selected": int(use_newcomer.sum()), "rescues": int((use_newcomer & opportunity).sum()),
            "harms": int((use_newcomer & (newcomer < incumbent)).sum()), "opportunities": int(opportunity.sum())}, outcomes


def run(args):
    dataset, split, config, _, _, _ = input_context(args.run)
    original = read_json(args.competitive / "validation_selection.json")
    protocol = original["protocol"]
    learning = read_json(args.learning_selection)
    margin, tolerance = protocol["margin"], protocol["evpi_tolerance"]
    if tolerance != TOLERANCES[0]:
        raise ValueError("Frozen threshold differs from the first tolerance in the predeclared grid.")
    with (args.competitive / "validation_by_arrival.csv").open(newline="", encoding="utf-8") as handle:
        historical = {(int(row["model_id"]), int(row["seed"]), int(row["newcomer_id"]), row["method"]): row
                      for row in csv.DictReader(handle)}
    fixed_count = original["fixed_count"]
    positions = dict(zip(dataset.prompt_ids, np.arange(dataset.n_questions)))
    reference = np.array([positions[prompt] for prompt in protocol["reference_prompt_ids"]])
    candidates = np.array([positions[prompt] for prompt in protocol["candidate_prompt_ids"]])
    target, newcomers = split.val_items, split.val_models
    domains = dataset.benchmark_ids[target]
    weights = 1 / (len(dataset.benchmark_names) * np.bincount(domains)[domains])
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    anchors = protocol["anchors"]
    competitive = [anchor["model_id"] for anchor in anchors if anchor["used_in_descriptive_competitive_average"]]
    aggregate, stats = defaultdict(list), defaultdict(list)
    grid = defaultdict(list)
    for anchor in anchors:
        incumbent = int(np.flatnonzero(dataset.model_ids == anchor["model_id"])[0])
        for seed in config["seeds"]:
            episodes = seed_episodes(dataset, split, config, learning, incumbent, newcomers, target,
                                     reference, candidates, seed, args)
            for model, episode in zip(newcomers, episodes):
                truth = dataset.labels[np.ix_([incumbent, model], target)]
                old_key = (anchor["model_id"], seed, episode["model_id"])
                _, random = run_episode(episode, margin, tolerance, options, "random", "validation")
                paths = {"smooth": smooth_path(episode, margin, options),
                         "random": (random[0], np.asarray(random[1]))}
                for method, path in paths.items():
                    evpi = path[1][:, 0]
                    stopped = {value: int(stop_indices(evpi, value)) + WARMUP for value in TOLERANCES}
                    own_stop = stopped[tolerance]
                    fixed = [(f"fixed_{budget}", budget) for budget in BUDGETS]
                    for condition, budget in [*fixed, ("frozen_own_stop", own_stop),
                                              (DETAIL_CONDITION, stopped[DETAIL_TOLERANCE])]:
                        row, outcomes = metrics(episode, path, budget, margin, truth, weights)
                        key = anchor["model_id"], method, condition
                        if method == "smooth" and condition in ("fixed_200", "frozen_own_stop"):
                            old = historical[(*old_key, EXACT_CAP if condition == "fixed_200" else EXACT_STOP)]
                            row["macro_gain_over_exact"] = row["macro"] - float(old["macro_accuracy"])
                            row["micro_gain_over_exact"] = row["micro"] - float(old["micro_accuracy"])
                            row["exact_probe_attempts"] = int(old["probe_attempts"])
                            row["exact_local_cpu_elapsed_seconds"] = float(old["local_cpu_elapsed_seconds"])
                        stats[key].append(row)
                        aggregate[(key, seed)].append(outcomes.astype(np.float64))
                    old_stop = historical[(*old_key, RANDOM_STOP)]
                    if method == "random" and int(old_stop["probe_attempts"]) != own_stop:
                        raise ValueError("Random stopping replay differs from the earlier validation.")
                    comparator = float(historical[(*old_key, VALIDATED_K)]["macro_accuracy"])
                    for value, budget in stopped.items():
                        row, _ = metrics(episode, path, budget, margin, truth, weights)
                        grid[method, value, anchor["model_id"]].append((budget, row["macro"] - comparator))
        print(f"Completed smooth-probe validation for incumbent {anchor['model_id']}.", flush=True)

    rows, contrasts, gates = [], [], []
    seeds = config["seeds"]

    def query_means(anchor, method, condition):
        return np.stack([np.mean(aggregate[((anchor, method, condition), seed)], axis=0) for seed in seeds])

    def compare(anchor, method, condition, comparator, comparator_condition):
        group = competitive if anchor == "competitive_average" else [anchor]
        left = np.mean([query_means(number, method, condition) for number in group], axis=0)
        right = np.mean([query_means(number, comparator, comparator_condition) for number in group], axis=0)
        return {"anchor_id": anchor, "selector": method, "condition": condition, "comparator": comparator,
                "comparator_condition": comparator_condition, **paired_queries(left, right, domains)}

    for (anchor, method, condition), records in stats.items():
        averages = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
        rows.append({"anchor_id": anchor, "selector": method, "condition": condition, "n_arrivals": len(records),
                     **averages, "median_probe_attempts": float(np.median([row["probe_attempts"] for row in records])),
                     "opportunity_capture": averages["rescues"] / averages["opportunities"]})
        if method == "smooth":
            contrasts.append(compare(anchor, method, condition, "random", condition))
    for anchor in [*[entry["model_id"] for entry in anchors], "competitive_average"]:
        for method in ("smooth", "random"):
            contrasts.append(compare(anchor, method, DETAIL_CONDITION, "random", f"fixed_{fixed_count}"))
    contrasts.append(compare("competitive_average", "smooth", DETAIL_CONDITION, "random", DETAIL_CONDITION))
    for method in ("smooth", "random"):
        for value in TOLERANCES:
            by_anchor = np.stack([np.mean(grid[method, value, anchor], axis=0) for anchor in competitive])
            mean_count, macro_gain = by_anchor.mean(axis=0)
            worst = by_anchor[:, 1].min()
            gates.append({"selector": method, "evpi": value, "mean_probe_attempts": mean_count,
                          "macro_gain_over_validation_best_fixed": macro_gain, "worst_anchor_macro_gain": worst,
                          **{f"anchor_{anchor}_macro_gain": by_anchor[i, 1] for i, anchor in enumerate(competitive)},
                          "gate_met": bool(mean_count <= 0.9 * fixed_count and macro_gain >= -0.001 and worst >= -0.0025)})
    for name, table in (("validation_summary", rows), ("validation_paired", contrasts), ("validation_stop_grid", gates)):
        atomic_csv(args.output / f"{name}.csv", table)
    atomic_json(args.output / "protocol.json", {
        "phase": "validation only", "anchors": anchors, "validation_queries": len(target), "margin": margin,
        "frozen_evpi_tolerance": tolerance, "fixed_budgets": BUDGETS, "tolerance_grid": TOLERANCES,
        "development_detail_evpi_tolerance": DETAIL_TOLERANCE,
        "shortlist": SCREENED_PER_STEP, "comparison_fixed_count": fixed_count,
        "minimum_probe_saving": 0.10, "maximum_macro_loss": 0.001, "maximum_one_anchor_loss": 0.0025,
        "gate_scope": "development only; unweighted q50/q75/q90 mean",
        "paired_query_interval": "conditional on arrivals, seeds and shared validation episodes",
        "rescue_opportunities": "incumbent wrong, newcomer correct on held-out validation query",
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--learning-selection", type=Path, default=ROOT / "reports/routing/learning/selection.json")
    parser.add_argument("--competitive", type=Path, default=ROOT / "reports/routing/competitive_onboarding")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/smooth_probe")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
