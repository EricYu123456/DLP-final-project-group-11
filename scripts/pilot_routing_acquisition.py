"""Compare routing-aware and generic probe acquisition on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import torch

from run_routing import (ADD_ONE, ROOT, VARIANT, add_one_outcomes, benchmark_means,
                         newcomer_choices, summarize_outcomes)
from coldstart.acquisition import benchmark_score_gradients, score_variance_reduction
from coldstart.adaptation import adapt_map, sigmoid
from coldstart.probes import select_probe_order
from embed_new_model import load_run
from route_query import interaction_scores
from run_experiment import atomic_csv, atomic_json, load_features


RANDOM = "balanced_random"
FISHER = "score_fisher"
D_OPTIMAL = "d_optimal"
ROUTING = "routing_expected_improvement"
POLICIES = (RANDOM, FISHER, D_OPTIMAL, ROUTING)
SEED = 11
BANK_SEED = 314159
BUDGETS = (50, 100)
WARMUP = 10
REFERENCE_PER_BENCHMARK = 32
CANDIDATES_PER_BENCHMARK = 100


def acquisition_banks(support: np.ndarray, domains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint, equally weighted reference and candidate banks within the support pool."""
    reference, candidates = [], []
    for domain in np.unique(domains[support]):
        rng = np.random.default_rng(np.random.SeedSequence([BANK_SEED, int(domain)]))
        order = rng.permutation(support[domains[support] == domain])
        reference.append(order[:REFERENCE_PER_BENCHMARK])
        candidates.append(order[REFERENCE_PER_BENCHMARK:REFERENCE_PER_BENCHMARK + CANDIDATES_PER_BENCHMARK])
    return np.concatenate(reference), np.concatenate(candidates)


def expected_switch_value(gap: np.ndarray, standard_deviation: np.ndarray) -> np.ndarray:
    """Expected gain from moving a Gaussian score difference across zero."""
    distance = np.abs(gap)
    normalized = np.divide(distance, standard_deviation,
                           out=np.full_like(standard_deviation, np.inf), where=standard_deviation > 0)
    density = np.exp(-0.5 * normalized ** 2) / np.sqrt(2 * np.pi)
    return np.maximum(standard_deviation * density - distance * ndtr(-normalized), 0)


def acquisition_scores(policy, design, offsets, reference_design, reference_offsets,
                       reference_domains, incumbent_probability, margin, fit):
    if policy == FISHER:
        gradients = benchmark_score_gradients(reference_design, reference_offsets,
                                              reference_domains, fit.latent)
        return score_variance_reduction(design, offsets, fit.latent, fit.covariance, gradients)

    probability = sigmoid(offsets + design @ fit.latent)
    weight = probability * (1 - probability)
    projected = design @ fit.covariance
    information = weight * np.einsum("ij,ij->i", projected, design)
    if policy == D_OPTIMAL:
        return np.log1p(information)

    reference_probability = sigmoid(reference_offsets + reference_design @ fit.latent)
    slope = reference_probability * (1 - reference_probability)
    gradients = slope[:, None] * reference_design
    standard_deviation = np.sqrt(weight / (1 + information))[:, None] * np.abs(projected @ gradients.T)
    gap = reference_probability - incumbent_probability - margin
    return expected_switch_value(gap, standard_deviation).mean(axis=1)


def acquire(policy, order, domains, design, offsets, observe, reference, incumbent_probability,
            margin, prior_mean, prior_precision, solver_options):
    selected = order[:WARMUP].tolist()
    observed = [observe(candidate) for candidate in selected]
    available = np.ones(len(order), dtype=bool)
    available[selected] = False
    fit = adapt_map(design[selected], offsets[selected], np.asarray(observed), prior_mean,
                    prior_precision, **solver_options)
    latents = {}
    for step in range(WARMUP, max(BUDGETS)):
        domain = domains[order[step]]
        eligible = order[available[order] & (domains[order] == domain)]
        if policy == RANDOM:
            chosen = eligible[0]
        else:
            values = acquisition_scores(policy, design[eligible], offsets[eligible],
                                         *reference, incumbent_probability, margin, fit)
            chosen = eligible[values.argmax()]
        selected.append(int(chosen))
        observed.append(observe(chosen))
        available[chosen] = False
        if policy != RANDOM or step + 1 in BUDGETS:
            fit = adapt_map(design[selected], offsets[selected], np.asarray(observed), prior_mean,
                            prior_precision, initial=fit.latent, **solver_options)
        if step + 1 in BUDGETS:
            latents[step + 1] = fit.latent
    return latents, selected


def write_paired(outcomes, domains, output):
    rows = []
    for budget in BUDGETS:
        for rule in ("fixed_margin", "argmax"):
            current = outcomes[(ADD_ONE, budget, f"{ROUTING}/{rule}")]["correct"]
            for policy in (RANDOM, FISHER, D_OPTIMAL):
                reference = outcomes[(ADD_ONE, budget, f"{policy}/{rule}")]["correct"]
                difference = current - reference
                rows.append({"budget": budget, "routing_rule": rule, "reference_policy": policy,
                             "micro_gain": float(difference.mean(dtype=np.float64)),
                             "macro_gain": float(benchmark_means(difference, domains).mean()),
                             "gained_correct_per_trial": float((difference > 0).sum(axis=-1).mean()),
                             "lost_correct_per_trial": float((difference < 0).sum(axis=-1).mean())})
    atomic_csv(output / "paired.csv", rows)


def run(run_path: Path, validation: Path, output: Path) -> None:
    dataset, split, config, lock, seed = load_run(run_path, VARIANT, SEED)
    frozen = json.loads(validation.read_text(encoding="utf-8"))
    incumbent_id = frozen["protocol"]["incumbent_model_id"]
    margin = frozen["margins"]["compact_embedding/100"]["minimum_gain"]
    incumbent = int(np.flatnonzero(dataset.model_ids == incumbent_id)[0])
    reference, candidates = acquisition_banks(split.train_items, dataset.benchmark_ids)
    feature = load_features(dataset, config, run_path, [VARIANT], seed, split.train_items)[VARIANT]
    decoder = feature["decoder"]
    design, offsets = feature["design"].astype(np.float64), feature["offsets"].astype(np.float64)
    incumbent_index = feature["metadata"]["training_model_ids"].index(incumbent_id)
    incumbent_latent = decoder.model_latents.weight[incumbent_index].detach().cpu().numpy()
    incumbent_reference = sigmoid(offsets[reference] + design[reference] @ incumbent_latent)
    reference_data = (design[reference], offsets[reference], dataset.benchmark_ids[reference])
    domains = dataset.benchmark_ids[candidates]
    order = select_probe_order(design[candidates], domains, np.arange(len(candidates)),
                               len(candidates), RANDOM, seed)
    prior_precision = lock["selection"]["prior_strengths"][VARIANT] * feature["precision"]
    solver_options = {name: config["adaptation"][name] for name in ("max_iterations", "tolerance")}
    latents = {(budget, policy): [] for budget in BUDGETS for policy in POLICIES}
    probe_orders = {}
    for model in split.val_models:
        model_id = str(dataset.model_ids[model])
        probe_orders[model_id] = {}

        def observe(candidate_local):
            return dataset.labels[model, candidates[candidate_local]]

        for policy in POLICIES:
            fits, selected = acquire(policy, order, domains, design[candidates], offsets[candidates],
                                     observe, reference_data, incumbent_reference, margin,
                                     feature["mean"], prior_precision, solver_options)
            probe_orders[model_id][policy] = dataset.prompt_ids[candidates[selected]].tolist()
            for budget, latent in fits.items():
                latents[(budget, policy)].append(latent)

    # Validation query text and labels enter only after all probe decisions and fits.
    target = split.val_items
    target_domains = dataset.benchmark_ids[target]
    target_offset, target_design = decoder.export_features(dataset.embeddings[target], target_domains)
    truth = dataset.labels[np.ix_(np.r_[incumbent, split.val_models], target)]
    outcomes, candidates_csv = {}, []
    for (budget, policy), candidate_latents in latents.items():
        logits = interaction_scores(np.vstack([incumbent_latent, *candidate_latents]), target_design)
        probabilities = sigmoid(logits + target_offset)[None, ...]
        for rule, selected_margin in (("fixed_margin", margin), ("argmax", 0.0)):
            chosen = newcomer_choices(probabilities, selected_margin)
            result = add_one_outcomes(chosen, truth, target_domains)
            outcomes[(ADD_ONE, budget, f"{policy}/{rule}")] = result
            for position, model in enumerate(split.val_models):
                rescue = int(result["rescue"][0, position].sum())
                harm = int(result["harm"][0, position].sum())
                opportunity = int(result["opportunity"][position].sum())
                candidates_csv.append({"budget": budget, "acquisition_policy": policy, "routing_rule": rule,
                    "model_id": int(dataset.model_ids[model]), "newcomer_selected": int(chosen[0, position].sum()),
                    "opportunities": opportunity, "rescues": rescue, "harms": harm,
                    "net_correct": rescue - harm, "capture": rescue / opportunity if opportunity else None})
    summary, _ = summarize_outcomes(outcomes, target_domains, dataset.benchmark_names, len(split.val_models), [seed])
    atomic_csv(output / "summary.csv", summary)
    atomic_csv(output / "by_candidate.csv", candidates_csv)
    write_paired(outcomes, target_domains, output)
    atomic_json(output / "protocol.json", {
        "phase": "validation_only", "decoder": VARIANT, "seed": seed, "bank_seed": BANK_SEED,
        "incumbent_model_id": incumbent_id, "validation_model_ids": dataset.model_ids[split.val_models],
        "validation_queries": len(target), "budgets": BUDGETS, "warmup": WARMUP,
        "probes_per_benchmark": "Equal round-robin quotas in every policy and budget",
        "acquisition_margin": margin, "fixed_routing_margin": margin,
        "reference_prompt_ids": dataset.prompt_ids[reference], "candidate_prompt_ids": dataset.prompt_ids[candidates],
        "warmup_prompt_ids": dataset.prompt_ids[candidates[order[:WARMUP]]],
        "balanced_random_prompt_ids": dataset.prompt_ids[candidates[order[:max(BUDGETS)]]],
        "probe_orders": probe_orders,
    })
    print(f"Saved validation-only acquisition pilot to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts" / "main_v2")
    parser.add_argument("--validation-selection", type=Path,
                        default=ROOT / "reports" / "routing" / "validation_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "routing" / "acquisition_pilot")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args.run, args.validation_selection, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
