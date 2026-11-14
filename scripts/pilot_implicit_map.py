"""Test whether differentiating through onboarding improves validation routing."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from route_query import ROOT, choose_models, interaction_scores
from coldstart.adaptation import adapt_many, sigmoid
from coldstart.models import load_decoder, save_decoder
from coldstart.probes import select_probe_order
from coldstart.splits import dataset_fingerprint
from pilot_routing_learning import B, REGRET_WEIGHT, TEMPERATURE, input_context, latent_prior
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries
from run_routing import benchmark_means, pool_outcomes


BUDGETS = (100, 200)
STRENGTHS = (0.3, 1.0, 3.0)
UPDATES = 1000
SELECTION_INTERVAL = 50
LEARNING_RATE = 1e-4
MODELS_PER_EPISODE = 13
OUTER_QUERIES_PER_BENCHMARK = 50
REFERENCE_PER_BENCHMARK = 64
REFERENCE_SEED = 314159
M0 = "M0_detached_map"
M1 = "M1_implicit_map"
FROZEN = "B_validation_selected_prior"
SUPPORT_FIXED = "support_best_fixed"
DOMAIN_FIXED = "support_best_per_benchmark"


@torch.no_grad()
def frozen_hidden(decoder, vectors):
    vectors = torch.as_tensor(vectors, dtype=torch.float32)
    return decoder.encoder((vectors - decoder.input_mean) / decoder.input_scale)


def features(decoder, hidden, domains):
    """Read the same two low-rank heads used during normal decoder inference."""
    offsets = decoder.difficulty_head(hidden).squeeze(-1) + decoder.benchmark_bias[domains]
    design = torch.tanh(decoder.interaction_head(hidden)) / math.sqrt(decoder.config.rank)
    return offsets.double(), design.double()


def adapt(offsets, design, labels, prior, options):
    offsets = offsets.detach().numpy()
    design = design.detach().numpy()
    fits = adapt_many(design, offsets, labels, *prior, **options)
    latents = torch.from_numpy(np.stack([fit.latent for fit in fits]))
    covariances = torch.from_numpy(np.stack([fit.covariance for fit in fits]))
    return latents, covariances


def routed_loss(decoder, hidden, domains, labels, episode, prior, options, implicit):
    rows, probes, target = episode
    responses = labels[np.ix_(rows, probes)]
    offsets, design = features(decoder, hidden[probes], domains[probes])
    latents, covariances = adapt(offsets, design, responses, prior, options)
    if implicit:
        response_targets = torch.as_tensor(responses, dtype=latents.dtype)
        fitted = (latents @ design.T + offsets).sigmoid()
        stationarity = (fitted - response_targets) @ design
        # Identical forward MAP point. The only new backward path is
        # -H^-1 times the derivative of the support stationarity equation.
        change = stationarity - stationarity.detach()
        latents = latents - (covariances @ change.unsqueeze(-1)).squeeze(-1)

    offsets, design = features(decoder, hidden[target], domains[target])
    interactions = latents @ design.T
    truth = torch.as_tensor(labels[np.ix_(rows, target)], dtype=latents.dtype)
    cross_entropy = F.binary_cross_entropy_with_logits(interactions + offsets, truth)
    choice_probability = (interactions / TEMPERATURE).softmax(dim=0)
    regret = (truth.amax(dim=0) - (choice_probability * truth).sum(dim=0)).mean()
    return cross_entropy + REGRET_WEIGHT * regret


def draw_episode(rng, by_domain, n_models, update):
    longest_budget = max(BUDGETS)
    probes_per_domain = longest_budget // len(by_domain)
    paths = [rng.choice(questions, probes_per_domain + OUTER_QUERIES_PER_BENCHMARK, replace=False)
             for questions in by_domain]
    support = np.stack([path[:probes_per_domain] for path in paths], axis=1).ravel()
    target = np.concatenate([path[probes_per_domain:] for path in paths])
    model_rows = rng.choice(n_models, MODELS_PER_EPISODE, replace=False)
    return model_rows, support[:BUDGETS[update % len(BUDGETS)]], target


def optimized_parameters(decoder):
    return [*decoder.difficulty_head.parameters(), *decoder.interaction_head.parameters(), decoder.benchmark_bias]


def gradient_check(decoder, hidden, domains, labels, episode, prior, options):
    """Compare one full finite difference, re-solving MAP in both directions."""
    parameters = optimized_parameters(decoder)
    generator = torch.Generator().manual_seed(23)
    directions = [torch.randn(parameter.shape, generator=generator) for parameter in parameters]
    scale = torch.sqrt(sum(direction.square().sum() for direction in directions))
    directions = [direction / scale for direction in directions]
    gradients = torch.autograd.grad(routed_loss(decoder, hidden, domains, labels, episode,
                                                prior, options, implicit=True), parameters)
    analytic = float(sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions)))
    originals = [parameter.detach().clone() for parameter in parameters]
    epsilon = 3e-3
    values = []
    with torch.no_grad():
        for sign in (-1, 1):
            for parameter, original, direction in zip(parameters, originals, directions):
                parameter.copy_(original + sign * epsilon * direction)
            values.append(float(routed_loss(decoder, hidden, domains, labels, episode,
                                            prior, options, implicit=False)))
        for parameter, original in zip(parameters, originals):
            parameter.copy_(original)
    finite_difference = (values[1] - values[0]) / (2 * epsilon)
    if not np.isclose(analytic, finite_difference, rtol=5e-3, atol=2e-5):
        raise RuntimeError(f"Implicit MAP directional derivative mismatch: {analytic} vs {finite_difference}.")
    return {"epsilon": epsilon, "implicit_gradient": analytic, "finite_difference": finite_difference,
            "absolute_error": abs(analytic - finite_difference)}


def fits_on_validation(decoder, context, budget, prior_strength, options):
    support = context["probe_order"][:budget]
    offsets, design = features(decoder, context["hidden"][support], context["domains"][support])
    mean, base_precision = context["base_prior"]
    latents, _ = adapt(offsets, design, context["labels"][:, support],
                      (mean, prior_strength * base_precision), options)
    target = context["target"]
    target_offsets, target_design = features(decoder, context["hidden"][target], context["domains"][target])
    scores = interaction_scores(latents.numpy(), target_design.detach().numpy())
    answers = context["labels"][:, target]
    target_domains = context["domains"][target].numpy()
    predicted = sigmoid(scores + target_offsets.detach().numpy())
    score_mae = np.abs(benchmark_means(answers, target_domains) - benchmark_means(predicted, target_domains)).mean()
    return latents.numpy(), scores, float(score_mae)


@torch.no_grad()
def validation_score(decoder, context, options):
    _, scores, mae = fits_on_validation(decoder, context, BUDGETS[-1], STRENGTHS[0], options)
    truth = context["labels"][:, context["target"]]
    choices = choose_models(scores)
    return float(benchmark_means(truth[choices, np.arange(len(choices))],
                                 context["domains"][context["target"]].numpy()).mean()), mae


def fit_seed(checkpoint, training, context, config, seed, options, output):
    decoder, _ = load_decoder(checkpoint)
    decoder.requires_grad_(False)
    hidden = frozen_hidden(decoder, training.embeddings)
    domains = torch.as_tensor(training.benchmark_ids)
    by_domain = [np.flatnonzero(training.benchmark_ids == domain) for domain in np.unique(training.benchmark_ids)]
    rng = np.random.default_rng(seed)
    mean, precision = context["base_prior"]
    prior = mean, STRENGTHS[0] * precision
    models = {arm: copy.deepcopy(decoder) for arm in (M0, M1)}
    optimizers = {}
    for arm, model in models.items():
        parameters = optimized_parameters(model)
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizers[arm] = torch.optim.AdamW(parameters, lr=LEARNING_RATE,
                                            weight_decay=config["model"]["weight_decay"])
    history, best, finite_difference = [], {}, None
    for update in range(UPDATES + 1):
        if update:
            episode = draw_episode(rng, by_domain, training.n_models, update - 1)
            if update == 1 and seed == config["seeds"][0]:
                finite_difference = gradient_check(models[M1], hidden, domains, training.labels,
                                                   episode, prior, options)
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                loss = routed_loss(model, hidden, domains, training.labels, episode, prior, options, implicit=arm == M1)
                loss.backward()
                optimizer.step()
        if update % SELECTION_INTERVAL:
            continue
        for arm, model in models.items():
            accuracy, mae = validation_score(model, context, options)
            history.append({"arm": arm, "seed": seed, "update": update,
                            "validation_macro_accuracy": accuracy, "validation_macro_score_mae": mae})
            if arm not in best or accuracy > best[arm][0]:
                best[arm] = accuracy, update, copy.deepcopy(model.state_dict())
        if update in (250, 500, 750, 1000):
            print(f"seed {seed}, update {update}: M0 {100*best[M0][0]:.2f}%, M1 {100*best[M1][0]:.2f}%", flush=True)
    selection = []
    for arm, model in models.items():
        accuracy, update, state = best[arm]
        model.load_state_dict(state)
        entry = {"arm": arm, "seed": seed, "update": update, "validation_macro_accuracy": accuracy}
        save_decoder(output / "checkpoints" / f"{arm}_{seed}.pt", model,
                     {**entry, "training_model_ids": training.model_ids.tolist(),
                      "training_prompt_ids": training.prompt_ids.tolist(), "prior_strength": STRENGTHS[0],
                      "dataset_fingerprint": context["dataset_fingerprint"]})
        selection.append(entry)
    return models, history, selection, finite_difference


@torch.no_grad()
def compare_validation(models_by_seed, contexts, frozen_strength, dataset, options, output):
    all_outcomes, by_seed, by_benchmark, counts = {}, [], [], []
    for seed, (models, context) in models_by_seed.items():
        target = context["target"]
        truth = context["labels"][:, target]
        query_domains = context["domains"][target].numpy()
        reference = context["reference"]
        reference_domains = context["domains"][reference].numpy()
        for budget in BUDGETS:
            probes = context["probe_order"][:budget]
            revealed = context["labels"][:, probes]
            support_means = benchmark_means(revealed, context["domains"][probes].numpy())
            choices = {SUPPORT_FIXED: np.full(len(target), support_means.mean(axis=1).argmax()),
                       DOMAIN_FIXED: support_means.argmax(axis=0)[query_domains]}
            errors = {}
            for arm, decoder in models.items():
                strength = frozen_strength if arm == FROZEN else STRENGTHS[0]
                latents, scores, mae = fits_on_validation(decoder, context, budget, strength, options)
                reference_offsets, reference_design = features(decoder, context["hidden"][reference],
                                                                context["domains"][reference])
                reference_p = sigmoid(interaction_scores(latents, reference_design.numpy()) + reference_offsets.numpy())
                winner = int(benchmark_means(reference_p, reference_domains).mean(axis=1).argmax())
                choices.update({arm: choose_models(scores), arm + "_embedding_mean_fixed": np.full(len(target), winner)})
                errors[arm] = mae
            for name, selected in choices.items():
                result = pool_outcomes(selected[None, :], truth, query_domains)
                values = result["correct"][0]
                all_outcomes.setdefault((budget, name), []).append(values)
                identity = {"seed": seed, "budget_per_candidate": budget, "method": name, "n_queries": len(target),
                            "n_candidates": len(truth), "total_probe_responses": len(truth) * budget}
                per_domain = benchmark_means(values, query_domains)
                shuffle = benchmark_means(result["shuffled"][0], query_domains)
                by_seed.append({**identity, "macro_accuracy": per_domain.mean(), "micro_accuracy": values.mean(dtype=np.float64),
                                "macro_score_mae": errors.get(name), "macro_gain_over_shuffle": (per_domain - shuffle).mean()})
                for domain, label in enumerate(dataset.benchmark_names):
                    by_benchmark.append({**identity, "benchmark": label, "n_queries": int((query_domains == domain).sum()),
                                         "accuracy": per_domain[domain], "shuffle_accuracy": shuffle[domain]})
            candidate_y = truth[choices[M1], np.arange(len(target))]
            for name in (M0, FROZEN, SUPPORT_FIXED, DOMAIN_FIXED, M1 + "_embedding_mean_fixed"):
                reference_y = truth[choices[name], np.arange(len(target))]
                counts.append({"seed": seed, "budget_per_candidate": budget, "comparator": name,
                               "rescues": int((candidate_y > reference_y).sum()),
                               "harms": int((candidate_y < reference_y).sum()),
                               "oracle_opportunities": int((truth.max(axis=0) > reference_y).sum())})

    summary, paired = [], []
    for (budget, method), data in all_outcomes.items():
        values = np.stack(data)
        subset = [row for row in by_seed if row["budget_per_candidate"] == budget and row["method"] == method]
        maes = [row["macro_score_mae"] for row in subset if row["macro_score_mae"] is not None]
        summary.append({"budget_per_candidate": budget, "method": method, "n_queries": len(query_domains),
                        "n_candidates": len(truth), "total_probe_responses": len(truth) * budget,
                        "macro_accuracy": benchmark_means(values, query_domains).mean(),
                        "micro_accuracy": values.mean(dtype=np.float64),
                        "macro_score_mae": float(np.mean(maes)) if maes else None,
                        "macro_gain_over_shuffle": float(np.mean([row["macro_gain_over_shuffle"] for row in subset]))})
        if method != M1:
            comparison = paired_queries(np.stack(all_outcomes[(budget, M1)]), values, query_domains)
            paired.append({"budget_per_candidate": budget, "method": M1, "comparator": method, **comparison})
    for name, rows in (("summary", summary), ("paired", paired), ("by_seed", by_seed),
                       ("by_benchmark", by_benchmark), ("rescues_harms", counts)):
        atomic_csv(output / f"validation_{name}.csv", rows)
    return paired


def run(args):
    dataset, split, config, _, _, _ = input_context(args.run)
    selection = json.loads(args.learning_selection.read_text(encoding="utf-8"))
    if selection["protocol"]["prior_strength"] != STRENGTHS[0]:
        raise ValueError("The implicit pilot expects the historical B prior strength.")
    if selection["protocol"]["training_model_ids"] != dataset.model_ids[split.train_models].tolist():
        raise ValueError("The initial checkpoint belongs to a different training roster.")
    training = dataset.subset(split.train_models, split.train_items)
    target = split.val_items
    orders = {seed: select_probe_order(dataset.embeddings, dataset.benchmark_ids, split.train_items, BUDGETS[-1],
                                       "balanced_random", seed) for seed in config["seeds"]}
    remaining = np.setdiff1d(split.train_items, np.concatenate(list(orders.values())))
    reference = select_probe_order(dataset.embeddings, dataset.benchmark_ids, remaining,
                                   REFERENCE_PER_BENCHMARK * len(dataset.benchmark_names),
                                   "balanced_random", REFERENCE_SEED)
    allowed = np.r_[split.train_items, target]
    local = {int(question): position for position, question in enumerate(allowed)}
    def local_positions(indices):
        return np.array([local[int(index)] for index in indices])

    val_labels = dataset.labels[np.ix_(split.val_models, allowed)]
    val_domains = torch.as_tensor(dataset.benchmark_ids[allowed])
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    contexts, models, history, checkpoints, prior_comparisons = {}, {}, [], [], []
    for seed in config["seeds"]:
        path = args.checkpoints / f"{B}_{seed}.pt"
        decoder, metadata = load_decoder(path)
        if not any(all(metadata[key] == value for key, value in row.items()) for row in selection["checkpoints"]):
            raise ValueError(f"Checkpoint differs from the routing-learning selection: {path}")
        contexts[seed] = {"hidden": frozen_hidden(decoder, dataset.embeddings[allowed]),
                          "labels": val_labels, "domains": val_domains,
                          "probe_order": local_positions(orders[seed]), "target": local_positions(target),
                          "reference": local_positions(reference), "base_prior": latent_prior(decoder, config, 1.0),
                          "dataset_fingerprint": dataset_fingerprint(dataset)}
        for strength in STRENGTHS:
            _, scores, mae = fits_on_validation(decoder, contexts[seed], BUDGETS[-1], strength, options)
            accuracy = benchmark_means(dataset.labels[split.val_models[choose_models(scores)], target],
                                       dataset.benchmark_ids[target]).mean()
            prior_comparisons.append({"seed": seed, "prior_strength": strength, "macro_accuracy": accuracy,
                                      "macro_score_mae": mae})
        fitted, records, selected, check = fit_seed(path, training, contexts[seed], config, seed, options,
                                                   args.checkpoint_output)
        models[seed] = ({FROZEN: decoder, **fitted}, contexts[seed])
        history.extend(records)
        checkpoints.extend(selected)
        if check is not None:
            atomic_json(args.output / "gradient_check.json", check)
    means = {strength: np.mean([row["macro_accuracy"] for row in prior_comparisons if row["prior_strength"] == strength])
             for strength in STRENGTHS}
    best_strength = max(STRENGTHS, key=lambda strength: (means[strength], strength))
    paired = compare_validation(models, contexts, best_strength, dataset, options, args.output)
    primary = [row for row in paired if row["budget_per_candidate"] == BUDGETS[-1]]
    strongest_routed = min((row for row in primary if row["comparator"] in (M0, FROZEN)),
                           key=lambda row: row["macro_gain"])
    strongest_fixed = min((row for row in primary if row["comparator"] not in (M0, FROZEN)),
                          key=lambda row: row["macro_gain"])
    gate_met = all(row["macro_gain"] >= 0.01 and row["positive_seed_count"] >= 2
                   for row in (strongest_routed, strongest_fixed))
    atomic_csv(args.output / "validation_history.csv", history)
    atomic_csv(args.output / "validation_baseline_prior_grid.csv", prior_comparisons)
    atomic_json(args.output / "selection.json", {
        "phase": "validation_only", "upstream_checkpoint": B, "seeds": config["seeds"],
        "train_model_ids": dataset.model_ids[split.train_models], "validation_model_ids": dataset.model_ids[split.val_models],
        "validation_query_count": len(target), "validation_query_prompt_ids": dataset.prompt_ids[target],
        "episode": {"training_models": MODELS_PER_EPISODE, "alternating_probe_budgets": BUDGETS,
                    "outer_queries_per_benchmark": OUTER_QUERIES_PER_BENCHMARK,
                    "partition": "balanced disjoint support and outer queries from historical training groups only"},
        "optimization": {"updates": UPDATES, "learning_rate": LEARNING_RATE, "evaluate_every": SELECTION_INTERVAL,
                         "frozen": "input normalization, dropout-free hidden projector, training latents and empirical Gaussian prior",
                         "trainable": "question interaction head, difficulty head and benchmark intercepts",
                         "training_prior_strength": STRENGTHS[0], "regret_weight": REGRET_WEIGHT, "softmax_temperature": TEMPERATURE},
        "checkpoints": checkpoints, "B_prior_grid": STRENGTHS, "B_selected_prior_strength": best_strength,
        "selection_rule": "max validation macro routing at 200; earlier update wins epoch ties, stronger prior wins B ties",
        "probe_prompt_ids": {str(seed): dataset.prompt_ids[order] for seed, order in orders.items()},
        "fixed_embedding_reference_prompt_ids": dataset.prompt_ids[reference],
        "prospective_gate": {"macro_gain_minimum": 0.01, "positive_seeds_minimum": 2,
                             "strongest_fixed": strongest_fixed["comparator"],
                             "strongest_routed": strongest_routed["comparator"],
                             "met": gate_met},
    })
    print(f"Completed validation-only implicit MAP comparison. Gate met: {gate_met}.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--learning-selection", type=Path, default=ROOT / "reports/routing/learning/selection.json")
    parser.add_argument("--checkpoint-output", type=Path, default=ROOT / "artifacts/implicit_map")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/implicit_map")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
