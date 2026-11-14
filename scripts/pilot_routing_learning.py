"""Compare learning and checkpoint objectives for cold-start query routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from run_routing import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, ROOT, VARIANT,
                         benchmark_means, pool_outcomes)
from coldstart.adaptation import adapt_many, sigmoid
from coldstart.data import load_dataset
from coldstart.models import create_decoder, load_decoder, save_decoder, select_device
from coldstart.probes import select_probe_order
from coldstart.splits import load_family_table, load_split
from evaluate_metadata_excluded import load_oracle
from route_query import choose_models, interaction_scores
from run_experiment import atomic_csv, atomic_json, decoder_configuration, training_configuration


A = "A_bce_score_selection"
B = "B_bce_routing_selection"
C = "C_bce_regret_routing_selection"
ARMS = (A, B, C)
ROUTER = "query_router"
EMBEDDING_FIXED = "embedding_mean_fixed"
SUPPORT_FIXED = "support_best_fixed"
BUDGETS = (100, 200)
PRIMARY_BUDGET = 200
ROSTER_SIZE = 13
REGRET_WEIGHT = 1.0
TEMPERATURE = 1.0
REFERENCE_PER_BENCHMARK = 64
REFERENCE_SEED = 314159


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def macro_weights(domains: np.ndarray) -> np.ndarray:
    counts = np.bincount(domains)
    return len(domains) / (np.count_nonzero(counts) * counts[domains])


def fit_trajectory(spec, settings, training, validation, evaluate_checkpoint, add_regret):
    """Run one fixed training schedule and retain checkpoints by validation objective."""
    decoder = create_decoder(spec, settings.seed).to(select_device(settings.device))
    device = next(decoder.parameters()).device
    vectors = torch.as_tensor(training.embeddings, device=device)
    domains = torch.as_tensor(training.benchmark_ids, device=device)
    labels = torch.as_tensor(training.labels, device=device)
    weights = torch.as_tensor(macro_weights(training.benchmark_ids), dtype=torch.float32, device=device)
    decoder.fit_input_normalizer(vectors)
    decoder.initialize_abilities(labels, domains)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    item_rng = np.random.default_rng(settings.seed)
    roster_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 13]))
    best = {}
    history = []
    for epoch in range(1, settings.epochs + 1):
        decoder.train()
        order = item_rng.permutation(training.n_questions)
        for start in range(0, len(order), settings.batch_questions):
            indices = torch.as_tensor(order[start:start + settings.batch_questions], device=device)
            targets, question_weights = labels[:, indices], weights[indices]
            offsets, design = decoder.item_features(vectors[indices], domains[indices])
            interactions = decoder.model_latents.weight @ design.T
            logits = interactions + offsets
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=0)
            loss = (bce * question_weights).mean()
            if add_regret:
                roster = roster_rng.choice(spec.num_models, size=ROSTER_SIZE, replace=False)
                roster = torch.as_tensor(roster, device=device)
                choice_probability = (interactions[roster] / TEMPERATURE).softmax(dim=0)
                realized = targets[roster]
                regret = realized.amax(dim=0) - (choice_probability * realized).sum(dim=0)
                loss = loss + REGRET_WEIGHT * (regret * question_weights).mean()
            penalty = settings.latent_l2 * decoder.model_latents.weight.square().sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss + penalty).backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

        if epoch != 1 and epoch % settings.selection_interval:
            continue
        macro_accuracy, macro_mae = evaluate_checkpoint(decoder, *validation)
        row = {"seed": settings.seed, "training_objective": "bce_plus_regret" if add_regret else "bce",
               "epoch": epoch, "macro_routing_accuracy": macro_accuracy, "macro_score_mae": macro_mae}
        history.append(row)
        criteria = {C: macro_accuracy} if add_regret else {A: -macro_mae, B: macro_accuracy}
        for arm, value in criteria.items():
            if arm not in best or value > best[arm]["selection_value"]:
                best[arm] = {**row, "selection_value": value,
                             "state": {key: parameter.detach().cpu().clone()
                                       for key, parameter in decoder.state_dict().items()}}
    return best, history


def weighted_map(design, offsets, responses, domains, mean, precision):
    """Fit a MAP latent with equal total likelihood weight per observed benchmark."""
    observed = np.isfinite(responses)
    design, offsets, responses = design[observed], offsets[observed], responses[observed]
    weights = macro_weights(domains[observed])

    def objective(latent):
        logits = offsets + design @ latent
        difference = latent - mean
        likelihood = weights @ np.logaddexp(0, (1.0 - 2.0 * responses) * logits)
        return float(likelihood + 0.5 * difference @ precision @ difference)

    latent = mean.copy()
    for _ in range(25):
        probabilities = sigmoid(offsets + design @ latent)
        gradient = design.T @ (weights * (probabilities - responses)) + precision @ (latent - mean)
        curvature = weights * probabilities * (1 - probabilities)
        hessian = design.T @ (curvature[:, None] * design) + precision
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step, ord=np.inf) <= 1e-7:
            return latent - step
        before = objective(latent)
        descent = float(gradient @ step)
        step_size = 1.0
        for _ in range(30):
            candidate = latent - step_size * step
            if objective(candidate) <= before - 1e-4 * step_size * descent:
                latent = candidate
                break
            step_size *= 0.5
        else:
            raise RuntimeError("No descent step for equal-benchmark full-support MAP.")
    raise RuntimeError("Equal-benchmark full-support MAP reached its iteration limit.")


def model_latents(features, responses, domains, prior, config, balanced=False):
    offsets, design = features
    mean, precision = prior
    if balanced:
        design, offsets = design.astype(np.float64), offsets.astype(np.float64)
        return np.stack([weighted_map(design, offsets, row, domains, mean, precision) for row in responses])
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    fits = adapt_many(design, offsets, responses, mean, precision, **options)
    return np.stack([fit.latent for fit in fits])


def latent_prior(decoder, config, prior_strength):
    mean, precision = decoder.latent_prior(
        variance_floor=config["model"]["prior_scale_floor"],
        variance_shrinkage=config["model"]["prior_variance_shrinkage"])
    return mean, precision * prior_strength


def checkpoint_eval(decoder, support, target, config, prior_strength):
    prior = latent_prior(decoder, config, prior_strength)
    features = decoder.export_features(support.embeddings, support.benchmark_ids)
    latents = model_latents(features, support.labels, support.benchmark_ids, prior, config)
    offsets, design = decoder.export_features(target.embeddings, target.benchmark_ids)
    scores = interaction_scores(latents, design)
    routed = target.labels[choose_models(scores), np.arange(target.n_questions)]
    probabilities = sigmoid(scores + offsets)
    actual = benchmark_means(target.labels, target.benchmark_ids)
    predicted = benchmark_means(probabilities, target.benchmark_ids)
    return float(benchmark_means(routed, target.benchmark_ids).mean()), float(np.abs(actual - predicted).mean())


def train_all(dataset, split, config, orders, prior_strength, output, device):
    training = dataset.subset(split.train_models, split.train_items)
    target = dataset.subset(split.val_models, split.val_items)
    spec = decoder_configuration(dataset, split, config, VARIANT)
    rows, selection = [], []
    for seed in config["seeds"]:
        support = dataset.subset(split.val_models, orders[seed][:PRIMARY_BUDGET])
        settings = training_configuration(config, seed, str(device))
        for add_regret in (False, True):
            best, history = fit_trajectory(spec, settings, training, (support, target, config, prior_strength),
                                           checkpoint_eval, add_regret)
            rows.extend(history)
            for arm, checkpoint in best.items():
                decoder = create_decoder(spec, seed)
                decoder.load_state_dict(checkpoint.pop("state"))
                checkpoint["arm"] = arm
                save_decoder(output / "checkpoints" / f"{arm}_{seed}.pt", decoder, checkpoint)
                selection.append(checkpoint)
                print(f"{arm} seed={seed} epoch={checkpoint['epoch']} "
                      f"routing={checkpoint['macro_routing_accuracy']:.5f} "
                      f"score_mae={checkpoint['macro_score_mae']:.5f}", flush=True)
    return rows, selection


def evaluate_roster(dataset, split, config, orders, reference, prior_strength, checkpoints,
                    model_ids, labels, target, roster, device):
    model_order = model_ids.argsort()
    model_ids, labels = model_ids[model_order], labels[model_order]
    truth = labels[:, target]
    common = np.isfinite(truth).all(axis=0)
    truth, target = truth[:, common], target[common]
    domains = dataset.benchmark_ids[target]
    all_support = split.train_items
    full_budget = len(all_support)
    subsets = [(budget, "ordinary") for budget in (*BUDGETS, full_budget)]
    subsets.append((full_budget, "equal_benchmark"))
    per_seed, by_benchmark, outcomes = [], [], {}
    contrast_counts = []
    for seed in config["seeds"]:
        def record(arm, policy, budget, weighting, choices):
            result = pool_outcomes(choices[None, :], truth, domains)
            identity = {"roster": roster, "n_candidates": len(model_ids), "n_queries": len(target), "seed": seed,
                        "arm": arm, "policy": policy, "budget_per_candidate": budget,
                        "total_probe_attempts": budget * len(model_ids), "map_weighting": weighting}
            accuracy = result["correct"][0]
            oracle = result["oracle"]
            shuffled = result["shuffled"][0]
            accuracy_by_domain = benchmark_means(accuracy, domains)
            oracle_by_domain = benchmark_means(oracle, domains)
            shuffle_by_domain = benchmark_means(shuffled, domains)
            counts = np.bincount(choices, minlength=len(model_ids))
            per_seed.append({**identity, "macro_accuracy": accuracy_by_domain.mean(), "micro_accuracy": accuracy.mean(dtype=np.float64),
                             "macro_oracle": oracle_by_domain.mean(), "macro_regret": (oracle_by_domain - accuracy_by_domain).mean(),
                             "macro_gain_over_shuffle": (accuracy_by_domain - shuffle_by_domain).mean(),
                             "most_selected_model_id": int(model_ids[counts.argmax()]),
                             "max_selection_share": counts.max() / len(target)})
            for i, domain in enumerate(np.unique(domains)):
                by_benchmark.append({**identity, "benchmark": dataset.benchmark_names[domain],
                                     "benchmark_queries": int((domains == domain).sum()),
                                     "accuracy": accuracy_by_domain[i], "oracle": oracle_by_domain[i],
                                     "gain_over_shuffle": accuracy_by_domain[i] - shuffle_by_domain[i]})
            key = (arm, policy, budget, weighting)
            outcomes.setdefault(key, []).append(accuracy)

        for budget in (*BUDGETS, full_budget):
            support = all_support if budget == full_budget else orders[seed][:budget]
            observed = labels[:, support]
            support_scores = np.stack([np.nanmean(observed[:, dataset.benchmark_ids[support] == domain],
                                                  axis=1, dtype=np.float64)
                                       for domain in np.unique(domains)], axis=1).mean(axis=1)
            support_fixed = int(np.nanargmax(support_scores))
            record("shared", SUPPORT_FIXED, budget, "ordinary", np.full(len(target), support_fixed))

        for arm in ARMS:
            decoder, _ = load_decoder(checkpoints / f"{arm}_{seed}.pt", str(device))
            prior = latent_prior(decoder, config, prior_strength)
            _, design = decoder.export_features(dataset.embeddings[target], domains)
            ref_domains = dataset.benchmark_ids[reference]
            ref_offsets, ref_design = decoder.export_features(dataset.embeddings[reference], ref_domains)
            full_features = None
            for budget, weighting in subsets:
                support = all_support if budget == full_budget else orders[seed][:budget]
                support_domains = dataset.benchmark_ids[support]
                if budget == full_budget and full_features is not None:
                    features = full_features
                else:
                    features = decoder.export_features(dataset.embeddings[support], support_domains)
                    if budget == full_budget:
                        full_features = features
                latents = model_latents(features, labels[:, support], support_domains, prior, config,
                                        balanced=weighting == "equal_benchmark")
                probabilities = sigmoid(interaction_scores(latents, ref_design) + ref_offsets)
                fixed_index = int(benchmark_means(probabilities, ref_domains).mean(axis=1).argmax())
                record(arm, EMBEDDING_FIXED, budget, weighting, np.full(len(target), fixed_index))
                record(arm, ROUTER, budget, weighting, choose_models(interaction_scores(latents, design)))

    for budget, weighting in subsets:
        for arm in ARMS:
            router = np.stack(outcomes[(arm, ROUTER, budget, weighting)])
            for comparator in (("shared", SUPPORT_FIXED, budget, "ordinary"),
                               (arm, EMBEDDING_FIXED, budget, weighting)):
                fixed = np.stack(outcomes[comparator])
                gain = router - fixed
                opportunities = truth.max(axis=0) > fixed
                for index, seed in enumerate(config["seeds"]):
                    contrast_counts.append({"roster": roster, "seed": seed, "arm": arm,
                        "budget_per_candidate": budget, "map_weighting": weighting,
                        "comparator": comparator[1], "n_queries": len(target),
                        "rescues": int((gain[index] > 0).sum()), "harms": int((gain[index] < 0).sum()),
                        "opportunities": int(opportunities[index].sum())})

    summary = []
    for (arm, policy, budget, weighting), values in outcomes.items():
        correctness = np.stack(values)
        macro = benchmark_means(correctness, domains).mean()
        oracle = benchmark_means(truth.max(axis=0), domains).mean()
        summary.append({"roster": roster, "n_candidates": len(model_ids), "n_queries": len(target),
                        "n_seeds": len(config["seeds"]), "arm": arm, "policy": policy,
                        "budget_per_candidate": budget, "total_probe_attempts": budget * len(model_ids),
                        "map_weighting": weighting, "macro_accuracy": macro,
                        "micro_accuracy": correctness.mean(dtype=np.float64), "macro_oracle": oracle,
                        "macro_regret": oracle - macro})
    return (summary, per_seed, by_benchmark, paired_contrasts(outcomes, domains, roster, subsets, full_budget),
            contrast_counts)


def paired_contrasts(outcomes, domains, roster, subsets, full_budget):
    pairs = []
    for budget, weighting in subsets:
        for arm in ARMS:
            left = (arm, ROUTER, budget, weighting)
            pairs.extend(((left, ("shared", SUPPORT_FIXED, budget, "ordinary")),
                          (left, (arm, EMBEDDING_FIXED, budget, weighting))))
        pairs.extend((((B, ROUTER, budget, weighting), (A, ROUTER, budget, weighting)),
                      ((C, ROUTER, budget, weighting), (B, ROUTER, budget, weighting))))
        if weighting == "equal_benchmark":
            pairs.extend(((arm, ROUTER, full_budget, weighting), (arm, ROUTER, full_budget, "ordinary"))
                         for arm in ARMS)
    differences = np.stack([np.stack(outcomes[left]) - np.stack(outcomes[right]) for left, right in pairs])
    averaged = differences.mean(axis=1, dtype=np.float64)
    by_seed = benchmark_means(differences, domains).mean(axis=-1)
    replicates = np.zeros((len(pairs), BOOTSTRAP_DRAWS))
    unique = np.unique(domains)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for domain in unique:
        values = averaged[:, domains == domain]
        for start in range(0, BOOTSTRAP_DRAWS, 128):
            stop = min(start + 128, BOOTSTRAP_DRAWS)
            drawn = rng.integers(values.shape[1], size=(stop - start, values.shape[1]))
            replicates[:, start:stop] += values[:, drawn].mean(axis=-1) / len(unique)
    low, high = np.quantile(replicates, [0.025, 0.975], axis=1)
    return [{"roster": roster, "arm": left[0], "comparator_arm": right[0],
             "comparator_policy": right[1], "budget_per_candidate": left[2], "map_weighting": left[3],
             "comparator_map_weighting": right[3], "macro_gain": by_seed[index].mean(),
             "micro_gain": averaged[index].mean(), "macro_ci_low": low[index], "macro_ci_high": high[index],
             "positive_seed_count": int((by_seed[index] > 0).sum())}
            for index, (left, right) in enumerate(pairs)]


def input_context(run: Path):
    config = read_json(run / "effective_config.json")
    data = config["data"]
    prepared = ROOT / data["prepared_path"]
    family_file = ROOT / data["family_table"]
    dataset = load_dataset(prepared, label_policy=data["label_policy"], model_ids=sorted(load_family_table(family_file)),
                           benchmarks=data["candidate_benchmarks"], deduplicate_groups=data["deduplicate_groups"])
    split = load_split(run / "split.json", dataset)
    prior_strength = read_json(run / "test_lock.json")["selection"]["prior_strengths"][VARIANT]
    return dataset, split, config, prior_strength, prepared, family_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train_validate", "validation", "test"), required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts" / "main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts" / "routing_learning")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "routing" / "learning")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args)
    return 0


def run(args):
    dataset, split, config, prior_strength, prepared, family_file = input_context(args.run)
    device = select_device(args.device)
    support = split.train_items
    orders = {seed: select_probe_order(dataset.embeddings, dataset.benchmark_ids, support,
                                      PRIMARY_BUDGET, "balanced_random", seed) for seed in config["seeds"]}
    remaining = np.setdiff1d(support, np.concatenate(list(orders.values())))
    reference = select_probe_order(dataset.embeddings, dataset.benchmark_ids, remaining,
                                   REFERENCE_PER_BENCHMARK * len(dataset.benchmark_names),
                                   "balanced_random", REFERENCE_SEED)
    protocol = {"split": split.protocol, "training_model_ids": dataset.model_ids[split.train_models].tolist(),
                "validation_model_ids": dataset.model_ids[split.val_models].tolist(),
                "audited_test_model_ids": dataset.model_ids[split.test_models].tolist(),
                "expanded_roster": "union of audited test IDs and all 24 metadata-excluded IDs, common observed query support",
                "query_counts": {"validation": len(split.val_items), "test_before_common_mask": len(split.test_items)},
                "seed_order": config["seeds"], "epochs": config["model"]["epochs"],
                "selection_epochs": [1, *range(config["model"]["selection_interval"], config["model"]["epochs"] + 1,
                                               config["model"]["selection_interval"])],
                "selection_budget": PRIMARY_BUDGET, "evaluation_budgets": list(BUDGETS),
                "checkpoint_ties": "earlier epoch", "model_ties": "lower source model ID",
                "trajectory": {A: "macro BCE, minimum validation score MAE", B: "same macro BCE, maximum validation routing macro accuracy",
                               C: "macro BCE plus expected regret, maximum validation routing macro accuracy"},
                "training_weights": "inverse benchmark frequency, mean weight 1 for both training losses",
                "training_roster_size": ROSTER_SIZE, "regret_weight": REGRET_WEIGHT, "softmax_temperature": TEMPERATURE,
                "prior_strength": prior_strength, "full_support_budget": len(support),
                "full_support_equal_benchmark_fit": "inverse benchmark frequency over observed labels, total likelihood weight unchanged",
                "probe_prompt_ids": {str(seed): dataset.prompt_ids[order].tolist() for seed, order in orders.items()},
                "embedding_mean_reference_prompt_ids": dataset.prompt_ids[reference].tolist(),
                "query_bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                                    "unit": "paired query within benchmark after averaging seeds, conditional on roster and checkpoints"},
                "test_scope": "retrospective, prior benchmark score and routing evaluations used these partitions"}
    saved = args.output / "selection.json"
    checkpoint_dir = args.checkpoints / "checkpoints"
    if args.phase == "train_validate":
        history, selection = train_all(dataset, split, config, orders, prior_strength, args.checkpoints, device)
        atomic_csv(args.output / "validation_trajectory.csv", history)
        atomic_json(saved, {"protocol": protocol, "checkpoints": selection})
    elif read_json(saved)["protocol"] != protocol:
        raise ValueError("Scoring inputs must match the saved validation protocol.")
    if args.phase in {"train_validate", "validation"}:
        phase = "validation"
        rosters = [("validation_13", dataset.model_ids[split.val_models], dataset.labels[split.val_models], split.val_items)]
    else:
        phase = "test"
        excluded = load_oracle(dataset, prepared / "embedllm.npz", family_file)
        model_ids = dataset.model_ids[split.test_models]
        labels = dataset.labels[split.test_models]
        combined_ids = np.r_[model_ids, excluded.model_ids]
        rosters = [("audited_13", model_ids, labels, split.test_items),
                   ("expanded_37", combined_ids, np.vstack([labels, excluded.labels]), split.test_items)]

    outputs = [[], [], [], [], []]
    for roster, model_ids, labels, target in rosters:
        print(f"Evaluating {roster}: {len(model_ids)} candidates, {len(target)} queries before common mask", flush=True)
        rows = evaluate_roster(dataset, split, config, orders, reference, prior_strength, checkpoint_dir,
                               model_ids, labels, target, roster, device)
        for destination, records in zip(outputs, rows):
            destination.extend(records)
    for suffix, records in zip(("summary", "by_seed", "by_benchmark", "paired", "rescues_harms_by_seed"), outputs):
        atomic_csv(args.output / f"{phase}_{suffix}.csv", records)
    print(f"Saved {phase} routing-learning results to {args.output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
