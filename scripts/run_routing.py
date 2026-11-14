"""Evaluate offline few-shot onboarding and query-by-query LLM routing."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from route_query import ROOT, choose_models, interaction_scores
from coldstart.adaptation import adapt_many, sigmoid
from coldstart.data import load_dataset
from coldstart.models import MeanBaselines, question_knn
from coldstart.probes import select_probe_order
from coldstart.splits import load_family_table, load_split
from run_experiment import atomic_csv, atomic_json, load_features


VARIANT = "neural_lowrank"
EMBEDDING = "compact_embedding"
FIXED = "fixed_candidate"
LOWEST_ID = "lowest_id_zero_probe"
UNIFORM = "uniform_zero_probe_expected"
DOMAIN = "domain_probe_mean"
KNN = "question_knn"
INCUMBENT = "incumbent_only"
ARGMAX = "compact_embedding_argmax"
ADD_ONE = "add_one"
POOL = "all_held_out"
MARGINS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 8675309


def benchmark_means(values: np.ndarray, domains: np.ndarray) -> np.ndarray:
    return np.stack([values[..., domains == domain].mean(axis=-1, dtype=np.float64)
                     for domain in np.unique(domains)], axis=-1)


def score_matrices(dataset, split, config, selection, run, rows, target, incumbent):
    """Build predictions from the support responses and the frozen question decoder."""
    support = split.train_items
    budgets = config["budgets"]
    support_domains = dataset.benchmark_ids[support]
    target_domains = dataset.benchmark_ids[target]
    target_vectors = dataset.embeddings[target]
    incumbent_support = dataset.labels[incumbent, support][None, :]
    incumbent_accuracy = benchmark_means(incumbent_support, support_domains).mean()
    baseline = MeanBaselines(len(dataset.benchmark_names), selection["mean_smoothing"][DOMAIN]).fit(
        dataset.labels[np.ix_(split.train_models, support)], support_domains)
    incumbent_domain = baseline.probe_mean(incumbent_support, support_domains, target_domains)
    knn_prior = baseline.benchmark_prior[target_domains]
    incumbent_knn = question_knn(dataset.embeddings[support], incumbent_support, target_vectors,
                                  prior_probabilities=knn_prior, batch_size=256, **selection["knn"])
    matrices = defaultdict(list)
    probe_orders = {}
    for seed in config["seeds"]:
        feature = load_features(dataset, config, run, [VARIANT], seed, support)[VARIANT]
        decoder = feature["decoder"]
        train_ids = dataset.model_ids[split.train_models].tolist()
        if decoder.config.mode != "lowrank" or decoder.item_bias is not None:
            raise ValueError("Routing requires a low-rank decoder for unseen questions.")
        if feature["metadata"]["training_model_ids"] != train_ids:
            raise ValueError("Training model IDs differ from the decoder's rows.")
        incumbent_latent = decoder.model_latents.weight.detach().cpu().numpy()[
            train_ids.index(int(dataset.model_ids[incumbent]))]
        prior_precision = selection["prior_strengths"][VARIANT] * feature["precision"]
        query_offset, query_design = decoder.export_features(target_vectors, target_domains)
        order = select_probe_order(dataset.embeddings, dataset.benchmark_ids, support,
                                   max(budgets), "balanced_random", seed)
        probe_orders[str(seed)] = dataset.prompt_ids[order].tolist()
        for budget in [*budgets, len(support)]:
            selected = support if budget == len(support) else order[:budget]
            responses = dataset.labels[np.ix_(rows, selected)]
            domains = dataset.benchmark_ids[selected]
            fitted = adapt_many(feature["design"][selected], feature["offsets"][selected],
                               responses, feature["mean"], prior_precision,
                               max_iterations=config["adaptation"]["max_iterations"],
                               tolerance=config["adaptation"]["tolerance"])
            latents = np.vstack([incumbent_latent, *[fit.latent for fit in fitted]])
            logits = interaction_scores(latents, query_design)
            matrices[(budget, EMBEDDING)].append(sigmoid(logits + query_offset))
            candidate_accuracy = benchmark_means(responses, domains).mean(axis=1)
            fixed_scores = np.r_[incumbent_accuracy, candidate_accuracy]
            matrices[(budget, FIXED)].append(np.broadcast_to(
                fixed_scores[:, None], (len(fixed_scores), len(target))))
            if budget != len(support):
                domain = baseline.probe_mean(responses, domains, target_domains)
                matrices[(budget, DOMAIN)].append(np.vstack([incumbent_domain, domain]))
                knn = question_knn(dataset.embeddings[selected], responses, target_vectors,
                                   prior_probabilities=knn_prior, **selection["knn"])
                matrices[(budget, KNN)].append(np.vstack([incumbent_knn, knn]))
    return {key: np.stack(values) for key, values in matrices.items()}, probe_orders


def newcomer_choices(scores: np.ndarray, margin: float) -> np.ndarray:
    incumbent = np.broadcast_to(scores[:, :1], scores[:, 1:].shape)
    return choose_models(np.stack([incumbent, scores[:, 1:]]), margin).astype(bool)


def select_margins(matrices, truth, domains):
    """Select a common margin per method and budget on validation queries only."""
    deltas = truth[1:] - truth[0]
    selection = {}
    for (budget, method), scores in matrices.items():
        if method == FIXED:
            continue
        values = {}
        for margin in MARGINS:
            gains = newcomer_choices(scores, margin) * deltas
            values[margin] = benchmark_means(gains.mean(axis=(0, 1)), domains).mean()
        chosen = max(MARGINS, key=lambda margin: (values[margin], margin))
        selection[f"{method}/{budget}"] = {
            "minimum_gain": chosen,
            "incumbent_only": chosen == 1.0,
            "validation_macro_gain": float(values[chosen]),
            "validation_macro_gain_at_argmax": float(values[0.0]),
        }
    return selection


def add_one_outcomes(chosen, truth, domains):
    """Measure rescue and harm against the incumbent on identical queries."""
    incumbent, newcomer = truth[0], truth[1:]
    opportunity = newcomer > incumbent
    rescue = chosen & opportunity
    harm = chosen & (newcomer < incumbent)
    correct = np.where(chosen, newcomer, incumbent)
    shuffled = np.empty_like(correct, dtype=np.float64)
    for domain in np.unique(domains):
        mask = domains == domain
        share = chosen[..., mask].mean(axis=-1)
        expected = ((1 - share) * incumbent[mask].mean(dtype=np.float64)
                    + share * newcomer[:, mask].mean(axis=-1, dtype=np.float64))
        shuffled[..., mask] = expected[..., None]
    return {"correct": correct, "oracle": np.maximum(incumbent, newcomer),
            "shuffled": shuffled, "selected": chosen, "rescue": rescue,
            "harm": harm, "opportunity": opportunity}


def pool_outcomes(chosen, truth, domains):
    """Score the simultaneous candidate pool and its same-usage shuffle."""
    correct = truth[chosen, np.arange(truth.shape[1])]
    shuffled = np.empty_like(correct, dtype=np.float64)
    for domain in np.unique(domains):
        mask = domains == domain
        rates = truth[:, mask].mean(axis=-1, dtype=np.float64)
        expected = [(np.bincount(row[mask], minlength=len(truth)) @ rates) / mask.sum()
                    for row in chosen]
        shuffled[:, mask] = np.asarray(expected)[:, None]
    return {"correct": correct, "oracle": truth.max(axis=0), "shuffled": shuffled}


def paired_intervals(outcomes, domains, full_budget):
    """Resample paired queries within benchmark after averaging seeds and trials."""
    keys, effects = [], []
    few_budgets = sorted(key[1] for key in outcomes if key[0] == ADD_ONE and key[2] == EMBEDDING
                         and key[1] != full_budget)
    for scenario in (ADD_ONE, POOL):
        fixed_key = (ADD_ONE, 0, INCUMBENT) if scenario == ADD_ONE else (POOL, 0, LOWEST_ID)
        for budget in [*few_budgets, full_budget]:
            key = (scenario, budget, EMBEDDING)
            current = outcomes[key]
            comparators = [(fixed_key, outcomes[fixed_key]["correct"]),
                           ((scenario, budget, FIXED), outcomes[(scenario, budget, FIXED)]["correct"])]
            if scenario == POOL:
                uniform_key = (POOL, 0, UNIFORM)
                comparators.append((uniform_key, outcomes[uniform_key]["correct"]))
            if budget != full_budget:
                comparators.extend(((scenario, budget, method), outcomes[(scenario, budget, method)]["correct"])
                                   for method in (DOMAIN, KNN))
                full_key = (scenario, full_budget, EMBEDDING)
                comparators.append((full_key, outcomes[full_key]["correct"]))
            if scenario == ADD_ONE:
                raw_key = (scenario, budget, ARGMAX)
                comparators.append((raw_key, outcomes[raw_key]["correct"]))
            for other, reference in comparators:
                difference = current["correct"] - reference
                effects.append(difference.mean(axis=tuple(range(difference.ndim - 1))))
                keys.append((key, other))

    effects = np.asarray(effects)
    replicates = np.zeros((len(effects), BOOTSTRAP_DRAWS))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        subset = effects[:, domains == domain]
        for start in range(0, BOOTSTRAP_DRAWS, 128):
            stop = min(start + 128, BOOTSTRAP_DRAWS)
            sampled = rng.integers(subset.shape[1], size=(stop - start, subset.shape[1]))
            replicates[:, start:stop] += subset[:, sampled].mean(axis=-1) / len(unique_domains)
    estimates = benchmark_means(effects, domains).mean(axis=-1)
    low, high = np.quantile(replicates, [0.025, 0.975], axis=1)
    return [{"scenario": key[0], "budget": key[1], "method": key[2],
             "reference_budget": other[1], "reference_method": other[2],
             "micro_gain": float(effect.mean()), "macro_gain": float(estimate),
             "macro_ci_low": float(left), "macro_ci_high": float(right)}
            for (key, other), effect, estimate, left, right in zip(keys, effects, estimates, low, high)]


def summarize_outcomes(outcomes, domains, benchmark_names, n_candidates, seeds):
    summary, by_benchmark = [], []
    for (scenario, budget, method), result in sorted(outcomes.items()):
        identity = {"scenario": scenario, "budget": budget, "method": method}
        values = result["correct"]
        accuracy = benchmark_means(values, domains).reshape(-1, len(benchmark_names)).mean(axis=0)
        oracle = benchmark_means(result["oracle"], domains).reshape(-1, len(benchmark_names)).mean(axis=0)
        shuffled = benchmark_means(result["shuffled"], domains).reshape(-1, len(benchmark_names)).mean(axis=0)
        micro_accuracy = float(values.mean(dtype=np.float64))
        micro_oracle = float(result["oracle"].mean(dtype=np.float64))
        row = {**identity, "n_queries": len(domains), "n_candidates": 2 if scenario == ADD_ONE else n_candidates,
               "n_newcomer_trials": n_candidates if scenario == ADD_ONE else 1, "n_seeds": len(seeds),
               "total_probe_responses_per_trial": budget if scenario == ADD_ONE else budget * n_candidates,
               "total_probe_responses_across_newcomers": budget * n_candidates,
               "micro_accuracy": micro_accuracy, "macro_accuracy": float(accuracy.mean()),
               "micro_oracle": micro_oracle, "macro_oracle": float(oracle.mean()),
               "micro_regret": micro_oracle - micro_accuracy,
               "macro_regret": float((oracle - accuracy).mean()),
               "macro_shuffled_accuracy": float(shuffled.mean()),
               "macro_gain_over_shuffle": float((accuracy - shuffled).mean())}
        if scenario == ADD_ONE:
            opportunity = result["opportunity"].sum()
            rescue = result["rescue"].sum() / len(seeds)
            harm = result["harm"].sum() / len(seeds)
            row.update(newcomer_share=float(result["selected"].mean()),
                       opportunities_per_trial=float(opportunity / n_candidates),
                       rescues_per_trial=float(rescue / n_candidates),
                       harms_per_trial=float(harm / n_candidates),
                       opportunity_capture=float(rescue / opportunity) if opportunity else None,
                       micro_gain_over_incumbent=float((rescue - harm) / (len(domains) * n_candidates)))
        summary.append(row)
        for domain, name in enumerate(benchmark_names):
            by_benchmark.append({**identity, "benchmark": name, "n_queries": int((domains == domain).sum()),
                                 "accuracy": float(accuracy[domain]), "oracle": float(oracle[domain]),
                                 "regret": float(oracle[domain] - accuracy[domain]),
                                 "shuffled_accuracy": float(shuffled[domain])})
    return summary, by_benchmark


def evaluate(dataset, rows, incumbent, target, seeds, matrices, margins, full_budget, output, phase):
    truth = dataset.labels[np.ix_(np.r_[incumbent, rows], target)]
    domains = dataset.benchmark_ids[target]
    outcomes, add_one_rows, pool_rows = {}, [], []
    empty = np.zeros((len(seeds), len(rows), len(target)), dtype=bool)
    outcomes[(ADD_ONE, 0, INCUMBENT)] = add_one_outcomes(empty, truth, domains)
    first = np.zeros((len(seeds), len(target)), dtype=np.int64)
    outcomes[(POOL, 0, LOWEST_ID)] = pool_outcomes(first, truth[1:], domains)
    uniform = np.broadcast_to(truth[1:].mean(axis=0, dtype=np.float64), first.shape)
    outcomes[(POOL, 0, UNIFORM)] = {"correct": uniform, "oracle": truth[1:].max(axis=0), "shuffled": uniform}
    pool_choices = {(0, LOWEST_ID): first}
    for (budget, method), scores in matrices.items():
        margin = 0.0 if method == FIXED else margins[f"{method}/{budget}"]["minimum_gain"]
        outcomes[(ADD_ONE, budget, method)] = add_one_outcomes(newcomer_choices(scores, margin), truth, domains)
        if method == EMBEDDING:
            outcomes[(ADD_ONE, budget, ARGMAX)] = add_one_outcomes(newcomer_choices(scores, 0.0), truth, domains)
        chosen = choose_models(scores[:, 1:].swapaxes(0, 1))
        outcomes[(POOL, budget, method)] = pool_outcomes(chosen, truth[1:], domains)
        pool_choices[(budget, method)] = chosen

    summary, benchmarks = summarize_outcomes(outcomes, domains, dataset.benchmark_names, len(rows), seeds)
    atomic_csv(output / f"{phase}_summary.csv", summary)
    atomic_csv(output / f"{phase}_by_benchmark.csv", benchmarks)
    for (scenario, budget, method), result in sorted(outcomes.items()):
        if scenario != ADD_ONE:
            continue
        for local_seed, seed in enumerate(seeds):
            for local_model, row in enumerate(rows):
                rescue = int(result["rescue"][local_seed, local_model].sum())
                harm = int(result["harm"][local_seed, local_model].sum())
                opportunity = int(result["opportunity"][local_model].sum())
                add_one_rows.append({"seed": seed, "budget": budget, "method": method,
                    "model_id": int(dataset.model_ids[row]), "model_name": dataset.model_names[row],
                    "n_queries": len(target), "newcomer_selected": int(result["selected"][local_seed, local_model].sum()),
                    "opportunities": opportunity, "rescues": rescue, "harms": harm,
                    "net_correct": rescue - harm, "capture": rescue / opportunity if opportunity else None})
    for (budget, method), choices in sorted(pool_choices.items()):
        counts = np.stack([np.bincount(row, minlength=len(rows)) for row in choices])
        for position, row in enumerate(rows):
            pool_rows.append({"budget": budget, "method": method, "model_id": int(dataset.model_ids[row]),
                              "model_name": dataset.model_names[row],
                              "selection_share": float(counts[:, position].mean() / len(target))})
    pool_rows.extend({"budget": 0, "method": UNIFORM, "model_id": int(dataset.model_ids[row]),
                      "model_name": dataset.model_names[row], "selection_share": 1 / len(rows)}
                     for row in rows)
    atomic_csv(output / f"{phase}_add_one_by_candidate.csv", add_one_rows)
    atomic_csv(output / f"{phase}_pool_selections.csv", pool_rows)
    if phase == "test":
        atomic_csv(output / "test_paired.csv", paired_intervals(outcomes, domains, full_budget))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts" / "main_v2",
                        help="Validated benchmark-prediction run containing the frozen decoders.")
    parser.add_argument("--data", type=Path, help="Prepared data directory or .npz file if moved.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "routing")
    parser.add_argument("--phase", required=True, choices=("validation", "test"))
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return run(args)


def run(args):
    config = json.loads((args.run / "effective_config.json").read_text(encoding="utf-8"))
    lock = json.loads((args.run / "test_lock.json").read_text(encoding="utf-8"))
    data = config["data"]
    model_ids = data.get("model_ids")
    if model_ids is None and data.get("model_scope") == "audited_lineage_included_only":
        model_ids = sorted(load_family_table(ROOT / data["family_table"]))
    dataset = load_dataset(args.data or ROOT / data["prepared_path"], label_policy=data["label_policy"],
                           model_ids=model_ids, benchmarks=data["candidate_benchmarks"],
                           deduplicate_groups=data["deduplicate_groups"])
    split = load_split(args.run / "split.json", dataset)
    if split.protocol != "strict_unseen_item" or data["label_policy"] != "strict":
        raise ValueError("This protocol requires the audited strict unseen-question run.")
    val_training = dataset.labels[np.ix_(split.train_models, split.val_items)]
    val_accuracy = benchmark_means(val_training, dataset.benchmark_ids[split.val_items]).mean(axis=1)
    ordered = np.lexsort((dataset.model_ids[split.train_models], -val_accuracy))
    incumbent = split.train_models[ordered[0]]
    chosen_rows, target = ((split.val_models, split.val_items) if args.phase == "validation"
                           else (split.test_models, split.test_items))
    rows = chosen_rows[np.argsort(dataset.model_ids[chosen_rows])]
    full_budget = len(split.train_items)
    protocol = {"checkpoint_run": args.run.name, "decoder": VARIANT, "embedding_dimension": 8,
                "source_repository": dataset.metadata["source_repository"],
                "source_revision": dataset.metadata["source_revision"],
                "data_protocol": split.protocol, "benchmarks": list(dataset.benchmark_names),
                "training_models": len(split.train_models), "support_questions": full_budget,
                "validation_candidates": dataset.model_ids[split.val_models].tolist(),
                "validation_queries": len(split.val_items),
                "test_candidates": dataset.model_ids[split.test_models].tolist(), "test_queries": len(split.test_items),
                "incumbent_model_id": int(dataset.model_ids[incumbent]), "incumbent_model_name": dataset.model_names[incumbent],
                "incumbent_validation_macro_accuracy": float(val_accuracy.max()),
                "incumbent_selection": "Best training model on validation queries by macro accuracy, ties lowest source ID",
                "fixed_candidate_selection": "Highest macro accuracy on available support probes, ties lowest source ID",
                "add_one_fixed_candidate_selection": "Compare incumbent full-support macro accuracy with newcomer probe macro accuracy, ties incumbent",
                "few_shot_budgets": config["budgets"], "full_support_budget": full_budget, "seeds": config["seeds"],
                "probe_policy": "balanced_random_nested", "validation_margin_grid": list(MARGINS),
                "validation_margin_selection": "Max mean add-one validation macro accuracy, ties larger margin",
                "paired_bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                                     "unit": "paired query within benchmark, conditional on candidates and support draws"}}
    selection_path = args.output / "validation_selection.json"
    if args.phase == "test":
        if not selection_path.exists():
            raise ValueError("Run the validation phase to select the incumbent and margins before test.")
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        if protocol != frozen["protocol"]:
            raise ValueError("Test protocol differs from the validation run.")

    print(f"{args.phase}: {len(rows)} newcomers, {len(target)} queries, incumbent {dataset.model_names[incumbent]}", flush=True)
    matrices, probe_orders = score_matrices(dataset, split, config, lock["selection"],
                                           args.run, rows, target, incumbent)
    if args.phase == "validation":
        truth = dataset.labels[np.ix_(np.r_[incumbent, rows], target)]
        margins = select_margins(matrices, truth, dataset.benchmark_ids[target])
        atomic_json(selection_path, {"protocol": protocol, "margins": margins, "probe_prompt_ids": probe_orders})
    else:
        if probe_orders != frozen["probe_prompt_ids"]:
            raise ValueError("Test probe selection differs from the validation run.")
        margins = frozen["margins"]
    evaluate(dataset, rows, incumbent, target, config["seeds"], matrices, margins,
             full_budget, args.output, args.phase)
    print(f"Saved {args.phase} routing results to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
