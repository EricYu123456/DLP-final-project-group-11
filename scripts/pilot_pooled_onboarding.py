"""Allocate a shared onboarding budget among held-out validation newcomers."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from route_query import choose_models, interaction_scores
from coldstart.adaptation import adapt_map, sigmoid
from coldstart.models import load_decoder
from coldstart.probes import select_probe_order
from pilot_routing_acquisition import expected_switch_value
from pilot_routing_learning import B, input_context, latent_prior
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries, read_json
from run_routing import benchmark_means, pool_outcomes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/routing/pooled_onboarding"
RUN = ROOT / "artifacts/main_v2"
CHECKPOINTS = ROOT / "artifacts/routing_learning/checkpoints"
SUPPORT_PROTOCOL = ROOT / "reports/routing/competitive_onboarding/validation_selection.json"
HORIZONS = (650, 1300, 2600)
PRIMARY = 1300
THRESHOLDS = (0.002, 0.005, 0.01, 0.02)
WARMUP = 10
CAP = 200
SHORTLIST = 48
PRIOR_STRENGTH = 0.3
CONTROL = "round_robin_random"
ACTIVE = "joint_smooth"
QUESTION = "question_selection_only"
MODEL = "model_allocation_only"
ROUTER = "query_router"
FIXED = ("probe_macro_fixed", "probe_benchmark_fixed", "reference_macro_fixed", "reference_benchmark_fixed")
MIN_GAIN = 0.005
MIN_STOP_SAVING = 0.20
MIN_STOP_LOWER_BOUND = -0.005


def reference_state(latents, covariances, design, offsets):
    """Pairwise-sum upper bound and its curvature at the current reference winners."""
    p = sigmoid(interaction_scores(latents, design) + offsets)
    slope = p * (1 - p)
    variance = ((design @ covariances) * design).sum(axis=-1) * slope ** 2
    refs = np.arange(len(offsets))
    best = p.argmax(axis=0)
    gap = p[best, refs] - p
    sigma = np.sqrt(np.maximum(variance[best, refs] + variance, 0))
    potential = expected_switch_value(gap, sigma)
    potential[best, refs] = 0
    valid = sigma > np.finfo(np.float64).eps
    ratio = np.divide(gap, sigma, out=np.zeros_like(sigma), where=valid)
    weights = np.divide(np.exp(-0.5 * ratio ** 2), 2 * np.sqrt(2 * np.pi) * sigma,
                        out=np.zeros_like(sigma), where=valid)
    weights[best, refs] = 0
    weights[best, refs] = weights.sum(axis=0)
    return float(potential.sum(axis=0).mean()), weights * slope ** 2


def pick_probe(policy, order, ranks, counts, available, latents, covariance, design, offsets,
               ref_design, ref_weights):
    selectable = np.flatnonzero(counts < CAP)
    if policy == MODEL:
        eligible = order[counts[selectable], None]
    else:
        if policy == QUESTION:
            selectable = np.array([counts.argmin()])
        priority = ranks[selectable, counts[selectable] - WARMUP]
        eligible = np.stack([row[available[model, row]][:SHORTLIST] for model, row in zip(selectable, priority)])
    candidates = design[eligible]
    p = sigmoid((candidates * latents[selectable, None]).sum(axis=-1) + offsets[eligible])
    w = p * (1 - p)
    projected = candidates @ covariance[selectable]
    scale = w / (1 + w * (projected * candidates).sum(axis=-1))
    weighted_gram = ref_design.T @ (ref_weights[selectable, :, None] * ref_design) / len(ref_design)
    acquisition = scale * ((projected @ weighted_gram) * projected).sum(axis=-1)
    if acquisition.max() <= 0:
        row = counts[selectable].argmin()
        return selectable[row], eligible[row, 0]
    row, col = np.unravel_index(acquisition.argmax(), acquisition.shape)
    return selectable[row], eligible[row, col]


def allocate(policy, model_ids, seed, order, candidate_domains, candidate, reference, prior, observe, options,
             horizons, thresholds):
    started = perf_counter()
    offsets, design = candidate
    ref_offsets, ref_design = reference
    models = len(model_ids)
    limit = max(horizons)
    ranks = None
    if policy in (ACTIVE, QUESTION):
        by_domain = [np.flatnonzero(candidate_domains == b) for b in np.unique(candidate_domains)]
        ranks = []
        for model in model_ids:
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(model)]))
            ranks.append([rng.permutation(by_domain[domain]) for domain in candidate_domains[order[WARMUP:CAP]]])
        ranks = np.asarray(ranks)
    available = np.ones((models, len(order)), dtype=bool)
    chosen = np.full((models, CAP), -1, dtype=np.int64)
    revealed = np.zeros((models, CAP), dtype=np.float64)
    latents = np.empty((models, len(prior[0])), dtype=np.float64)
    covariance = np.empty((models, len(prior[0]), len(prior[0])), dtype=np.float64)
    counts = np.full(models, WARMUP, dtype=np.int64)

    def fit_row(model, initial=None):
        budget = counts[model]
        ids = chosen[model, :budget]
        fit = adapt_map(design[ids], offsets[ids], revealed[model, :budget], *prior, initial=initial, **options)
        return fit.latent, fit.covariance

    chosen[:, :WARMUP] = order[:WARMUP]
    available[:, order[:WARMUP]] = False
    for model in range(models):
        revealed[model, :WARMUP] = [observe(model, question) for question in order[:WARMUP]]
        latents[model], covariance[model] = fit_row(model)
    states, first_crossing, trace = {}, {}, []
    for total in range(WARMUP * models, limit + 1):
        bound, weights = reference_state(latents, covariance, ref_design, ref_offsets)
        trace.append((total, bound, float(perf_counter() - started)))

        def snapshot():
            return counts.copy(), chosen.copy(), revealed.copy(), latents.copy(), trace[-1][2], bound

        if total in horizons:
            states[total] = snapshot()
        for threshold in thresholds:
            if threshold not in first_crossing and (bound <= threshold or total == limit):
                first_crossing[threshold] = (total, snapshot())
        if total == limit:
            break
        if policy == CONTROL:
            model = counts.argmin()
            question = order[counts[model]]
        else:
            model, question = pick_probe(policy, order, ranks, counts, available, latents, covariance,
                                         design, offsets, ref_design, weights)
        budget = counts[model]
        chosen[model, budget] = question
        revealed[model, budget] = observe(model, question)
        available[model, question] = False
        counts[model] += 1
        latents[model], covariance[model] = fit_row(model, initial=latents[model])
    if counts.sum() != limit or counts.max() > CAP or any(
        len(np.unique(row[:count])) != count for row, count in zip(chosen, counts)
    ):
        raise RuntimeError("The pooled run violated its per-model cap or repeated a probe.")
    if policy == QUESTION and not np.all(counts == limit // models):
        raise RuntimeError("Question selection must keep a round-robin allocation.")
    if policy == MODEL and any(
        not np.array_equal(row[:count], order[:count]) for row, count in zip(chosen, counts)
    ):
        raise RuntimeError("Model allocation must retain the balanced-random prefix for each model.")
    if any(not np.array_equal(candidate_domains[row[:count]], candidate_domains[order[:count]])
           for row, count in zip(chosen, counts)):
        raise RuntimeError("The pooled run changed a model's benchmark quota order.")
    return states, first_crossing, trace


def score(snapshot, query, reference, truth, domains, reference_domains, candidate_domains):
    counts, selected, labels, latents, elapsed, bound = snapshot
    query_offsets, query_design = query
    ref_offsets, ref_design = reference
    predictions = interaction_scores(latents, query_design) + query_offsets
    predicted_ref = sigmoid(interaction_scores(latents, ref_design) + ref_offsets)
    ref_means = benchmark_means(predicted_ref, reference_domains)
    empirical = np.stack([benchmark_means(row[:count], candidate_domains[ids[:count]])
                          for count, ids, row in zip(counts, selected, labels)])
    choices = {
        ROUTER: choose_models(predictions),
        FIXED[0]: np.full(len(domains), empirical.mean(axis=1).argmax()),
        FIXED[1]: empirical.argmax(axis=0)[domains],
        FIXED[2]: np.full(len(domains), ref_means.mean(axis=1).argmax()),
        FIXED[3]: ref_means.argmax(axis=0)[domains],
    }
    result = {name: pool_outcomes(picks[None, :], truth, domains) for name, picks in choices.items()}
    return result


def evaluate(results, domains):
    summary, paired = [], []
    for (policy, budget, method), runs in results.items():
        actual = np.stack([run["correct"][0] for run in runs])
        oracle = runs[0]["oracle"]
        shuffled = np.stack([run["shuffled"][0] for run in runs])
        summary.append({"selection": policy, "total_probes": budget, "method": method,
                        "macro_accuracy": benchmark_means(actual, domains).mean(),
                        "micro_accuracy": actual.mean(dtype=np.float64),
                        "hindsight_oracle_macro": benchmark_means(oracle, domains).mean(),
                        "hindsight_oracle_micro": oracle.mean(dtype=np.float64),
                        "gain_over_same_usage_shuffle": benchmark_means(actual - shuffled, domains).mean()})
        if policy != ACTIVE or method != ROUTER:
            continue
        for baseline, other in [(CONTROL, ROUTER), *[(p, f) for p in (ACTIVE, CONTROL) for f in FIXED]]:
            comparison = np.stack([run["correct"][0] for run in results[baseline, budget, other]])
            paired.append({"selection": policy, "total_probes": budget, "method": method,
                           "comparator_selection": baseline, "comparator": other,
                           **paired_queries(actual, comparison, domains)})
    return summary, paired


def mechanism_report(output, new_rows, archived_rows, results, domains):
    summary = [{**row, "source": source} for rows, source in
               ((archived_rows, "archived_main"), (new_rows, "decomposition")) for row in rows]
    queries = {row["selection"]: row for row in summary if row["method"] == ROUTER}
    contrasts = []
    for method, comparator in ((QUESTION, CONTROL), (MODEL, CONTROL), (ACTIVE, MODEL), (ACTIVE, QUESTION),
                               (QUESTION, MODEL)):
        contrast = {"selection": method, "comparator_selection": comparator,
                    "macro_gain": float(queries[method]["macro_accuracy"]) - float(queries[comparator]["macro_accuracy"]),
                    "micro_gain": float(queries[method]["micro_accuracy"]) - float(queries[comparator]["micro_accuracy"]),
                    "macro_ci_low": "", "macro_ci_high": "", "positive_seed_count": ""}
        if method == QUESTION and comparator == MODEL:
            correctness = [np.stack([row["correct"][0] for row in results[policy, PRIMARY, ROUTER]])
                           for policy in (method, comparator)]
            contrast.update(paired_queries(*correctness, domains))
        contrasts.append(contrast)
    atomic_csv(output / "mechanism_summary.csv", summary)
    atomic_csv(output / "mechanism_contrasts.csv", contrasts)


def run(run_dir, checkpoints, support_protocol, output, part, trace_output):
    main_study = part == "main"
    if main_study:
        policies, budgets, thresholds = (CONTROL, ACTIVE), HORIZONS, THRESHOLDS
    else:
        policies, budgets, thresholds = (QUESTION, MODEL), (PRIMARY,), ()
        with (output / "summary.csv").open(encoding="utf-8", newline="") as file:
            archived_rows = [row for row in csv.DictReader(file) if int(row["total_probes"]) == PRIMARY]
    dataset, split, config, _, _, _ = input_context(run_dir)
    protocol = read_json(support_protocol)["protocol"]
    positions = {int(dataset.prompt_ids[row]): row for row in split.train_items}
    support = np.array([positions[item] for item in protocol["candidate_prompt_ids"]])
    refs = np.array([positions[item] for item in protocol["reference_prompt_ids"]])
    target = split.val_items
    rows = split.val_models[np.argsort(dataset.model_ids[split.val_models])]
    model_ids = dataset.model_ids[rows]
    candidate_domains, reference_domains, domains = [dataset.benchmark_ids[index] for index in (support, refs, target)]
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    results, stopping = defaultdict(list), defaultdict(list)
    allocation, paths, crossing, scoring = [], [], [], {}
    for seed in config["seeds"]:
        decoder, _ = load_decoder(checkpoints / f"{B}_{seed}.pt")
        candidate = decoder.export_features(dataset.embeddings[support], candidate_domains)
        reference = decoder.export_features(dataset.embeddings[refs], reference_domains)
        candidate = tuple(value.astype(np.float64) for value in candidate)
        reference = tuple(value.astype(np.float64) for value in reference)
        prior = latent_prior(decoder, config, PRIOR_STRENGTH)
        order = select_probe_order(candidate[1], candidate_domains, np.arange(len(support)), len(support),
                                   "balanced_random", seed)

        def observe(model, question):
            return float(dataset.labels[rows[model], support[question]])

        for policy in policies:
            horizons, stops, trace = allocate(policy, model_ids, seed, order, candidate_domains, candidate,
                                               reference, prior, observe, options, budgets, thresholds)
            if trace_output is not None:
                paths.extend({"seed": seed, "selector": policy, "total_probes": total,
                              "posterior_upper_bound": bound, "local_cpu_elapsed_seconds": seconds}
                             for total, bound, seconds in trace)
            for threshold, (total, state) in stops.items():
                crossing.append({"seed": seed, "selector": policy, "threshold": threshold, "total_probes": total,
                                 "posterior_upper_bound": state[-1], "reason": "cap" if total == HORIZONS[-1] else "threshold"})
            # The acquisition function only receives the callback and historical banks.
            if seed not in scoring:
                scoring[seed] = decoder.export_features(dataset.embeddings[target], domains), reference
                truth = dataset.labels[np.ix_(rows, target)]
            query, reference = scoring[seed]
            for total, state in horizons.items():
                counts, _, _, _, runtime, bound = state
                allocation.extend({"seed": seed, "selector": policy, "total_probes": total, "model_id": int(model),
                                   "probe_count": int(count), "posterior_upper_bound": bound,
                                   "local_cpu_elapsed_seconds": runtime} for model, count in zip(model_ids, counts))
                for method, outcome in score(state, query, reference, truth, domains, reference_domains, candidate_domains).items():
                    results[policy, total, method].append(outcome)
            stopping[policy].append(stops)
            print(f"Seed {seed} used {budgets[-1]} responses in {trace[-1][2]:.2f} seconds ({policy})", flush=True)
    summary, paired = evaluate(results, domains)
    if trace_output is not None:
        atomic_csv(trace_output, paths)
    if not main_study:
        mechanism_report(output, summary, archived_rows, results, domains)
        atomic_csv(output / "mechanism_allocation.csv", allocation)
        atomic_json(output / "mechanism_protocol.json", {
            "phase": "Exploratory EmbedLLM validation decomposition at the archived primary budget.",
            "main_comparator_source": "summary.csv in the same output directory. It stores aggregate accuracy, without per-query correctness.",
            "model_ids": model_ids, "seeds": config["seeds"], "questions": len(target), "total_probes": PRIMARY,
            "question_selection_only": "Lowest-count model (source-ID tie break) and its best smooth probe from the same seeded shortlist as joint selection.",
            "model_allocation_only": "Choose the best smooth score among each eligible model's next probe from the exact archived balanced-random order.",
            "shared_protocol": "Same warmup, bank, benchmark quota order, checkpoint, cap, MAP and query selection as the archived main study.",
            "paired_interval": "Only between the two new arms, conditional on the three seeds, with queries resampled within benchmark.",
            "stopping_quality_scored": False,
        })
        return
    primary = [row for row in paired if row["total_probes"] == PRIMARY]
    gate = all(row["macro_gain"] >= MIN_GAIN and row["macro_ci_low"] > 0 and row["positive_seed_count"] >= 2
               for row in primary)
    stop_comparisons = []
    if gate:
        best = max(summary, key=lambda row: (row["macro_accuracy"], -row["total_probes"]))
        baseline = np.stack([value["correct"][0] for value in results[best["selection"], best["total_probes"], best["method"]]])
        for policy, runs in stopping.items():
            for threshold in THRESHOLDS:
                correctness, totals = [], []
                for seed, run_states in zip(config["seeds"], runs):
                    total, state = run_states[threshold]
                    query, reference = scoring[seed]
                    correctness.append(score(state, query, reference, truth, domains,
                                             reference_domains, candidate_domains)[ROUTER]["correct"][0])
                    totals.append(total)
                effect = paired_queries(np.stack(correctness), baseline, domains)
                saving = 1 - np.mean(totals) / best["total_probes"]
                stop_comparisons.append({"selector": policy, "threshold": threshold, "mean_total": np.mean(totals),
                                         "saving_vs_best_fixed": saving, **effect,
                                         "met": bool(saving >= MIN_STOP_SAVING and effect["macro_ci_low"] >= MIN_STOP_LOWER_BOUND)})
        atomic_csv(output / "stopping_quality.csv", stop_comparisons)
    for name, records in (("summary", summary), ("paired", paired), ("allocation", allocation),
                          ("first_crossing", crossing)):
        atomic_csv(output / f"{name}.csv", records)
    atomic_json(output / "protocol.json", {
        "phase": "EmbedLLM validation only", "model_ids": model_ids, "seeds": config["seeds"],
        "questions": len(target), "checkpoint": B, "prior_strength": PRIOR_STRENGTH,
        "warmup_per_model": WARMUP, "cap_per_model": CAP, "horizons": HORIZONS, "primary": PRIMARY,
        "candidates": len(support), "text_reference": len(refs), "shortlist": SHORTLIST,
        "round_robin": "Fixed balanced order per model, lowest-count model first. Equal ties follow source ID order.",
        "joint": "Deterministically shuffled shortlist per seed, model, next count and benchmark. Rank-one smooth reduction in Gaussian pairwise-sum upper bound. Zero-score fallback favors lowest count and first shortlist item.",
        "query_policy": "Select one model by maximum predicted correctness, with no incumbent or admission margin.",
        "fixed_policies": "One model for all queries or one model per benchmark, using empirical probe means or predicted reference means. Per-benchmark choices use the incoming benchmark label.",
        "bootstrap": "Paired validation queries resampled within benchmark after averaging seeds.",
        "fixed_count_gate": {"minimum_macro_gain": MIN_GAIN, "minimum_lower_ci_above": 0,
                             "minimum_positive_seeds": 2, "met": bool(gate)},
        "stopping_gate": {"thresholds": THRESHOLDS, "minimum_saving": MIN_STOP_SAVING,
                          "minimum_lower_ci": MIN_STOP_LOWER_BOUND, "quality_scored": bool(gate)},
    })
    print(f"Validation matched-budget improvement gate met: {gate}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--support-protocol", type=Path, default=SUPPORT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--part", choices=("main", "mechanisms"), default="main",
                        help="Mechanisms uses archived main rows from summary.csv in the same output directory.")
    parser.add_argument("--trace-output", type=Path, help="Optional path for the full per-probe uncertainty trace.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.run, args.checkpoints, args.support_protocol, args.output, args.part, args.trace_output)


if __name__ == "__main__":
    main()
