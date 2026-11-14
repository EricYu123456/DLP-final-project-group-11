"""Select both onboarding questions and their count for a frozen LLM router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from run_routing import (ADD_ONE, MARGINS, ROOT, add_one_outcomes, benchmark_means,
                         summarize_outcomes)
from coldstart.adaptation import adapt_map, sigmoid
from coldstart.models import load_decoder
from coldstart.probes import select_probe_order
from pilot_routing_acquisition import acquisition_banks, expected_switch_value
from pilot_routing_learning import B, input_context, latent_prior, model_latents
from route_query import choose_models, interaction_scores
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries


WARMUP = 10
CAP = 200
SCREENED_PER_STEP = 48
EXACT_PER_STEP = 8
FIXED_BUDGETS = (50, 100, CAP)
THRESHOLDS = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05)
MAX_VALIDATION_LOSS = 0.001
ZERO = "incumbent_only"
ADAPTIVE = "sequential_probes_and_count"
ADAPTIVE_FIXED = "sequential_probes_fixed_200"
MATCHED_RANDOM = "balanced_random_same_attempts"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_summary(fit, reference_design, reference_offsets, incumbent, margin):
    probability = sigmoid(reference_offsets + reference_design @ fit.latent)
    gap = probability - incumbent - margin
    gradient = (probability * (1 - probability))[:, None] * reference_design
    variance = np.einsum("ij,ij->i", gradient @ fit.covariance, gradient)
    evpi = expected_switch_value(gap, np.sqrt(np.maximum(variance, 0))).mean()
    return probability, gap, gradient, float(evpi)


def select_probe(eligible, design, offsets, selected_design, selected_offsets, observed,
                 reference_design, reference_offsets, fit, prior, state, solver_options):
    """Screen candidates locally, then compare two hypothetical outcomes by MAP."""
    reference_probability, gap, gradients, _ = state
    candidates = design[eligible]
    probability = sigmoid(offsets[eligible] + candidates @ fit.latent)
    weight = probability * (1 - probability)
    projected = candidates @ fit.covariance
    information = weight * np.einsum("ij,ij->i", projected, candidates)
    deviations = np.sqrt(weight / (1 + information))[:, None] * np.abs(projected @ gradients.T)
    fisher = expected_switch_value(gap, deviations).mean(axis=1)
    best = np.argsort(-fisher, kind="stable")[:EXACT_PER_STEP]
    best_value = -np.inf
    for index in best:
        candidate = eligible[index]
        new_design = np.vstack([selected_design, design[candidate]])
        new_offsets = np.r_[selected_offsets, offsets[candidate]]
        outcomes = [adapt_map(new_design, new_offsets, np.r_[observed, response], *prior,
                              initial=fit.latent, **solver_options) for response in (0, 1)]
        q = probability[index]
        after = sigmoid(np.stack([value.latent for value in outcomes]) @ reference_design.T + reference_offsets)
        # Center the two plug-in MAP predictions so their expected value is the
        # current prediction. This removes otherwise spurious mean drift.
        delta = after - reference_probability
        centered = delta - ((1 - q) * delta[0] + q * delta[1])
        expected = ((1 - q) * np.maximum(gap + centered[0], 0)
                    + q * np.maximum(gap + centered[1], 0))
        value = (expected - np.maximum(gap, 0)).mean()
        if value > best_value:
            best_value, chosen, chosen_outcomes = value, candidate, outcomes
    return chosen, chosen_outcomes


def probe_steps(order, domains, design, offsets, reference, incumbent, prior, margin,
                solver_options, shortlist_seed, tolerance, *, selection="exact"):
    """Yield one selected probe and receive its response before selecting again."""
    if selection not in ("exact", "random"):
        raise ValueError("Probe selection must be exact or random.")
    rng = np.random.default_rng(shortlist_seed)
    reference_design, reference_offsets = reference
    selected = list(order[:WARMUP])
    started = perf_counter()
    observed = np.empty(WARMUP, dtype=np.float64)
    for index, probe in enumerate(selected):
        observed[index] = yield probe
    fit = adapt_map(design[selected], offsets[selected], observed, *prior, **solver_options)
    available = np.ones(len(order), dtype=bool)
    available[selected] = False
    trace, latents = [], []
    for budget in range(WARMUP, CAP + 1):
        state = state_summary(fit, reference_design, reference_offsets, incumbent, margin)
        trace.append((state[-1], float(state[1].max()), perf_counter() - started))
        latents.append(fit.latent)
        if budget == CAP or (tolerance > 0 and state[-1] <= tolerance):
            if budget == CAP:
                stop_reason = "cap"
            else:
                stop_reason = "sufficient_reference_evidence" if state[1].max() > 0 else "weak_on_reference"
            return np.array(latents), trace, selected, stop_reason
        if selection == "exact":
            domain = domains[order[budget]]
            eligible = np.flatnonzero(available & (domains == domain))
            screened = rng.permutation(eligible)[:SCREENED_PER_STEP]
            chosen, after = select_probe(screened, design, offsets, design[selected], offsets[selected],
                                         observed, reference_design, reference_offsets, fit, prior, state, solver_options)
        else:
            chosen = order[budget]
        actual = yield chosen
        selected.append(chosen)
        observed = np.r_[observed, actual]
        available[chosen] = False
        if np.isfinite(actual):
            if selection == "exact":
                fit = after[int(actual)]
            else:
                fit = adapt_map(design[selected], offsets[selected], observed, *prior,
                                initial=fit.latent, **solver_options)


def trajectory(order, domains, design, offsets, observe, reference, incumbent, prior, margin,
               solver_options, shortlist_seed, tolerance, *, selection="exact"):
    """Replay one arrival, invoking the response callback only for chosen probes."""
    steps = probe_steps(order, domains, design, offsets, reference, incumbent, prior, margin,
                        solver_options, shortlist_seed, tolerance, selection=selection)
    try:
        candidate = next(steps)
        while True:
            candidate = steps.send(observe(candidate))
    except StopIteration as done:
        return done.value


def stop_indices(records, tolerance):
    if tolerance == 0:
        return np.full(records.shape[:-1], CAP - WARMUP, dtype=np.int64)
    meets = records <= tolerance
    return np.where(meets.any(axis=-1), meets.argmax(axis=-1), CAP - WARMUP)


def margin_choice(advantage, delta, domains):
    values = [benchmark_means(((advantage > value) * delta).mean(axis=(0, 1)), domains).mean()
              for value in MARGINS]
    winner = max(range(len(values)), key=lambda i: (values[i], MARGINS[i]))
    return MARGINS[winner], [{"margin": margin, "macro_gain_over_incumbent": value}
                              for margin, value in zip(MARGINS, values)]


def tuning(episodes, delta, domains, margin):
    evpi = np.asarray([[np.asarray(trial["trace"])[:, 0] for trial in seed] for seed in episodes])
    rows = []
    for tolerance in THRESHOLDS:
        prefix = stop_indices(evpi, tolerance)
        correct = []
        for seed, trials in enumerate(episodes):
            per_model = []
            for model, trial in enumerate(trials):
                latent = trial["latents"][prefix[seed, model]]
                prob = sigmoid(interaction_scores(latent, trial["design"]) + trial["offsets"])
                per_model.append((prob - trial["incumbent"] > margin) * delta[model])
            correct.append(per_model)
        macro = benchmark_means(np.asarray(correct).mean(axis=(0, 1)), domains).mean()
        rows.append({"evpi_tolerance": tolerance, "mean_probe_attempts": float(prefix.mean() + WARMUP),
                     "macro_gain_over_incumbent": float(macro)})
    reference = rows[0]["macro_gain_over_incumbent"]
    for row in rows:
        row["macro_gain_over_fixed_200"] = row["macro_gain_over_incumbent"] - reference
        row["eligible"] = row["macro_gain_over_fixed_200"] >= -MAX_VALIDATION_LOSS
    selected = min((row for row in rows if row["eligible"]), key=lambda row: (
        row["mean_probe_attempts"], -row["macro_gain_over_incumbent"], row["evpi_tolerance"]))
    return selected["evpi_tolerance"], rows


def probabilities(latent, episode):
    return sigmoid(interaction_scores(latent, episode["design"]) + episode["offsets"])


def fit_random(episode, budget, solver_options):
    probes = episode["random_order"][:budget]
    started = perf_counter()
    fit = adapt_map(episode["probe_design"][probes], episode["probe_offsets"][probes],
                    episode["observe"](probes), *episode["prior"], **solver_options)
    return fit.latent, perf_counter() - started


def onboarding_context(dataset, split, config, learning, incumbent, reference, candidates, seed, checkpoint):
    decoder, metadata = load_decoder(checkpoint)
    if not any(all(metadata[key] == value for key, value in row.items()) for row in learning["checkpoints"]):
        raise ValueError(f"Checkpoint does not match the frozen routing selection: {checkpoint}")
    support = split.train_items
    support_features = decoder.export_features(dataset.embeddings[support], dataset.benchmark_ids[support])
    prior = latent_prior(decoder, config, learning["protocol"]["prior_strength"])
    incumbent_latent = model_latents(support_features, dataset.labels[incumbent, support][None, :],
                                    dataset.benchmark_ids[support], prior, config)[0]
    ref_offsets, ref_design = decoder.export_features(dataset.embeddings[reference], dataset.benchmark_ids[reference])
    ref_probability = sigmoid(interaction_scores(incumbent_latent, ref_design) + ref_offsets)
    candidate_offsets, candidate_design = decoder.export_features(dataset.embeddings[candidates], dataset.benchmark_ids[candidates])
    candidate_domains = dataset.benchmark_ids[candidates]
    order = select_probe_order(candidate_design, candidate_domains, np.arange(len(candidates)),
                               len(candidates), "balanced_random", seed)
    return decoder, incumbent_latent, {
        "reference": (ref_design.astype(np.float64), ref_offsets.astype(np.float64)),
        "reference_incumbent": ref_probability, "random_order": order, "probe_domains": candidate_domains,
        "prior": prior, "probe_design": candidate_design.astype(np.float64),
        "probe_offsets": candidate_offsets.astype(np.float64)}


def seed_episodes(dataset, split, config, learning, incumbent, rows, target, reference, candidates, seed, args):
    checkpoint = args.checkpoints / f"{B}_{seed}.pt"
    decoder, incumbent_latent, shared = onboarding_context(dataset, split, config, learning, incumbent,
                                                           reference, candidates, seed, checkpoint)
    query_offsets, query_design = decoder.export_features(dataset.embeddings[target], dataset.benchmark_ids[target])
    incumbent_query = sigmoid(interaction_scores(incumbent_latent, query_design) + query_offsets)
    shared.update(incumbent=incumbent_query, offsets=query_offsets, design=query_design)
    episodes = []
    for row in rows:
        # Explicit callback keeps each model's unrevealed support and all query
        # labels outside the policy. Retrospective comparators use their own calls.
        def observe(probe, model=row):
            return dataset.labels[model, candidates[probe]]

        episodes.append({**shared, "seed": seed, "model_id": int(dataset.model_ids[row]), "observe": observe})
    return episodes


def add_trajectories(episodes, solver_options, margin, tolerance, phase):
    for trials in episodes:
        for episode in trials:
            seed = np.random.SeedSequence([episode["seed"], episode["model_id"]])
            calls = []

            def observe(probe):
                calls.append(probe)
                return episode["observe"](probe)

            inputs = (episode["random_order"], episode["probe_domains"], episode["probe_design"],
                      episode["probe_offsets"], observe, episode["reference"],
                      episode["reference_incumbent"], episode["prior"], margin, solver_options, seed)
            latents, trace, selected, reason = trajectory(*inputs, tolerance)
            if calls != selected:
                raise RuntimeError("Online response callbacks differ from the chosen probe order.")
            episode.update(latents=latents, trace=trace, selected=selected, stop_reason=reason,
                           response_callback_count=len(calls))
            if phase == "test" and len(selected) < CAP:
                # Fixed-200 is a separate retrospective comparator. Retain only
                # its terminal model fit, and verify it follows the online prefix.
                calls.clear()
                fixed_latents, fixed_trace, fixed_order, _ = trajectory(*inputs, 0.0)
                if calls != fixed_order or fixed_order[:len(selected)] != selected:
                    raise RuntimeError("The fixed comparator did not reproduce its online prefix.")
                episode["cap_latent"] = fixed_latents[-1]
                episode["cap_seconds"] = fixed_trace[-1][2]
            else:
                episode["cap_latent"] = latents[-1]
                episode["cap_seconds"] = trace[-1][2]
        print(f"{phase}: finished routing-aware trajectories for seed {trials[0]['seed']}", flush=True)


def score_episodes(episodes, candidates, dataset, truth, domains, seeds, margin, tolerance, solver_options, output, phase):
    n_seeds, n_models = len(seeds), len(episodes[0])
    target_count = len(domains)
    policies = (ZERO, ADAPTIVE, MATCHED_RANDOM, ADAPTIVE_FIXED, "balanced_random_50",
                "balanced_random_100", "balanced_random_200")
    chosen = {method: np.zeros((n_seeds, n_models, target_count), dtype=bool) for method in policies}
    trials, histories = [], []
    for si, seed in enumerate(episodes):
        for mi, episode in enumerate(seed):
            trace = np.asarray(episode["trace"], dtype=np.float64)
            if phase == "validation":
                prefix = int(stop_indices(trace[:, 0], tolerance))
            else:
                prefix = len(trace) - 1
            attempts = WARMUP + prefix
            reference_edge = trace[prefix, 1]
            reason = ("cap" if attempts == CAP else
                      "sufficient_reference_evidence" if reference_edge > 0 else "weak_on_reference")
            assessments = {
                ADAPTIVE: (episode["latents"][prefix], attempts, trace[prefix, 2]),
                ADAPTIVE_FIXED: (episode["cap_latent"], CAP, episode["cap_seconds"]),
            }
            random_fits = {}
            for budget in sorted({attempts, *FIXED_BUDGETS}):
                random_fits[budget] = fit_random(episode, budget, solver_options)
                if budget in FIXED_BUDGETS:
                    latent, seconds = random_fits[budget]
                    assessments[f"balanced_random_{budget}"] = (latent, budget, seconds)
            fit, seconds = random_fits[attempts]
            assessments[MATCHED_RANDOM] = (fit, attempts, seconds)
            incumbent = episode["incumbent"]
            for method, (latent, budget, seconds) in assessments.items():
                newcomer = probabilities(latent, episode)
                chosen[method][si, mi] = choose_models(np.stack([incumbent, newcomer]), margin).astype(bool)
                trials.append({"seed": episode["seed"], "model_id": episode["model_id"], "method": method,
                               "probe_attempts": budget, "local_cpu_elapsed_seconds": seconds,
                               "stop_reason": reason if method == ADAPTIVE else None,
                               "reference_evpi_at_stop": trace[prefix, 0] if method == ADAPTIVE else None,
                               "online_response_callback_count": episode["response_callback_count"]
                               if method == ADAPTIVE and phase == "test" else None})
            trials.append({"seed": episode["seed"], "model_id": episode["model_id"], "method": ZERO,
                           "probe_attempts": 0, "local_cpu_elapsed_seconds": 0.0})
            histories.append({"seed": episode["seed"], "model_id": episode["model_id"], "phase": phase,
                              "adaptive_prompt_ids": dataset.prompt_ids[candidates[episode["selected"]]].tolist(),
                              "attempts_decided_for_adaptive_arm": attempts,
                              "first_evpi_budget": WARMUP,
                              "reference_evpi": trace[:, 0].tolist(),
                              "max_reference_margin_edge": trace[:, 1].tolist()})

    outcomes = {method: add_one_outcomes(policy, truth, domains) for method, policy in chosen.items()}
    trial_index = {(row["seed"], row["model_id"], row["method"]): row for row in trials}
    for method, result in outcomes.items():
        for si, seed in enumerate(episodes):
            for mi, episode in enumerate(seed):
                row = trial_index[(episode["seed"], episode["model_id"], method)]
                rescue = int(result["rescue"][si, mi].sum())
                harm = int(result["harm"][si, mi].sum())
                opportunities = int(result["opportunity"][mi].sum())
                row.update(newcomer_selected=int(result["selected"][si, mi].sum()), rescues=rescue,
                           harms=harm, opportunities=opportunities, missed_opportunities=opportunities - rescue,
                           macro_accuracy=float(benchmark_means(result["correct"][si, mi], domains).mean()),
                           micro_accuracy=float(result["correct"][si, mi].mean(dtype=np.float64)))
    fixed_budget = {method: budget for method, budget in zip(policies, (0, 0, 0, CAP, *FIXED_BUDGETS))}
    summary, by_benchmark = summarize_outcomes({(ADD_ONE, fixed_budget[method], method): result
                                               for method, result in outcomes.items()},
                                              domains, dataset.benchmark_names, n_models, seeds)
    for row in summary:
        records = [trial for trial in trials if trial["method"] == row["method"]]
        probes = [trial["probe_attempts"] for trial in records]
        elapsed = [trial["local_cpu_elapsed_seconds"] for trial in records if trial["local_cpu_elapsed_seconds"] is not None]
        del row["total_probe_responses_per_trial"], row["total_probe_responses_across_newcomers"]
        if row["method"] in (ADAPTIVE, MATCHED_RANDOM):
            row["budget"] = None
        row.update(mean_probe_attempts=float(np.mean(probes)), median_probe_attempts=float(np.median(probes)),
                   min_probe_attempts=min(probes), max_probe_attempts=max(probes),
                   total_probe_attempts_across_arrivals=sum(probes),
                   mean_local_cpu_elapsed_seconds=float(np.mean(elapsed)) if elapsed else None,
                   missed_opportunities_per_trial=row["opportunities_per_trial"] - row["rescues_per_trial"])
        for reason in ("cap", "sufficient_reference_evidence", "weak_on_reference"):
            row[f"stopped_{reason}"] = sum(record.get("stop_reason") == reason for record in records)
    for row in by_benchmark:
        if row["method"] in (ADAPTIVE, MATCHED_RANDOM):
            row["budget"] = None
    paired = []
    current = outcomes[ADAPTIVE]["correct"]
    for method, result in outcomes.items():
        if method == ADAPTIVE:
            continue
        left = current.reshape(-1, target_count)
        right = result["correct"].reshape(-1, target_count)
        comparison = paired_queries(left, right, domains)
        comparison.pop("positive_seed_count")
        seed_gain = benchmark_means((current - result["correct"]).mean(axis=1), domains).mean(axis=1)
        paired.append({"method": ADAPTIVE, "comparator": method,
                       "positive_seed_count": int((seed_gain > 0).sum()), **comparison})
    for suffix, records in (("summary", summary), ("by_benchmark", by_benchmark), ("by_arrival", trials), ("paired", paired)):
        atomic_csv(output / f"{phase}_{suffix}.csv", records)
    trace_path = output / f"{phase}_probe_trace.jsonl"
    trace_path.write_text("".join(json.dumps(trial, separators=(",", ":")) + "\n" for trial in histories),
                          encoding="utf-8")


def run(args):
    dataset, split, config, _, _, _ = input_context(args.run)
    learning = read_json(args.learning_selection)
    original = read_json(args.incumbent_selection)
    incumbent_id = original["protocol"]["incumbent_model_id"]
    incumbent = int(np.flatnonzero(dataset.model_ids == incumbent_id)[0])
    if incumbent not in split.train_models:
        raise ValueError("The incumbent must belong to the established training roster.")
    reference, candidates = acquisition_banks(split.train_items, dataset.benchmark_ids)
    rows = split.val_models if args.phase == "validation" else split.test_models
    target = split.val_items if args.phase == "validation" else split.test_items
    seeds = config["seeds"]
    domains = dataset.benchmark_ids[target]
    solver_options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    episodes = [seed_episodes(dataset, split, config, learning, incumbent, rows, target, reference,
                              candidates, seed, args) for seed in seeds]
    protocol = {"incumbent_model_id": incumbent_id, "checkpoint_arm": B, "seeds": seeds,
                "selection": "Laplace-screened expected improvement, Bernoulli two-outcome warm-start MAP, centered reference effects",
                "stop": "mean posterior EVPI falls at or below tolerance; 0 disables early stopping",
                "stop_reasons": "early stop = weak if no favorable reference edge, sufficient otherwise; at ceiling = cap",
                "warmup": WARMUP, "cap": CAP, "laplace_shortlist_size": SCREENED_PER_STEP,
                "exact_lookahead_size": EXACT_PER_STEP, "benchmark_quotas": "round-robin",
                "admission_margin_grid": MARGINS, "tolerance_grid": THRESHOLDS,
                "validation_quality_constraint": MAX_VALIDATION_LOSS,
                "comparison": "Same newcomer, seed and realized probe count for matched balanced random",
                "reference_prompt_ids": dataset.prompt_ids[reference].tolist(),
                "candidate_prompt_ids": dataset.prompt_ids[candidates].tolist(),
                "balanced_random_prompt_ids": {
                    str(seed): dataset.prompt_ids[candidates[trials[0]["random_order"][:CAP]]].tolist()
                    for seed, trials in zip(seeds, episodes)},
                "validation_model_ids": dataset.model_ids[split.val_models].tolist(),
                "test_model_ids": dataset.model_ids[split.test_models].tolist(),
                "validation_queries": len(split.val_items), "test_queries": len(split.test_items),
                "runtime": "Local elapsed selection and MAP fit on CPU with archived response lookups, no downstream inference",
                "interval": "paired queries stratified by benchmark after averaging arrivals and seeds, conditional on roster and learned selection"}
    protocol["admission_margin_grid"] = list(MARGINS)
    protocol["tolerance_grid"] = list(THRESHOLDS)
    selection_path = args.output / "validation_selection.json"
    if args.phase == "validation":
        truth = dataset.labels[np.ix_(np.r_[incumbent, rows], target)]
        delta = truth[1:] - truth[0]
        random_advantage = np.asarray([[probabilities(fit_random(episode, CAP, solver_options)[0], episode)
                                        - episode["incumbent"] for episode in trials] for trials in episodes])
        margin, margin_rows = margin_choice(random_advantage, delta, domains)
        print(f"Validation admission margin: {margin:.4f}", flush=True)
        add_trajectories(episodes, solver_options, margin, 0.0, args.phase)
        tolerance, tolerance_rows = tuning(episodes, delta, domains, margin)
        atomic_csv(args.output / "validation_margin_grid.csv", margin_rows)
        atomic_csv(args.output / "validation_tolerance_grid.csv", tolerance_rows)
        atomic_json(selection_path, {"protocol": protocol, "admission_margin": margin, "evpi_tolerance": tolerance})
    else:
        selection = read_json(selection_path)
        if selection["protocol"] != protocol:
            raise ValueError("Test protocol differs from the validation plan.")
        margin, tolerance = selection["admission_margin"], selection["evpi_tolerance"]
        add_trajectories(episodes, solver_options, margin, tolerance, args.phase)
        truth = dataset.labels[np.ix_(np.r_[incumbent, rows], target)]
    score_episodes(episodes, candidates, dataset, truth, domains, seeds, margin, tolerance,
                   solver_options, args.output, args.phase)
    print(f"Saved {args.phase} sequential routing results to {args.output}.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--learning-selection", type=Path, default=ROOT / "reports/routing/learning/selection.json")
    parser.add_argument("--incumbent-selection", type=Path, default=ROOT / "reports/routing/validation_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/sequential")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
