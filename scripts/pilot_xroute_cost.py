"""Fit and score the xRouteBench training-only cost study in two stages."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldstart.adaptation import sigmoid
from coldstart.models import load_decoder
from coldstart.probes import select_probe_order
from pilot_routing_learning import B, latent_prior, model_latents
from route_query import interaction_scores, routing_features
from run_experiment import atomic_csv, atomic_json
from run_external_routing import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_queries, read_json
from run_routing import benchmark_means
from run_xroute_onboarding import (EXCLUDED, PRIOR_STRENGTH, SEEDS, TASKS,
                                   anchor_roster, embedllm_overlap_keys, parquet, partition)


RAW = ROOT / "data/raw/xroutebench"
FEATURES = ROOT / "artifacts/xroute_onboarding/features.npz"
CHECKPOINTS = ROOT / "artifacts/routing_learning/checkpoints"
CONFIG = ROOT / "artifacts/main_v2/effective_config.json"
OUT = ROOT / "artifacts/xroute_cost"
PROBES = 100
QUALITY_TOLERANCE = 0.005
MIN_BILL_SAVING = 0.10
LAMBDA = (0, 0.01, 0.025, 0.05, 0.10, 0.20)
RATE_SCALE = 1_000_000
PRICE_STEP = 100
ANCHOR = 0.75
INCUMBENT = "incumbent_only"
GLOBAL = "global_convex"
DOMAIN = "domain_convex"
LENGTH = "domain_length_price"
SCALAR = "scalar_difficulty_price"
EMBEDDING = "embedding_query_price"
STRONGEST = "support_selected_control"
BUDGET_MATCH = "support_cost_matched_control"
BROADER = "broader_roster_fixed"
CONTROLS = (GLOBAL, DOMAIN, LENGTH, SCALAR)
METHODS = (INCUMBENT, *CONTROLS, EMBEDDING, STRONGEST, BUDGET_MATCH)


def release():
    rates = pq.read_table(RAW / "models.parquet",
                          columns=["model_name", "input_price_per_1m", "output_price_per_1m"]).to_pylist()
    roster = sorted(row["model_name"] for row in rates if row["model_name"] not in EXCLUDED)
    price = {row["model_name"]: (row["input_price_per_1m"], row["output_price_per_1m"]) for row in rates}
    overlap = embedllm_overlap_keys(ROOT / "data/raw/question_order.csv", ROOT / "data/processed/prompts.csv")
    identity = ["task_name", "query", "model_name"]
    clean, _, _ = partition(parquet(RAW / "train.parquet", identity), set(price), overlap, labeled=False)
    domains = np.array([TASKS.index(key[0]) for key, _, _ in clean])
    return clean, domains, np.array(roster), np.array([price[model] for model in roster])


def records(clean, positions, models, prices):
    """Read outcomes and token measurements only for explicitly listed official-training groups."""
    model_rows = {model: row for row, model in enumerate(models)}
    keys = {(row["task_name"], row["query"], row["model_name"]): (model_rows[row["model_name"]], column)
            for column, position in enumerate(positions) for row in clean[position][2]
            if row["model_name"] in model_rows}
    measured = ("performance", "input_tokens", "output_tokens")
    raw = pq.read_table(RAW / "train.parquet",
                        columns=["task_name", "query", "model_name", *measured],
                        filters=[("task_name", "in", list(TASKS)),
                                 ("query", "in", sorted({key[1] for key in keys})),
                                 ("model_name", "in", models.tolist())])
    values = np.full((len(models), len(positions), len(measured)), np.nan)
    for row in raw.to_pylist():
        cell = keys.get((row["task_name"], row["query"], row["model_name"]))
        if cell is not None:
            values[cell] = [row[key] for key in measured]
    labels, inputs, outputs = np.moveaxis(values, -1, 0)
    if (not np.isin(labels, [0, 1]).all() or not np.isfinite(values).all()
            or (inputs <= 0).any() or (outputs < 0).any() or not (inputs == inputs[0]).all()):
        raise ValueError("Expected complete binary labels and shared prompt input counts in xRouteBench training.")
    bill = inputs * prices[:, 0, None] + outputs * prices[:, 1, None]
    return labels, inputs[0], outputs, bill


def interpolate(values, mixture):
    low, high, share = mixture
    return values[low] + share * (values[high] - values[low])


def minimize_mixture(objective, constraint, floor):
    """Minimize one linear mean over the portfolio above a single constraint."""
    high = np.flatnonzero(constraint >= floor - 1e-12)
    if not len(high):
        candidate = int(np.lexsort((objective, -constraint))[0])
        return (candidate, candidate, 0.0), False
    low = np.flatnonzero(constraint < floor - 1e-12)
    candidates = [(int(i), int(i), 0.0) for i in high]
    candidates += [(int(i), int(j), float((floor - constraint[i]) / (constraint[j] - constraint[i])))
                   for i in low for j in high if constraint[j] > floor]
    best = min(candidates, key=lambda m: (interpolate(objective, m), -interpolate(constraint, m), m))
    return best, True


def predictions(decoder_features, priors, config, domains, chars, prices, bank_rows, bank, reference, probe, apply):
    labels, inputs, outputs, _ = bank
    slots, profile_domains = bank_rows[probe], domains[probe]
    response = labels[:, slots]
    offsets, design = decoder_features
    fitted = model_latents((offsets[probe], design[probe]), response, profile_domains, priors, config)
    embedding = sigmoid(interaction_scores(fitted, design[apply]) + offsets[apply])
    ref_offsets = offsets[reference]
    ref_domains = domains[reference]
    reference_weight = 1 / (len(TASKS) * np.bincount(ref_domains)[ref_domains])
    reference_accuracy = sigmoid(interaction_scores(fitted, design[reference]) + ref_offsets) @ reference_weight
    abilities = np.array([brentq(lambda a: sigmoid(ref_offsets + a) @ reference_weight - target, -50, 50)
                          for target in reference_accuracy])
    if not np.allclose(sigmoid(ref_offsets + abilities[:, None]) @ reference_weight, reference_accuracy,
                       rtol=0, atol=1e-9):
        raise RuntimeError("Scalar reference calibration did not match the embedding mean.")
    scalar = sigmoid(offsets[apply] + abilities[:, None])
    per_task = benchmark_means(response, profile_domains)
    length_design = np.stack([np.ones(len(chars)), chars], axis=1)
    coefficients = np.linalg.lstsq(length_design[probe], inputs[slots], rcond=None)[0]
    estimated_input = np.maximum(length_design[apply] @ coefficients, 1.0)
    estimated_output = benchmark_means(outputs[:, slots], profile_domains)[:, domains[apply]]
    cost = prices[:, 0, None] * estimated_input + prices[:, 1, None] * estimated_output
    return embedding, scalar, per_task[:, domains[apply]], cost


def policies(prediction, domains, incumbent, newcomer):
    embedding, scalar, per_task, cost = prediction
    fixed = np.broadcast_to(np.array([[0.0], [1.0]]), (2, len(domains)))
    task_masks = ((np.arange(1 << len(TASKS))[:, None] >> domains) & 1).astype(np.float64)
    delta_cost = (cost[newcomer] - cost[incumbent]) / PRICE_STEP
    cheapest = delta_cost < 0
    choices = {INCUMBENT: fixed[:1], GLOBAL: fixed, DOMAIN: task_masks}
    for method, quality in ((EMBEDDING, embedding), (LENGTH, per_task), (SCALAR, scalar)):
        gap = quality[newcomer] - quality[incumbent]
        threshold_choices = gap - np.array(LAMBDA)[:, None] * delta_cost > 0
        choices[method] = np.vstack((fixed, threshold_choices, cheapest))
    return choices


def support_frontier(policies, labels, bills, domains, incumbent, newcomer):
    quality, bill = {}, {}
    for method, choices in policies.items():
        observed_quality = np.where(choices, labels[newcomer], labels[incumbent])
        observed_cost = np.where(choices, bills[newcomer], bills[incumbent])
        quality[method] = benchmark_means(observed_quality, domains).mean(axis=-1)
        bill[method] = benchmark_means(observed_cost, domains).mean(axis=-1)
    return quality, bill


def select_policies(quality, bill, floor):
    result = {}
    for method in (INCUMBENT, *CONTROLS, EMBEDDING):
        mix, feasible = minimize_mixture(bill[method], quality[method], floor)
        if not feasible:
            raise ValueError("Always-incumbent should make the support quality floor attainable.")
        result[method] = method, mix, feasible
    strongest = min(CONTROLS, key=lambda method: interpolate(bill[method], result[method][1]))
    result[STRONGEST] = result[strongest]
    budget = interpolate(bill[EMBEDDING], result[EMBEDDING][1])
    candidates = []
    for method in CONTROLS:
        mix, feasible = minimize_mixture(-quality[method], -bill[method], -budget)
        candidates.append((method, mix, feasible))
    feasible = [row for row in candidates if row[2]]
    if feasible:
        result[BUDGET_MATCH] = max(feasible, key=lambda row: interpolate(quality[row[0]], row[1]))
    else:
        result[BUDGET_MATCH] = min(candidates, key=lambda row: interpolate(bill[row[0]], row[1]))
    return result


def fit():
    clean, domains, models, prices = release()
    with np.load(FEATURES, allow_pickle=False) as data:
        vectors, candidates, reserved = data["train_vectors"], data["candidates"], data["reference"]
        if not np.array_equal(data["train_domains"], domains):
            raise ValueError("Training features have different xRouteBench group positions.")
    diagnosis = np.setdiff1d(np.arange(len(clean)), np.r_[candidates, reserved])
    if (len(candidates), len(reserved), len(diagnosis), len(models)) != (500, 160, 1438, 15):
        raise ValueError("Unexpected grouped xRouteBench training partition.")
    chars = np.array([len(rows[0]["query"]) for _, _, rows in clean], dtype=np.float64)
    bank = records(clean, candidates, models, prices)
    bank_rows = np.full(len(clean), -1)
    bank_rows[candidates] = np.arange(len(candidates))
    incumbent_name = next(a["model"] for a in anchor_roster(models, bank[0], domains[candidates])
                          if a["quantile"] == ANCHOR)
    incumbent = int(np.flatnonzero(models == incumbent_name)[0])
    newcomers = np.flatnonzero(np.arange(len(models)) != incumbent)
    config = read_json(CONFIG)
    routes = np.empty((len(METHODS), len(SEEDS), len(newcomers), len(diagnosis)))
    probe_bill = np.empty((len(SEEDS), len(newcomers)))
    broader, selected, orders = [], [], []

    for row, seed in enumerate(SEEDS):
        decoder, _ = load_decoder(CHECKPOINTS / f"{B}_{seed}.pt")
        decoder_features = routing_features(decoder, vectors)
        prior = latent_prior(decoder, config, PRIOR_STRENGTH)
        probe = select_probe_order(vectors, domains, candidates, PROBES, "balanced_random", seed)
        folds = np.split(probe, 2)
        if any(not np.all(np.bincount(domains[fold], minlength=len(TASKS)) == 10) for fold in folds):
            raise ValueError("The support folds must each have exactly ten prompts per task.")
        arguments = decoder_features, prior, config, domains, chars, prices, bank_rows, bank, reserved
        crossfit = [predictions(*arguments, train, holdout) for train, holdout in (folds[::-1], folds)]
        oof = tuple(np.concatenate(arrays, axis=-1) for arrays in zip(*crossfit))
        trained = predictions(*arguments, probe, diagnosis)
        labels, bills = bank[0][:, bank_rows[probe]], bank[3][:, bank_rows[probe]]
        support_quality = benchmark_means(labels, domains[probe]).mean(axis=-1)
        support_bill = benchmark_means(bills, domains[probe]).mean(axis=-1)
        floor = support_quality[incumbent] - QUALITY_TOLERANCE
        qualified = np.flatnonzero(support_quality >= floor)
        broader.append(qualified[support_bill[qualified].argmin()])
        orders.append(probe)
        for column, newcomer in enumerate(newcomers):
            probe_bill[row, column] = bills[newcomer].sum()
            oof_policies = policies(oof, domains[probe], incumbent, newcomer)
            deployment = policies(trained, domains[diagnosis], incumbent, newcomer)
            quality, bill = support_frontier(oof_policies, labels, bills, domains[probe], incumbent, newcomer)
            fitted = select_policies(quality, bill, floor)
            for m, method in enumerate(METHODS):
                family, mix, feasible = fitted[method]
                routes[m, row, column] = interpolate(deployment[family], mix)
                selected.append({"seed": seed, "newcomer": models[newcomer], "method": method, "family": family,
                                 "endpoint_a": mix[0], "endpoint_b": mix[1], "probability_b": mix[2],
                                 "support_cost_micro_units": interpolate(bill[family], mix),
                                 "support_macro_accuracy": interpolate(quality[family], mix),
                                 "support_quality_floor": floor, "support_constraint_feasible": feasible})
        print(f"Seed {seed} froze {len(newcomers)} arrivals on 100 support labels per model", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "frozen_routes.npz", routes=routes, incumbent=incumbent, newcomers=newcomers,
                        broader=broader, probe_bill=probe_bill, models=models, diagnosis=diagnosis,
                        domains=domains[diagnosis], probe_groups=orders)
    atomic_csv(OUT / "selection.csv", selected)
    print(f"Support fit saved to {OUT}", flush=True)


def bill_interval(left, right, domains):
    left = left.mean(axis=(0, 1))
    right = right.mean(axis=(0, 1))
    means = np.zeros((2, BOOTSTRAP_DRAWS))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for domain in np.unique(domains):
        group = np.flatnonzero(domains == domain)
        samples = rng.choice(group, (BOOTSTRAP_DRAWS, len(group)))
        means += np.array([left[samples].mean(axis=1), right[samples].mean(axis=1)])
    return np.quantile(1 - means[0] / means[1], [.025, .975])


def score():
    with np.load(OUT / "frozen_routes.npz", allow_pickle=False) as frozen:
        routes, incumbent, newcomers, broader, probe_bill, models, diagnosis, domains = (
            frozen[key] for key in ("routes", "incumbent", "newcomers", "broader", "probe_bill",
                                    "models", "diagnosis", "domains"))
    clean, release_domains, release_models, prices = release()
    if not np.array_equal(models, release_models) or not np.array_equal(domains, release_domains[diagnosis]):
        raise ValueError("The released training data and frozen routes differ.")
    labels, _, _, bills = records(clean, diagnosis, models, prices)
    correctness = labels[incumbent] + routes * (labels[newcomers] - labels[incumbent])
    spending = bills[incumbent] + routes * (bills[newcomers] - bills[incumbent])
    method_scores = dict(zip(METHODS, zip(correctness, spending, routes.mean(axis=(1, 2, 3)))))
    method_scores[BROADER] = (labels[broader, None, :], bills[broader, None, :],
                              float((broader != incumbent).mean()))
    bill_means = {method: benchmark_means(c, domains).mean() for method, (_, c, _) in method_scores.items()}
    summary = [{"method": method, "macro_accuracy": benchmark_means(q, domains).mean(),
                "micro_accuracy": q.mean(), "mean_bill_per_query": bill_means[method] / RATE_SCALE,
                "mean_bill_per_query_micro": c.mean() / RATE_SCALE,
                "nonincumbent_dispatch_share": share}
               for method, (q, c, share) in method_scores.items()]
    q, c, _ = method_scores[EMBEDDING]
    paired = []
    for comparator in (STRONGEST, INCUMBENT, *CONTROLS, BUDGET_MATCH):
        base_q, base_c, _ = method_scores[comparator]
        saving = 1 - bill_means[EMBEDDING] / bill_means[comparator]
        lo, hi = bill_interval(c, base_c, domains)
        paired.append({"method": EMBEDDING, "comparator": comparator,
                       **paired_queries(q.mean(axis=1), base_q.mean(axis=1), domains),
                       "relative_bill_saving": saving, "bill_saving_ci_low": lo, "bill_saving_ci_high": hi,
                       "mean_bill_gain_per_query": (bill_means[comparator] - bill_means[EMBEDDING]) / RATE_SCALE})
    contrasts = {row["comparator"]: row for row in paired}
    achieved = (contrasts[STRONGEST]["relative_bill_saving"] >= MIN_BILL_SAVING
                and all(contrasts[method]["macro_ci_low"] >= -QUALITY_TOLERANCE
                        for method in (STRONGEST, INCUMBENT)))
    inc_q, inc_c, _ = method_scores[INCUMBENT]
    quality_delta = benchmark_means(q - inc_q, domains).mean(axis=-1)
    bill_delta = benchmark_means(inc_c - c, domains).mean(axis=-1)
    per_arrival = []
    for index, seed in enumerate(SEEDS):
        for row, newcomer in enumerate(newcomers):
            delta_quality = quality_delta[index, row]
            delta_bill = bill_delta[index, row]
            onboarding = probe_bill[index, row]
            break_even = (int(np.ceil(onboarding / delta_bill))
                          if delta_quality >= -QUALITY_TOLERANCE and delta_bill > 0 else "")
            per_arrival.append({"seed": seed, "newcomer": models[newcomer],
                                "probe_bill": onboarding / RATE_SCALE,
                                "macro_gain_vs_incumbent": delta_quality,
                                "meets_observed_quality_floor": bool(delta_quality >= -QUALITY_TOLERANCE),
                                "savings_per_query_vs_incumbent": delta_bill / RATE_SCALE,
                                "point_forecast_break_even_queries": break_even})
    atomic_csv(OUT / "summary.csv", summary)
    atomic_csv(OUT / "paired.csv", paired)
    atomic_csv(OUT / "amortization.csv", per_arrival)
    atomic_json(OUT / "criterion.json", {
        "outcome": "passed" if achieved else "failed",
        "relative_bill_saving": contrasts[STRONGEST]["relative_bill_saving"],
        "macro_quality_lower_vs_control": contrasts[STRONGEST]["macro_ci_low"],
        "macro_quality_lower_vs_incumbent": contrasts[INCUMBENT]["macro_ci_low"],
        "macro_quality_point_vs_incumbent": contrasts[INCUMBENT]["macro_gain"],
        "point_floor_vs_incumbent_met": bool(contrasts[INCUMBENT]["macro_gain"] >= -QUALITY_TOLERANCE),
        "scope": "Official xRouteBench training, conditional on models, seeds and fitted policies.",
        "amortization": "Point forecasts use actual released newcomer tokens on 100 probes, positive incumbent bill savings, and the per-arrival observed quality floor. Matched-evidence controls share probe costs.",
    })
    print(f"Development gate {'passed' if achieved else 'failed'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("fit", "score"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (fit if args.stage == "fit" else score)()
