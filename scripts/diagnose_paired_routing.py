"""Compare two paired 8D routing representations on EmbedLLM validation questions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from route_query import ROOT
from coldstart.models import load_decoder
from pilot_routing_learning import B, input_context
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries, read_json
from run_routing import benchmark_means


B_RIDGE = "frozen_b"
SUPERVISED = "supervised_reduced_rank"
MARGIN = 0.025
FIT_RIDGE = 50.0
MAP_RIDGE = 0.01
RANK = 7
METHODS = ("incumbent", "newcomer", "hindsight_best_fixed", "query_oracle",
           "paired_ridge", "same_fit_reference_fixed", "direct_support_fixed")
PER_BENCHMARK = "per_benchmark_support_fixed"
COMPETITIVE_MEAN = "competitive_average"


def reduced_rank_features(embeddings, labels, domains, parts):
    """Learn seven axes from centered historical responses with equal benchmark weights."""
    x = embeddings.astype(np.float64)
    y = labels.T.astype(np.float64)
    weights = 1 / (len(np.unique(domains)) * np.bincount(domains)[domains])
    mean = weights @ x
    x -= mean
    sd = np.sqrt(weights @ x ** 2)
    active = sd > 0
    scale = sd[active] * np.sqrt(x.shape[1])
    x = x[:, active] / scale
    y -= y.mean(axis=1, keepdims=True)
    y -= weights @ y
    # Cholesky whitening turns the reduced-rank regression into a truncated SVD.
    chol = np.linalg.cholesky(x.T @ (weights[:, None] * x) + MAP_RIDGE * np.eye(x.shape[1]))
    cross = x.T @ (weights[:, None] * y)
    left, _, _ = np.linalg.svd(solve_triangular(chol, cross, lower=True), full_matrices=False)
    projection = solve_triangular(chol.T, left[:, :RANK])
    return [(vectors[:, active].astype(np.float64) - mean[active]) / scale @ projection for vectors in parts]


def design_matrices(matrices, *, principal_axes):
    source = matrices[0]
    center = source.mean(axis=0)
    centered = source - center
    if principal_axes:
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        projection = basis[:RANK].T
        deviation = (centered @ projection).std(axis=0)
        axes = [(matrix - center) @ projection / deviation for matrix in matrices]
    else:
        deviation = source.std(axis=0)
        axes = [(matrix - center) / deviation for matrix in matrices]
    return [np.column_stack((np.ones(len(matrix)), matrix)) for matrix in axes]


def score(matrices, pairs, domains, methods):
    support_features, query_features, reference_features = matrices
    support_domains, query_domains, reference_domains = domains
    lhs = support_features.T @ support_features + np.diag([0] + [FIT_RIDGE] * RANK)
    outcomes = {}
    for quantile, (support_delta, query_delta) in pairs.items():
        weights = np.linalg.solve(lhs, support_features.T @ support_delta.T)
        predicted_gap = (query_features @ weights).T
        reference_gap = benchmark_means((reference_features @ weights).T, reference_domains).mean(axis=1)
        domain_support = benchmark_means(support_delta, support_domains)
        best_fixed = benchmark_means(query_delta, query_domains).mean(axis=1) > 0
        gains = (
            np.zeros_like(query_delta), query_delta, query_delta * best_fixed[:, None], np.maximum(query_delta, 0),
            query_delta * (predicted_gap > MARGIN), query_delta * (reference_gap > MARGIN)[:, None],
            query_delta * (domain_support.mean(axis=1) > MARGIN)[:, None],
        )
        if PER_BENCHMARK in methods:
            gains = (*gains, query_delta * (domain_support[:, query_domains] > MARGIN))
        outcomes[quantile] = dict(zip(methods, gains))
    return outcomes


def metrics(runs, domains, representation):
    summary, paired = [], []
    for quantile, result in runs.items():
        identity = {"representation": representation, "anchor_quantile": quantile}
        for method, actual in result.items():
            summary.append({**identity, "method": method,
                            "macro_gain_over_incumbent": benchmark_means(actual, domains).mean(),
                            "micro_gain_over_incumbent": actual.mean(dtype=np.float64),
                            "rescues": int((actual > 0).sum()), "harms": int((actual < 0).sum())})
        contrasts = [("query_oracle", "hindsight_best_fixed")]
        contrasts.extend(("paired_ridge", method) for method in (
            "same_fit_reference_fixed", "direct_support_fixed", "hindsight_best_fixed"))
        if PER_BENCHMARK in result:
            contrasts.append(("paired_ridge", PER_BENCHMARK))
        for method, comparator in contrasts:
            left = result[method].mean(axis=1, dtype=np.float64)
            right = result[comparator].mean(axis=1, dtype=np.float64)
            effect = paired_queries(left, right, domains)
            if representation == SUPERVISED:
                effect.pop("positive_seed_count")
            paired.append({**identity, "method": method, "comparator": comparator, **effect})
    return summary, paired


def decision_gate(paired, primary):
    fixed = ("same_fit_reference_fixed", "direct_support_fixed", PER_BENCHMARK)
    primary_rows = [row for row in paired if row["anchor_quantile"] == COMPETITIVE_MEAN and row["comparator"] in fixed]
    strongest = min(primary_rows, key=lambda row: row["macro_gain"])
    positive = sum(all(row["macro_gain"] > 0 for row in paired
                       if row["anchor_quantile"] == quantile and row["comparator"] in fixed) for quantile in primary)
    return {"maximizing_fixed": strongest["comparator"], "positive_anchors_vs_all_fixed": positive,
            "required_gain": 0.005, "required_lower_bound": 0, "required_positive_anchors": 2,
            "met": bool(strongest["macro_gain"] >= 0.005
                        and min(row["macro_ci_low"] for row in primary_rows) > 0 and positive >= 2)}


def run(args):
    dataset, split, config, _, _, _ = input_context(args.run)
    protocol = read_json(args.competitive_selection)["protocol"]
    support_index = {int(dataset.prompt_ids[i]): i for i in split.train_items}
    support = np.array([support_index[pid] for pid in protocol["candidate_prompt_ids"]])
    reference = np.array([support_index[pid] for pid in protocol["reference_prompt_ids"]])
    target, newcomers = split.val_items, split.val_models
    parts = support, target, reference
    positions = np.concatenate(parts)
    if len(np.unique(dataset.group_ids[positions])) != len(positions):
        raise ValueError("Probe, reference and validation query groups must be disjoint.")
    domains = tuple(dataset.benchmark_ids[index] for index in parts)
    anchors = {entry["quantile"]: np.flatnonzero(dataset.model_ids == entry["model_id"])[0]
               for entry in protocol["anchors"]}
    pairs = {q: tuple(dataset.labels[np.ix_(newcomers, items)] - dataset.labels[incumbent, items]
                      for items in (support, target)) for q, incumbent in anchors.items()}
    b_runs = []
    seeds = config["seeds"]
    for seed in seeds:
        decoder, _ = load_decoder(args.checkpoints / f"{B}_{seed}.pt")
        b_axes = [decoder.export_features(dataset.embeddings[index], categories)[1].astype(np.float64)
                  for index, categories in zip(parts, domains)]
        b_runs.append(score(design_matrices(b_axes, principal_axes=True), pairs, domains, METHODS))
    b_outcomes = {q: {method: np.stack([result[q][method] for result in b_runs]) for method in METHODS}
                  for q in anchors}
    summary, paired = metrics(b_outcomes, domains[1], B_RIDGE)

    train = split.train_items
    supervised_axes = reduced_rank_features(dataset.embeddings[train],
                                            dataset.labels[np.ix_(split.train_models, train)],
                                            dataset.benchmark_ids[train],
                                            [dataset.embeddings[index] for index in parts])
    supervised = score(design_matrices(supervised_axes, principal_axes=False), pairs, domains, (*METHODS, PER_BENCHMARK))
    primary = [entry["quantile"] for entry in protocol["anchors"] if entry["used_in_descriptive_competitive_average"]]
    supervised[COMPETITIVE_MEAN] = {method: np.concatenate([supervised[q][method] for q in primary], axis=0)
                                   for method in (*METHODS, PER_BENCHMARK)}
    supervised = {q: {method: result[None, ...] for method, result in methods.items()}
                  for q, methods in supervised.items()}
    supervised_summary, supervised_paired = metrics(supervised, domains[1], SUPERVISED)
    # Retain the reported comparison to the archived three-seed B mean next to
    # the deterministic supervised map, with no extra interval or seed claim.
    b_macro = {row["anchor_quantile"]: row["macro_gain_over_incumbent"] for row in summary if row["method"] == "paired_ridge"}
    b_macro[COMPETITIVE_MEAN] = float(np.mean([b_macro[q] for q in primary]))
    for row in supervised_summary:
        summary.append(row)
        if row["method"] == PER_BENCHMARK:
            summary.append({"representation": SUPERVISED, "anchor_quantile": row["anchor_quantile"],
                            "method": "frozen_b_ridge_archived_mean",
                            "macro_gain_over_incumbent": b_macro[row["anchor_quantile"]]})
    paired.extend(supervised_paired)
    atomic_csv(args.output / "summary.csv", summary)
    atomic_csv(args.output / "paired.csv", paired)
    atomic_json(args.output / "protocol.json", {
        "phase": "validation_only", "anchors": protocol["anchors"], "frozen_b_checkpoint": B,
        "frozen_b_seeds": seeds, "supervised_map": "one deterministic fit",
        "fit": "Newcomer correctness minus incumbent correctness. Ridge on seven support-centered and standardized axes with an unpenalized intercept.",
        "frozen_b_axes": "First seven principal axes of B's interaction features on the paired support bank.",
        "supervised_axes": "Cholesky-whitened, weighted reduced-rank ridge on historical model-by-question responses and MPNet features. Features are standardized and divided by sqrt(input dimension). Responses are double-centered.",
        "historical_map_ridge": MAP_RIDGE, "historical_train_models": len(split.train_models),
        "historical_train_questions": len(train), "fit_ridge_sum_loss": FIT_RIDGE, "admission_margin": MARGIN,
        "paired_support_questions": len(support), "unlabeled_reference_questions": len(reference),
        "validation_newcomers": len(newcomers), "validation_questions": len(target),
        "partition_source": "competitive_onboarding/validation_selection.json",
        "static": "Choose one model from equal-benchmark mean reference predictions or observed support advantage at the same admission margin.",
        "per_benchmark_static": "Uses the incoming benchmark label and selects one model per prompt. All prompts in a benchmark use the same model, determined by paired support advantage and the admission margin.",
        "hindsight": "Best fixed validation macro identity per pair. The query oracle takes the maximum observed binary correctness for each query.",
        "interval": "Conditional paired query bootstrap within benchmark after averaging newcomer trials and the three frozen B seeds. The supervised map is deterministic.",
        "gate": decision_gate(supervised_paired, primary),
    })
    print(f"Scored {len(target)} validation questions with {len(newcomers)} newcomers for both paired representations.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--competitive-selection", type=Path,
                        default=ROOT / "reports/routing/competitive_onboarding/validation_selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/paired_routing")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
