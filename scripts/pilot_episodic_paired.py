"""Train topic axes through an 8D paired ridge fit and score validation only."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from diagnose_paired_routing import COMPETITIVE_MEAN, MARGIN, RANK, design_matrices
from pilot_routing_learning import B, input_context
from coldstart.models import load_decoder
from coldstart.probes import select_probe_order
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries, read_json
from run_routing import benchmark_means


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/routing/episodic_paired"
RUN = ROOT / "artifacts/main_v2"
COMPETITIVE = ROOT / "reports/routing/competitive_onboarding/validation_selection.json"
CHECKPOINTS = ROOT / "artifacts/routing_learning/checkpoints"
BUDGETS = (100, 200)
UPDATES = 1000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
RIDGE_PER_PROBE = 0.1
HISTORICAL_PAIRS = 13
OUTER_PER_BENCHMARK = 50
LEARNED = "episodic_paired"
FROZEN = "frozen_b_paired"
INCUMBENT = "incumbent"
GLOBAL = "direct_support_fixed"
DOMAIN = "per_benchmark_support_fixed"
REF_B = "frozen_b_reference_fixed"
REF_LEARNED = "episodic_reference_fixed"
HINDSIGHT = "hindsight_best_fixed"
ORACLE = "query_oracle"
FIXED = (INCUMBENT, GLOBAL, DOMAIN, REF_B, REF_LEARNED)


def normalize_historical(dataset, train):
    features = dataset.embeddings[train].astype(np.float64)
    domains = dataset.benchmark_ids[train]
    weights = 1 / (len(np.unique(domains)) * np.bincount(domains)[domains])
    mean = weights @ features
    scale = np.sqrt(weights @ (features - mean) ** 2)
    return lambda indices: ((dataset.embeddings[indices] - mean) / scale).astype(np.float32)


def fit_head(vectors, labels, domains, seed):
    """Fit one fresh head. Only historical training labels enter this function."""
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(vectors.shape[1], 64), nn.GELU(), nn.Linear(64, RANK))
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    questions = [np.flatnonzero(domains == domain) for domain in np.unique(domains)]
    anchor, challenger = np.triu_indices(len(labels), k=1)
    rng = np.random.default_rng(seed)
    feature_tensor, outcomes = torch.from_numpy(vectors), torch.from_numpy(labels)
    per_benchmark = max(BUDGETS) // len(questions)
    eye = torch.eye(RANK)
    first_parameters = torch.cat([p.detach().flatten() for p in head.parameters()])
    losses, gradient_norm = [], None
    for update in range(UPDATES):
        budget = BUDGETS[update % len(BUDGETS)]
        sampled = [rng.choice(domain, per_benchmark + OUTER_PER_BENCHMARK, replace=False) for domain in questions]
        probes = np.stack([ids[:per_benchmark] for ids in sampled], axis=1).ravel()[:budget]
        queries = np.concatenate([ids[per_benchmark:] for ids in sampled])
        selected_pairs = rng.choice(len(anchor), HISTORICAL_PAIRS, replace=False)
        pair_rows = anchor[selected_pairs], challenger[selected_pairs]
        outcomes_in_pair = outcomes[pair_rows[1]] - outcomes[pair_rows[0]]
        support_labels, query_labels = outcomes_in_pair[:, probes], outcomes_in_pair[:, queries]
        axes = head(feature_tensor[np.r_[probes, queries]])
        support, target = axes[:budget], axes[budget:]
        center = support.mean(dim=0)
        sd = support.std(dim=0, correction=0).clamp_min(1e-6)
        support, target = (support - center) / sd, (target - center) / sd
        demeaned_support_labels = support_labels - support_labels.mean(dim=1, keepdim=True)
        slopes = torch.linalg.solve(support.T @ support + (RIDGE_PER_PROBE * budget) * eye,
                                    support.T @ demeaned_support_labels.T)
        predicted = slopes.T @ target.T
        prediction_residual = predicted - predicted.mean(dim=1, keepdim=True)
        truth_residual = query_labels - query_labels.mean(dim=1, keepdim=True)
        loss = F.mse_loss(prediction_residual, truth_residual)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update == 0:
            gradient_norm = float(torch.cat([p.grad.flatten() for p in head.parameters()]).norm())
        optimizer.step()
        losses.append(float(loss.detach()))
    final_parameters = torch.cat([p.detach().flatten() for p in head.parameters()])
    displacement = float((first_parameters - final_parameters).norm())
    if not (np.isfinite(losses).all() and np.isfinite(gradient_norm) and gradient_norm > 0
            and np.isfinite(displacement) and displacement > 0):
        raise RuntimeError(f"Episodic gradient or parameter update failed at seed {seed}.")
    return head.eval(), {"seed": seed, "initial_gradient_norm": gradient_norm, "parameter_change_norm": displacement,
                         "mean_loss_first_100_updates": float(np.mean(losses[:100])),
                         "mean_loss_last_100_updates": float(np.mean(losses[-100:]))}


def paired_policy(axes, support_delta, reference_domains):
    support, target, reference = axes
    ridge = np.diag([0] + [RIDGE_PER_PROBE * len(support)] * RANK)
    coefficients = np.linalg.solve(support.T @ support + ridge, support.T @ support_delta.T)
    predicted = (target @ coefficients).T
    reference_gap = benchmark_means((reference @ coefficients).T, reference_domains).mean(axis=1)
    return predicted > MARGIN, reference_gap > MARGIN


def aggregate(outcomes, query_domains, primary):
    results = {q: {budget: {method: np.stack(scores) for method, scores in policies.items()}
                   for budget, policies in budgets.items()} for q, budgets in outcomes.items()}
    results[COMPETITIVE_MEAN] = {
        budget: {method: np.concatenate([results[q][budget][method] for q in primary], axis=1)
                 for method in results[primary[0]][budget]} for budget in BUDGETS}
    summary, paired = [], []
    for quantile, budgets in results.items():
        for budget, methods in budgets.items():
            identity = {"anchor_quantile": quantile, "probes_per_candidate": budget}
            for name, matrix in methods.items():
                summary.append({**identity, "method": name,
                                "macro_gain_over_incumbent": benchmark_means(matrix, query_domains).mean(),
                                "micro_gain_over_incumbent": matrix.mean(dtype=np.float64),
                                "rescues": int((matrix > 0).sum()), "harms": int((matrix < 0).sum())})
            learned = methods[LEARNED].mean(axis=1, dtype=np.float64)
            for comparator in (*FIXED, FROZEN, HINDSIGHT):
                contrast = paired_queries(learned, methods[comparator].mean(axis=1, dtype=np.float64), query_domains)
                paired.append({**identity, "method": LEARNED, "comparator": comparator, **contrast})
    return summary, paired


def run(run_dir, checkpoints, support_protocol, output):
    dataset, split, config, _, _, _ = input_context(run_dir)
    protocol = read_json(support_protocol)["protocol"]
    train, target, newcomers = split.train_items, split.val_items, split.val_models
    positions = {int(dataset.prompt_ids[row]): row for row in train}
    support = np.array([positions[item] for item in protocol["candidate_prompt_ids"]])
    reference = np.array([positions[item] for item in protocol["reference_prompt_ids"]])
    query_domains, reference_domains = dataset.benchmark_ids[target], dataset.benchmark_ids[reference]
    primary = [a["quantile"] for a in protocol["anchors"] if a["used_in_descriptive_competitive_average"]]
    transform = normalize_historical(dataset, train)
    train_vectors = transform(train)
    train_labels = dataset.labels[np.ix_(split.train_models, train)]
    train_domains = dataset.benchmark_ids[train]
    models = {anchor["quantile"]: int(np.flatnonzero(dataset.model_ids == anchor["model_id"])[0])
              for anchor in protocol["anchors"]}
    paired_targets = {quantile: dataset.labels[np.ix_(newcomers, target)] - dataset.labels[model, target]
                      for quantile, model in models.items()}
    outcomes = {q: {budget: defaultdict(list) for budget in BUDGETS} for q in models}
    training, orders = [], {}
    for seed in config["seeds"]:
        head, row = fit_head(train_vectors, train_labels, train_domains, seed)
        training.append(row)
        print(f"Seed {seed}: finished {UPDATES} training updates", flush=True)
        order = select_probe_order(dataset.embeddings, dataset.benchmark_ids, support, max(BUDGETS),
                                   "balanced_random", seed)
        orders[str(seed)] = dataset.prompt_ids[order].tolist()
        decoder, _ = load_decoder(checkpoints / f"{B}_{seed}.pt")
        parts = order, target, reference
        frozen = [decoder.export_features(dataset.embeddings[index], dataset.benchmark_ids[index])[1].astype(np.float64)
                  for index in parts]
        with torch.no_grad():
            learned = [head(torch.from_numpy(transform(index))).numpy().astype(np.float64) for index in parts]
        for budget in BUDGETS:
            observed = order[:budget]
            axes_b = design_matrices([frozen[0][:budget], *frozen[1:]], principal_axes=True)
            axes_learned = design_matrices([learned[0][:budget], *learned[1:]], principal_axes=False)
            for quantile, incumbent in models.items():
                truth = paired_targets[quantile]
                support_delta = dataset.labels[np.ix_(newcomers, observed)] - dataset.labels[incumbent, observed]
                domains = benchmark_means(support_delta, dataset.benchmark_ids[observed])
                use_b, ref_b = paired_policy(axes_b, support_delta, reference_domains)
                use_learned, ref_learned = paired_policy(axes_learned, support_delta, reference_domains)
                policies = {LEARNED: truth * use_learned, FROZEN: truth * use_b, INCUMBENT: np.zeros_like(truth),
                            GLOBAL: truth * (domains.mean(axis=1) > MARGIN)[:, None],
                            DOMAIN: truth * (domains[:, query_domains] > MARGIN),
                            REF_B: truth * ref_b[:, None], REF_LEARNED: truth * ref_learned[:, None],
                            HINDSIGHT: truth * (benchmark_means(truth, query_domains).mean(axis=1) > 0)[:, None],
                            ORACLE: np.maximum(truth, 0)}
                for method, gain in policies.items():
                    outcomes[quantile][budget][method].append(gain)
    summary, paired = aggregate(outcomes, query_domains, primary)
    gate_rows = {row["comparator"]: row for row in paired
                 if row["anchor_quantile"] == COMPETITIVE_MEAN and row["probes_per_candidate"] == max(BUDGETS)}
    strongest = min(FIXED, key=lambda method: gate_rows[method]["macro_gain"])
    passed = (all(gate_rows[method]["macro_gain"] >= 0.01 and gate_rows[method]["macro_ci_low"] > 0
                  and gate_rows[method]["positive_seed_count"] >= 2 for method in FIXED)
              and gate_rows[FROZEN]["macro_gain"] >= 0)
    atomic_csv(output / "training.csv", training)
    atomic_csv(output / "summary.csv", summary)
    atomic_csv(output / "paired.csv", paired)
    atomic_json(output / "protocol.json", {
        "phase": "validation_only", "seeds": config["seeds"], "updates_per_seed": UPDATES,
        "architecture": "fixed MPNet, weighted historical standardization, Linear(768,64), GELU, Linear(64,7), paired intercept",
        "train_models": dataset.model_ids[split.train_models], "train_questions": len(train),
        "support_budgets": BUDGETS, "schedule": "alternating 100 and 200. Source benchmark-balanced support and 50 disjoint query items per benchmark.",
        "historical_pairs_per_update": HISTORICAL_PAIRS, "pairs": "sample uniformly without replacement over unordered historical model pairs",
        "ridge_per_probe": RIDGE_PER_PROBE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss": "Outer MSE of within-pair centered predicted and actual correctness differences. Exact differentiable ridge on support-standardized axes.",
        "reference": len(reference), "validation_models": dataset.model_ids[newcomers], "validation_questions": len(target),
        "anchor_quantiles": list(models), "primary_anchor_quantiles": primary, "probe_ids": orders,
        "admission_margin": MARGIN, "interval": "Conditional paired queries within benchmark; seed and observed partner trials averaged.",
        "gate": {"strongest_static": strongest, "met": bool(passed), "gain_over_each_fixed": 0.01,
                 "positive_seeds_minimum": 2, "required_ci_lower_bound": 0, "no_drop_vs_frozen_b": True},
    })
    print(f"Finished validation. The episodic representation met its gate: {passed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--support-protocol", type=Path, default=COMPETITIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args.run, args.checkpoints, args.support_protocol, args.output)


if __name__ == "__main__":
    main()
