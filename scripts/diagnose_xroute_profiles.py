"""Compare unequal and matched model-profile support on xRouteBench training data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from route_query import ROOT, interaction_scores, routing_features
from coldstart.adaptation import sigmoid
from coldstart.models import load_decoder
from coldstart.probes import select_probe_order
from pilot_routing_learning import B, latent_prior, model_latents
from run_experiment import atomic_csv
from run_external_routing import read_json
from run_routing import benchmark_means
from run_xroute_onboarding import MARGIN, PRIOR_STRENGTH, SEEDS


BUDGETS = (10, 50, 200, 500)
ASYMMETRIC = "500_anchor_few_new"
MATCHED = "matched_support"
METRICS = ("true_mean_gap", "mean_predicted_gap", "newcomer_admitted_share", "routed_gain_macro",
           "both_fit_reference_static_gain_macro", "opportunity_macro")


def diagnose(args):
    protocol = read_json(args.protocol)
    config = read_json(args.run / "effective_config.json")
    with np.load(args.features, allow_pickle=False) as data:
        vectors, labels, domains = data["train_vectors"], data["training_labels"], data["train_domains"]
        candidates, reference = data["candidates"], data["reference"]
    target = np.setdiff1d(np.arange(len(vectors)), np.r_[candidates, reference])
    models = protocol["eligible_models"]
    rows = []
    for seed in SEEDS:
        decoder, _ = load_decoder(args.checkpoints / f"{B}_{seed}.pt")
        offsets, design = routing_features(decoder, vectors)
        prior = latent_prior(decoder, config, PRIOR_STRENGTH)
        # Preserve the existing candidate order. Sorting it would change the
        # seeded question permutations and therefore every smaller prefix.
        order = select_probe_order(vectors, domains, candidates, len(candidates), "balanced_random", seed)
        predictions = {}
        for budget in BUDGETS:
            support = order[:budget]
            latents = model_latents((offsets[support], design[support]), labels[:, support],
                                    domains[support], prior, config)
            predictions[budget] = (sigmoid(interaction_scores(latents, design[target]) + offsets[target]),
                                   sigmoid(interaction_scores(latents, design[reference]) + offsets[reference]))
        for anchor in protocol["anchors"]:
            incumbent = models.index(anchor["model"])
            for budget, (short_target, short_reference) in predictions.items():
                for mode, incumbent_budget in ((ASYMMETRIC, BUDGETS[-1]), (MATCHED, budget)):
                    full_target, full_reference = predictions[incumbent_budget]
                    for challenger, model in enumerate(models):
                        if challenger == incumbent:
                            continue
                        truth = labels[challenger, target] - labels[incumbent, target]
                        predicted = short_target[challenger] - full_target[incumbent]
                        reference_gap = short_reference[challenger] - full_reference[incumbent]
                        admit = predicted > MARGIN
                        true_gap = benchmark_means(truth, domains[target]).mean()
                        rows.append({
                            "seed": seed, "quantile": anchor["quantile"], "arriving_model": model,
                            "budget": budget, "mode": mode, "true_mean_gap": float(true_gap),
                            "mean_predicted_gap": float(benchmark_means(predicted, domains[target]).mean()),
                            "newcomer_admitted_share": float(admit.mean()),
                            "routed_gain_macro": float(benchmark_means(truth * admit, domains[target]).mean()),
                            "both_fit_reference_static_gain_macro": float(true_gap * (reference_gap.mean() > MARGIN)),
                            "opportunity_macro": float(benchmark_means(truth > 0, domains[target]).mean()),
                        })
    return rows, (len(candidates), len(reference), len(target))


def summarize(rows, sizes):
    summary = []
    for quantile in sorted({row["quantile"] for row in rows}):
        for budget in BUDGETS:
            for mode in (ASYMMETRIC, MATCHED):
                if budget == BUDGETS[-1] and mode == ASYMMETRIC:
                    continue
                paired = [row for row in rows if (row["quantile"], row["budget"], row["mode"]) == (quantile, budget, mode)]
                summary.append({"anchor_quantile": quantile, "budget_per_newcomer": budget, "mode": mode,
                                "candidate_questions": sizes[0], "unlabeled_reference_questions": sizes[1],
                                "training_diagnostic_questions": sizes[2], "arrivals": len(paired),
                                **{name: float(np.mean([row[name] for row in paired])) for name in METRICS}})
    return summary


def verify(rows, archive):
    identity = ("seed", "quantile", "arriving_model", "budget", "mode")
    with archive.open(newline="", encoding="utf-8") as handle:
        original = {tuple(row[key] for key in identity): row for row in csv.DictReader(handle)}
    if len(rows) != len(original):
        raise ValueError("Diagnostic roster, budgets or seeds differ from the archived replay.")
    for row in rows:
        key = tuple(str(row[name]) for name in identity)
        expected = original[key]
        for name in METRICS:
            tolerance = 1e-8 if name == "mean_predicted_gap" else 1e-12
            np.testing.assert_allclose(row[name], float(expected[name]), atol=tolerance, rtol=0,
                                       err_msg=f"{key} {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/routing/xroute_onboarding/protocol.json")
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/xroute_onboarding/features.npz")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/xroute_onboarding/train_profile_diagnostic.csv")
    parser.add_argument("--verify-against", type=Path, help="Optional prior per-arrival CSV from the same training-only analysis.")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        rows, sizes = diagnose(args)
    if args.verify_against is not None:
        verify(rows, args.verify_against)
    atomic_csv(args.output, summarize(rows, sizes))
    print(f"Summarized {len(rows)} paired arrivals and profile fits on {sizes[2]} training-only diagnosis questions.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
