"""Reproduce the rejected post-primary, validation-only evidence-blend pilot.

This reads stored validation profiles and scores. It never trains a decoder or
reads the 24-model coverage cohort. The pilot did not change the final method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def laplace_log_evidence(objectives, prior_precision, posterior_covariances):
    """Conditional evidence of frozen decoders, integrating Gaussian latents."""
    objectives = np.atleast_1d(np.asarray(objectives, dtype=np.float64))
    precision = np.asarray(prior_precision, dtype=np.float64)
    covariance = np.asarray(posterior_covariances, dtype=np.float64)
    if (precision.ndim != 2 or precision.shape[0] != precision.shape[1]
            or covariance.shape != (len(objectives), *precision.shape)):
        raise ValueError("Expected objectives[M], precision[D,D], covariance[M,D,D]")
    if not all(np.isfinite(x).all() for x in (objectives, precision, covariance)):
        raise ValueError("Evidence inputs must be finite")
    if (not np.allclose(precision, precision.T)
            or not np.allclose(covariance, covariance.transpose(0, 2, 1))):
        raise ValueError("Precision and covariance must be symmetric")
    np.linalg.cholesky(precision)
    np.linalg.cholesky(covariance)
    return -objectives + .5 * (np.linalg.slogdet(precision)[1]
                               + np.linalg.slogdet(covariance)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/main_v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/exploratory_hybrid/validation_only"))
    args = parser.parse_args()
    config = json.loads((args.run / "effective_config.json").read_text(encoding="utf-8"))
    fits = pd.read_csv(args.run / "validation_adaptation.csv")
    domains = pd.read_csv(args.run / "validation_per_domain_metrics.csv")
    rows = []
    for seed in config["seeds"]:
        for budget in config["budgets"]:
            embeddings, evidence = {}, {}
            for variant in ["neural_lowrank", "rank0"]:
                path = args.run / "validation_embeddings" / f"seed{seed}_balanced_random_k{budget}_{variant}.npz"
                with np.load(path, allow_pickle=False) as source:
                    embeddings[variant] = {key: source[key] for key in
                                           ["prior_precision", "covariances", "model_ids", "probe_prompt_ids"]}
                saved = embeddings[variant]
                fit = fits[(fits.seed == seed) & (fits.budget == budget)
                           & (fits.strategy == "balanced_random")
                           & (fits.predictor == "coldstart_" + variant)].set_index("model_id").reindex(saved["model_ids"])
                if not fit.converged.all() or not fit.observed_responses.eq(budget).all():
                    raise ValueError("The pilot requires converged fits on identical fully observed supports")
                evidence[variant] = laplace_log_evidence(fit.objective.to_numpy(),
                                                        saved["prior_precision"], saved["covariances"])
            one, two = embeddings.values()
            for key in ["model_ids", "probe_prompt_ids"]:
                if not np.array_equal(one[key], two[key]):
                    raise ValueError("Parent profiles do not use identical models/probes")
            delta = evidence["neural_lowrank"] - evidence["rank0"]
            weights = 1 / (1 + np.exp(-np.clip(delta, -700, 700)))
            for index, model_id in enumerate(one["model_ids"]):
                scores = []
                for variant in ["neural_lowrank", "rank0"]:
                    selected = domains[(domains.seed == seed) & (domains.budget == budget)
                                       & (domains.strategy == "balanced_random")
                                       & (domains.predictor == "coldstart_" + variant)
                                       & (domains.model_id == model_id)].sort_values("benchmark")
                    scores.append(selected)
                full, rank0 = scores
                if (not np.array_equal(full.benchmark, rank0.benchmark)
                        or not np.array_equal(full.true_score, rank0.true_score)):
                    raise ValueError("Parent validation targets differ")
                truth, p8, p0 = full.true_score.to_numpy(), full.pred_score.to_numpy(), rank0.pred_score.to_numpy()
                candidates = [("evidence_gate", weights[index]), ("uniform_blend", .5)]
                candidates += [(f"constant_{weight:g}", weight) for weight in [0, .25, .5, .75, 1]]
                for name, weight in candidates:
                    rows.append({"seed": seed, "budget": budget, "model_id": int(model_id),
                                 "candidate": name, "weight_lowrank": float(weight),
                                 "macro_mae": float(np.abs(weight * p8 + (1 - weight) * p0 - truth).mean()),
                                 "log_evidence_delta": float(delta[index])})
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "per_model.csv", index=False)
    summary = results.groupby(["candidate", "budget"], as_index=False).agg(
        macro_mae=("macro_mae", "mean"), mean_weight_lowrank=("weight_lowrank", "mean"))
    summary.to_csv(args.output / "summary.csv", index=False)
    scope = {"status": "validation_only_candidate_probe", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "idea_generated_after_main_and_original_and_secondary_family_test_inspection": True,
             "validation_data_only": True, "metadata_excluded24_performance_unseen_at_decision": True,
             "definition": "logE=-MAPobjective+.5*(logdetpriorprecision+logdetposteriorcovariance);w=sigmoid(E8-E0);p=w*p8+(1-w)*p0",
             "temperature": 1, "prior_model_weight": .5, "constant_grid": [0, .25, .5, .75, 1],
             "no_core_or_parent_weight_changes": True, "no_new_network_training": True,
             "decision": "rejected: evidence gate did not outperform the best constant blend; no extra tuning or cohort predictor added"}
    (args.output / "scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")
    print(summary.pivot(index="candidate", columns="budget", values="macro_mae").round(6).to_string())


if __name__ == "__main__":
    main()
