"""Prepare and gate a descriptive same-release metadata-excluded cohort.

The original 88-model dataset remains the checkpoint/feature/prior context.
The 24 excluded models supply a separate correctness oracle only. No shared
model is trained or tuned, and no confidence intervals are computed here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from coldstart.metrics import benchmark_metrics
from coldstart.models import MeanBaselines, ScalarLogitShift, question_knn
from coldstart.probes import select_probe_order
from coldstart.runtime import Deadline, register_job
from coldstart.splits import dataset_fingerprint
from embed_new_model import load_run
from run_experiment import (VARIANTS, atomic_csv, atomic_json, config_hash,
    infer_probabilities, load_features, python_source_sha256, scientific_source_hashes, sha256, summarize)

PREDICTORS = ("coldstart_structured8", "coldstart_rank0", "coldstart_neural_lowrank",
              "domain_probe_mean", "global_probe_mean", "question_knn", "scalar_logit_shift")
COMPLETE_DOMAIN_FIELDS = ("macro_mae", "worst_domain_mae", "worst_overestimate", "macro_brier", "macro_nll")


@dataclass
class Oracle:
    model_ids: np.ndarray
    model_names: tuple[str, ...]
    labels: np.ndarray
    prompt_ids: np.ndarray
    benchmark_ids: np.ndarray


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_context(main_output: Path):
    config = read_json(main_output / "effective_config.json")
    dataset, split, config, lock, _ = load_run(main_output, "neural_lowrank", config["seeds"][0])
    if dataset.n_models != 88 or len(dataset.benchmark_names) != 5:
        raise ValueError("The frozen predictor context must remain the original audited 88 models and five benchmarks")
    if config["seeds"] != [11, 23, 37] or config["budgets"] != [50, 100, 200]:
        raise ValueError("Reuse the exact frozen main seeds and budgets")
    if lock["source_sha256"] != scientific_source_hashes():
        raise ValueError("Frozen main scientific source changed")
    return dataset, split, config, lock


def load_oracle(base, prepared_path: Path, family_path: Path) -> Oracle:
    """Align all excluded rows to the original bank; never filter by their labels."""
    with family_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    included, excluded = set(), set()
    seen = set()
    for row in rows:
        mid = int(row["model_id"])
        if mid in seen:
            raise ValueError("Duplicate source ID in the metadata table")
        seen.add(mid)
        decision = row["inclusion"].strip().lower()
        if decision == "include":
            included.add(mid)
        elif decision == "exclude":
            excluded.add(mid)
        else:
            raise ValueError("Every metadata model must have an explicit include/exclude decision")
    if len(included) != 88 or len(excluded) != 24 or included.intersection(excluded) or included != set(base.model_ids.tolist()):
        raise ValueError("Cohort must be exactly the 24 excluded IDs, disjoint from the frozen audited 88")
    metadata = read_json(prepared_path.with_suffix(".json"))
    with np.load(prepared_path, allow_pickle=False) as archive:
        model_ids, prompt_ids = archive["model_ids"], archive["prompt_ids"]
        if set(model_ids.tolist()) != included.union(excluded) or len(model_ids) != 112:
            raise ValueError("Prepared source rows must contain the same complete 112-model release")
        model_positions = {int(value): i for i, value in enumerate(model_ids)}
        prompt_positions = {int(value): i for i, value in enumerate(prompt_ids)}
        if len(prompt_positions) != len(prompt_ids) or not set(base.prompt_ids.tolist()).issubset(prompt_positions):
            raise ValueError("Frozen source prompt IDs are not uniquely represented")
        cols = np.array([prompt_positions[int(value)] for value in base.prompt_ids])
        cohort_ids = np.array(sorted(excluded), dtype=np.int64)
        row_positions = np.array([model_positions[int(value)] for value in cohort_ids])
        labels = archive["strict_labels"][np.ix_(row_positions, cols)].astype(np.float32)
        raw_names = np.asarray(metadata["benchmark_names"])[archive["benchmark_ids"][cols]]
        expected_names = np.asarray(base.benchmark_names)[base.benchmark_ids]
        if not np.array_equal(raw_names, expected_names) or not np.array_equal(archive["embeddings"][cols], base.embeddings):
            raise ValueError("Oracle alignment must preserve exact main benchmark IDs and text embeddings")
        if np.isinf(labels).any() or not np.isin(labels[np.isfinite(labels)], [0, 1]).all():
            raise ValueError("Strict source oracle must be binary or structurally unavailable")
        names = tuple(str(metadata["model_names"][int(position)]) for position in row_positions)
    return Oracle(cohort_ids, names, labels, base.prompt_ids.copy(), base.benchmark_ids.copy())


def fixed_metrics(truth: np.ndarray, probabilities: np.ndarray, domains: np.ndarray) -> dict:
    """Keep the five-domain denominator; do not reward incomplete target rows."""
    result = benchmark_metrics(truth, probabilities, domains, require_all_domains=False)
    incomplete = result["per_model"]["domain_count"] != len(np.unique(domains))
    for key in COMPLETE_DOMAIN_FIELDS:
        result["per_model"][key][incomplete] = np.nan
        if np.any(incomplete):
            result["summary"][key] = np.nan
    result["target_complete"] = ~incomplete
    return result


def restore_baselines(base, split, main_config: dict, main_output: Path):
    """Use original 62-model priors; no excluded row enters prior estimation."""
    mean = MeanBaselines(len(base.benchmark_names)).fit(
        base.labels[np.ix_(split.train_models, split.train_items)], base.benchmark_ids[split.train_items])
    first = None
    for seed in main_config["seeds"]:
        saved = read_json(main_output / "baseline_fits" / f"scalar_seed_{seed}.json")
        if saved["training_model_ids"] != base.model_ids[split.train_models].tolist() or saved["training_prompt_ids"] != base.prompt_ids[split.train_items].tolist() or saved["dataset_fingerprint"] != dataset_fingerprint(base):
            raise ValueError("Saved scalar prior does not belong to the frozen 62-model training pool")
        fields = {name: saved[name] for name in ("benchmark_offsets", "prior_mean", "prior_precision", "training_offsets", "training_counts", "training_successes")}
        if first is None:
            first = fields
        elif fields != first:
            raise ValueError("Frozen scalar priors differ across main seeds")
    spec = main_config["baselines"]["scalar_logit_shift"]
    scalar = ScalarLogitShift(len(base.benchmark_names), spec["variance_floor"], spec["training_ridge"])
    for name, values in first.items():
        setattr(scalar, name, np.asarray(values, dtype=np.float64))
    return mean, scalar


def prior_snapshots(mean, scalar):
    return ({"global": mean.global_prior, "benchmark": mean.benchmark_prior.tolist()},
            {"mean": scalar.prior_mean.tolist(), "precision": scalar.prior_precision.tolist(),
             "benchmark_offsets": scalar.benchmark_offsets.tolist()})


def verify_prior_bindings(main_output: Path, manifest: dict, mean, scalar) -> None:
    if set(manifest["scalar_fit_sha256"]) != {"11", "23", "37"}:
        raise ValueError("All three frozen scalar fit files must be bound")
    for seed, expected in manifest["scalar_fit_sha256"].items():
        if expected != sha256(main_output / "baseline_fits" / f"scalar_seed_{seed}.json"):
            raise ValueError("Saved scalar prior file changed")
    mean_snapshot, scalar_snapshot = prior_snapshots(mean, scalar)
    if mean_snapshot != manifest["frozen_mean_prior"] or scalar_snapshot != manifest["frozen_scalar_prior"]:
        raise ValueError("Restored frozen prior attributes changed")


def verify_context_loader(manifest: dict, path: Path | None = None) -> None:
    path = Path(__file__).with_name("embed_new_model.py") if path is None else path
    if manifest["context_loader_sha256"] != python_source_sha256(path):
        raise ValueError("Frozen context loader source changed")


def prepare(output: Path, main_output: Path, deadline: Deadline) -> dict:
    output, main_output = output.resolve(), main_output.resolve()
    if (output / "pre_score_manifest.json").exists():
        return verify(output, deadline)[-1]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing a nonempty cohort output without a pre-score manifest")
    base, split, main_config, lock = load_context(main_output)
    if base.n_questions != 22475:
        raise ValueError("This evaluation must reuse exactly the frozen 22,475 source prompts")
    if len(np.unique(base.benchmark_ids[split.test_items])) != 5:
        raise ValueError("The frozen target bank must contain all five benchmarks")
    prepared = Path(main_config["data"]["prepared_path"])
    prepared = (prepared / "embedllm.npz" if prepared.is_dir() else prepared).resolve()
    family_path = Path(main_config["data"].get("family_table", "data/metadata/model_families.csv")).resolve()
    deadline.check()
    oracle = load_oracle(base, prepared, family_path)
    mean, scalar = restore_baselines(base, split, main_config, main_output)
    output.mkdir(parents=True, exist_ok=True)
    orders, availability = {}, []
    for seed in main_config["seeds"]:
        selected = select_probe_order(base.embeddings, base.benchmark_ids, split.train_items, 200, "balanced_random", seed,
                                      check_deadline=deadline.check)
        orders[str(seed)] = base.prompt_ids[selected].tolist()
        for model, mid in enumerate(oracle.model_ids):
            availability.append({"model_id": int(mid), "seed": seed,
                "observed_probe_labels": {str(k): int(np.isfinite(oracle.labels[model, selected[:k]]).sum()) for k in main_config["budgets"]}})
    target_counts = [{"model_id": int(mid), "target_benchmark_counts": {
        name: int(np.isfinite(oracle.labels[model, split.test_items][base.benchmark_ids[split.test_items] == domain]).sum())
        for domain, name in enumerate(base.benchmark_names)}} for model, mid in enumerate(oracle.model_ids)]
    atomic_json(output / "probe_orders.json", orders)
    atomic_json(output / "oracle_availability.json", {"probe_counts": availability, "target_counts": target_counts,
        "correctness_totals_inspected": False, "models_dropped": 0})
    effective = {"schema_version": "1.0", "experiment": {"name": "same_release_metadata_excluded_cohort",
        "regime": "strict_unseen_item", "secondary_descriptive": True, "training_performed": False,
        "tuning_performed": False, "external_dataset": False, "live_llm_queries": False, "family_blind_claim": False},
        "data": {"label_policy": "strict", "source_repository": main_config["data"]["source_repository"],
            "source_revision": main_config["data"]["source_revision"], "evaluation_model_scope": "all metadata inclusion==exclude",
            "model_ids": oracle.model_ids.tolist(), "prompt_ids_source": "exact frozen main bank; no cohort-dependent filter or representative choice"},
        "budgets": main_config["budgets"], "seeds": main_config["seeds"], "probe_strategies": ["balanced_random"],
        "evaluation": {"primary_predictor": "coldstart_neural_lowrank", "primary_metric": "macro_mae",
            "score_scale": "proportion", "report_intervals": False,
            "uncertainty_note": "No intervals: dependence and ancestry of excluded models are uncertain."},
        "frozen_predictor_training_configuration_source": str(main_output / "effective_config.json"),
        "frozen_selection": lock["selection"], "missing_data_rule": {
            "probes": "Fixed attempted questions; unavailable labels consume budget and are not replaced.",
            "targets": "All 24 model rows remain. Primary macro/worst/macro-proper scores are unavailable if any of five target benchmarks is unobserved.",
            "item_proper_scores": "Secondary Brier/NLL use observed target cells only."}}
    atomic_json(output / "effective_config.json", effective)
    mean_snapshot, scalar_snapshot = prior_snapshots(mean, scalar)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output), "main_output": str(main_output), "main_lock_sha256": sha256(main_output / "test_lock.json"),
        "main_config_sha256": config_hash(main_config), "main_split_sha256": sha256(main_output / "split.json"),
        "source_hash_policy": "utf8_lf", "orchestrator_sha256": python_source_sha256(Path(__file__).resolve()),
        "context_loader_sha256": python_source_sha256(Path(__file__).with_name("embed_new_model.py")),
        "scientific_source_sha256": scientific_source_hashes(), "prepared_path": str(prepared),
        "prepared_data_sha256": sha256(prepared), "family_table_path": str(family_path), "family_table_sha256": sha256(family_path),
        "prepared_metadata_sha256": sha256(prepared.with_suffix(".json")),
        "scalar_fit_sha256": {str(seed): sha256(main_output / "baseline_fits" / f"scalar_seed_{seed}.json") for seed in main_config["seeds"]},
        "original_dataset_fingerprint": dataset_fingerprint(base), "original_model_ids": base.model_ids.tolist(),
        "original_training_model_ids": base.model_ids[split.train_models].tolist(), "cohort_model_ids": oracle.model_ids.tolist(),
        "cohort_model_names": list(oracle.model_names), "model_count": 24, "prompt_count": base.n_questions,
        "benchmark_names": list(base.benchmark_names), "target_prompt_ids": base.prompt_ids[split.test_items].tolist(),
        "predictors": list(PREDICTORS), "zero_probe_predictor": "training_domain_mean", "expected_model_rows": 1584,
        "expected_domain_rows": 7920, "effective_config_sha256": sha256(output / "effective_config.json"),
        "probe_orders_sha256": sha256(output / "probe_orders.json"), "oracle_availability_sha256": sha256(output / "oracle_availability.json"),
        "frozen_mean_prior": mean_snapshot, "frozen_scalar_prior": scalar_snapshot,
        "scope": "Secondary descriptive same-release out-of-curated-cohort evaluation; no training, tuning, family-blind or external-data claim.",
        "target_scores_computed": False, "confidence_intervals": False}
    manifest["manifest_sha256"] = hash_json(manifest)
    atomic_json(output / "pre_score_manifest.json", manifest)
    return manifest


def verify(output: Path, deadline: Deadline):
    output = output.resolve()
    manifest = read_json(output / "pre_score_manifest.json")
    stored = manifest["manifest_sha256"]
    if stored != hash_json({key: value for key, value in manifest.items() if key != "manifest_sha256"}) or manifest["output_root"] != str(output):
        raise ValueError("Cohort pre-score manifest changed")
    if manifest["orchestrator_sha256"] != python_source_sha256(Path(__file__).resolve()) or manifest["scientific_source_sha256"] != scientific_source_hashes():
        raise ValueError("Cohort or frozen scientific source changed")
    verify_context_loader(manifest)
    main_output = Path(manifest["main_output"])
    base, split, main_config, lock = load_context(main_output)
    for expected, path in (("main_lock_sha256", main_output / "test_lock.json"), ("main_split_sha256", main_output / "split.json"),
        ("prepared_data_sha256", Path(manifest["prepared_path"])), ("family_table_sha256", Path(manifest["family_table_path"])),
        ("prepared_metadata_sha256", Path(manifest["prepared_path"]).with_suffix(".json")),
        ("effective_config_sha256", output / "effective_config.json"), ("probe_orders_sha256", output / "probe_orders.json"),
        ("oracle_availability_sha256", output / "oracle_availability.json")):
        deadline.check()
        if manifest[expected] != sha256(path):
            raise ValueError(f"Bound cohort input changed: {expected}")
    if manifest["original_dataset_fingerprint"] != dataset_fingerprint(base) or manifest["main_config_sha256"] != config_hash(main_config):
        raise ValueError("Original 88-model feature/prior binding changed")
    oracle = load_oracle(base, Path(manifest["prepared_path"]), Path(manifest["family_table_path"]))
    if oracle.model_ids.tolist() != manifest["cohort_model_ids"] or oracle.prompt_ids.tolist() != base.prompt_ids.tolist():
        raise ValueError("Cohort identities changed")
    mean, scalar = restore_baselines(base, split, main_config, main_output)
    verify_prior_bindings(main_output, manifest, mean, scalar)
    return base, split, main_config, lock, oracle, manifest


def score(output: Path, gate_path: Path, deadline: Deadline, device: str):
    base, split, main_config, lock, oracle, manifest = verify(output, deadline)
    gate = read_json(gate_path)
    if not str(gate.get("status", "")).upper().startswith("PASS") or gate.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Scoring requires a passing independent gate bound to this exact pre-score manifest")
    if (output / "test_completed.json").exists():
        raise ValueError("The one-pass frozen cohort evaluation is already complete")
    main_output = Path(manifest["main_output"])
    mean, scalar = restore_baselines(base, split, main_config, main_output)
    verify_prior_bindings(main_output, manifest, mean, scalar)
    selection, targets = lock["selection"], split.test_items
    truth, domains = oracle.labels[:, targets], base.benchmark_ids[targets]
    records, domain_records, adaptation_records, probe_records = [], [], [], []
    effective = read_json(output / "effective_config.json")
    orders = read_json(output / "probe_orders.json")
    positions = {int(value): i for i, value in enumerate(base.prompt_ids)}
    started = time.perf_counter()

    def save_predictions(probabilities, identity, attempts, observed_counts):
        metrics = fixed_metrics(truth, probabilities, domains)
        for row, mid in enumerate(oracle.model_ids):
            complete = bool(metrics["target_complete"][row])
            base_identity = {**identity, "model_id": int(mid), "model_name": oracle.model_names[row],
                "probe_attempts": attempts, "observed_probes": int(observed_counts[row]),
                "unavailable_probe_labels": attempts - int(observed_counts[row]),
                "target_status": "complete_five_benchmarks" if complete else "incomplete_target_benchmarks",
                "primary_macro_available": complete}
            records.append({**base_identity, **{name: float(values[row]) for name, values in metrics["per_model"].items()}})
            for local_domain, domain in enumerate(metrics["domain_ids"]):
                domain_records.append({**base_identity, "benchmark": base.benchmark_names[int(domain)],
                    "true_score": float(metrics["true_scores"][row, local_domain]),
                    "pred_score": float(metrics["pred_scores"][row, local_domain]),
                    "error": float(metrics["absolute_error"][row, local_domain]),
                    "signed_error": float(metrics["signed_error"][row, local_domain]),
                    "count": int(metrics["counts"][row, local_domain])})
        prediction_dir = output / "test_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_dir / f"seed{identity['seed']}_{identity['strategy']}_k{identity['budget']}_{identity['predictor']}.npz",
            probabilities=probabilities, model_ids=oracle.model_ids, prompt_ids=base.prompt_ids[targets])

    for seed in main_config["seeds"]:
        deadline.check()
        # Crucially, pass the original 88-model object and its original fingerprint.
        features = load_features(base, main_config, main_output, list(VARIANTS), seed, split.train_items, device)
        identity = {"experiment_id": effective["experiment"]["name"], "protocol": "strict_unseen_item",
            "label_policy": "strict", "model_split": "metadata_excluded_cohort", "phase": "test", "seed": seed,
            "secondary_descriptive": True, "confidence_intervals": False}
        save_predictions(mean.training_mean(domains, len(oracle.model_ids)), {**identity,
            "strategy": "zero_probe", "budget": 0, "predictor": "training_domain_mean"}, 0, np.zeros(24, dtype=int))
        order = np.array([positions[int(value)] for value in orders[str(seed)]])
        for model, mid in enumerate(oracle.model_ids):
            probe_records.append({"seed": seed, "model_id": int(mid), "strategy": "balanced_random", "budget": 200,
                "prompt_ids": base.prompt_ids[order].tolist(), "observed_labels": int(np.isfinite(oracle.labels[model, order]).sum())})
        for budget in main_config["budgets"]:
            deadline.check()
            selected = order[:budget]
            responses = oracle.labels[:, selected]
            observed = np.isfinite(responses).sum(axis=1)
            probabilities = {}
            for predictor, by_benchmark in (("domain_probe_mean", True), ("global_probe_mean", False)):
                mean.smoothing = selection["mean_smoothing"][predictor]
                probabilities[predictor] = mean.probe_mean(responses, base.benchmark_ids[selected], domains, by_benchmark=by_benchmark)
            probabilities["question_knn"] = question_knn(base.embeddings[selected], responses, base.embeddings[targets],
                prior_probabilities=mean.benchmark_prior[domains], **selection["knn"])
            probabilities["scalar_logit_shift"] = scalar.predict(responses, base.benchmark_ids[selected], domains,
                selection["scalar_logit_shift"]["prior_strength"], check_deadline=deadline.check)
            for variant in VARIANTS:
                values, latents, covariances = [], [], []
                for model, response in enumerate(responses):
                    predicted, fit = infer_probabilities(features[variant], selected, response, targets,
                        selection["prior_strengths"][variant], main_config, deadline)
                    values.append(predicted); latents.append(fit["latent"]); covariances.append(fit["covariance"])
                    adaptation_records.append({"seed": seed, "budget": budget, "strategy": "balanced_random",
                        "predictor": VARIANTS[variant]["predictor"], "model_id": int(oracle.model_ids[model]),
                        **{key: value for key, value in fit.items() if key not in {"latent", "covariance"}}})
                probabilities[VARIANTS[variant]["predictor"]] = np.stack(values)
                embedding_dir = output / "test_embeddings"
                embedding_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(embedding_dir / f"seed{seed}_balanced_random_k{budget}_{variant}.npz",
                    latents=np.stack(latents), covariances=np.stack(covariances), model_ids=oracle.model_ids,
                    probe_prompt_ids=np.broadcast_to(base.prompt_ids[selected], (24, budget)),
                    prior_mean=features[variant]["mean"],
                    prior_precision=selection["prior_strengths"][variant] * features[variant]["precision"])
            if set(probabilities) != set(PREDICTORS):
                raise ValueError("Frozen cohort predictor list changed")
            for predictor, values in probabilities.items():
                save_predictions(values, {**identity, "strategy": "balanced_random", "budget": budget,
                    "predictor": predictor}, budget, observed)
            atomic_csv(output / "per_model_metrics.csv", records)
            atomic_csv(output / "per_domain_metrics.csv", domain_records)
            atomic_csv(output / "test_adaptation.csv", adaptation_records)
            atomic_json(output / "test_probes.json", probe_records)
    if len(records) != manifest["expected_model_rows"] or len(domain_records) != manifest["expected_domain_rows"]:
        raise ValueError("Cohort evaluation did not retain every model/predictor/budget row")
    summary = {"phase": "test", "elapsed_seconds": time.perf_counter() - started,
        "num_records": len(records), "summary": summarize(records), "secondary_descriptive": True,
        "confidence_intervals": False}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "test_completed.json", {"phase": "test", "manifest_sha256": manifest["manifest_sha256"],
        "finished_at": datetime.now(timezone.utc).isoformat(), "secondary_descriptive": True,
        "model_count": 24, "num_records": len(records), "domain_rows": len(domain_records),
        "models_dropped": 0, "training_performed": False, "tuning_performed": False,
        "confidence_intervals": False, "elapsed_seconds": summary["elapsed_seconds"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "score"), default="prepare")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/metadata_excluded_cohort")
    parser.add_argument("--main", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--gate", type=Path, help="Independent PASS report containing this manifest_sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", help="Explicit timezone-aware deadline, required for scoring")
    args = parser.parse_args(argv)
    if args.phase == "score" and (not args.deadline or args.gate is None):
        parser.error("score requires --deadline and a bound independent --gate")
    deadline = Deadline(args.deadline)
    deadline.check()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    output = args.output.resolve()
    with register_job(deadline=args.deadline, metadata={"task": "metadata_excluded_same_release_cohort", "phase": args.phase}):
        if args.phase == "prepare":
            manifest = prepare(output, args.main.resolve(), deadline)
        else:
            score(output, args.gate.resolve(), deadline, args.device)
            manifest = read_json(output / "pre_score_manifest.json")
    print(json.dumps({"phase": args.phase, "manifest_sha256": manifest["manifest_sha256"], "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
