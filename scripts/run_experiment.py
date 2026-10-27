"""Train, validate, lock, and evaluate cold-start benchmark predictors.

Example: python scripts/run_experiment.py --phase train_validate --output artifacts/main
Test evaluation is a separate command and requires a matching validation lock.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from coldstart.adaptation import adapt_map, predict_from_latent
from coldstart.data import PreparedDataset, load_dataset
from coldstart.metrics import benchmark_metrics
from coldstart.models import (
    ColdStartDecoder, DecoderConfig, MeanBaselines, ScalarLogitShift, TrainConfig, create_decoder,
    fit_decoder, load_decoder, question_knn, save_decoder, select_device,
)
from coldstart.probes import STRATEGIES, select_probe_order
from coldstart.runtime import Deadline, DeadlineExceeded, atomic_replace, register_job
from coldstart.splits import (
    SplitManifest, dataset_fingerprint, load_family_table, load_split, make_split, save_split,
)


VARIANTS = {
    "structured8": {"mode": "structured", "rank": 8, "predictor": "coldstart_structured8"},
    "rank0": {"mode": "structured", "rank": 0, "predictor": "coldstart_rank0"},
    "neural_lowrank": {"mode": "lowrank", "rank": 8, "predictor": "coldstart_neural_lowrank"},
}
METRIC_NAMES = (
    "macro_mae", "worst_domain_mae", "worst_overestimate", "micro_score_mae",
    "weighted_domain_mae", "brier", "macro_brier", "nll", "macro_nll",
    "domain_count", "item_count",
)
SOURCE_HASH_POLICY = "utf8_lf"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    atomic_replace(temporary, path)


def atomic_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fields = list(dict.fromkeys(key for record in records for key in record))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    atomic_replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_source_sha256(path: Path) -> str:
    """Hash UTF-8 Python text with LF endings on every checkout platform."""
    source = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def scientific_source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "src" / "coldstart").glob("*.py")) + [Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): python_source_sha256(path) for path in paths}


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def checkpoint_path(output: Path, variant: str, seed: int) -> Path:
    return output / "checkpoints" / variant / f"seed_{seed}.pt"


def decoder_configuration(dataset: PreparedDataset, split: SplitManifest, config: dict, variant: str) -> DecoderConfig:
    specification = VARIANTS[variant]
    return DecoderConfig(
        input_dim=dataset.embeddings.shape[1], num_benchmarks=len(dataset.benchmark_names),
        num_models=len(split.train_models), rank=int(specification["rank"]),
        hidden_dim=int(config["model"].get("hidden_dim", 128)), mode=specification["mode"],
        dropout=float(config["model"].get("dropout", 0.1)),
        num_item_biases=dataset.n_questions if split.protocol == "fixed_bank" else 0,
    )


def training_configuration(config: dict, seed: int, device: str) -> TrainConfig:
    keys = TrainConfig.__dataclass_fields__
    values = {key: value for key, value in config["model"].items() if key in keys}
    return TrainConfig(**{**values, "seed": seed, "device": device})


def train_models(dataset: PreparedDataset, split: SplitManifest, config: dict,
                 output: Path, variants: list[str], deadline: Deadline, device: str,
                 resume: bool) -> dict:
    metadata = {}
    # No final-item labels, including those of old models, enter this trainer.
    permitted_items = np.concatenate([split.train_items, split.val_items])
    labels = np.full((len(split.train_models), dataset.n_questions), np.nan, dtype=np.float32)
    labels[:, permitted_items] = dataset.labels[np.ix_(split.train_models, permitted_items)]
    fit_train_items = split.train_items
    fit_val_items = split.val_items
    if split.protocol == "fixed_bank":
        # Separate protocol: old-model query responses are explicitly visible.
        labels = dataset.labels[split.train_models].copy()
        fit_train_items = np.concatenate([split.train_items, split.test_items])
    for variant in variants:
        for seed in config["seeds"]:
            deadline.check()
            path = checkpoint_path(output, variant, seed)
            if resume and path.exists():
                _, existing = load_decoder(path)
                if existing.get("config_sha256") != config_hash(config) or existing.get("dataset_fingerprint") != dataset_fingerprint(dataset):
                    raise ValueError(f"Checkpoint cannot be resumed with a different config/data: {path}")
                if existing.get("split_sha256") != sha256(output / "split.json"):
                    raise ValueError(f"Checkpoint belongs to a different split: {path}")
                if existing.get("training_model_ids") != dataset.model_ids[split.train_models].tolist() or existing.get("training_prompt_ids") != dataset.prompt_ids[fit_train_items].tolist():
                    raise ValueError(f"Checkpoint training IDs differ: {path}")
                if existing.get("validation_prompt_ids") != dataset.prompt_ids[fit_val_items].tolist():
                    raise ValueError(f"Checkpoint validation IDs differ: {path}")
                if existing.get("training_complete", False):
                    metadata[f"{variant}/{seed}"] = existing
                    log(f"resume checkpoint {variant} seed={seed}")
                    continue
                log(f"retrain interrupted checkpoint from its deterministic initialization: {variant} seed={seed}")
            decoder = create_decoder(decoder_configuration(dataset, split, config, variant), seed)
            train_config = training_configuration(config, seed, device)
            provenance = {
                "config_sha256": config_hash(config), "dataset_fingerprint": dataset_fingerprint(dataset),
                "split_sha256": sha256(output / "split.json"),
                "prepared_data_sha256": dataset.metadata.get("verified_prepared_sha256"),
                "source_sha256": scientific_source_hashes(),
                "source_hash_policy": SOURCE_HASH_POLICY,
                "variant": variant, "seed": seed, "protocol": split.protocol,
                "training_model_ids": dataset.model_ids[split.train_models].tolist(),
                "training_prompt_ids": dataset.prompt_ids[fit_train_items].tolist(),
                "validation_prompt_ids": dataset.prompt_ids[fit_val_items].tolist(),
                "implementation": "independent neural logistic factor model; not a faithful EmbedLLM reproduction",
            }
            selection_callback = None
            if train_config.selection_metric in {"coldstart_macro_mae", "macro_mae"}:
                support_budget = min(train_config.selection_budget, len(split.probe_items))
                support = select_probe_order(dataset.embeddings, dataset.benchmark_ids, split.probe_items,
                    support_budget, "balanced_random", seed, check_deadline=deadline.check)
                validation_responses = dataset.labels[np.ix_(split.val_models, support)]
                validation_truth = dataset.labels[np.ix_(split.val_models, split.val_items)]
                provenance["checkpoint_selection"] = {
                    "metric": "coldstart_macro_mae", "model_ids": dataset.model_ids[split.val_models].tolist(),
                    "probe_prompt_ids": dataset.prompt_ids[support].tolist(),
                    "target_prompt_ids": dataset.prompt_ids[split.val_items].tolist(),
                    "prior_strengths": list(train_config.selection_prior_strengths),
                }

                def on_selection(model: ColdStartDecoder, epoch: int) -> dict:
                    support_offsets, support_design = model.export_features(dataset.embeddings[support],
                        dataset.benchmark_ids[support], support if model.item_bias is not None else None)
                    target_offsets, target_design = model.export_features(dataset.embeddings[split.val_items],
                        dataset.benchmark_ids[split.val_items], split.val_items if model.item_bias is not None else None)
                    mean, base_precision = model.latent_prior(
                        variance_floor=config["model"].get("prior_scale_floor", 0.2),
                        variance_shrinkage=config["model"].get("prior_variance_shrinkage", 0.2))
                    candidates = []
                    for strength in train_config.selection_prior_strengths:
                        probabilities = []
                        for response in validation_responses:
                            fit = adapt_map(support_design, support_offsets, response, mean,
                                strength * base_precision, check_deadline=deadline.check)
                            probabilities.append(predict_from_latent(fit.latent, target_offsets, target_design))
                        metric = benchmark_metrics(validation_truth, np.stack(probabilities),
                            dataset.benchmark_ids[split.val_items])["summary"]["macro_mae"]
                        candidates.append((float(metric), float(strength)))
                    value, strength = min(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
                    return {"metric": "coldstart_macro_mae", "value": value, "prior_strength": strength}

                selection_callback = on_selection

            def on_checkpoint(model: ColdStartDecoder, state: dict) -> None:
                save_decoder(path, model, {**provenance, **state, "training_complete": False})

            def on_progress(row: dict) -> None:
                if row["epoch"] == 1 or row["epoch"] % 5 == 0:
                    log(f"train {variant} seed={seed} epoch={row['epoch']} "
                        f"BCE={row['train_bce']:.5f} val={row['validation_bce']:.5f} "
                        f"seconds={row['elapsed_seconds']:.1f} selection={row.get('selection_value')}")

            result = fit_decoder(
                decoder, dataset.embeddings, dataset.benchmark_ids, labels,
                fit_train_items, fit_val_items, train_config,
                check_deadline=deadline.check, checkpoint_callback=on_checkpoint,
                progress_callback=on_progress, selection_callback=selection_callback,
            )
            completed = {**provenance, **result, "training_complete": True}
            save_decoder(path, decoder, completed)
            atomic_json(output / "training" / f"{variant}_seed_{seed}.json", completed)
            metadata[f"{variant}/{seed}"] = completed
            log(f"trained {variant} seed={seed} best_epoch={result['best_epoch']} "
                f"val_BCE={result['best_validation_bce']:.5f} {result['best_selection_metric']}="
                f"{result['best_selection_value']:.5f} seconds={result['elapsed_seconds']:.1f}")
    return metadata


def load_features(dataset: PreparedDataset, config: dict, output: Path, variants: list[str],
                  seed: int, permitted_items: np.ndarray, device: str = "cpu") -> dict:
    features = {}
    for variant in variants:
        path = checkpoint_path(output, variant, seed)
        decoder, metadata = load_decoder(path, str(select_device(device)))
        if not metadata.get("training_complete", False):
            raise ValueError(f"Incomplete checkpoint requires retraining: {path}")
        if metadata.get("config_sha256") != config_hash(config):
            raise ValueError(f"Checkpoint config differs: {path}")
        if metadata.get("dataset_fingerprint") != dataset_fingerprint(dataset):
            raise ValueError(f"Checkpoint dataset differs: {path}")
        if metadata.get("split_sha256") != sha256(output / "split.json"):
            raise ValueError(f"Checkpoint split differs: {path}")
        item_ids = permitted_items if decoder.item_bias is not None else None
        encoded_offsets, encoded_design = decoder.export_features(
            dataset.embeddings[permitted_items], dataset.benchmark_ids[permitted_items], item_ids)
        # Acquisition receives only support slices of these arrays. Held-out
        # features are computed lazily, after support MAP, solely for prediction.
        offsets = np.full(dataset.n_questions, np.nan, dtype=np.float32)
        design = np.full((dataset.n_questions, decoder.config.latent_dim), np.nan, dtype=np.float32)
        offsets[permitted_items], design[permitted_items] = encoded_offsets, encoded_design
        mean, precision = decoder.latent_prior(
            variance_floor=config["model"].get("prior_scale_floor", 0.2),
            variance_shrinkage=config["model"].get("prior_variance_shrinkage", 0.2),
        )
        features[variant] = {"offsets": offsets, "design": design, "mean": mean,
                             "precision": precision, "metadata": metadata, "decoder": decoder,
                             "dataset": dataset, "prediction_cache": {}}
    return features


def infer_probabilities(feature: dict, selected: np.ndarray, responses: np.ndarray,
                        target_items: np.ndarray, strength: float, config: dict,
                        deadline: Deadline) -> tuple[np.ndarray, dict]:
    fit = adapt_map(
        feature["design"][selected], feature["offsets"][selected], responses,
        feature["mean"], strength * feature["precision"],
        max_iterations=config["adaptation"].get("max_iterations", 25),
        tolerance=config["adaptation"].get("tolerance", 1e-7), check_deadline=deadline.check,
    )
    cache_key = tuple(int(item) for item in target_items)
    if cache_key not in feature["prediction_cache"]:
        decoder, dataset = feature["decoder"], feature["dataset"]
        item_ids = target_items if decoder.item_bias is not None else None
        feature["prediction_cache"][cache_key] = decoder.export_features(
            dataset.embeddings[target_items], dataset.benchmark_ids[target_items], item_ids)
    target_offsets, target_design = feature["prediction_cache"][cache_key]
    probabilities = predict_from_latent(fit.latent, target_offsets, target_design)
    return probabilities, {**fit.metadata(), "latent": fit.latent, "covariance": fit.covariance}


def tune(dataset: PreparedDataset, split: SplitManifest, config: dict, output: Path,
         variants: list[str], deadline: Deadline, device: str) -> dict:
    """Choose priors and baseline smoothing on held-out models/validation items."""
    target = split.val_items
    rows = split.val_models
    budgets = config["tuning"].get("reference_budgets", config["budgets"])
    strategy = config["tuning"].get("reference_strategy", "balanced_random")
    grid = []
    prior_scores: dict[str, dict[float, list[float]]] = {variant: defaultdict(list) for variant in variants}
    baseline_scores: dict[str, dict[float, list[float]]] = {
        "domain_probe_mean": defaultdict(list), "global_probe_mean": defaultdict(list),
    }
    knn_scores: dict[tuple[int, float], list[float]] = defaultdict(list)
    scalar_scores: dict[float, list[float]] = defaultdict(list)
    scalar_config = config.get("baselines", {}).get("scalar_logit_shift", {})
    for seed in config["seeds"]:
        deadline.check()
        features = load_features(dataset, config, output, variants, seed, split.probe_items, device)
        order = select_probe_order(dataset.embeddings, dataset.benchmark_ids, split.probe_items,
                                   max(budgets), strategy, seed, device=device,
                                   check_deadline=deadline.check)
        baseline = MeanBaselines(len(dataset.benchmark_names)).fit(
            dataset.labels[np.ix_(split.train_models, split.train_items)],
            dataset.benchmark_ids[split.train_items],
        )
        scalar = None
        if scalar_config.get("enabled", False):
            scalar = ScalarLogitShift(len(dataset.benchmark_names),
                variance_floor=scalar_config.get("variance_floor", 0.2),
                training_ridge=scalar_config.get("training_ridge", 1e-6)).fit(
                dataset.labels[np.ix_(split.train_models, split.train_items)],
                dataset.benchmark_ids[split.train_items])
            atomic_json(output / "baseline_fits" / f"scalar_seed_{seed}.json", {
                "training_model_ids": dataset.model_ids[split.train_models],
                "training_prompt_ids": dataset.prompt_ids[split.train_items],
                "dataset_fingerprint": dataset_fingerprint(dataset), **scalar.metadata()})
        y_target = dataset.labels[np.ix_(rows, target)]
        domains_target = dataset.benchmark_ids[target]
        for budget in budgets:
            selected = order[:budget]
            responses = dataset.labels[np.ix_(rows, selected)]
            for variant in variants:
                for strength in config["adaptation"].get("prior_strengths", [0.3, 1.0, 3.0]):
                    deadline.check()
                    probabilities = np.stack([
                        infer_probabilities(features[variant], selected, responses[i], target,
                                            float(strength), config, deadline)[0]
                        for i in range(len(rows))
                    ])
                    score = benchmark_metrics(y_target, probabilities, domains_target)["summary"]["macro_mae"]
                    prior_scores[variant][float(strength)].append(float(score))
                    grid.append({"kind": "prior", "predictor": VARIANTS[variant]["predictor"],
                                 "variant": variant, "seed": seed, "budget": budget,
                                 "value": strength, "macro_mae": score})
            if scalar is not None:
                for strength in scalar_config.get("prior_strengths", [0.1, 0.3, 1.0, 3.0, 10.0]):
                    probabilities = scalar.predict(responses, dataset.benchmark_ids[selected], domains_target,
                                                   float(strength), check_deadline=deadline.check)
                    score = benchmark_metrics(y_target, probabilities, domains_target)["summary"]["macro_mae"]
                    scalar_scores[float(strength)].append(float(score))
                    grid.append({"kind": "scalar_prior", "predictor": "scalar_logit_shift",
                                 "seed": seed, "budget": budget, "value": strength, "macro_mae": score})
            for smoothing in config.get("baselines", {}).get("mean_smoothing_grid", [0.5, 2.0, 5.0, 10.0, 20.0]):
                baseline.smoothing = smoothing
                for name, by_domain in (("domain_probe_mean", True), ("global_probe_mean", False)):
                    probabilities = baseline.probe_mean(responses, dataset.benchmark_ids[selected],
                                                        domains_target, by_benchmark=by_domain)
                    score = benchmark_metrics(y_target, probabilities, domains_target)["summary"]["macro_mae"]
                    baseline_scores[name][float(smoothing)].append(float(score))
                    grid.append({"kind": "baseline_smoothing", "predictor": name,
                                 "seed": seed, "budget": budget, "value": smoothing, "macro_mae": score})
            for neighbors in config.get("baselines", {}).get("knn_neighbors_grid", [5, 10, 20]):
                for prior_weight in config.get("baselines", {}).get("knn_prior_weight_grid", [0.0, 1.0, 3.0]):
                    deadline.check()
                    probabilities = question_knn(
                        dataset.embeddings[selected], responses, dataset.embeddings[target],
                        prior_probabilities=baseline.benchmark_prior[domains_target],
                        n_neighbors=neighbors, prior_weight=prior_weight,
                    )
                    score = benchmark_metrics(y_target, probabilities, domains_target)["summary"]["macro_mae"]
                    knn_scores[(neighbors, prior_weight)].append(float(score))
                    grid.append({"kind": "knn", "predictor": "question_knn", "seed": seed,
                                 "budget": budget, "neighbors": neighbors,
                                 "prior_weight": prior_weight, "macro_mae": score})
            log(f"validated seed={seed} budget={budget} reference={strategy}")
        atomic_csv(output / "validation_grid.csv", grid)

    def best(mapping: dict) -> Any:
        # Higher shrinkage wins an exact tie; this rule is fixed before testing.
        return min(mapping, key=lambda value: (float(np.mean(mapping[value])), -value))

    selected_priors = {variant: float(best(scores)) for variant, scores in prior_scores.items()}
    selected_means = {name: float(best(scores)) for name, scores in baseline_scores.items()}
    selected_knn = min(knn_scores, key=lambda value: (float(np.mean(knn_scores[value])), -value[1], value[0]))
    selection = {
        "prior_strengths": selected_priors, "mean_smoothing": selected_means,
        "knn": {"n_neighbors": int(selected_knn[0]), "prior_weight": float(selected_knn[1]), "temperature": 0.1},
        "scalar_logit_shift": ({"enabled": True, "prior_strength": float(best(scalar_scores))}
                               if scalar_scores else {"enabled": False}),
        "validation_summary": {
            "priors": {variant: {str(strength): float(np.mean(scores)) for strength, scores in values.items()}
                       for variant, values in prior_scores.items()},
            "means": {name: {str(value): float(np.mean(scores)) for value, scores in values.items()}
                      for name, values in baseline_scores.items()},
            "knn": {str(value): float(np.mean(scores)) for value, scores in knn_scores.items()},
            "scalar_prior_strengths": {str(value): float(np.mean(scores)) for value, scores in scalar_scores.items()},
        },
        "reference_strategy": strategy, "reference_budgets": budgets,
        "validation_model_ids": dataset.model_ids[rows].tolist(),
        "validation_prompt_ids": dataset.prompt_ids[target].tolist(),
        "test_labels_used": False,
    }
    atomic_json(output / "selected_hyperparameters.json", selection)
    atomic_csv(output / "validation_grid.csv", grid)
    log(f"selected priors={selected_priors}, means={selected_means}, KNN={selection['knn']}")
    return selection


def create_lock(dataset: PreparedDataset, config: dict, output: Path,
                variants: list[str], selection: dict) -> dict:
    checkpoint_hashes = {
        f"{variant}/{seed}": sha256(checkpoint_path(output, variant, seed))
        for variant in variants for seed in config["seeds"]
    }
    lock = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash(config), "split_sha256": sha256(output / "split.json"),
            "source_sha256": scientific_source_hashes(),
            "source_hash_policy": SOURCE_HASH_POLICY,
            "prepared_data_sha256": dataset.metadata.get("verified_prepared_sha256"),
            "dataset_fingerprint": dataset_fingerprint(dataset), "checkpoint_sha256": checkpoint_hashes,
            "variants": variants, "selection": selection, "test_labels_used": False}
    atomic_json(output / "test_lock.json", lock)
    return lock


def verify_lock(dataset: PreparedDataset, config: dict, output: Path, variants: list[str]) -> dict:
    path = output / "test_lock.json"
    if not path.exists():
        raise ValueError("Test evaluation requires validation and test_lock.json")
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = {"config_sha256": config_hash(config), "split_sha256": sha256(output / "split.json"),
                "dataset_fingerprint": dataset_fingerprint(dataset), "variants": variants,
                "source_sha256": scientific_source_hashes(),
                "source_hash_policy": SOURCE_HASH_POLICY,
                "prepared_data_sha256": dataset.metadata.get("verified_prepared_sha256")}
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(f"Test lock mismatch for {key}; use a separate output directory")
    for variant in variants:
        for seed in config["seeds"]:
            if lock["checkpoint_sha256"][f"{variant}/{seed}"] != sha256(checkpoint_path(output, variant, seed)):
                raise ValueError("Checkpoint changed after test lock")
    return lock


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["predictor"], row["strategy"], row["budget"], row["phase"])].append(row)
    summary = []
    for (predictor, strategy, budget, phase), rows in sorted(groups.items()):
        entry = {"predictor": predictor, "strategy": strategy, "budget": budget, "phase": phase,
                 "num_models": len({row["model_id"] for row in rows}),
                 "num_seeds": len({row["seed"] for row in rows}), "num_rows": len(rows)}
        for metric in METRIC_NAMES:
            values_by_model: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if metric in row:
                    values_by_model[row["model_id"]].append(float(row[metric]))
            values = np.array([np.mean(values) for values in values_by_model.values()])
            entry[metric] = float(np.mean(values))
            entry[metric + "_model_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def evaluate(dataset: PreparedDataset, split: SplitManifest, config: dict, output: Path,
             variants: list[str], selection: dict, deadline: Deadline, device: str,
             phase: str, save_predictions: bool) -> dict:
    rows = split.test_models if phase == "test" else split.val_models
    target = split.test_items if phase == "test" else split.val_items
    records, domain_records, probe_records, adaptation_records, acquisition_records = [], [], [], [], []
    budgets = config["budgets"]
    prediction_dir = output / f"{phase}_predictions"
    started = time.perf_counter()

    def score(probabilities: np.ndarray, identity: dict, extra: dict | None = None) -> None:
        metrics = benchmark_metrics(dataset.labels[np.ix_(rows, target)], probabilities,
                                    dataset.benchmark_ids[target])
        for local_model, dataset_row in enumerate(rows):
            base = {**identity, "model_id": int(dataset.model_ids[dataset_row]),
                    "model_name": dataset.model_names[dataset_row]}
            records.append({**base, **{name: float(values[local_model]) for name, values in metrics["per_model"].items()},
                            **(extra or {})})
            for local_domain, domain in enumerate(metrics["domain_ids"]):
                domain_records.append({**base, "benchmark": dataset.benchmark_names[int(domain)],
                    "true_score": float(metrics["true_scores"][local_model, local_domain]),
                    "pred_score": float(metrics["pred_scores"][local_model, local_domain]),
                    "error": float(metrics["absolute_error"][local_model, local_domain]),
                    "signed_error": float(metrics["signed_error"][local_model, local_domain]),
                    "count": int(metrics["counts"][local_model, local_domain])})
        if save_predictions:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            name = f"seed{identity['seed']}_{identity['strategy']}_k{identity['budget']}_{identity['predictor']}.npz"
            np.savez_compressed(prediction_dir / name, probabilities=probabilities,
                                model_ids=dataset.model_ids[rows], prompt_ids=dataset.prompt_ids[target])

    for seed in config["seeds"]:
        deadline.check()
        features = load_features(dataset, config, output, variants, seed, split.probe_items, device)
        baseline = MeanBaselines(len(dataset.benchmark_names)).fit(
            dataset.labels[np.ix_(split.train_models, split.train_items)], dataset.benchmark_ids[split.train_items])
        scalar_config = config.get("baselines", {}).get("scalar_logit_shift", {})
        scalar = None
        if scalar_config.get("enabled", False):
            if not selection.get("scalar_logit_shift", {}).get("enabled", False):
                raise ValueError("Enabled scalar baseline requires validation-selected prior strength")
            scalar = ScalarLogitShift(len(dataset.benchmark_names),
                variance_floor=scalar_config.get("variance_floor", 0.2),
                training_ridge=scalar_config.get("training_ridge", 1e-6)).fit(
                dataset.labels[np.ix_(split.train_models, split.train_items)],
                dataset.benchmark_ids[split.train_items])
        base_identity = {"experiment_id": config["experiment"]["name"], "protocol": split.protocol,
                         "label_policy": config["data"]["label_policy"],
                         "model_split": split.metadata.get("model_split", split.metadata.get("split_kind", "random")),
                         "arm": split.metadata.get("arm", ""),
                         "target_family": split.metadata.get("target_family", ""),
                         "validation_family": split.metadata.get("validation_family", ""),
                         "seed": seed, "phase": phase}
        score(baseline.training_mean(dataset.benchmark_ids[target], len(rows)),
              {**base_identity, "budget": 0, "strategy": "zero_probe", "predictor": "training_domain_mean",
               "rank": "", "mode": "", "prior_strength": ""})
        for strategy in config["probe_strategies"]:
            deadline.check()
            selection_started = time.perf_counter()
            if strategy in STRATEGIES:
                order = select_probe_order(dataset.embeddings, dataset.benchmark_ids, split.probe_items,
                    max(budgets), strategy, seed, device=device, check_deadline=deadline.check)
                orders = np.broadcast_to(order, (len(rows), len(order)))
                selection_seconds = time.perf_counter() - selection_started
            elif strategy in {"fisher", "balanced_fisher"}:
                from coldstart.acquisition import select_adaptive_probe_order
                fisher_config = config.get("probes", {}).get("fisher", {})
                selector_variant = fisher_config.get("selector_variant", "structured8")
                if selector_variant not in features:
                    raise ValueError(f"Fisher acquisition needs its declared {selector_variant} decoder")
                feature = features[selector_variant]
                pool = split.probe_items
                reference_parts = []
                for domain in np.unique(dataset.benchmark_ids[pool]):
                    eligible = pool[dataset.benchmark_ids[pool] == domain]
                    rng = np.random.default_rng(np.random.SeedSequence([
                        fisher_config.get("reference_seed", 314159), int(domain)]))
                    reference_parts.append(rng.permutation(eligible)[:fisher_config.get("reference_items_per_benchmark", 64)])
                reference = np.concatenate(reference_parts)
                atomic_json(output / "fisher_reference.json", {"pool": "train_items",
                    "prompt_ids": dataset.prompt_ids[reference], "benchmark_ids": dataset.benchmark_ids[reference],
                    "seed": fisher_config.get("reference_seed", 314159), "selector_variant": selector_variant})
                orders = []
                for dataset_row in rows:
                    acquired = select_adaptive_probe_order(
                        candidate_design=feature["design"][pool], candidate_offsets=feature["offsets"][pool],
                        candidate_benchmark_ids=dataset.benchmark_ids[pool],
                        responses=lambda local_index, row=int(dataset_row): float(dataset.labels[row, pool[local_index]]),
                        reference_design=feature["design"][reference], reference_offsets=feature["offsets"][reference],
                        reference_benchmark_ids=dataset.benchmark_ids[reference], prior_mean=feature["mean"],
                        prior_precision=selection["prior_strengths"][selector_variant] * feature["precision"],
                        budget=max(budgets), strategy=strategy, seed=seed,
                        check_deadline=deadline.check, warmup=fisher_config.get("warmup", 10),
                    )
                    if acquired.response_count != max(budgets):
                        raise ValueError("Acquisition response count differs from its declared budget")
                    acquisition_records.append({**base_identity, "strategy": strategy,
                        "selector_variant": selector_variant,
                        "model_id": int(dataset.model_ids[dataset_row]), "response_count": acquired.response_count,
                        "seconds": acquired.seconds, "trace": [{**step,
                            "prompt_id": int(dataset.prompt_ids[pool[step["candidate_index"]]])}
                            for step in acquired.trace]})
                    trace_path = output / f"{phase}_acquisition.jsonl"
                    temporary = trace_path.with_suffix(".jsonl.tmp")
                    with temporary.open("w", encoding="utf-8") as handle:
                        for record in acquisition_records:
                            handle.write(json.dumps(record, default=_json_default) + "\n")
                    atomic_replace(temporary, trace_path)
                    orders.append(pool[acquired.indices])
                orders = np.asarray(orders)
                selection_seconds = time.perf_counter() - selection_started
            else:
                raise ValueError(f"Unknown acquisition policy: {strategy}")
            for local_model, dataset_row in enumerate(rows):
                probe_records.append({**base_identity, "strategy": strategy,
                    "model_id": int(dataset.model_ids[dataset_row]), "budget": max(budgets),
                    "prompt_ids": dataset.prompt_ids[orders[local_model]].tolist(),
                    "selection_seconds_all_models": selection_seconds})
            for budget in budgets:
                deadline.check()
                selected_orders = orders[:, :budget]
                probabilities_by_predictor: dict[str, list[np.ndarray]] = defaultdict(list)
                adaptation_seconds = defaultdict(float)
                fitted_latents: dict[str, list[np.ndarray]] = defaultdict(list)
                fitted_covariances: dict[str, list[np.ndarray]] = defaultdict(list)
                for local_model, dataset_row in enumerate(rows):
                    selected = selected_orders[local_model]
                    responses = dataset.labels[dataset_row, selected]
                    for name, by_benchmark in (("domain_probe_mean", True), ("global_probe_mean", False)):
                        baseline.smoothing = selection["mean_smoothing"][name]
                        probabilities_by_predictor[name].append(baseline.probe_mean(
                            responses, dataset.benchmark_ids[selected], dataset.benchmark_ids[target],
                            by_benchmark=by_benchmark)[0])
                    if scalar is not None:
                        scalar_started = time.perf_counter()
                        probabilities_by_predictor["scalar_logit_shift"].append(scalar.predict(responses,
                            dataset.benchmark_ids[selected], dataset.benchmark_ids[target],
                            selection["scalar_logit_shift"]["prior_strength"], check_deadline=deadline.check)[0])
                        adaptation_seconds["scalar_logit_shift"] += time.perf_counter() - scalar_started
                    probabilities_by_predictor["question_knn"].append(question_knn(
                        dataset.embeddings[selected], responses, dataset.embeddings[target],
                        prior_probabilities=baseline.benchmark_prior[dataset.benchmark_ids[target]],
                        **selection["knn"])[0])
                    for variant in variants:
                        before = time.perf_counter()
                        probability, info = infer_probabilities(features[variant], selected, responses, target,
                            selection["prior_strengths"][variant], config, deadline)
                        name = VARIANTS[variant]["predictor"]
                        adaptation_seconds[name] += time.perf_counter() - before
                        probabilities_by_predictor[name].append(probability)
                        fitted_latents[variant].append(info["latent"])
                        fitted_covariances[variant].append(info["covariance"])
                        adaptation_records.append({**base_identity, "strategy": strategy, "budget": budget,
                            "predictor": name, "model_id": int(dataset.model_ids[dataset_row]),
                            **{key: value for key, value in info.items() if key not in {"latent", "covariance"}}})
                        permutation = config.get("diagnostics", {}).get("within_benchmark_response_permutation", {})
                        if permutation.get("enabled", False) and strategy == permutation.get("policy", "balanced_random") and variant in permutation.get("variants", list(VARIANTS)):
                            from coldstart.diagnostics import permute_support_within_benchmark
                            permutation_seed = int(np.random.SeedSequence([
                                seed, budget, int(dataset.model_ids[dataset_row]), 969]).generate_state(1)[0])
                            shuffled = permute_support_within_benchmark(
                                responses, dataset.benchmark_ids[selected], permutation_seed)
                            shuffled_started = time.perf_counter()
                            shuffled_probability, shuffled_info = infer_probabilities(features[variant], selected,
                                shuffled, target, selection["prior_strengths"][variant], config, deadline)
                            adaptation_seconds[name + "_permuted"] += time.perf_counter() - shuffled_started
                            probabilities_by_predictor[name + "_permuted"].append(shuffled_probability)
                            adaptation_records.append({**base_identity, "strategy": strategy, "budget": budget,
                                "predictor": name + "_permuted", "model_id": int(dataset.model_ids[dataset_row]),
                                **{key: value for key, value in shuffled_info.items() if key not in {"latent", "covariance"}}})
                embedding_dir = output / f"{phase}_embeddings"
                embedding_dir.mkdir(parents=True, exist_ok=True)
                for variant in variants:
                    np.savez_compressed(embedding_dir / f"seed{seed}_{strategy}_k{budget}_{variant}.npz",
                        latents=np.stack(fitted_latents[variant]), covariances=np.stack(fitted_covariances[variant]),
                        prior_mean=features[variant]["mean"],
                        prior_precision=selection["prior_strengths"][variant] * features[variant]["precision"],
                        model_ids=dataset.model_ids[rows], probe_prompt_ids=dataset.prompt_ids[selected_orders],
                        checkpoint_sha256=np.array(sha256(checkpoint_path(output, variant, seed))))
                for predictor, values in probabilities_by_predictor.items():
                    variant = next((value for value in variants if VARIANTS[value]["predictor"] == predictor.removesuffix("_permuted")), None)
                    score(np.stack(values), {**base_identity, "budget": budget, "strategy": strategy,
                        "predictor": predictor, "rank": VARIANTS[variant]["rank"] if variant else "",
                        "mode": VARIANTS[variant]["mode"] if variant else "scalar" if predictor == "scalar_logit_shift" else "",
                        "prior_strength": selection["prior_strengths"][variant] if variant else
                            selection["scalar_logit_shift"]["prior_strength"] if predictor == "scalar_logit_shift" else ""},
                        {"adaptation_seconds_mean": adaptation_seconds[predictor] / len(rows),
                         "selection_seconds_all_models": selection_seconds})
                log(f"{phase} seed={seed} strategy={strategy} budget={budget} models={len(rows)}")
                atomic_csv(output / f"{phase}_per_model_metrics.csv", records)
                atomic_csv(output / f"{phase}_per_domain_metrics.csv", domain_records)
                atomic_json(output / f"{phase}_summary.json", summarize(records))
                atomic_json(output / f"{phase}_probes.json", probe_records)
                atomic_csv(output / f"{phase}_adaptation.csv", adaptation_records)
    summary = {"elapsed_seconds": time.perf_counter() - started, "num_records": len(records),
               "summary": summarize(records), "phase": phase}
    atomic_json(output / f"{phase}_completed.json", summary)
    if phase == "test":
        # Stable convenience filenames for report/plot consumers.
        atomic_csv(output / "per_model_metrics.csv", records)
        atomic_csv(output / "per_domain_metrics.csv", domain_records)
        atomic_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/protocol.json")
    parser.add_argument("--data", help="Prepared data directory or NPZ; defaults to config")
    parser.add_argument("--output", default="artifacts/main")
    parser.add_argument("--split", help="Predeclared split manifest (for family studies)")
    parser.add_argument("--phase", choices=["train", "validate", "train_validate", "test", "all"], default="train_validate")
    parser.add_argument("--variants", default="structured8,rank0,neural_lowrank")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deadline", help="Optional timezone-aware ISO deadline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--evaluate-validation", action="store_true", help="Run all policies on validation after selection")
    parser.add_argument("--selection-file", help="Preselected shared validation hyperparameters, e.g. a paired family study")
    parser.add_argument("--epochs", type=int, help="Development override recorded in effective config")
    parser.add_argument("--seeds", help="Comma-separated development override recorded in effective config")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deadline = Deadline(args.deadline)
    deadline.check()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["model"]["epochs"] = args.epochs
    if args.seeds:
        config["seeds"] = [int(value) for value in args.seeds.split(",")]
    variants = args.variants.split(",")
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"Available model variants: {', '.join(VARIANTS)}")
    torch.set_num_threads(args.cpu_threads)
    # Avoid severe oversubscription during the small NumPy Newton solves.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.cpu_threads)
    data_path = args.data or config["data"].get("prepared_path", "data/processed")
    requested_models = config["data"].get("model_ids")
    if requested_models is None and config["data"].get("model_scope") == "audited_lineage_included_only":
        requested_models = sorted(load_family_table(config["data"].get("family_table", config["data"].get("model_family_table", "data/metadata/model_families.csv"))))
    dataset = load_dataset(data_path, label_policy=config["data"]["label_policy"], model_ids=requested_models,
                           benchmarks=config["data"].get("candidate_benchmarks"),
                           deduplicate_groups=config["data"].get("deduplicate_groups"))
    prepared_path = Path(data_path) / "embedllm.npz" if Path(data_path).is_dir() else Path(data_path)
    actual_data_sha256 = sha256(prepared_path)
    if dataset.metadata.get("prepared_sha256") not in (None, actual_data_sha256):
        raise ValueError("Prepared dataset bytes differ from their recorded checksum")
    dataset.metadata["verified_prepared_sha256"] = actual_data_sha256
    if args.split:
        split = load_split(args.split, dataset)
        if (output / "split.json").exists():
            supplied = json.loads(Path(args.split).read_text(encoding="utf-8"))["manifest_sha256"]
            existing = json.loads((output / "split.json").read_text(encoding="utf-8"))["manifest_sha256"]
            if supplied != existing:
                raise ValueError("Supplied split differs from the output's saved split; choose a new output directory")
    elif (output / "split.json").exists():
        split = load_split(output / "split.json", dataset)
    else:
        split_config = config["split"]
        split = make_split(dataset, item_seed=split_config.get("item_seed", 1729),
                           model_seed=split_config.get("model_seed", 2718),
                           item_ratios=split_config.get("item_ratios", [0.7, 0.15, 0.15]),
                           model_ratios=split_config.get("model_ratios", [0.7, 0.15, 0.15]),
                           protocol=config["experiment"]["regime"])
    if split.protocol != config["experiment"]["regime"]:
        raise ValueError("Config and split information protocols differ")
    if (output / "test_completed.json").exists() and args.phase != "test":
        raise ValueError("This output already has test results; development requires a new directory")
    effective_path = output / "effective_config.json"
    if effective_path.exists() and config_hash(json.loads(effective_path.read_text(encoding="utf-8"))) != config_hash(config):
        raise ValueError("Output directory belongs to another config; choose a new directory")
    atomic_json(effective_path, config)
    if not (output / "split.json").exists():
        save_split(output / "split.json", dataset, split)
    atomic_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "command": sys.argv, "dataset_fingerprint": dataset_fingerprint(dataset)})
    log(f"data models={dataset.n_models} questions={dataset.n_questions} "
        f"benchmarks={list(dataset.benchmark_names)} split_models="
        f"{len(split.train_models)}/{len(split.val_models)}/{len(split.test_models)} "
        f"split_items={len(split.train_items)}/{len(split.val_items)}/{len(split.test_items)}")
    with register_job(deadline=args.deadline, metadata={"output": str(output), "phase": args.phase}):
        try:
            if args.phase in {"train", "train_validate", "all"}:
                train_models(dataset, split, config, output, variants, deadline, args.device, args.resume)
            if args.phase in {"validate", "train_validate", "all"}:
                if args.selection_file:
                    selection = json.loads(Path(args.selection_file).read_text(encoding="utf-8"))
                    selection["shared_selection_source"] = str(Path(args.selection_file).resolve())
                    selection["shared_selection_sha256"] = sha256(Path(args.selection_file))
                    atomic_json(output / "selected_hyperparameters.json", selection)
                else:
                    selection = tune(dataset, split, config, output, variants, deadline, args.device)
                create_lock(dataset, config, output, variants, selection)
                if args.evaluate_validation:
                    evaluate(dataset, split, config, output, variants, selection, deadline,
                             args.device, "validation", args.save_predictions)
            if args.phase in {"test", "all"}:
                lock = verify_lock(dataset, config, output, variants)
                evaluate(dataset, split, config, output, variants, lock["selection"], deadline,
                         args.device, "test", args.save_predictions)
            atomic_json(output / "run_status.json", {"status": "completed", "phase": args.phase,
                        "finished_at": datetime.now(timezone.utc).isoformat()})
        except DeadlineExceeded:
            atomic_json(output / "run_status.json", {"status": "deadline_exceeded", "phase": args.phase,
                        "finished_at": datetime.now(timezone.utc).isoformat()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
