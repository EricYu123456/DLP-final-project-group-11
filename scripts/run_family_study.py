"""Prepare, jointly validate, then explicitly test matched ancestry contrasts.

Prepare never trains. Validate runs both arms without final testing and chooses
one shared prior/baseline setting per pair using Qwen validation models only.
Testing is a separate, explicit --phase test invocation after review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import psutil

from coldstart.data import PreparedDataset, load_dataset, sha256_file
from coldstart.runtime import Deadline, atomic_replace, register_job
from coldstart.splits import (dataset_fingerprint, load_family_table, load_split,
                              make_paired_family_shift_splits, make_question_split, save_split)

VARIANTS = ("structured8", "rank0")
PREDICTORS = {"structured8": "coldstart_structured8", "rank0": "coldstart_rank0"}
ARMS = ("excluded", "related_allowed")
CASES = {"Llama": (8, 8), "Mistral": (6, 8), "Yi": (3, 6)}
SEEDS = (11, 23, 37)
BUDGETS = (50, 100, 200)
SOURCE_HASH_POLICY = "utf8_lf"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _python_source_sha256(path: Path) -> str:
    source = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atomic_replace(temporary, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_replace(temporary, path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_family_config(config: dict) -> None:
    """Reject accidental use of a main-grid-selected configuration."""
    if config["experiment"]["regime"] != "strict_unseen_item" or config["data"]["label_policy"] != "strict":
        raise ValueError("Family study requires the strict unseen-item protocol")
    if config["data"].get("deduplicate_groups") is not True:
        raise ValueError("Family study requires the frozen global group representatives")
    if tuple(config["seeds"]) != SEEDS or tuple(config["budgets"]) != BUDGETS:
        raise ValueError("Family seeds and budgets must match the predeclared protocol")
    if config["probe_strategies"] != ["balanced_random"]:
        raise ValueError("Family core comparison permits balanced_random only")
    if tuple(config["experiment"].get("variants", ())) != VARIANTS:
        raise ValueError("Family configuration must declare structured8 and rank0 only")
    if config["tuning"].get("reference_strategy") != "balanced_random" or tuple(config["tuning"].get("reference_budgets", [])) != BUDGETS:
        raise ValueError("Family tuning must cover the same balanced_random budgets")
    fixed = {"hidden_dim": 128, "learning_rate": 0.002, "latent_l2": 0.002, "warmup_epochs": 0}
    if any(config["model"].get(key) != value for key, value in fixed.items()):
        raise ValueError("Use the dedicated pre-grid family hyperparameters, not the primary grid winner")
    if config["model"].get("selection_metric") not in ("coldstart_macro_mae", "macro_mae"):
        raise ValueError("Family checkpoints must be selected by held-out-model macro MAE")
    if config.get("diagnostics", {}).get("within_benchmark_response_permutation", {}).get("enabled", False):
        raise ValueError("Family core configuration must disable extra permutation diagnostics")
    if config.get("family_hyperparameter_origin", {}).get("main_grid_results_used_for_family_choice") is not False:
        raise ValueError("Family hyperparameter origin must explicitly exclude primary grid results")
    stress = config["split"]["family_stress"]
    declared = {case["family"]: (case["target_models"], case["related_training_models"]) for case in stress["target_cases"]}
    if stress["validation_family"] != "Qwen" or stress["training_models"] != 40 or declared != CASES:
        raise ValueError("Family target/relative/training counts differ from the declared matched design")


def load_inputs(config: dict, deadline: Deadline) -> tuple[PreparedDataset, dict[int, str], Path, str]:
    deadline.check()
    family_path = _root_path(config["data"].get("family_table", "data/metadata/model_families.csv"))
    families = load_family_table(family_path)
    requested = config["data"].get("model_ids") or sorted(families)
    if set(requested) != set(families):
        raise ValueError("Family study must use the complete audited inclusion scope once")
    path = _root_path(config["data"]["prepared_path"])
    dataset = load_dataset(path, label_policy="strict", model_ids=requested,
                           benchmarks=config["data"].get("candidate_benchmarks"),
                           deduplicate_groups=True)
    array_path = path / "embedllm.npz" if path.is_dir() else path
    actual_sha = sha256_file(array_path, deadline.check)
    if dataset.metadata.get("prepared_sha256") not in (None, actual_sha):
        raise ValueError("Prepared dataset bytes differ from the recorded checksum")
    dataset.metadata["verified_prepared_sha256"] = actual_sha
    if not dataset.mask.all() or len(np.unique(dataset.group_ids)) != dataset.n_questions:
        raise ValueError("Family study needs one common observed representative per group")
    if len([mid for mid in dataset.model_ids if families[int(mid)] == "Qwen"]) != 6:
        raise ValueError("Family validation must contain exactly the six declared Qwen models")
    selection_path = config["data"].get("primary_selection_manifest")
    if selection_path:
        selection = _read_json(_root_path(selection_path))
        if selection.get("dataset_fingerprint") != dataset_fingerprint(dataset):
            raise ValueError("Family dataset differs from the frozen primary selection")
    return dataset, families, family_path, actual_sha


def _study_hash(manifest: dict) -> str:
    return _hash_json({key: value for key, value in manifest.items() if key != "study_sha256"})


def prepare_study(config_path: Path, output: Path, deadline: Deadline,
                  device: str = "auto", cpu_threads: int = 4) -> dict:
    config_path, output = config_path.resolve(), output.resolve()
    config = _read_json(config_path)
    validate_family_config(config)
    dataset, families, family_path, prepared_sha = load_inputs(config, deadline)
    if (output / "study.json").exists():
        manifest = verify_study(config_path, output, deadline, dataset=dataset)
        if manifest["device"] != device or manifest["cpu_threads"] != cpu_threads:
            raise ValueError("Output already belongs to a different execution-device plan")
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to prepare into a nonempty output without study.json")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", deepcopy(config))
    split_config = config["split"]
    items = make_question_split(dataset, seed=split_config.get("item_seed", 1729),
                                 ratios=split_config.get("item_ratios", (0.7, 0.15, 0.15)))
    if len(items["train"]) < max(BUDGETS):
        raise ValueError("Common training/probe pool cannot support the largest budget")
    cases = {}
    for family, (target_count, relative_count) in CASES.items():
        deadline.check()
        pairs = make_paired_family_shift_splits(
            dataset, families, family, validation_family="Qwen", n_test=target_count,
            n_train=40, n_related=relative_count, seed=split_config.get("model_seed", 2718),
            item_seed=split_config.get("item_seed", 1729), item_splits=items)
        case = {"target_models": target_count, "training_models": 40,
                "related_training_models": relative_count, "validation_family": "Qwen", "arms": {}}
        for arm, split in pairs.items():
            relative = Path("splits") / f"{family}_{arm}.json"
            saved = save_split(output / relative, dataset, split)
            case["arms"][arm] = {"split_path": relative.as_posix(), "manifest_sha256": saved["manifest_sha256"],
                                  "split_file_sha256": sha256_file(output / relative, deadline.check),
                                  "run_path": (Path(family) / arm).as_posix()}
        case["test_model_ids"] = dataset.model_ids[pairs["excluded"].test_models].tolist()
        case["validation_model_ids"] = dataset.model_ids[pairs["excluded"].val_models].tolist()
        declared_targets = config["split"]["family_stress"].get("fixed_target_source_ids", {}).get(family)
        if declared_targets is not None and declared_targets != case["test_model_ids"]:
            raise ValueError("Generated family targets differ from the predeclared source model IDs")
        cases[family] = case
    manifest = {"schema_version": 1, "created_at": _now(), "output_root": str(output),
                "orchestrator_source_path": str(Path(__file__).resolve()),
                "orchestrator_source_sha256": _python_source_sha256(Path(__file__).resolve()),
                "orchestrator_source_hash_policy": SOURCE_HASH_POLICY,
                "source_config_path": str(config_path), "source_config_sha256": _hash_json(config),
                "source_config_file_sha256": sha256_file(config_path, deadline.check),
                "family_config_path": "config.json", "family_config_sha256": _hash_json(config),
                "family_config_file_sha256": sha256_file(output / "config.json", deadline.check),
                "family_table_path": str(family_path), "family_table_sha256": sha256_file(family_path, deadline.check),
                "dataset_fingerprint": dataset_fingerprint(dataset), "prepared_data_sha256": prepared_sha,
                "variants": list(VARIANTS), "seeds": list(SEEDS), "budgets": list(BUDGETS),
                "strategy": "balanced_random", "device": device, "cpu_threads": cpu_threads,
                "fixed_training_hyperparameters": {key: config["model"][key] for key in
                    ("hidden_dim", "learning_rate", "latent_l2", "warmup_epochs", "selection_metric")},
                "hyperparameter_origin": "Original protocol values; family configuration is not selected using the primary development grid.",
                "joint_selection_unit": "equal average across both arms, seeds and budgets, using Qwen validation models only",
                "test_stage": "Separate explicit invocation after joint-selection review; validate never tests.",
                "cases": cases}
    manifest["study_sha256"] = _study_hash(manifest)
    _write_json(output / "study.json", manifest)
    return manifest


def verify_study(config_path: Path, output: Path, deadline: Deadline,
                 dataset: PreparedDataset | None = None) -> dict:
    output, config_path = output.resolve(), config_path.resolve()
    manifest = _read_json(output / "study.json")
    if manifest.get("study_sha256") != _study_hash(manifest):
        raise ValueError("Family study manifest checksum mismatch")
    if (manifest.get("orchestrator_source_path") != str(Path(__file__).resolve())
            or manifest.get("orchestrator_source_hash_policy") != SOURCE_HASH_POLICY
            or manifest.get("orchestrator_source_sha256") != _python_source_sha256(Path(__file__).resolve())):
        raise ValueError("Family orchestrator source changed or lacks a normalized source binding")
    if manifest["output_root"] != str(output) or manifest["source_config_path"] != str(config_path):
        raise ValueError("Family output/configuration path mismatch")
    config = _read_json(config_path)
    validate_family_config(config)
    if manifest["source_config_sha256"] != _hash_json(config) or manifest["source_config_file_sha256"] != sha256_file(config_path, deadline.check):
        raise ValueError("Family source config changed after preparation")
    snapshot = output / manifest["family_config_path"]
    if manifest["family_config_sha256"] != _hash_json(_read_json(snapshot)) or manifest["family_config_file_sha256"] != sha256_file(snapshot, deadline.check):
        raise ValueError("Family frozen config changed after preparation")
    if manifest["family_table_sha256"] != sha256_file(manifest["family_table_path"], deadline.check):
        raise ValueError("Family ancestry table changed after preparation")
    if dataset is None:
        dataset, _, _, prepared_sha = load_inputs(config, deadline)
        if prepared_sha != manifest["prepared_data_sha256"]:
            raise ValueError("Family prepared data changed after preparation")
    elif dataset.metadata.get("verified_prepared_sha256") != manifest["prepared_data_sha256"]:
        raise ValueError("Family prepared data changed after preparation")
    if manifest["dataset_fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("Family dataset fingerprint changed after preparation")
    for family, case in manifest["cases"].items():
        splits = {}
        for arm, record in case["arms"].items():
            path = output / record["split_path"]
            if record["split_file_sha256"] != sha256_file(path, deadline.check):
                raise ValueError("Family split changed after preparation")
            splits[arm] = load_split(path, dataset)
        left, right = splits["excluded"], splits["related_allowed"]
        for field in ("test_models", "val_models", "train_items", "val_items", "test_items"):
            if not np.array_equal(getattr(left, field), getattr(right, field)):
                raise ValueError(f"Paired family splits differ in {field}")
        if len(left.train_models) != len(right.train_models) or len(left.train_models) != 40:
            raise ValueError("Paired family training model counts differ")
    return manifest


def build_child_command(study: dict, output: Path, family: str, arm: str,
                        phase: str, deadline: str, selection_file: Path | None = None,
                        save_predictions: bool = False) -> list[str]:
    if phase not in ("train_validate", "validate", "test"):
        raise ValueError("Family subprocess phase must be train_validate, validate or test")
    record = study["cases"][family]["arms"][arm]
    command = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
               "--config", str(output / study["family_config_path"]), "--output", str(output / record["run_path"]),
               "--split", str(output / record["split_path"]), "--phase", phase,
               "--variants", ",".join(VARIANTS), "--device", study["device"],
               "--cpu-threads", str(study["cpu_threads"]), "--deadline", deadline]
    if phase == "train_validate":
        command.append("--resume")
    if selection_file is not None:
        if phase != "validate":
            raise ValueError("Shared selection is installed only by the validation phase")
        command.extend(["--selection-file", str(selection_file)])
    if phase == "test" and save_predictions:
        command.append("--save-predictions")
    return command


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except psutil.Error:
            pass


def run_child(command: list[str], log_path: Path, deadline: Deadline) -> dict:
    """Run exactly one child at a time; terminate its tree on deadline/error."""
    deadline.check()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            while process.poll() is None:
                deadline.check()
                time.sleep(min(0.5, max(0.01, deadline.remaining_seconds())))
        except BaseException:
            _stop_child(process)
            raise
    result = {"command": command, "pid": process.pid, "started_at": started,
              "finished_at": _now(), "returncode": process.returncode, "log_path": str(log_path)}
    _write_json(log_path.with_suffix(".json"), result)
    if process.returncode != 0:
        raise RuntimeError(f"Family subprocess failed ({process.returncode}); inspect {log_path}")
    return result


def aggregate_joint_selection(grid_paths: dict[str, Path], config: dict,
                              validation_model_ids: list[int], validation_prompt_ids: list[int]) -> tuple[dict, list[dict], list[dict]]:
    """Require a complete paired grid before averaging arm/seed/budget cells."""
    if set(grid_paths) != set(ARMS):
        raise ValueError("Joint selection requires both complete paired arms")
    strengths = [float(value) for value in config["adaptation"]["prior_strengths"]]
    smoothing = [float(value) for value in config["baselines"]["mean_smoothing_grid"]]
    neighbors = [int(value) for value in config["baselines"]["knn_neighbors_grid"]]
    weights = [float(value) for value in config["baselines"]["knn_prior_weight_grid"]]
    candidates: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    seen: set[tuple] = set()
    pooled = []
    expected = set()
    for arm in ARMS:
        for seed in SEEDS:
            for budget in BUDGETS:
                for variant in VARIANTS:
                    for value in strengths:
                        expected.add((arm, "prior", variant, seed, budget, value))
                for predictor in ("domain_probe_mean", "global_probe_mean"):
                    for value in smoothing:
                        expected.add((arm, "baseline_smoothing", predictor, seed, budget, value))
                for count in neighbors:
                    for weight in weights:
                        expected.add((arm, "knn", "question_knn", seed, budget, count, weight))
    for arm in ARMS:
        for row in _read_csv(grid_paths[arm]):
            kind, predictor = row["kind"], row["predictor"]
            seed, budget, score = int(row["seed"]), int(row["budget"]), float(row["macro_mae"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Validation grid macro MAE must be finite and within [0,1]")
            if kind == "prior":
                variant, value = row.get("variant", ""), float(row["value"])
                if variant not in PREDICTORS or predictor != PREDICTORS[variant]:
                    raise ValueError("Validation prior row has an unexpected predictor/variant")
                identity = (arm, kind, variant, seed, budget, value)
                candidate = (kind, variant, value)
            elif kind == "baseline_smoothing":
                value = float(row["value"])
                identity = (arm, kind, predictor, seed, budget, value)
                candidate = (kind, predictor, value)
            elif kind == "knn":
                count, weight = int(row["neighbors"]), float(row["prior_weight"])
                identity = (arm, kind, predictor, seed, budget, count, weight)
                candidate = (kind, predictor, count, weight)
            else:
                raise ValueError(f"Unexpected validation grid kind: {kind}")
            if identity not in expected or identity in seen:
                raise ValueError("Validation grid contains an unexpected or duplicate arm/seed/budget/candidate cell")
            seen.add(identity)
            candidates[candidate].append((arm, score))
            pooled.append({"arm": arm, **row})
    if seen != expected:
        raise ValueError(f"Incomplete paired validation grid: missing {len(expected - seen)} cells")
    means = {key: float(np.mean([value for _, value in values])) for key, values in candidates.items()}
    priors = {variant: min(strengths, key=lambda value: (means[("prior", variant, value)], -value)) for variant in VARIANTS}
    mean_settings = {name: min(smoothing, key=lambda value: (means[("baseline_smoothing", name, value)], -value))
                     for name in ("domain_probe_mean", "global_probe_mean")}
    knn = min(((count, weight) for count in neighbors for weight in weights),
              key=lambda value: (means[("knn", "question_knn", *value)], -value[1], value[0]))
    summaries = []
    for key, values in sorted(candidates.items()):
        summaries.append({"kind": key[0], "predictor_or_variant": key[1], "candidate": list(key[2:]),
                          "macro_mae": means[key], "cells": len(values),
                          "excluded_macro_mae": float(np.mean([value for arm, value in values if arm == "excluded"])),
                          "related_allowed_macro_mae": float(np.mean([value for arm, value in values if arm == "related_allowed"]))})
    selection = {"schema_version": 1, "prior_strengths": priors, "mean_smoothing": mean_settings,
                 "knn": {"n_neighbors": knn[0], "prior_weight": knn[1],
                         "temperature": float(config["baselines"].get("knn_temperature", 0.1))},
                 "validation_summary": {"joint_candidates": summaries},
                 "reference_strategy": "balanced_random", "reference_budgets": list(BUDGETS),
                 "validation_model_ids": validation_model_ids, "validation_prompt_ids": validation_prompt_ids,
                 "selection_unit": "equal mean of paired-arm x seed x budget validation macro MAE",
                 "tie_break": "larger shrinkage; KNN then fewer neighbors",
                 "source_grids_sha256": {arm: sha256_file(path) for arm, path in grid_paths.items()},
                 "test_labels_used": False}
    return selection, pooled, summaries


def _pair_paths(study: dict, output: Path, family: str) -> dict[str, Path]:
    return {arm: output / study["cases"][family]["arms"][arm]["run_path"] for arm in ARMS}


def _verify_joint_locks(study: dict, output: Path, family: str) -> dict:
    marker = _read_json(output / family / "validation_completed.json")
    if marker["study_sha256"] != study["study_sha256"]:
        raise ValueError("Joint validation belongs to a different family study")
    joint_path = output / family / "joint_selection.json"
    joint_sha = sha256_file(joint_path)
    if marker["joint_selection_sha256"] != joint_sha:
        raise ValueError("Shared joint selection changed after paired validation")
    joint = _read_json(joint_path)
    for arm, path in _pair_paths(study, output, family).items():
        if marker["grid_sha256"][arm] != sha256_file(path / "validation_grid.csv"):
            raise ValueError("Family validation grid changed after selection")
        lock_path = path / "test_lock.json"
        if marker["test_lock_sha256"][arm] != sha256_file(lock_path):
            raise ValueError("Family test lock changed after joint validation")
        lock = _read_json(lock_path)
        selection = lock["selection"]
        if lock.get("test_labels_used") is not False or selection.get("test_labels_used") is not False:
            raise ValueError("Family lock must be established without final test labels")
        if selection.get("shared_selection_sha256") != joint_sha:
            raise ValueError("Paired arms do not share the joint selection file")
        if any(selection.get(key) != joint.get(key) for key in ("prior_strengths", "mean_smoothing", "knn")):
            raise ValueError("Paired arms use different selected hyperparameters")
    return marker


def validate_stage(study: dict, output: Path, families: list[str], deadline_text: str,
                   execute: Callable = run_child) -> None:
    deadline = Deadline(deadline_text)
    config = _read_json(output / study["family_config_path"])
    for family in families:
        paths = _pair_paths(study, output, family)
        if any((path / "test_completed.json").exists() for path in paths.values()):
            raise ValueError("Family outputs already contain final tests; validation cannot be rerun")
        marker_path = output / family / "validation_completed.json"
        if marker_path.exists():
            _verify_joint_locks(study, output, family)
            continue
        for arm in ARMS:
            deadline.check()
            command = build_child_command(study, output, family, arm, "train_validate", deadline_text)
            execute(command, output / family / "logs" / f"{arm}_train_validate.log", deadline)
            if (paths[arm] / "test_completed.json").exists():
                raise ValueError("Validation subprocess unexpectedly produced final test results")
        case = study["cases"][family]
        split = _read_json(output / case["arms"]["excluded"]["split_path"])
        grids = {arm: path / "validation_grid.csv" for arm, path in paths.items()}
        joint, pooled, summary = aggregate_joint_selection(grids, config, case["validation_model_ids"], split["val_prompt_ids"])
        joint.update({"target_family": family, "study_sha256": study["study_sha256"], "created_at": _now()})
        joint_path = output / family / "joint_selection.json"
        _write_json(joint_path, joint)
        _write_csv(output / family / "joint_validation_grid.csv", pooled)
        _write_csv(output / family / "joint_validation_summary.csv", summary)
        for arm in ARMS:
            command = build_child_command(study, output, family, arm, "validate", deadline_text, joint_path)
            execute(command, output / family / "logs" / f"{arm}_joint_relock.log", deadline)
            if (paths[arm] / "test_completed.json").exists():
                raise ValueError("Shared-selection relock unexpectedly produced final test results")
        marker = {"schema_version": 1, "created_at": _now(), "status": "awaiting_explicit_test_stage",
                  "study_sha256": study["study_sha256"], "joint_selection_sha256": sha256_file(joint_path),
                  "grid_sha256": {arm: sha256_file(path / "validation_grid.csv") for arm, path in paths.items()},
                  "test_lock_sha256": {arm: sha256_file(path / "test_lock.json") for arm, path in paths.items()},
                  "test_labels_used": False}
        _write_json(marker_path, marker)
        _verify_joint_locks(study, output, family)


def _verify_shared_probes(paths: dict[str, Path], case: dict, train_prompt_ids: set[int]) -> None:
    identities = []
    for arm in ARMS:
        rows = _read_json(paths[arm] / "test_probes.json")
        selected = {(int(row["seed"]), int(row["model_id"]), row["strategy"]): row["prompt_ids"] for row in rows}
        expected = {(seed, int(model_id), "balanced_random") for seed in SEEDS for model_id in case["test_model_ids"]}
        if len(selected) != len(rows) or set(selected) != expected:
            raise ValueError("Family test probe identities are invalid")
        if any(len(order) != max(BUDGETS) or len(set(order)) != len(order) or not set(order).issubset(train_prompt_ids)
               for order in selected.values()):
            raise ValueError("Family probes must be distinct training-pool items at the declared budget")
        identities.append(selected)
    if identities[0] != identities[1]:
        raise ValueError("Paired family arms used different static probe IDs")


def _verify_test_metrics(rows: list[dict], case: dict, benchmarks: list[str] | None = None) -> None:
    expected = set()
    predictors = list(PREDICTORS.values()) + ["domain_probe_mean", "global_probe_mean", "question_knn"]
    for seed in SEEDS:
        for model_id in case["test_model_ids"]:
            expected.add((seed, int(model_id), "zero_probe", 0, "training_domain_mean"))
            for budget in BUDGETS:
                for predictor in predictors:
                    expected.add((seed, int(model_id), "balanced_random", budget, predictor))
    if benchmarks is not None:
        expected = {(*key, benchmark) for key in expected for benchmark in benchmarks}
    seen = set()
    for row in rows:
        key = (int(row["seed"]), int(row["model_id"]), row["strategy"], int(row["budget"]), row["predictor"])
        if benchmarks is not None:
            key += (row["benchmark"],)
        if key not in expected or key in seen or row.get("phase") != "test":
            raise ValueError("Family final metrics contain duplicate or unexpected cells")
        seen.add(key)
    if seen != expected:
        raise ValueError("Family final metrics are incomplete")


def test_stage(study: dict, output: Path, families: list[str], deadline_text: str,
               execute: Callable = run_child, save_predictions: bool = False) -> None:
    # Audit all requested locks before the first explicit final-test invocation.
    for family in families:
        _verify_joint_locks(study, output, family)
    deadline = Deadline(deadline_text)
    all_model_rows, all_domain_rows = [], []
    config = _read_json(output / study["family_config_path"])
    for family in families:
        paths = _pair_paths(study, output, family)
        for arm in ARMS:
            deadline.check()
            marker_path = output / family / f"{arm}_test_completed.json"
            if marker_path.exists():
                marker = _read_json(marker_path)
                files = ("per_model_metrics.csv", "per_domain_metrics.csv", "test_probes.json", "test_completed.json")
                if marker["study_sha256"] != study["study_sha256"] or any(marker["result_sha256"].get(name) != sha256_file(paths[arm] / name) for name in files):
                    raise ValueError("Saved family final-test result changed")
            else:
                command = build_child_command(study, output, family, arm, "test", deadline_text, save_predictions=save_predictions)
                execute(command, output / family / "logs" / f"{arm}_test.log", deadline)
                _write_json(marker_path, {"study_sha256": study["study_sha256"], "created_at": _now(),
                    "result_sha256": {name: sha256_file(paths[arm] / name) for name in
                                      ("per_model_metrics.csv", "per_domain_metrics.csv", "test_probes.json", "test_completed.json")}})
            model_rows, domain_rows = _read_csv(paths[arm] / "per_model_metrics.csv"), _read_csv(paths[arm] / "per_domain_metrics.csv")
            _verify_test_metrics(model_rows, study["cases"][family])
            _verify_test_metrics(domain_rows, study["cases"][family], config["data"]["candidate_benchmarks"])
            all_model_rows.extend({"family_case": family, "condition": arm, **row} for row in model_rows)
            all_domain_rows.extend({"family_case": family, "condition": arm, **row} for row in domain_rows)
        split = _read_json(output / study["cases"][family]["arms"]["excluded"]["split_path"])
        _verify_shared_probes(paths, study["cases"][family], set(split["train_prompt_ids"]))
    _write_csv(output / "family_per_model_metrics.csv", all_model_rows)
    _write_csv(output / "family_per_domain_metrics.csv", all_domain_rows)
    _write_json(output / "test_completed.json", {"schema_version": 1, "study_sha256": study["study_sha256"],
                "finished_at": _now(), "families": families, "model_metric_rows": len(all_model_rows),
                "domain_metric_rows": len(all_domain_rows), "static_probe_ids_matched_within_pairs": True})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "family_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "family_study")
    parser.add_argument("--phase", choices=("prepare", "validate", "test"), default="prepare")
    parser.add_argument("--families", default=",".join(CASES), help="Comma-separated declared cases to execute")
    parser.add_argument("--device", help="One device for all sequential subprocesses; fixed at prepare")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--deadline", help="Explicit timezone-aware ISO stop time; required for validate/test")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(argv)
    families = args.families.split(",")
    if not families or len(set(families)) != len(families) or set(families) - set(CASES):
        parser.error("--families must contain distinct declared cases Llama,Mistral,Yi")
    if args.phase != "prepare" and args.deadline is None:
        parser.error("validate/test require an explicit --deadline")
    deadline = Deadline(args.deadline)
    deadline.check()
    config_path, output = _root_path(args.config), _root_path(args.output)
    with register_job(deadline=args.deadline, metadata={"task": "matched_family_study", "phase": args.phase, "families": families}):
        if args.phase == "prepare":
            study = prepare_study(config_path, output, deadline, device=args.device or "auto", cpu_threads=args.cpu_threads)
        else:
            study = verify_study(config_path, output, deadline)
            if args.device is not None and args.device != study["device"]:
                raise ValueError("Execution device differs from the prepared same-device plan")
            if args.cpu_threads != study["cpu_threads"]:
                raise ValueError("CPU thread setting differs from the prepared execution plan")
            if args.phase == "validate":
                validate_stage(study, output, families, args.deadline, execute=run_child)
            else:
                test_stage(study, output, families, args.deadline, execute=run_child, save_predictions=args.save_predictions)
    print(json.dumps({"phase": args.phase, "study_sha256": study["study_sha256"],
                      "output": str(output), "families": families,
                      "final_tests_automatically_started": args.phase == "test"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
