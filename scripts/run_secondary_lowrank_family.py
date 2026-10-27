"""Exploratory family coverage for the frozen eight-dimensional predictor.

This separate study reuses the completed family study's exact split bytes and
baseline choices. Its scope was finalized after the main final table was seen.
Only the new predictor's paired Qwen validation scores select its prior strength.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from coldstart.data import sha256_file
from coldstart.metrics import benchmark_metrics
from coldstart.probes import select_probe_order
from coldstart.runtime import Deadline, register_job
from coldstart.splits import dataset_fingerprint, load_split
from run_experiment import infer_probabilities, load_features, scientific_source_hashes
import run_family_study as control

VARIANT = "neural_lowrank"
PREDICTOR = "coldstart_neural_lowrank"
STRENGTHS = (0.3, 1.0, 3.0)
SEEDS, BUDGETS, ARMS = control.SEEDS, control.BUDGETS, control.ARMS


def _hash_study(study: dict) -> str:
    return control._hash_json({key: value for key, value in study.items() if key != "study_sha256"})


def validate_config(config: dict) -> None:
    required = {"hidden_dim": 128, "learning_rate": 0.002, "latent_l2": 0.002, "mode": "lowrank",
                "rank": 8, "warmup_epochs": 0, "selection_budget": 100, "selection_interval": 2}
    if any(config["model"].get(key) != value for key, value in required.items()):
        raise ValueError("Supplement uses only the frozen original family architecture")
    if config["experiment"]["regime"] != "strict_unseen_item" or config["data"]["label_policy"] != "strict" or config["data"].get("deduplicate_groups") is not True:
        raise ValueError("Supplement must preserve the original strict information protocol")
    if tuple(config["model"]["selection_prior_strengths"]) != STRENGTHS or tuple(config["adaptation"]["prior_strengths"]) != STRENGTHS:
        raise ValueError("Supplement checkpoint and final prior grids must remain the original three values")
    if tuple(config["seeds"]) != SEEDS or tuple(config["budgets"]) != BUDGETS or config["probe_strategies"] != ["balanced_random"]:
        raise ValueError("Supplement must use the original seeds, budgets, and static policy")
    if config["experiment"]["variants"] != [VARIANT] or not config["experiment"].get("secondary_exploratory"):
        raise ValueError("Supplement must explicitly be exploratory and lowrank-only")
    if config["baselines"].get("scalar_logit_shift", {}).get("enabled", False) or config["diagnostics"]["within_benchmark_response_permutation"]["enabled"]:
        raise ValueError("Supplement must not add scalar or permutation experiments")


def prepare(output: Path, control_output: Path, main_output: Path, deadline: Deadline) -> dict:
    output, control_output, main_output = output.resolve(), control_output.resolve(), main_output.resolve()
    if (output / "study.json").exists():
        return verify(output, deadline)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing a nonempty supplemental output without a manifest")
    original = control._read_json(control_output / "study.json")
    original = control.verify_study(Path(original["source_config_path"]), control_output, deadline)
    if not (control_output / "test_completed.json").exists() or not (main_output / "test_completed.json").exists():
        raise ValueError("This exploratory scope must record that the original final studies are already complete")
    base = control._read_json(control_output / original["family_config_path"])
    dataset, _, _, data_sha = control.load_inputs(base, deadline)
    main_split = control._read_json(main_output / "split.json")
    targets = set(value for case in original["cases"].values() for value in case["test_model_ids"])
    overlaps = {name: sorted(targets.intersection(main_split[f"{name}_model_ids"])) for name in ("train", "val", "test")}
    config = deepcopy(base)
    config["experiment"].update(name="embedllm_secondary_lowrank_family_coverage", variants=[VARIANT],
        secondary_exploratory=True, status="scope_frozen_after_main_final_inspection")
    config["model"]["mode"] = "lowrank"
    inherited_origin = config.pop("family_hyperparameter_origin", {})
    inherited_origin["applies_to"] = "Historical original-family numeric defaults only; this record does not describe creation or variant selection of the supplemental study."
    config["inherited_original_family_numeric_defaults_origin"] = inherited_origin
    config["secondary_scope"] = {
        "created_at_utc": control._now(),
        "scope_finalized_after_main_final_table_inspection": True,
        "timing_note": "Coverage was proposed before main-table inspection; full scope was frozen after the root inspected the main final table around 04:33 Taipei.",
        "variant_choice_used_main_validation": True, "main_partition_overlap_with_family_targets": overlaps,
        "variant_selection_is_not_fully_family_target_blind": bool(overlaps["val"]),
        "architecture_and_checkpoint_rule": "Original fixed family settings, no new architecture search",
        "new_prior_selection": "Qwen validation only; one strength shared across each paired condition",
        "baseline_choices": "Reused unchanged from the original completed family joint validation, bound by hash",
        "pooled_outputs": "Only coldstart_neural_lowrank; baseline rows are not duplicated",
        "interpretation": "Secondary exploratory coverage; separate from the original confirmatory tables.",
    }
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    control._write_json(output / "config.json", config)
    cases = {}
    for family, case in original["cases"].items():
        control._verify_joint_locks(original, control_output, family)
        copied = deepcopy(case)
        for arm, record in copied["arms"].items():
            source = control_output / record["split_path"]
            destination = output / record["split_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            if sha256_file(destination) != record["split_file_sha256"]:
                raise ValueError("Supplement must reuse the exact original split bytes")
        copied["baseline_selection_path"] = str(control_output / family / "joint_selection.json")
        copied["baseline_selection_sha256"] = sha256_file(Path(copied["baseline_selection_path"]))
        cases[family] = copied
    study = {"schema_version": 1, "created_at": control._now(), "secondary_exploratory": True,
        "output_root": str(output), "config_file_sha256": sha256_file(output / "config.json"),
        "orchestrator_source_path": str(Path(__file__).resolve()),
        "orchestrator_source_sha256": control._python_source_sha256(Path(__file__).resolve()),
        "scientific_source_sha256": scientific_source_hashes(),
        "source_hash_policy": "utf8_lf", "control_output": str(control_output),
        "control_study_sha256": original["study_sha256"], "control_manifest_file_sha256": sha256_file(control_output / "study.json"),
        "control_test_marker_sha256": sha256_file(control_output / "test_completed.json"),
        "main_output": str(main_output), "main_lock_sha256": sha256_file(main_output / "test_lock.json"),
        "main_split_sha256": sha256_file(main_output / "split.json"),
        "main_test_marker_sha256": sha256_file(main_output / "test_completed.json"),
        "dataset_fingerprint": dataset_fingerprint(dataset), "prepared_data_sha256": data_sha,
        "device": "cuda", "cpu_threads": 1, "variant": VARIANT, "seeds": list(SEEDS), "budgets": list(BUDGETS),
        "strengths": list(STRENGTHS), "paired_prior_cells_per_case": 54,
        "baseline_rows_in_pooled_outputs": False, "scope": config["secondary_scope"], "cases": cases}
    study["study_sha256"] = _hash_study(study)
    control._write_json(output / "study.json", study)
    return study


def verify(output: Path, deadline: Deadline) -> dict:
    output = output.resolve()
    study = control._read_json(output / "study.json")
    if study.get("study_sha256") != _hash_study(study) or study["output_root"] != str(output):
        raise ValueError("Supplement manifest changed")
    if study["orchestrator_source_sha256"] != control._python_source_sha256(Path(__file__).resolve()):
        raise ValueError("Supplement orchestrator source changed")
    if study["scientific_source_sha256"] != scientific_source_hashes():
        raise ValueError("Frozen scientific source changed")
    if study["config_file_sha256"] != sha256_file(output / "config.json"):
        raise ValueError("Supplement configuration changed")
    config = control._read_json(output / "config.json")
    validate_config(config)
    original_output = Path(study["control_output"])
    if study["control_manifest_file_sha256"] != sha256_file(original_output / "study.json") or study["control_test_marker_sha256"] != sha256_file(original_output / "test_completed.json"):
        raise ValueError("Original family study changed after the supplement was prepared")
    original = control._read_json(original_output / "study.json")
    control.verify_study(Path(original["source_config_path"]), original_output, deadline)
    dataset, _, _, data_sha = control.load_inputs(config, deadline)
    if data_sha != study["prepared_data_sha256"] or dataset_fingerprint(dataset) != study["dataset_fingerprint"]:
        raise ValueError("Supplement data changed")
    main_output = Path(study["main_output"])
    for key, name in (("main_lock_sha256", "test_lock.json"), ("main_split_sha256", "split.json"), ("main_test_marker_sha256", "test_completed.json")):
        if study[key] != sha256_file(main_output / name):
            raise ValueError("Bound original main artifact changed")
    for family, case in study["cases"].items():
        control._verify_joint_locks(original, original_output, family)
        if case["baseline_selection_sha256"] != sha256_file(Path(case["baseline_selection_path"])):
            raise ValueError("Reused baseline choices changed")
        for arm, record in case["arms"].items():
            if record["split_file_sha256"] != sha256_file(output / record["split_path"]):
                raise ValueError("Supplement split bytes changed")
            load_split(output / record["split_path"], dataset)
    return study


def child_command(study: dict, output: Path, family: str, arm: str, phase: str,
                  deadline_text: str, selection_file: Path | None = None) -> list[str]:
    record = study["cases"][family]["arms"][arm]
    command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config", str(output / "config.json"),
        "--output", str(output / record["run_path"]), "--split", str(output / record["split_path"]),
        "--phase", phase, "--variants", VARIANT, "--device", study["device"], "--cpu-threads", "1", "--deadline", deadline_text]
    if phase == "train":
        command.append("--resume")
    if selection_file is not None:
        command.extend(["--selection-file", str(selection_file)])
    if phase == "test":
        command.append("--save-predictions")
    return command


def evaluate_prior_grid(dataset, split, config: dict, arm_output: Path, deadline: Deadline, device: str) -> list[dict]:
    rows = []
    target = split.val_items
    truth = dataset.labels[np.ix_(split.val_models, target)]
    for seed in SEEDS:
        feature = load_features(dataset, config, arm_output, [VARIANT], seed, split.probe_items, device)[VARIANT]
        order = select_probe_order(dataset.embeddings, dataset.benchmark_ids, split.probe_items,
                                   max(BUDGETS), "balanced_random", seed, check_deadline=deadline.check)
        for budget in BUDGETS:
            selected = order[:budget]
            responses = dataset.labels[np.ix_(split.val_models, selected)]
            for strength in STRENGTHS:
                probabilities = np.stack([infer_probabilities(feature, selected, response, target,
                    strength, config, deadline)[0] for response in responses])
                score = benchmark_metrics(truth, probabilities, dataset.benchmark_ids[target])["summary"]["macro_mae"]
                rows.append({"kind": "prior", "variant": VARIANT, "predictor": PREDICTOR,
                    "seed": seed, "budget": budget, "value": strength, "macro_mae": float(score)})
    return rows


def joint_selection(grids: dict[str, list[dict]], baseline: dict) -> tuple[dict, list[dict]]:
    expected = {(arm, seed, budget, strength) for arm in ARMS for seed in SEEDS for budget in BUDGETS for strength in STRENGTHS}
    seen, scores, pooled = set(), {strength: [] for strength in STRENGTHS}, []
    if set(grids) != set(ARMS):
        raise ValueError("Supplement requires both paired prior grids")
    for arm, rows in grids.items():
        for row in rows:
            key = (arm, int(row["seed"]), int(row["budget"]), float(row["value"]))
            score = float(row["macro_mae"])
            if key not in expected or key in seen or row.get("kind") != "prior" or row.get("variant") != VARIANT or row.get("predictor") != PREDICTOR or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Unexpected or duplicate supplemental prior cell")
            seen.add(key)
            scores[key[-1]].append(score)
            pooled.append({"arm": arm, **row})
    if seen != expected:
        raise ValueError("Incomplete supplemental prior grid; exactly 54 cells are required")
    means = {value: float(np.mean(values)) for value, values in scores.items()}
    strength = min(STRENGTHS, key=lambda value: (means[value], -value))
    result = {"prior_strengths": {VARIANT: strength},
        "mean_smoothing": deepcopy(baseline["mean_smoothing"]), "knn": deepcopy(baseline["knn"]),
        "scalar_logit_shift": {"enabled": False}, "validation_summary": {"paired_prior_means": means},
        "reference_strategy": "balanced_random", "reference_budgets": list(BUDGETS),
        "validation_model_ids": baseline["validation_model_ids"], "validation_prompt_ids": baseline["validation_prompt_ids"],
        "test_labels_used": False, "secondary_exploratory": True,
        "selection_unit": "equal mean of both arms x three seeds x three budgets, Qwen validation only"}
    return result, pooled


def verify_case(study: dict, output: Path, family: str) -> dict:
    marker = control._read_json(output / family / "validation_completed.json")
    joint_path = output / family / "joint_selection.json"
    if marker["study_sha256"] != study["study_sha256"] or marker["joint_sha256"] != sha256_file(joint_path):
        raise ValueError("Supplement joint selection changed after validation")
    joint = control._read_json(joint_path)
    if marker.get("test_labels_used") is not False or joint.get("test_labels_used") is not False or set(joint["prior_strengths"]) != {VARIANT} or joint["prior_strengths"][VARIANT] not in STRENGTHS:
        raise ValueError("Supplement requires a validation-only lowrank prior selection")
    if joint.get("reused_baseline_selection_sha256") != study["cases"][family]["baseline_selection_sha256"]:
        raise ValueError("Supplement baseline reuse is not bound to the original choice file")
    baseline = control._read_json(Path(study["cases"][family]["baseline_selection_path"]))
    if joint["mean_smoothing"] != baseline["mean_smoothing"] or joint["knn"] != baseline["knn"]:
        raise ValueError("Supplement must reuse the original baseline choices")
    for arm, record in study["cases"][family]["arms"].items():
        path = output / record["run_path"]
        if marker["prior_grid_sha256"][arm] != sha256_file(path / "prior_validation_grid.csv") or marker["lock_sha256"][arm] != sha256_file(path / "test_lock.json"):
            raise ValueError("Supplement grid or test lock changed")
        lock = control._read_json(path / "test_lock.json")
        if lock["selection"].get("shared_selection_sha256") != marker["joint_sha256"] or any(lock["selection"][key] != joint[key] for key in ("prior_strengths", "mean_smoothing", "knn")):
            raise ValueError("Supplement paired arms do not share the selected settings")
    return marker


def validate(study: dict, output: Path, deadline_text: str, execute=control.run_child,
             grid_evaluator=evaluate_prior_grid) -> None:
    deadline = Deadline(deadline_text)
    config = control._read_json(output / "config.json")
    dataset, _, _, _ = control.load_inputs(config, deadline)
    for family, case in study["cases"].items():
        if any((output / record["run_path"] / "test_completed.json").exists() for record in case["arms"].values()):
            raise ValueError("Supplement already has final tests; validation cannot be rerun")
        if (output / family / "validation_completed.json").exists():
            verify_case(study, output, family)
            continue
        grids = {}
        for arm, record in case["arms"].items():
            execute(child_command(study, output, family, arm, "train", deadline_text),
                    output / family / "logs" / f"{arm}_train.log", deadline)
            split = load_split(output / record["split_path"], dataset)
            grids[arm] = grid_evaluator(dataset, split, config, output / record["run_path"], deadline, study["device"])
            control._write_csv(output / record["run_path"] / "prior_validation_grid.csv", grids[arm])
        baseline = control._read_json(Path(case["baseline_selection_path"]))
        selected, pooled = joint_selection(grids, baseline)
        selected.update(target_family=family, study_sha256=study["study_sha256"],
            reused_baseline_selection_sha256=case["baseline_selection_sha256"])
        joint_path = output / family / "joint_selection.json"
        control._write_json(joint_path, selected)
        control._write_csv(output / family / "joint_prior_grid.csv", pooled)
        for arm in ARMS:
            execute(child_command(study, output, family, arm, "validate", deadline_text, joint_path),
                    output / family / "logs" / f"{arm}_relock.log", deadline)
        marker = {"study_sha256": study["study_sha256"], "status": "awaiting_explicit_secondary_test",
            "test_labels_used": False, "joint_sha256": sha256_file(joint_path),
            "prior_grid_sha256": {arm: sha256_file(output / case["arms"][arm]["run_path"] / "prior_validation_grid.csv") for arm in ARMS},
            "lock_sha256": {arm: sha256_file(output / case["arms"][arm]["run_path"] / "test_lock.json") for arm in ARMS}}
        control._write_json(output / family / "validation_completed.json", marker)
        verify_case(study, output, family)


def test(study: dict, output: Path, deadline_text: str, execute=control.run_child) -> None:
    for family in study["cases"]:
        verify_case(study, output, family)
    deadline = Deadline(deadline_text)
    all_models, all_domains = [], []
    config = control._read_json(output / "config.json")
    for family, case in study["cases"].items():
        paths = {}
        for arm, record in case["arms"].items():
            path = paths[arm] = output / record["run_path"]
            execute(child_command(study, output, family, arm, "test", deadline_text),
                    output / family / "logs" / f"{arm}_test.log", deadline)
            for filename, destination, by_domain in (("per_model_metrics.csv", all_models, False), ("per_domain_metrics.csv", all_domains, True)):
                rows = [row for row in control._read_csv(path / filename) if row["predictor"] == PREDICTOR]
                expected = {(seed, mid, budget) for seed in SEEDS for mid in case["test_model_ids"] for budget in BUDGETS}
                if by_domain:
                    expected = {(*key, domain) for key in expected for domain in config["data"]["candidate_benchmarks"]}
                seen = set()
                for row in rows:
                    key = (int(row["seed"]), int(row["model_id"]), int(row["budget"]))
                    if by_domain:
                        key += (row["benchmark"],)
                    if key not in expected or key in seen or row["strategy"] != "balanced_random" or row["phase"] != "test":
                        raise ValueError("Supplement contains unexpected or duplicate neural metric cells")
                    seen.add(key)
                    destination.append({"secondary_exploratory": True, "family_case": family, "condition": arm, **row})
                if seen != expected:
                    raise ValueError("Supplement final neural metrics are incomplete")
        split = control._read_json(output / case["arms"]["excluded"]["split_path"])
        control._verify_shared_probes(paths, case, set(split["train_prompt_ids"]))
        original_output = Path(study["control_output"])
        for arm in ARMS:
            current = control._read_json(paths[arm] / "test_probes.json")
            original = control._read_json(original_output / case["arms"][arm]["run_path"] / "test_probes.json")
            key = lambda row: (int(row["seed"]), int(row["model_id"]), row["strategy"])
            if {key(row): row["prompt_ids"] for row in current} != {key(row): row["prompt_ids"] for row in original}:
                raise ValueError("Supplement probes differ from the original frozen study")
    control._write_csv(output / "secondary_lowrank_per_model_metrics.csv", all_models)
    control._write_csv(output / "secondary_lowrank_per_domain_metrics.csv", all_domains)
    control._write_json(output / "test_completed.json", {"study_sha256": study["study_sha256"],
        "secondary_exploratory": True, "finished_at": control._now(), "metric_rows": len(all_models),
        "domain_rows": len(all_domains), "baseline_rows_pooled": False, "probes_match_original_control": True})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "validate", "test"), default="prepare")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/family_lowrank_secondary")
    parser.add_argument("--control", type=Path, default=ROOT / "artifacts/family_study_v4")
    parser.add_argument("--main", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--deadline", help="Explicit timezone-aware deadline, required for validation/testing")
    args = parser.parse_args(argv)
    if args.phase != "prepare" and not args.deadline:
        parser.error("validate/test require --deadline")
    deadline = Deadline(args.deadline)
    deadline.check()
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    output = control._root_path(args.output)
    with register_job(deadline=args.deadline, metadata={"task": "secondary_exploratory_lowrank_family", "phase": args.phase}):
        study = prepare(output, control._root_path(args.control), control._root_path(args.main), deadline) if args.phase == "prepare" else verify(output, deadline)
        if args.phase == "validate":
            validate(study, output, args.deadline)
        elif args.phase == "test":
            test(study, output, args.deadline)
    print(json.dumps({"phase": args.phase, "secondary_exploratory": True,
        "study_sha256": study["study_sha256"], "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
