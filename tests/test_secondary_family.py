"""Supplement orchestration uses synthetic control artifacts and fake models."""
import importlib.util
import json
from pathlib import Path

import pytest

from coldstart.runtime import Deadline
from test_family_study import prepared_study, fake_child_executor


def secondary_runner():
    source = Path(__file__).resolve().parents[1] / "scripts/run_secondary_lowrank_family.py"
    specification = importlib.util.spec_from_file_location("secondary_runner_under_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def secondary_study(prepared_study, tmp_path):
    original, _, _, control_output, original_study = prepared_study
    commands = []
    execute = fake_child_executor(original, commands)
    families = list(original.CASES)
    original.validate_stage(original_study, control_output, families, "2099-01-01T00:00:00+00:00", execute=execute)
    original.test_stage(original_study, control_output, families, "2099-01-01T00:00:00+00:00", execute=execute)
    main_output = tmp_path / "synthetic_main"
    main_output.mkdir()
    target_ids = [value for case in original_study["cases"].values() for value in case["test_model_ids"]]
    original._write_json(main_output / "split.json", {"train_model_ids": target_ids[2:],
        "val_model_ids": target_ids[:2], "test_model_ids": []})
    original._write_json(main_output / "test_lock.json", {"synthetic": True})
    original._write_json(main_output / "test_completed.json", {"synthetic": True})
    runner = secondary_runner()
    output = tmp_path / "secondary"
    study = runner.prepare(output, control_output, main_output, Deadline())
    return runner, output, study


def fake_grid(dataset, split, config, arm_output, deadline, device):
    values = {0.3: (0.01, 0.41), 1.0: (0.20, 0.10), 3.0: (0.30, 0.01)}
    related = split.metadata["arm"] == "related_allowed"
    return [{"kind": "prior", "variant": "neural_lowrank", "predictor": "coldstart_neural_lowrank",
             "seed": seed, "budget": budget, "value": strength, "macro_mae": values[strength][related]}
            for seed in (11, 23, 37) for budget in (50, 100, 200) for strength in (0.3, 1.0, 3.0)]


def fake_executor(runner, commands):
    def execute(command, log_path, deadline):
        commands.append(command)
        value = lambda name: command[command.index(name) + 1]
        output = Path(value("--output"))
        output.mkdir(parents=True, exist_ok=True)
        phase = value("--phase")
        split = runner.control._read_json(Path(value("--split")))
        config = runner.control._read_json(Path(value("--config")))
        if phase == "validate":
            selected_path = Path(value("--selection-file"))
            selected = runner.control._read_json(selected_path)
            selected["shared_selection_sha256"] = runner.sha256_file(selected_path)
            runner.control._write_json(output / "test_lock.json", {"selection": selected, "test_labels_used": False})
        elif phase == "test":
            models, domains, probes = [], [], []
            for seed in runner.SEEDS:
                for model_id in split["test_model_ids"]:
                    probes.append({"seed": seed, "model_id": model_id, "strategy": "balanced_random", "budget": 200,
                                   "prompt_ids": split["train_prompt_ids"][:200]})
                    for budget in runner.BUDGETS:
                        row = {"seed": seed, "model_id": model_id, "budget": budget, "predictor": runner.PREDICTOR,
                               "strategy": "balanced_random", "phase": "test", "macro_mae": 0.2}
                        models.append(row)
                        domains.extend({**row, "benchmark": domain} for domain in config["data"]["candidate_benchmarks"])
                    models.append({"seed": seed, "model_id": model_id, "budget": 0, "predictor": "training_domain_mean",
                                   "strategy": "zero_probe", "phase": "test", "macro_mae": 0.2})
            runner.control._write_csv(output / "per_model_metrics.csv", models)
            runner.control._write_csv(output / "per_domain_metrics.csv", domains)
            runner.control._write_json(output / "test_probes.json", probes)
            runner.control._write_json(output / "test_completed.json", {"synthetic": True})
        elif phase != "train":
            raise AssertionError(phase)
        return {"returncode": 0}
    return execute


def test_secondary_preparation_reuses_split_bytes_and_records_nonblind_scope(secondary_study):
    runner, output, study = secondary_study
    assert study["secondary_exploratory"] and study["paired_prior_cells_per_case"] == 54
    scope = study["scope"]
    assert scope["scope_finalized_after_main_final_table_inspection"]
    assert scope["variant_selection_is_not_fully_family_target_blind"]
    config = runner.control._read_json(output / "config.json")
    assert "family_hyperparameter_origin" not in config
    assert "Historical original-family numeric defaults only" in config["inherited_original_family_numeric_defaults_origin"]["applies_to"]
    assert scope["created_at_utc"]
    assert config["experiment"]["variants"] == ["neural_lowrank"]
    assert config["model"]["learning_rate"] == 0.002
    for case in study["cases"].values():
        for record in case["arms"].values():
            original = Path(study["control_output"]) / record["split_path"]
            assert (output / record["split_path"]).read_bytes() == original.read_bytes()
    assert runner.verify(output, Deadline())["study_sha256"] == study["study_sha256"]


def test_secondary_prior_selection_requires_all54_cells_and_reuses_baseline_choices(secondary_study):
    runner, output, study = secondary_study
    dataset, _, _, _ = runner.control.load_inputs(runner.control._read_json(output / "config.json"), Deadline())
    case = study["cases"]["Llama"]
    grids = {arm: fake_grid(dataset, runner.load_split(output / case["arms"][arm]["split_path"], dataset), {}, output, Deadline(), "cpu")
             for arm in runner.ARMS}
    baseline = runner.control._read_json(Path(case["baseline_selection_path"]))
    selected, pooled = runner.joint_selection(grids, baseline)
    assert len(pooled) == 54 and selected["prior_strengths"] == {"neural_lowrank": 1.0}
    assert selected["mean_smoothing"] == baseline["mean_smoothing"] and selected["knn"] == baseline["knn"]
    grids["excluded"].pop()
    with pytest.raises(ValueError, match="exactly 54"):
        runner.joint_selection(grids, baseline)


def test_secondary_validation_never_tests_and_pooled_final_rows_exclude_baselines(secondary_study):
    runner, output, study = secondary_study
    commands = []
    execute = fake_executor(runner, commands)
    with pytest.raises(FileNotFoundError):
        runner.test(study, output, "2099-01-01T00:00:00+00:00", execute=execute)
    assert commands == []
    runner.validate(study, output, "2099-01-01T00:00:00+00:00", execute=execute, grid_evaluator=fake_grid)
    assert set(command[command.index("--phase") + 1] for command in commands) == {"train", "validate"}
    assert not list(output.rglob("test_completed.json"))
    runner.test(study, output, "2099-01-01T00:00:00+00:00", execute=execute)
    rows = runner.control._read_csv(output / "secondary_lowrank_per_model_metrics.csv")
    assert len(rows) == 306 and {row["predictor"] for row in rows} == {"coldstart_neural_lowrank"}
    marker = runner.control._read_json(output / "test_completed.json")
    assert marker["baseline_rows_pooled"] is False and marker["probes_match_original_control"] is True


def test_secondary_rejects_changed_joint_selection_and_bound_control_baselines(secondary_study):
    runner, output, study = secondary_study
    commands = []
    execute = fake_executor(runner, commands)
    runner.validate(study, output, "2099-01-01T00:00:00+00:00", execute=execute, grid_evaluator=fake_grid)
    joint_path = output / "Yi/joint_selection.json"
    joint = runner.control._read_json(joint_path)
    joint["prior_strengths"]["neural_lowrank"] = 0.3
    runner.control._write_json(joint_path, joint)
    count = len(commands)
    with pytest.raises(ValueError, match="joint selection changed"):
        runner.test(study, output, "2099-01-01T00:00:00+00:00", execute=execute)
    assert len(commands) == count
    baseline_path = Path(study["cases"]["Llama"]["baseline_selection_path"])
    baseline = runner.control._read_json(baseline_path)
    baseline["mean_smoothing"]["domain_probe_mean"] = 0.5
    runner.control._write_json(baseline_path, baseline)
    with pytest.raises(ValueError, match="selection changed"):
        runner.verify(output, Deadline())
