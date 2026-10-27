"""Matched-family orchestration tests use synthetic records and fake runners."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from coldstart.runtime import Deadline, DeadlineExceeded


def family_runner():
    location = Path(__file__).resolve().parents[1] / "scripts" / "run_family_study.py"
    specification = importlib.util.spec_from_file_location("family_runner_under_test", location)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def prepared_study(tmp_path):
    runner = family_runner()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "family_protocol.json").read_text(encoding="utf-8"))
    data = tmp_path / "synthetic_data"
    data.mkdir()
    model_count, question_count = 88, 1000
    np.savez_compressed(data / "embedllm.npz", strict_labels=np.zeros((model_count, question_count)),
                        embeddings=np.ones((question_count, 4)), benchmark_ids=np.repeat(np.arange(5), 200),
                        group_ids=np.arange(question_count), model_ids=np.arange(model_count), prompt_ids=np.arange(question_count))
    (data / "embedllm.json").write_text(json.dumps({"raw_audit_complete": True,
        "model_names": [f"SYNTHETIC_model_{i}" for i in range(model_count)],
        "benchmark_names": config["data"]["candidate_benchmarks"]}), encoding="utf-8")
    table = tmp_path / "families.csv"
    assignments = ["Llama"] * 36 + ["Mistral"] * 18 + ["Yi"] * 9 + ["Qwen"] * 6 + ["other"] * 19
    table.write_text("model_id,family,inclusion\n" + "".join(f"{i},{name},include\n" for i, name in enumerate(assignments)), encoding="utf-8")
    config["data"].update(prepared_path=str(data), family_table=str(table), primary_selection_manifest=None)
    config["split"]["family_stress"].pop("fixed_target_source_ids", None)
    source = tmp_path / "family_config.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "family_outputs"
    study = runner.prepare_study(source, output, Deadline("2099-01-01T00:00:00+00:00"), device="cpu", cpu_threads=1)
    return runner, config, source, output, study


def grid_rows(runner, config, arm, tie=False):
    rows = []
    for seed in runner.SEEDS:
        for budget in runner.BUDGETS:
            for variant in runner.VARIANTS:
                for value in config["adaptation"]["prior_strengths"]:
                    score = {0.3: (0.01, 0.41), 1.0: (0.20, 0.10), 3.0: (0.30, 0.01)}[float(value)][arm == "related_allowed"]
                    rows.append({"kind": "prior", "predictor": runner.PREDICTORS[variant], "variant": variant,
                                 "seed": seed, "budget": budget, "value": value, "macro_mae": 0.2 if tie else score})
            for name in ("domain_probe_mean", "global_probe_mean"):
                for value in config["baselines"]["mean_smoothing_grid"]:
                    rows.append({"kind": "baseline_smoothing", "predictor": name, "seed": seed, "budget": budget,
                                 "value": value, "macro_mae": 0.2 if tie else 0.1 + 0.001 * abs(value - 5)})
            for count in config["baselines"]["knn_neighbors_grid"]:
                for weight in config["baselines"]["knn_prior_weight_grid"]:
                    rows.append({"kind": "knn", "predictor": "question_knn", "seed": seed, "budget": budget,
                                 "neighbors": count, "prior_weight": weight,
                                 "macro_mae": 0.2 if tie else 0.1 + 0.01 * count + 0.001 * weight})
    return rows


def fake_child_executor(runner, commands):
    def execute(command, log_path, deadline):
        deadline.check()
        commands.append(command)
        value = lambda flag: command[command.index(flag) + 1]
        phase, output = value("--phase"), Path(value("--output"))
        output.mkdir(parents=True, exist_ok=True)
        config = runner._read_json(Path(value("--config")))
        split = runner._read_json(Path(value("--split")))
        arm = split["metadata"]["arm"]
        if phase == "train_validate":
            runner._write_csv(output / "validation_grid.csv", grid_rows(runner, config, arm))
            runner._write_json(output / "test_lock.json", {"selection": {}, "test_labels_used": False})
        elif phase == "validate":
            selection_path = Path(value("--selection-file"))
            selection = runner._read_json(selection_path)
            selection.update(shared_selection_source=str(selection_path.resolve()), shared_selection_sha256=runner.sha256_file(selection_path))
            runner._write_json(output / "test_lock.json", {"selection": selection, "test_labels_used": False})
        elif phase == "test":
            models, domains, probes = [], [], []
            for seed in runner.SEEDS:
                for model_id in split["test_model_ids"]:
                    probes.append({"seed": seed, "model_id": model_id, "strategy": "balanced_random", "budget": 200,
                                   "prompt_ids": split["train_prompt_ids"][:200]})
                    identities = [("zero_probe", 0, "training_domain_mean")]
                    identities += [("balanced_random", budget, predictor) for budget in runner.BUDGETS
                                   for predictor in list(runner.PREDICTORS.values()) + ["domain_probe_mean", "global_probe_mean", "question_knn"]]
                    for strategy, budget, predictor in identities:
                        row = {"phase": "test", "seed": seed, "model_id": model_id, "strategy": strategy,
                               "budget": budget, "predictor": predictor, "macro_mae": 0.2}
                        models.append(row)
                        domains.extend({**row, "benchmark": domain} for domain in config["data"]["candidate_benchmarks"])
            runner._write_csv(output / "per_model_metrics.csv", models)
            runner._write_csv(output / "per_domain_metrics.csv", domains)
            runner._write_json(output / "test_probes.json", probes)
            runner._write_json(output / "test_completed.json", {"phase": "test", "synthetic": True})
        else:
            raise AssertionError(phase)
        return {"returncode": 0}
    return execute


def test_prepare_fixes_targets_counts_and_items_and_rejects_changed_inputs(prepared_study):
    runner, config, source, output, study = prepared_study
    for family, (test_count, relatives) in runner.CASES.items():
        case = study["cases"][family]
        left, right = [runner._read_json(output / case["arms"][arm]["split_path"]) for arm in runner.ARMS]
        assert len(left["train_model_ids"]) == len(right["train_model_ids"]) == 40
        assert len(left["test_model_ids"]) == test_count
        assert len(left["val_model_ids"]) == 6
        assert left["metadata"]["related_count"] == relatives
        assert left["test_model_ids"] == right["test_model_ids"]
        assert left["val_model_ids"] == right["val_model_ids"]
        assert left["test_prompt_ids"] == right["test_prompt_ids"]
    assert runner.prepare_study(source, output, Deadline(), device="cpu", cpu_threads=1)["study_sha256"] == study["study_sha256"]
    config["model"]["epochs"] += 1
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source config changed"):
        runner.verify_study(source, output, Deadline())


def test_orchestrator_hash_normalizes_line_endings_and_is_bound_to_study(prepared_study, tmp_path, monkeypatch):
    runner, _, source, output, study = prepared_study
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"value = 1\n")
    crlf.write_bytes(b"value = 1\r\n")
    assert runner._python_source_sha256(lf) == runner._python_source_sha256(crlf)
    assert study["orchestrator_source_hash_policy"] == "utf8_lf"
    assert len(study["orchestrator_source_sha256"]) == 64
    monkeypatch.setattr(runner, "_python_source_sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="orchestrator source changed"):
        runner.verify_study(source, output, Deadline())


def test_joint_selection_pools_both_arms_and_rejects_partial_or_duplicate_cells(prepared_study):
    runner, config, _, output, _ = prepared_study
    paths = {arm: output / f"grid_{arm}.csv" for arm in runner.ARMS}
    for arm, path in paths.items():
        runner._write_csv(path, grid_rows(runner, config, arm))
    selection, pooled, summary = runner.aggregate_joint_selection(paths, config, [63, 64], [1, 2])
    assert selection["prior_strengths"] == {"structured8": 1.0, "rank0": 1.0}
    assert selection["mean_smoothing"] == {"domain_probe_mean": 5.0, "global_probe_mean": 5.0}
    assert selection["knn"]["n_neighbors"] == 5
    assert len(pooled) == 450 and all(row["cells"] == 18 for row in summary)
    rows = runner._read_csv(paths["excluded"])
    runner._write_csv(paths["excluded"], rows[:-1])
    with pytest.raises(ValueError, match="Incomplete paired"):
        runner.aggregate_joint_selection(paths, config, [], [])
    runner._write_csv(paths["excluded"], rows + [rows[0]])
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        runner.aggregate_joint_selection(paths, config, [], [])


def test_joint_exact_ties_choose_predeclared_more_shrinkage_then_smaller_knn(prepared_study):
    runner, config, _, output, _ = prepared_study
    paths = {arm: output / f"tie_{arm}.csv" for arm in runner.ARMS}
    for arm, path in paths.items():
        runner._write_csv(path, grid_rows(runner, config, arm, tie=True))
    selection, _, _ = runner.aggregate_joint_selection(paths, config, [], [])
    assert selection["prior_strengths"] == {"structured8": 3.0, "rank0": 3.0}
    assert selection["mean_smoothing"]["domain_probe_mean"] == 20.0
    assert selection["knn"]["n_neighbors"] == 5 and selection["knn"]["prior_weight"] == 3.0


def test_validate_stops_before_test_and_explicit_test_uses_shared_lock(prepared_study):
    runner, _, _, output, study = prepared_study
    commands = []
    execute = fake_child_executor(runner, commands)
    with pytest.raises(FileNotFoundError):
        runner.test_stage(study, output, ["Llama"], "2099-01-01T00:00:00+00:00", execute=execute)
    assert commands == []
    runner.validate_stage(study, output, ["Llama"], "2099-01-01T00:00:00+00:00", execute=execute)
    assert [command[command.index("--phase") + 1] for command in commands] == ["train_validate", "train_validate", "validate", "validate"]
    assert not list(output.rglob("test_completed.json"))
    assert runner._read_json(output / "Llama" / "validation_completed.json")["test_labels_used"] is False
    runner.test_stage(study, output, ["Llama"], "2099-01-01T00:00:00+00:00", execute=execute)
    assert [command[command.index("--phase") + 1] for command in commands[-2:]] == ["test", "test"]
    assert runner._read_json(output / "test_completed.json")["static_probe_ids_matched_within_pairs"] is True
    with pytest.raises(ValueError, match="already contain final tests"):
        runner.validate_stage(study, output, ["Llama"], "2099-01-01T00:00:00+00:00", execute=execute)


def test_changed_shared_selection_or_split_is_rejected_before_test(prepared_study):
    runner, _, source, output, study = prepared_study
    commands = []
    runner.validate_stage(study, output, ["Yi"], "2099-01-01T00:00:00+00:00", execute=fake_child_executor(runner, commands))
    selection_path = output / "Yi" / "joint_selection.json"
    selection = runner._read_json(selection_path)
    selection["prior_strengths"]["structured8"] = 0.3
    runner._write_json(selection_path, selection)
    count = len(commands)
    with pytest.raises(ValueError, match="changed after paired validation"):
        runner.test_stage(study, output, ["Yi"], "2099-01-01T00:00:00+00:00", execute=fake_child_executor(runner, commands))
    assert len(commands) == count
    path = output / study["cases"]["Yi"]["arms"]["excluded"]["split_path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="split changed"):
        runner.verify_study(source, output, Deadline())


def test_child_is_terminated_on_deadline(tmp_path):
    runner = family_runner()
    deadline = Deadline(datetime.now(timezone.utc) + timedelta(seconds=0.15))
    with pytest.raises(DeadlineExceeded):
        runner.run_child([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "deadline.log", deadline)
