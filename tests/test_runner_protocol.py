"""Pipeline tests use synthetic labels only; these are not research results."""

import importlib.util
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from coldstart.adaptation import sigmoid


def _runner():
    location = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    specification = importlib.util.spec_from_file_location("runner_under_test", location)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_python_source_hash_normalizes_lf_and_crlf_but_detects_code_changes(tmp_path):
    runner = _runner()
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes("# 可重現\nvalue = 1\n".encode("utf-8"))
    crlf.write_bytes("# 可重現\r\nvalue = 1\r\n".encode("utf-8"))
    assert runner.python_source_sha256(lf) == runner.python_source_sha256(crlf)
    crlf.write_bytes("# 可重現\r\nvalue = 2\r\n".encode("utf-8"))
    assert runner.python_source_sha256(lf) != runner.python_source_sha256(crlf)


@pytest.mark.parametrize("selector_variant", ["structured8", "neural_lowrank"])
def test_synthetic_train_validate_lock_test_pipeline(tmp_path, monkeypatch, selector_variant):
    runner = _runner()
    rng = np.random.default_rng(2026)
    model_count, question_count = 12, 100
    x = rng.normal(size=(question_count, 5)).astype(np.float32)
    domains = np.arange(question_count) % 2
    logits = rng.normal(scale=0.8, size=(model_count, 1)) + 0.4 * x[:, 0]
    labels = rng.binomial(1, sigmoid(logits)).astype(np.float32)
    data = tmp_path / "synthetic_data"
    data.mkdir()
    np.savez_compressed(data / "embedllm.npz", strict_labels=labels, embeddings=x,
                        benchmark_ids=domains, group_ids=np.arange(question_count),
                        model_ids=np.arange(model_count), prompt_ids=np.arange(question_count))
    (data / "embedllm.json").write_text(json.dumps({"raw_audit_complete": True,
        "model_names": [f"SYNTHETIC_model_{i}" for i in range(model_count)],
        "benchmark_names": ["MMLU", "MathQA"], "source": "synthetic_smoke_only"}), encoding="utf-8")
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "protocol.json").read_text(encoding="utf-8"))
    config["experiment"]["name"] = "SYNTHETIC_PIPELINE_SMOKE"
    config["data"]["prepared_path"] = str(data)
    config["data"]["model_scope"] = "synthetic_smoke_only"
    config["data"]["candidate_benchmarks"] = ["MMLU", "MathQA"]
    config["budgets"] = [6, 12]
    config["seeds"] = [7]
    config["probe_strategies"] = ["random", "balanced_random", "fisher", "balanced_fisher"]
    config["probes"]["fisher_warmup"] = 4
    config["probes"].setdefault("fisher", {})["selector_variant"] = selector_variant
    config["model"].update(epochs=2, hidden_dim=8, batch_questions=32, patience=2, warmup_epochs=0)
    config["adaptation"]["prior_strengths"] = [0.3, 1]
    config["tuning"]["reference_budgets"] = [6, 12]
    config["baselines"] = {"mean_smoothing_grid": [2], "knn_neighbors_grid": [3], "knn_prior_weight_grid": [1],
        "scalar_logit_shift": {"enabled": True, "prior_strengths": [0.3, 1.0], "variance_floor": 0.2, "training_ridge": 1e-6}}
    config_path = tmp_path / "smoke_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "synthetic_results"
    original_fit = runner.fit_decoder
    training_labels = []
    original_scalar_fit = runner.ScalarLogitShift.fit
    scalar_training_inputs = []

    def checked_fit(decoder, embeddings, benchmark_ids, supplied_labels, train_items, val_items, *args, **kwargs):
        unseen = np.setdiff1d(np.arange(question_count), np.concatenate([train_items, val_items]))
        assert np.isnan(supplied_labels[:, unseen]).all()
        training_labels.append(supplied_labels.copy())
        return original_fit(decoder, embeddings, benchmark_ids, supplied_labels, train_items, val_items, *args, **kwargs)

    monkeypatch.setattr(runner, "fit_decoder", checked_fit)

    def checked_scalar_fit(self, supplied_labels, supplied_domains):
        split_content = json.loads((output / "split.json").read_text(encoding="utf-8"))
        expected = labels[np.ix_(split_content["train_models"], split_content["train_items"])]
        np.testing.assert_array_equal(supplied_labels, expected)
        np.testing.assert_array_equal(supplied_domains, domains[split_content["train_items"]])
        scalar_training_inputs.append(supplied_labels.copy())
        return original_scalar_fit(self, supplied_labels, supplied_domains)

    monkeypatch.setattr(runner.ScalarLogitShift, "fit", checked_scalar_fit)
    variants = ["structured8", "rank0"] + (["neural_lowrank"] if selector_variant == "neural_lowrank" else [])
    arguments = ["--config", str(config_path), "--output", str(output), "--device", "cpu",
                 "--variants", ",".join(variants), "--deadline", "2099-01-01T00:00:00+00:00",
                 "--cpu-threads", "1"]
    assert runner.main(arguments + ["--phase", "train_validate"]) == 0
    assert len(training_labels) == len(variants)
    lock = json.loads((output / "test_lock.json").read_text(encoding="utf-8"))
    assert lock["test_labels_used"] is False
    assert not (output / "test_completed.json").exists()
    assert runner.main(arguments + ["--phase", "test"]) == 0
    assert len(scalar_training_inputs) >= 2
    assert lock["selection"]["scalar_logit_shift"]["enabled"] is True
    assert (output / "per_model_metrics.csv").exists()
    assert json.loads((output / "test_completed.json").read_text(encoding="utf-8"))["num_records"] > 0
    dataset = runner.load_dataset(data, benchmarks=["MMLU", "MathQA"])
    altered = json.loads(json.dumps(config))
    altered["model"]["rank"] = 16
    with pytest.raises(ValueError, match="lock mismatch"):
        runner.verify_lock(dataset, altered, output, variants)
    split = runner.load_split(output / "split.json", dataset)
    alternate = runner.load_split(output / "split.json", dataset)
    alternate.train_models[0], alternate.val_models[0] = alternate.val_models[0], alternate.train_models[0]
    alternate_path = tmp_path / "alternate_split.json"
    runner.save_split(alternate_path, dataset, alternate)
    with pytest.raises(ValueError, match="Supplied split differs"):
        runner.main(arguments + ["--phase", "test", "--split", str(alternate_path)])

    # An external model supplies its own answers; no existing row is retrieved.
    location = Path(__file__).resolve().parents[1] / "scripts" / "embed_new_model.py"
    specification = importlib.util.spec_from_file_location("external_embedder_under_test", location)
    embedder = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(embedder)
    response_path = tmp_path / "external_answers.csv"
    with response_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_id", "response"])
        writer.writeheader()
        writer.writerows({"prompt_id": int(dataset.prompt_ids[item]), "response": int(i % 2)}
                         for i, item in enumerate(split.train_items[:8]))
    external_output = tmp_path / "external_embedding"
    assert embedder.main(["--deadline", "2099-01-01T00:00:00+00:00", "infer", "--run", str(output),
                          "--responses", str(response_path), "--model-name", "SYNTHETIC_external_unseen_model",
                          "--output", str(external_output), "--seed", "7"]) == 0
    artifact = np.load(external_output / "embedding.npz")
    assert artifact["latent"].shape == (11,)
    assert len(artifact["responses"]) == 8
    prediction = json.loads((external_output / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["model_name"] == "SYNTHETIC_external_unseen_model"
    assert len(prediction["benchmark_predictions"]) == 2
