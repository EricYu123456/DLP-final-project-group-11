"""Oracle separation tests use synthetic rows, never research result numbers."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from coldstart.data import PreparedDataset
from coldstart.models import ScalarLogitShift
from coldstart.runtime import Deadline
from coldstart.splits import dataset_fingerprint


def evaluator():
    source = Path(__file__).resolve().parents[1] / "scripts/evaluate_metadata_excluded.py"
    specification = importlib.util.spec_from_file_location("excluded_evaluator_under_test", source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def oracle_fixture(tmp_path):
    runner = evaluator()
    source_ids = np.roll(np.arange(112), 7)
    prompts = np.array([5, 0, 8, 3, 9, 1, 11, 4, 2, 10, 6, 7])
    embeddings = np.arange(24, dtype=np.float32).reshape(12, 2)
    domains = prompts % 5
    labels = ((source_ids[:, None] >> (prompts[None, :] % 7)) & 1).astype(np.float32)
    labels[np.flatnonzero(source_ids == 88)[0], np.flatnonzero(prompts == 3)[0]] = np.nan
    bank = np.array([0, 3, 8, 1, 11, 2, 4, 5, 9, 7])
    selected = np.array([int(np.flatnonzero(prompts == value)[0]) for value in bank])
    base = PreparedDataset(np.zeros((88, len(bank))), embeddings[selected], domains[selected], np.arange(len(bank)),
        np.arange(88), bank, [f"model_{i}" for i in range(88)], [f"b{i}" for i in range(5)],
        {"label_policy": "strict"})
    path = tmp_path / "source.npz"
    np.savez_compressed(path, model_ids=source_ids, prompt_ids=prompts, strict_labels=labels,
                        embeddings=embeddings, benchmark_ids=domains)
    path.with_suffix(".json").write_text(json.dumps({"model_names": [f"model_{mid}" for mid in source_ids],
        "benchmark_names": [f"b{i}" for i in range(5)]}), encoding="utf-8")
    family = tmp_path / "families.csv"
    family.write_text("model_id,inclusion\n" + "".join(f"{mid},{'include' if mid < 88 else 'exclude'}\n" for mid in range(112)), encoding="utf-8")
    return runner, base, path, family, labels, source_ids, selected


def test_oracle_aligns_all24_rows_to_fixed_bank_and_keeps_missing_without_mutating_base(oracle_fixture):
    runner, base, path, family, labels, source_ids, selected = oracle_fixture
    original = base.labels.copy()
    oracle = runner.load_oracle(base, path, family)
    np.testing.assert_array_equal(oracle.model_ids, np.arange(88, 112))
    np.testing.assert_array_equal(oracle.prompt_ids, base.prompt_ids)
    np.testing.assert_array_equal(oracle.benchmark_ids, base.benchmark_ids)
    expected_rows = np.array([int(np.flatnonzero(source_ids == mid)[0]) for mid in range(88, 112)])
    np.testing.assert_array_equal(oracle.labels, labels[np.ix_(expected_rows, selected)])
    assert np.isnan(oracle.labels[0]).sum() == 1
    assert oracle.model_names[0] == "model_88" and not hasattr(oracle, "embeddings")
    oracle.labels[:] = np.nan
    np.testing.assert_array_equal(base.labels, original)


def test_oracle_rejects_cohort_scope_or_feature_bank_changes(oracle_fixture):
    runner, base, path, family, *_ = oracle_fixture
    family.write_text(family.read_text().replace("111,exclude", "111,include"))
    with pytest.raises(ValueError, match="exactly the 24"):
        runner.load_oracle(base, path, family)
    family.write_text(family.read_text().replace("111,include", "111,exclude"))
    base.embeddings[0, 0] += 1
    with pytest.raises(ValueError, match="exact main benchmark IDs and text embeddings"):
        runner.load_oracle(base, path, family)


def test_fixed_macro_denominator_keeps_incomplete_and_all_missing_models_as_unavailable():
    runner = evaluator()
    domains = np.repeat(np.arange(5), 2)
    truth = np.tile([0, 1], (3, 5)).astype(float)
    truth[1, 6:8] = np.nan
    truth[2] = np.nan
    result = runner.fixed_metrics(truth, np.full((3, 10), 0.5), domains)
    np.testing.assert_array_equal(result["per_model"]["domain_count"], [5, 4, 0])
    assert len(result["per_model"]["macro_mae"]) == 3
    for field in runner.COMPLETE_DOMAIN_FIELDS:
        assert np.isfinite(result["per_model"][field][0])
        assert np.isnan(result["per_model"][field][1:]).all()
        assert np.isnan(result["summary"][field])
    assert result["per_model"]["brier"][1] == 0.25
    assert np.isnan(result["per_model"]["brier"][2])


def test_frozen_priors_restore_only_old62_statistics_and_detect_seed_changes(oracle_fixture, tmp_path):
    runner, base, *_ = oracle_fixture
    rng = np.random.default_rng(14)
    base.labels[:] = rng.integers(0, 2, size=base.labels.shape)
    split = SimpleNamespace(train_models=np.arange(62), train_items=np.arange(base.n_questions))
    scalar = ScalarLogitShift(5).fit(base.labels[:62], base.benchmark_ids)
    for seed in (11, 23, 37):
        runner.atomic_json(tmp_path / "baseline_fits" / f"scalar_seed_{seed}.json", {
            **scalar.metadata(), "training_model_ids": base.model_ids[:62].tolist(),
            "training_prompt_ids": base.prompt_ids.tolist(), "dataset_fingerprint": dataset_fingerprint(base)})
    config = {"seeds": [11, 23, 37], "baselines": {"scalar_logit_shift": {"variance_floor": 0.2, "training_ridge": 1e-6}}}
    mean, restored = runner.restore_baselines(base, split, config, tmp_path)
    expected = mean.benchmark_prior.copy()
    np.testing.assert_array_equal(restored.prior_precision, scalar.prior_precision)
    base.labels[62:] = np.nan
    again, _ = runner.restore_baselines(base, split, config, tmp_path)
    np.testing.assert_array_equal(again.benchmark_prior, expected)
    changed = runner.read_json(tmp_path / "baseline_fits/scalar_seed_37.json")
    changed["prior_mean"][0] += 1
    runner.atomic_json(tmp_path / "baseline_fits/scalar_seed_37.json", changed)
    with pytest.raises(ValueError, match="differ across main seeds"):
        runner.restore_baselines(base, split, config, tmp_path)


def test_scalar_file_hashes_and_restored_attributes_are_both_bound(tmp_path):
    runner = evaluator()
    mean = SimpleNamespace(global_prior=0.4, benchmark_prior=np.array([0.3, 0.5]))
    scalar = SimpleNamespace(prior_mean=np.array([0.1]), prior_precision=np.array([[2.0]]), benchmark_offsets=np.array([-0.2, 0.2]))
    for seed in (11, 23, 37):
        runner.atomic_json(tmp_path / "baseline_fits" / f"scalar_seed_{seed}.json", {"seed": seed, "prior": "frozen"})
    mean_snapshot, scalar_snapshot = runner.prior_snapshots(mean, scalar)
    manifest = {"scalar_fit_sha256": {str(seed): runner.sha256(tmp_path / "baseline_fits" / f"scalar_seed_{seed}.json") for seed in (11, 23, 37)},
                "frozen_mean_prior": mean_snapshot, "frozen_scalar_prior": scalar_snapshot}
    runner.verify_prior_bindings(tmp_path, manifest, mean, scalar)
    scalar.prior_mean[0] += 1
    with pytest.raises(ValueError, match="attributes changed"):
        runner.verify_prior_bindings(tmp_path, manifest, mean, scalar)
    scalar.prior_mean[0] -= 1
    runner.atomic_json(tmp_path / "baseline_fits/scalar_seed_23.json", {"changed": True})
    with pytest.raises(ValueError, match="file changed"):
        runner.verify_prior_bindings(tmp_path, manifest, mean, scalar)


def test_context_loader_hash_ignores_line_endings_but_rejects_code_changes(tmp_path):
    runner = evaluator()
    source = tmp_path / "context.py"
    source.write_bytes(b"context = 1\n")
    manifest = {"context_loader_sha256": runner.python_source_sha256(source)}
    source.write_bytes(b"context = 1\r\n")
    runner.verify_context_loader(manifest, source)
    source.write_bytes(b"context = 2\r\n")
    with pytest.raises(ValueError, match="context loader source changed"):
        runner.verify_context_loader(manifest, source)


@pytest.mark.parametrize("status,digest", [("FAIL", "correct"), ("PASS", "wrong")])
def test_score_refuses_missing_or_wrong_independent_gate_before_predictor_execution(tmp_path, monkeypatch, status, digest):
    runner = evaluator()
    monkeypatch.setattr(runner, "verify", lambda output, deadline: (None, None, None, None, None, {"manifest_sha256": "correct"}))
    monkeypatch.setattr(runner, "load_features", lambda *args, **kwargs: pytest.fail("Predictors must remain closed before the gate"))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"status": status, "manifest_sha256": digest}), encoding="utf-8")
    with pytest.raises(ValueError, match="passing independent gate"):
        runner.score(tmp_path, gate, Deadline(), "cpu")
