"""Checks for the model, underlying-item, and revealed-answer boundaries."""

import json

import numpy as np
import pytest

from coldstart.data import PreparedDataset, canonical_prompt_key
from coldstart.adaptation import adapt_map, predict_from_latent
from coldstart.diagnostics import permute_support_within_benchmark
from coldstart.splits import (
    SplitManifest,
    load_split,
    make_question_split,
    make_paired_family_shift_splits,
    make_random_model_split,
    save_split,
    validate_split,
)


def _dataset():
    # Two variants per underlying item; nonconsecutive source IDs ensure tests
    # exercise positional indices rather than accidentally using upstream IDs.
    groups = np.repeat(np.arange(18), 2)
    return PreparedDataset(
        labels=np.tile(np.arange(36) % 2, (9, 1)),
        embeddings=np.eye(36, 4, dtype=np.float32),
        benchmark_ids=np.repeat(np.arange(3), 12),
        group_ids=groups,
        model_ids=np.arange(9) * 13 + 100,
        prompt_ids=np.arange(36) * 7 + 200,
        model_names=[f"model_{i}" for i in range(9)],
        benchmark_names=["benchmark_a", "benchmark_b", "benchmark_c"],
        metadata={"label_policy": "strict"},
    )


def _manifest(dataset):
    items = make_question_split(dataset, seed=1729)
    models = make_random_model_split(dataset, seed=2718)
    return SplitManifest(
        models["train"], models["val"], models["test"],
        items["train"], items["val"], items["test"],
    )


def test_group_split_keeps_prompt_variants_together_and_covers_benchmarks():
    dataset = _dataset()
    manifest = _manifest(dataset)
    validate_split(dataset, manifest)
    group_sets = [set(dataset.group_ids[getattr(manifest, f"{name}_items")])
                  for name in ("train", "val", "test")]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    for name in ("train", "val", "test"):
        assert set(dataset.benchmark_ids[getattr(manifest, f"{name}_items")]) == {0, 1, 2}


def test_split_is_unchanged_when_heldout_outcomes_are_changed():
    dataset = _dataset()
    original = _manifest(dataset)
    changed = _dataset()
    changed.labels[original.test_models] = 1 - changed.labels[original.test_models]
    changed.labels[:, original.test_items] = 1 - changed.labels[:, original.test_items]
    updated = _manifest(changed)
    for name in ("train_models", "val_models", "test_models", "train_items", "val_items", "test_items"):
        np.testing.assert_array_equal(getattr(original, name), getattr(updated, name))


def test_validator_rejects_underlying_item_leakage_even_with_distinct_prompt_ids():
    dataset = _dataset()
    manifest = _manifest(dataset)
    left, right = int(manifest.train_items[0]), int(manifest.test_items[0])
    dataset.group_ids[right] = dataset.group_ids[left]
    with pytest.raises(ValueError, match="base-question group leakage"):
        validate_split(dataset, manifest)


def test_validator_rejects_model_and_known_lineage_overlap():
    dataset = _dataset()
    manifest = _manifest(dataset)
    manifest.test_models[0] = manifest.train_models[0]
    with pytest.raises(ValueError, match="Overlapping models"):
        validate_split(dataset, manifest)
    manifest = _manifest(dataset)
    families = {int(model): "shared_ancestor" for model in dataset.model_ids}
    with pytest.raises(ValueError, match="Family lineage leakage"):
        validate_split(dataset, manifest, families=families)


def test_validator_rejects_missing_domain_and_incomplete_item_partition():
    dataset = _dataset()
    manifest = _manifest(dataset)
    moved = []
    for name in ("val", "test"):
        items = getattr(manifest, f"{name}_items")
        moved.extend(items[dataset.benchmark_ids[items] == 2].tolist())
        setattr(manifest, f"{name}_items", items[dataset.benchmark_ids[items] != 2])
    manifest.train_items = np.sort(np.r_[manifest.train_items, moved])
    with pytest.raises(ValueError, match="benchmark coverage"):
        validate_split(dataset, manifest)
    manifest = _manifest(dataset)
    manifest.test_items = manifest.test_items[2:]
    with pytest.raises(ValueError, match="items partitions do not cover"):
        validate_split(dataset, manifest, require_coverage=False)


def test_manifest_roundtrip_binds_source_ids_and_detects_tampering(tmp_path):
    dataset = _dataset()
    manifest = _manifest(dataset)
    path = tmp_path / "split.json"
    content = save_split(path, dataset, manifest)
    assert content["train_model_ids"] == dataset.model_ids[manifest.train_models].tolist()
    assert content["test_prompt_ids"] == dataset.prompt_ids[manifest.test_items].tolist()
    loaded = load_split(path, dataset)
    np.testing.assert_array_equal(loaded.test_items, manifest.test_items)
    content["test_items"][0] = int(manifest.train_items[0])
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_split(path, dataset)


def test_cross_header_and_option_order_variants_group_as_one_question():
    first = (
        "The following are multiple choice questions (with answers) about biology.\n"
        "Which species survives?\nA. Species one\nB. Species two\nC. Species three\nD. Species four\nAnswer:"
    )
    second = (
        "The following are multiple choice questions (with answers) about physics.\n"
        "Which species survives?\nA. Species four\nB. Species two\nC. Species one\nD. Species three\nAnswer:"
    )
    other_source = (
        "Question: Which species survives?\nChoices:\nA. Species three\nB. Species four\n"
        "C. Species two\nD. Species one\nAnswer:"
    )
    key, _ = canonical_prompt_key(first, "MMLU")
    assert canonical_prompt_key(second, "MMLU")[0] == key
    assert canonical_prompt_key(other_source, "MedMCQA")[0] == key
    assert canonical_prompt_key(first.replace("survives", "does not survive"), "MMLU")[0] != key


def test_response_permutation_preserves_domain_successes_and_missingness():
    actual = np.array([0, 1, np.nan, 1, 0, 0, 1, 1], dtype=float)
    domains = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    original = actual.copy()
    shuffled = permute_support_within_benchmark(actual, domains, seed=11)
    np.testing.assert_array_equal(np.isnan(shuffled), np.isnan(actual))
    np.testing.assert_allclose(actual, original, equal_nan=True)
    for domain in (0, 1):
        selected = domains == domain
        assert np.nansum(shuffled[selected]) == np.nansum(actual[selected])
        assert np.isfinite(shuffled[selected]).sum() == np.isfinite(actual[selected]).sum()
    np.testing.assert_allclose(
        shuffled, permute_support_within_benchmark(actual, domains, seed=11), equal_nan=True
    )


def test_response_permutation_cannot_import_outside_budget_answers():
    domain_order = np.tile(np.arange(5), 40)
    answers = (np.arange(200) % 3 == 0).astype(float)
    k = 50
    before = permute_support_within_benchmark(answers[:k], domain_order[:k], seed=23)
    answers[k:] = 1 - answers[k:]
    after = permute_support_within_benchmark(answers[:k], domain_order[:k], seed=23)
    np.testing.assert_array_equal(before, after)
    assert len(after) == k


def test_rank_zero_map_depends_on_domain_totals_not_answer_item_alignment():
    rng = np.random.default_rng(103)
    domains = np.tile(np.arange(3), 15)
    design = np.column_stack([np.ones(len(domains)), np.eye(3)[domains]])
    offsets = rng.normal(scale=1.2, size=len(domains))
    responses = rng.binomial(1, 0.6, size=len(domains)).astype(float)
    responses[7] = np.nan
    shuffled = permute_support_within_benchmark(responses, domains, seed=37)
    matrix = rng.normal(size=(4, 4))
    precision = matrix.T @ matrix + np.eye(4)
    prior = rng.normal(scale=0.2, size=4)
    original_fit = adapt_map(design, offsets, responses, prior, precision)
    shuffled_fit = adapt_map(design, offsets, shuffled, prior, precision)
    assert original_fit.converged and shuffled_fit.converged
    np.testing.assert_allclose(original_fit.latent, shuffled_fit.latent, atol=1e-7)
    np.testing.assert_allclose(original_fit.covariance, shuffled_fit.covariance, atol=1e-8)
    np.testing.assert_allclose(
        predict_from_latent(original_fit.latent, offsets, design),
        predict_from_latent(shuffled_fit.latent, offsets, design), atol=1e-7,
    )


@pytest.mark.parametrize("target_family,n_test,n_related", [("Llama", 8, 8), ("Mistral", 6, 8), ("Yi", 3, 6)])
def test_family_shift_holds_targets_items_and_training_count_fixed(target_family, n_test, n_related):
    base = _dataset()
    families = np.repeat(["Llama", "Mistral", "Yi", "Qwen", "Other"], [36, 18, 9, 6, 19])
    dataset = PreparedDataset(
        labels=np.tile(base.labels[0], (88, 1)), embeddings=base.embeddings,
        benchmark_ids=base.benchmark_ids, group_ids=base.group_ids,
        model_ids=np.arange(88) * 13 + 100, prompt_ids=base.prompt_ids,
        model_names=[f"model_{i}" for i in range(88)], benchmark_names=base.benchmark_names,
    )
    arms = make_paired_family_shift_splits(
        dataset, families, target_family=target_family, n_test=n_test, n_related=n_related,
    )
    excluded, allowed = arms["excluded"], arms["related_allowed"]
    for name in ("val_models", "test_models", "train_items", "val_items", "test_items"):
        np.testing.assert_array_equal(getattr(excluded, name), getattr(allowed, name))
    assert len(excluded.train_models) == len(allowed.train_models) == 40
    assert len(excluded.test_models) == n_test
    assert set(families[excluded.test_models]) == {target_family}
    assert set(families[excluded.val_models]) == {"Qwen"}
    assert target_family not in families[excluded.train_models]
    assert "Qwen" not in families[excluded.train_models]
    assert "Qwen" not in families[allowed.train_models]
    relatives = np.setdiff1d(allowed.train_models, excluded.train_models)
    replacements = np.setdiff1d(excluded.train_models, allowed.train_models)
    assert len(relatives) == len(replacements) == n_related
    assert set(families[relatives]) == {target_family}
    assert target_family not in families[replacements]
    assert len(np.intersect1d(excluded.train_models, allowed.train_models)) == 40 - n_related
