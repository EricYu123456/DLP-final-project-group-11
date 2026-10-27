"""Data invariants that matter to the scientific information boundary."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from coldstart.data import PreparedDataset, canonical_prompt_key, load_dataset, prepare_dataset
from coldstart.splits import (SplitManifest, load_split, make_paired_family_shift_splits,
                              make_question_split, make_split, save_split, validate_split)


def toy_dataset(models: int = 9, items: int = 36) -> PreparedDataset:
    return PreparedDataset(np.zeros((models, items)), np.ones((items, 4)),
                           np.repeat(np.arange(2), items // 2), np.arange(items) // 2,
                           np.arange(models), np.arange(items), [f"m{i}" for i in range(models)],
                           ["first", "second"])


def test_template_grouping_preserves_full_choices_and_question_meaning():
    first = ("The following are multiple choice questions (with answers) about medicine.\n"
             "Which is NOT 10?\nA. 9\nB. 10\nC. 11\nD. 12\nAnswer:")
    second = ("The following are multiple choice questions (with answers) about biology.\n"
              "Which is NOT 10?\nA. 12\nB. 11\nC. 10\nD. 9\nAnswer:")
    medical = "Question: Which is NOT 10?\nChoices:\nA. 12\nB. 11\nC. 10\nD. 9\nAnswer:"
    assert canonical_prompt_key(first, "MMLU")[0] == canonical_prompt_key(second, "MMLU")[0]
    assert canonical_prompt_key(first, "MMLU")[0] == canonical_prompt_key(medical, "MedMCQA")[0]
    assert canonical_prompt_key(first.replace("NOT ", ""), "MMLU")[0] != canonical_prompt_key(first, "MMLU")[0]
    assert canonical_prompt_key(first.replace("NOT 10", "NOT 100"), "MMLU")[0] != canonical_prompt_key(first, "MMLU")[0]
    # This intentionally groups identical generic stems even when choice text
    # differs. The possible overgrouping is an explicit conservative policy.
    assert canonical_prompt_key(first.replace("A. 9", "A. 9."), "MMLU")[0] == canonical_prompt_key(first, "MMLU")[0]


def test_truthfulqa_last_question_excludes_examples():
    a = "Q: Example one?\nA: Example.\n\nQ: Actual question?\nA:"
    b = "Q: Another example?\nA: Different.\n\nQ: Actual question?\nA:"
    assert canonical_prompt_key(a, "TruthfulQA")[0] == canonical_prompt_key(b, "TruthfulQA")[0]


def test_structural_singletons_drop_agreeing_duplicates_and_preserve_missing(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    m, q = 3, 6
    pd.DataFrame({"model_id": range(m), "model_name": [f"org__model{i}" for i in range(m)]}).to_csv(raw / "model_order.csv", index=False)
    prompts = [f"Question {index}" for index in range(q)]
    pd.DataFrame({"prompt_id": range(q), "prompt": prompts}).to_csv(raw / "question_order.csv", index=False)
    torch.save(torch.ones((q, 4)), raw / "question_embeddings.pth")
    categories = ["mmlu_anatomy"] * 4 + ["medmcqa", "logiqa"]
    published = np.zeros((m, q), dtype=np.float32)
    for split, ids in (("train", [0, 1]), ("val", [2, 3]), ("test", [4, 5])):
        records = []
        for model in range(m):
            for prompt_id in ids:
                if (model, prompt_id) == (1, 1):
                    continue  # Unknown support is intentionally *not* incorrect.
                label = (model + prompt_id) % 2
                category = "mmlu_clinical_knowledge" if prompt_id == 3 and model > 0 else categories[prompt_id]
                row = {"prompt_id": prompt_id, "model_id": model, "category_id": categories.index(categories[prompt_id]),
                       "label": label, "prompt": prompts[prompt_id], "model_name": f"org__model{model}", "category": category}
                records.append(row)
                published[model, prompt_id] = label
                if (model, prompt_id) == (0, 0):
                    records.append(dict(row))  # Agreeing duplicates still fail strict policy.
                if (model, prompt_id) == (2, 2):
                    records.append({**row, "label": 1 - label})
                    published[model, prompt_id] = 1
        pd.DataFrame(records).to_csv(raw / f"{split}.csv", index=False)
        torch.save(published[:, ids], raw / f"{split}_y.pth")
    audit = prepare_dataset(raw, output, verify=False, chunksize=5)
    assert audit["raw_audit_complete"] is True
    assert audit["duplicate_cells"] == 2
    assert audit["missing_cells"] == 1
    assert audit["published_equals_raw_max_on_observed_cells"] is True
    with np.load(output / "embedllm.npz") as arrays:
        assert np.isnan(arrays["strict_labels"][0, 0])
        assert np.isnan(arrays["strict_labels"][1, 1])
        assert np.isnan(arrays["strict_labels"][:, 3]).all()
        assert arrays["raw_count"][1, 1] == 0
    strict = load_dataset(output, "strict")
    assert strict.prompt_ids.tolist() == [4, 5]
    assert strict.mask.all()
    public = load_dataset(output, "published")
    assert public.prompt_ids.tolist() == list(range(q))
    assert public.labels[1, 1] == 0  # Retains released target semantics explicitly.


def test_group_split_is_deterministic_and_cross_domain_safe():
    dataset = toy_dataset()
    dataset.group_ids[-1] = dataset.group_ids[0]  # Same item repeated in another domain.
    first = make_question_split(dataset, seed=21)
    second = make_question_split(dataset, seed=21)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert sorted(np.concatenate(list(first.values())).tolist()) == list(range(dataset.n_questions))
    groups = [set(dataset.group_ids[part]) for part in first.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_strict_representative_is_global_across_domains_and_lowest_source_id(tmp_path):
    # The two different domain labels share a candidate base item. Selecting a
    # representative independently inside each domain would duplicate a probe.
    np.savez_compressed(tmp_path / "embedllm.npz", strict_labels=np.zeros((3, 4)),
                        published_labels=np.zeros((3, 4)), embeddings=np.ones((4, 2)),
                        benchmark_ids=[0, 1, 0, 1], group_ids=[11, 11, 12, 13],
                        model_ids=[0, 1, 2], prompt_ids=[20, 10, 30, 40])
    metadata = {"raw_audit_complete": True, "model_names": ["a", "b", "c"],
                "benchmark_names": ["MMLU", "MedMCQA"]}
    (tmp_path / "embedllm.json").write_text(json.dumps(metadata))
    strict = load_dataset(tmp_path, label_policy="strict", benchmarks=("MMLU", "MedMCQA"))
    assert strict.prompt_ids.tolist() == [10, 30, 40]
    assert len(np.unique(strict.group_ids)) == strict.n_questions
    assert strict.metadata["representative_dropped_prompt_ids_by_benchmark"] == {"MMLU": [20]}
    public = load_dataset(tmp_path, label_policy="published")
    assert public.n_questions == 4


def test_manifest_roundtrip_rejects_tampering_and_wrong_dataset(tmp_path):
    dataset = toy_dataset()
    manifest = make_split(dataset)
    path = tmp_path / "split.json"
    save_split(path, dataset, manifest)
    assert np.array_equal(load_split(path, dataset).test_items, manifest.test_items)
    dataset.prompt_ids += 100
    with pytest.raises(ValueError, match="different prepared dataset"):
        load_split(path, dataset)
    content = json.loads(path.read_text())
    content["seed"] += 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="checksum"):
        load_split(path)


def test_paired_family_shift_uses_same_targets_items_and_model_count():
    dataset = toy_dataset(models=15)
    families = {i: "target" if i < 5 else "validation" if i < 7 else "other" for i in range(15)}
    pairs = make_paired_family_shift_splits(dataset, families, "target", "validation", n_test=2, n_train=6, n_related=3)
    excluded, related = pairs["excluded"], pairs["related_allowed"]
    assert len(excluded.train_models) == len(related.train_models) == 6
    assert np.array_equal(excluded.test_models, related.test_models)
    assert np.array_equal(excluded.val_models, related.val_models)
    assert np.array_equal(excluded.test_items, related.test_items)
    assert all(families[int(i)] != "target" for i in excluded.train_models)
    assert sum(families[int(i)] == "target" for i in related.train_models) == 3
    validate_split(dataset, excluded, require_coverage=False)
    with pytest.raises(ValueError, match="do not cover"):
        validate_split(dataset, excluded)
