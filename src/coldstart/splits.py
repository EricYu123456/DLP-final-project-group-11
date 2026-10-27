"""Deterministic model and candidate-base-item partitions with audit manifests."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import PreparedDataset


@dataclass
class SplitManifest:
    # These are positional indices in PreparedDataset, not upstream IDs.
    train_models: np.ndarray
    val_models: np.ndarray
    test_models: np.ndarray
    train_items: np.ndarray
    val_items: np.ndarray
    test_items: np.ndarray
    protocol: str = "strict_unseen_item"
    seed: int = 1729
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("train_models", "val_models", "test_models", "train_items", "val_items", "test_items"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.int64))

    @property
    def probe_items(self) -> np.ndarray:
        return self.train_items


def _ratios(ratios: Sequence[float]) -> np.ndarray:
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("Three nonnegative train/val/test ratios must sum to one")
    return values


def _allocation(count: int, ratios: Sequence[float]) -> np.ndarray:
    ratios = _ratios(ratios)
    exact = count * ratios
    allocated = np.floor(exact).astype(np.int64)
    remainder_order = np.argsort(-(exact - allocated), kind="stable")
    for index in remainder_order[:count - int(allocated.sum())]:
        allocated[index] += 1
    # When feasible, every requested partition gets at least one group/model.
    nonzero = np.flatnonzero(ratios > 0)
    if count >= len(nonzero):
        for empty in nonzero[allocated[nonzero] == 0]:
            donors = np.flatnonzero(allocated > 1)
            donor = donors[np.argmax(allocated[donors])]
            allocated[donor] -= 1
            allocated[empty] += 1
    return allocated


def make_question_split(dataset: PreparedDataset, seed: int = 1729,
                         ratios: Sequence[float] = (0.7, 0.15, 0.15)) -> dict[str, np.ndarray]:
    """Stratify candidate base groups by benchmark; never split a group.

    Ratios allocate *groups*, rather than individually weighting prompt variants.
    The returned indices refer to columns of this exact prepared dataset.
    """
    _ratios(ratios)
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    # Exact underlying questions can recur in two benchmarks. Give each domain
    # signature its own stratum, retaining domain labels but splitting jointly.
    group_domains: dict[int, set[int]] = {}
    for group, benchmark in zip(dataset.group_ids, dataset.benchmark_ids):
        group_domains.setdefault(int(group), set()).add(int(benchmark))
    strata: dict[tuple[int, ...], list[int]] = {}
    for group, domains in group_domains.items():
        strata.setdefault(tuple(sorted(domains)), []).append(group)
    for signature in sorted(strata):
        all_groups = np.array(sorted(strata[signature]), dtype=np.int64)
        items = np.flatnonzero(np.isin(dataset.group_ids, all_groups))
        groups = rng.permutation(all_groups)
        counts = _allocation(len(groups), ratios)
        start = 0
        for name, count in zip(result, counts):
            chosen = groups[start:start + int(count)]
            result[name].extend(items[np.isin(dataset.group_ids[items], chosen)].tolist())
            start += int(count)
    return {name: np.array(sorted(indices), dtype=np.int64) for name, indices in result.items()}


def make_split(dataset: PreparedDataset, item_seed: int = 1729, model_seed: int = 2718,
               item_ratios: Sequence[float] = (0.7, 0.15, 0.15),
               model_ratios: Sequence[float] = (0.7, 0.15, 0.15),
               protocol: str = "strict_unseen_item") -> SplitManifest:
    items = make_question_split(dataset, seed=item_seed, ratios=item_ratios)
    models = make_random_model_split(dataset, seed=model_seed, ratios=model_ratios)
    manifest = SplitManifest(models["train"], models["val"], models["test"],
                             items["train"], items["val"], items["test"],
                             protocol=protocol, seed=item_seed, metadata={"model_seed": model_seed})
    validate_split(dataset, manifest)
    return manifest


def make_random_model_split(dataset: PreparedDataset | int, seed: int = 2718,
                            ratios: Sequence[float] = (0.7, 0.15, 0.15),
                            include_indices: Sequence[int] | None = None,
                            counts: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    total = dataset if isinstance(dataset, int) else dataset.n_models
    indices = np.arange(total, dtype=np.int64) if include_indices is None else np.asarray(include_indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= total).any():
        raise ValueError("Invalid included model positions")
    if counts is None:
        allocated = _allocation(len(indices), ratios)
    else:
        allocated = np.asarray(counts, dtype=np.int64)
        if allocated.shape != (3,) or (allocated < 0).any() or allocated.sum() != len(indices):
            raise ValueError("Model counts must be three nonnegative counts summing to included models")
    shuffled = np.random.default_rng(seed).permutation(indices)
    result = {}
    start = 0
    for name, count in zip(("train", "val", "test"), allocated):
        result[name] = np.sort(shuffled[start:start + int(count)])
        start += int(count)
    return result


def load_family_table(path: str | Path) -> dict[int, str]:
    """Read only explicitly included ancestry assignments, keyed by source ID."""
    import csv
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("inclusion", "").strip().lower() == "include":
                if not row.get("family", "").strip():
                    raise ValueError("Included family record has no family")
                model_id = int(row["model_id"])
                if model_id in result:
                    raise ValueError(f"Duplicate family source model ID: {model_id}")
                result[model_id] = row["family"].strip()
    return result


def _family_vector(dataset: PreparedDataset, families: Mapping[int, str] | Sequence[str] | str | Path) -> np.ndarray:
    if isinstance(families, (str, Path)):
        families = load_family_table(families)
    if isinstance(families, Mapping):
        return np.array([str(families.get(int(value), "")) for value in dataset.model_ids])
    values = np.asarray(families, dtype=str)
    if values.shape != (dataset.n_models,):
        raise ValueError("Family vector must match prepared model rows")
    return values


def make_family_model_split(dataset: PreparedDataset,
                            families: Mapping[int, str] | Sequence[str] | str | Path,
                            test_family: str, val_family: str | None = None,
                            seed: int = 2718) -> dict[str, np.ndarray]:
    values = _family_vector(dataset, families)
    groups = sorted(set(values) - {""})
    if test_family not in groups:
        raise ValueError(f"Test family is absent: {test_family}")
    if val_family is None:
        candidates = [value for value in groups if value != test_family]
        if len(candidates) < 2:
            raise ValueError("Family holdout requires at least three included groups")
        # Deterministic choice; callers should predeclare val_family for studies.
        val_family = str(np.random.default_rng(seed).choice(candidates))
    if val_family == test_family or val_family not in groups:
        raise ValueError("Validation family must be a distinct included family")
    train = np.flatnonzero((values != "") & (values != test_family) & (values != val_family))
    val, test = np.flatnonzero(values == val_family), np.flatnonzero(values == test_family)
    if not len(train) or not len(val) or not len(test):
        raise ValueError("Every family split needs nonempty train/val/test model sets")
    return {"train": train, "val": val, "test": test}


def make_paired_family_shift_splits(
        dataset: PreparedDataset,
        families: Mapping[int, str] | Sequence[str] | str | Path,
        target_family: str,
        validation_family: str = "Qwen",
        n_test: int = 8,
        n_train: int = 40,
        n_related: int = 8,
        seed: int = 2718,
        item_seed: int = 1729,
        item_ratios: Sequence[float] = (0.7, 0.15, 0.15),
        item_splits: Mapping[str, Sequence[int]] | None = None,
        ) -> dict[str, SplitManifest]:
    """Hold targets fixed while swapping equal counts of relatives/unrelateds.

    Both arms share C, target models, validation models and item partitions.
    ``excluded`` trains on C+U without the target family; ``related_allowed``
    trains on C+R, where R are non-target relatives and |R|=|U|. Unused models
    remain explicitly outside all partitions. Filter the prepared dataset once
    before calling this function so both arms share the same singleton items.
    """
    values = _family_vector(dataset, families)
    if (values == "").any():
        raise ValueError("Filter unknown/excluded ancestry models before paired family stress")
    if target_family == validation_family:
        raise ValueError("Validation family must differ from target family")
    if min(n_test, n_train, n_related) <= 0:
        raise ValueError("Test, training and related counts must be positive")
    target_pool = np.flatnonzero(values == target_family)
    validation = np.flatnonzero(values == validation_family)
    if len(target_pool) <= n_test or len(validation) == 0:
        raise ValueError("Need held-out targets, additional relatives and validation models")
    unrelated_pool = np.flatnonzero((values != target_family) & (values != validation_family))
    if len(unrelated_pool) < n_train:
        raise ValueError("Insufficient unrelated models for the requested count-matched training arm")
    family_entropy = int.from_bytes(hashlib.sha256(target_family.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, family_entropy]))
    shuffled_targets = rng.permutation(target_pool)
    test = np.sort(shuffled_targets[:n_test])
    related_count = min(n_related, len(target_pool) - n_test, n_train)
    relatives = np.sort(shuffled_targets[n_test:n_test + related_count])
    unrelated = rng.permutation(unrelated_pool)
    common = np.sort(unrelated[:n_train - related_count])
    replacement = np.sort(unrelated[n_train - related_count:n_train])
    questions = make_question_split(dataset, seed=item_seed, ratios=item_ratios) if item_splits is None else item_splits
    result = {}
    for arm, training in (("excluded", np.sort(np.concatenate((common, replacement)))),
                          ("related_allowed", np.sort(np.concatenate((common, relatives))))):
        used = np.concatenate((training, validation, test))
        unused = np.setdiff1d(np.arange(dataset.n_models), used)
        metadata = {"model_seed": seed, "item_seed": item_seed, "split_kind": "paired_family_shift",
                    "arm": arm, "target_family": target_family, "validation_family": validation_family,
                    "n_train": n_train, "n_test": n_test, "related_count": related_count,
                    "require_coverage": False,
                    "common_train_model_ids": dataset.model_ids[common].tolist(),
                    "related_train_model_ids": dataset.model_ids[relatives].tolist(),
                    "replacement_train_model_ids": dataset.model_ids[replacement].tolist(),
                    "unused_model_ids": dataset.model_ids[unused].tolist(),
                    "family_interpretation": "Conservative declared ancestry-series block; not a causal estimate of all distribution shift."}
        manifest = SplitManifest(training, validation, test, questions["train"], questions["val"], questions["test"],
                                 seed=item_seed, metadata=metadata)
        validate_split(dataset, manifest, require_coverage=False)
        if len(np.unique(np.concatenate((manifest.train_items, manifest.val_items, manifest.test_items)))) != dataset.n_questions:
            raise ValueError("Paired stress must retain the same complete item partition")
        if np.any(values[training] == validation_family):
            raise ValueError("Validation family leaked into training")
        if arm == "excluded" and np.any(values[training] == target_family):
            raise ValueError("Target family leaked into the excluded arm")
        result[arm] = manifest
    return result


def validate_split(dataset: PreparedDataset, manifest: SplitManifest,
                   require_coverage: bool = True,
                   families: Mapping[int, str] | Sequence[str] | str | Path | None = None) -> None:
    """Reject positional, candidate-base-group, and optional lineage leakage.

    Items always partition the complete prepared question set. Setting
    require_coverage=False permits unused *models* for the paired stress study.
    """
    if manifest.protocol not in ("strict_unseen_item", "fixed_bank"):
        raise ValueError(f"Unknown information protocol: {manifest.protocol}")
    for axis, size in (("models", dataset.n_models), ("items", dataset.n_questions)):
        partitions = [getattr(manifest, f"{name}_{axis}") for name in ("train", "val", "test")]
        for positions in partitions:
            if positions.ndim != 1 or len(np.unique(positions)) != len(positions):
                raise ValueError(f"{axis} partitions must be unique one-dimensional positions")
            if (positions < 0).any() or (positions >= size).any():
                raise ValueError(f"{axis} position outside prepared dataset")
            if not len(positions):
                raise ValueError(f"Every split needs nonempty {axis} partitions")
        for left in range(3):
            for right in range(left + 1, 3):
                if np.intersect1d(partitions[left], partitions[right]).size:
                    raise ValueError(f"Overlapping {axis} partitions")
        if (require_coverage or axis == "items") and len(np.unique(np.concatenate(partitions))) != size:
            raise ValueError(f"{axis} partitions do not cover the prepared dataset")
    required_benchmarks = set(dataset.benchmark_ids.tolist())
    for name in ("train", "val", "test"):
        items = getattr(manifest, f"{name}_items")
        if set(dataset.benchmark_ids[items].tolist()) != required_benchmarks:
            raise ValueError(f"{name} item partition lacks benchmark coverage")
        for benchmark in required_benchmarks:
            domain_items = items[dataset.benchmark_ids[items] == benchmark]
            if not np.isfinite(dataset.labels[:, domain_items]).any(axis=1).all():
                raise ValueError(f"{name} benchmark has no observed labels for at least one model")
    item_groups = [set(dataset.group_ids[getattr(manifest, f"{name}_items")].tolist()) for name in ("train", "val", "test")]
    for left in range(3):
        for right in range(left + 1, 3):
            if item_groups[left] & item_groups[right]:
                raise ValueError("Candidate base-question group leakage across item partitions")
    if families is not None:
        values = _family_vector(dataset, families)
        family_sets = [set(values[getattr(manifest, f"{name}_models")]) for name in ("train", "val", "test")]
        if any("" in selected for selected in family_sets):
            raise ValueError("Unknown/excluded lineage entered a family split")
        if any(family_sets[left] & family_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("Family lineage leakage across model partitions")


def dataset_fingerprint(dataset: PreparedDataset) -> str:
    digest = hashlib.sha256()
    for name in ("model_ids", "prompt_ids", "benchmark_ids", "group_ids"):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(getattr(dataset, name), dtype="<i8").tobytes())
    digest.update(str(dataset.metadata.get("label_policy", "unspecified")).encode("utf-8"))
    digest.update(str(dataset.metadata.get("prepared_sha256", "")).encode("ascii"))
    return digest.hexdigest()


def save_split(path: str | Path, dataset: PreparedDataset, manifest: SplitManifest) -> dict[str, Any]:
    validate_split(dataset, manifest, require_coverage=manifest.metadata.get("require_coverage", True))
    content: dict[str, Any] = {"schema_version": 1, "protocol": manifest.protocol, "seed": manifest.seed,
                               "metadata": manifest.metadata, "dataset_fingerprint": dataset_fingerprint(dataset),
                               "index_semantics": "zero-based positions in the fingerprinted prepared dataset",
                               "item_group_semantics": "candidate base-item groups from published prompt templates"}
    for split in ("train", "val", "test"):
        models, items = getattr(manifest, f"{split}_models"), getattr(manifest, f"{split}_items")
        content[f"{split}_models"] = models.tolist()
        content[f"{split}_items"] = items.tolist()
        content[f"{split}_model_ids"] = dataset.model_ids[models].tolist()
        content[f"{split}_prompt_ids"] = dataset.prompt_ids[items].tolist()
        content[f"{split}_group_ids"] = sorted(set(dataset.group_ids[items].tolist()))
        content[f"{split}_benchmark_counts"] = {name: int(np.sum(dataset.benchmark_ids[items] == index))
                                                 for index, name in enumerate(dataset.benchmark_names)}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    content["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return content


def load_split(path: str | Path, dataset: PreparedDataset | None = None) -> SplitManifest:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_hash = content.pop("manifest_sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    if stored_hash != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("Split manifest checksum mismatch")
    if dataset is not None and content["dataset_fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("Split manifest belongs to a different prepared dataset")
    manifest = SplitManifest(**{name: content[name] for name in ("train_models", "val_models", "test_models",
                                                                "train_items", "val_items", "test_items")},
                             protocol=content["protocol"], seed=content["seed"], metadata=content["metadata"])
    if dataset is not None:
        validate_split(dataset, manifest, require_coverage=manifest.metadata.get("require_coverage", True))
    return manifest
