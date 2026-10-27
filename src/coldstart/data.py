"""Pinned EmbedLLM data, structural audit, and compact array access.

The published tensors are a *published proxy target*.  They are never silently
treated as recovered correctness, and missing raw rows are never filled with 0.
The strict policy uses raw singleton records, not a label-dependent repair rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence
import unicodedata

import numpy as np

SOURCE_REPOSITORY = "RZ412/EmbedLLM"
SOURCE_REVISION = "266ae3e180ccb533c86afc9cfb2f5db99d171aea"
STRICT_BENCHMARKS = ("MMLU", "MedMCQA", "MathQA", "TruthfulQA", "LogiQA")
BENCHMARKS = ("ASDiv", "GPQA", "GSM8K", "LogiQA", "MMLU", "MathQA",
              "MedMCQA", "PIQA", "SocialIQA", "TruthfulQA")
EXPECTED_FILES: dict[str, dict[str, Any]] = {
    "model_order.csv": {"size": 3925, "sha256": "bf7015fa775c1aed8955f1a7a1190f0e4c12f9ac2bdfdc896a051a090be4ad93"},
    "question_order.csv": {"size": 19777711, "sha256": "024a43fd073dda8d37c9dae0590fc42faf5f5538f3d5efb11aa5657afc96d1f2"},
    "question_embeddings.pth": {"size": 109588706, "sha256": "47c35522dc323d3f3bfcc77643965882f61021cf0b949cf86de6a4853acba556"},
    "train_y.pth": {"size": 26588016, "sha256": "96c9b3b0345587b6a6a3c6847f0435bc1bb58f0f012dbae72b9e446151361324"},
    "val_y.pth": {"size": 2689000, "sha256": "0ad3892c03a4f858af0b587b7676233b8ef1bebd49febbc79f1123875aa99eac"},
    "test_y.pth": {"size": 2689004, "sha256": "fee1000428a9df833fc15f333d9afe25a9cd0d2a666050b23533b2786c2c5bb3"},
    "train.csv": {"size": 2701750720, "sha256": "4ac81ebe312d2ca0290c78774ddc5f5852acb66cd10d28738efcb2a0dbd8e247"},
    "val.csv": {"size": 273417680, "sha256": "802f312809e7c7086233510b3c1e12619ab72051e06664ebb8bff24cf0f67fce"},
    "test.csv": {"size": 282192406, "sha256": "385e701c97e65285e6c687164a8ebd63e0afa3ea70b28cca0092d601c3d9d72a"},
}


def sha256_file(path: str | Path, check: Callable[[], None] | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            if check is not None:
                check()
            digest.update(block)
    return digest.hexdigest()


def verify_source_file(path: str | Path, check: Callable[[], None] | None = None) -> str:
    path = Path(path)
    expected = EXPECTED_FILES[path.name]
    if path.stat().st_size != expected["size"]:
        raise ValueError(f"Wrong size for {path.name}: {path.stat().st_size}")
    actual = sha256_file(path, check)
    if actual != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual}")
    return actual


def benchmark_from_category(category: str) -> str:
    category = str(category).strip().lower()
    if category.startswith("mmlu_"):
        return "MMLU"
    if category.startswith("gpqa_"):
        return "GPQA"
    mapping = {"asdiv": "ASDiv", "gsm8k": "GSM8K", "logiqa": "LogiQA",
               "mathqa": "MathQA", "medmcqa": "MedMCQA", "piqa": "PIQA",
               "social_iqa": "SocialIQA", "truthfulqa_mc1": "TruthfulQA"}
    if category not in mapping:
        raise ValueError(f"Unknown task category: {category!r}")
    return mapping[category]


def _normalise_text(text: str) -> str:
    # Preserve case, punctuation, numbers and negation; normalise whitespace only.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _prompt_keys(prompt: str, benchmark: str) -> tuple[str, str | None, str]:
    """Return complete-content and complete-stem keys plus extraction rule.

    Explicit task template markers permit removing instructions and sorting
    complete choice texts. Unrecognised formats retain the complete prompt;
    semantic paraphrases intentionally remain unmatched. Keys group conditions
    for splitting only; they never aggregate their labels.
    """
    text = unicodedata.normalize("NFKC", str(prompt)).replace("\r\n", "\n")
    rule = "normalised_exact_prompt_v1"
    payload = _normalise_text(text)
    key_namespace = "normalised_exact_prompt_v1"
    stem_payload: str | None = None
    if benchmark == "GPQA":
        if "Question:" in text:
            text = text.rsplit("Question:", 1)[1]
        elif "What is the correct answer to this question:" in text:
            text = text.split("What is the correct answer to this question:", 1)[1]
        if "Choices:" in text:
            stem, choices_text = text.rsplit("Choices:", 1)
            choices_text = re.split(r"\n?(?:Answer:|Let's think step by step:)", choices_text, maxsplit=1)[0]
            choices = re.split(r"(?:^|\n)\s*\([A-D]\)\s*", choices_text)
            choices = [_normalise_text(choice) for choice in choices if choice.strip()]
            if len(choices) == 4 and all(choices):
                payload = json.dumps([_normalise_text(stem), sorted(choices)], ensure_ascii=False)
                rule = "gpqa_last_question_sorted_complete_choices_v1"
                key_namespace = "complete_mcq_v1"
                stem_payload = _normalise_text(stem)
    elif benchmark in ("MMLU", "MedMCQA", "LogiQA"):
        text = re.sub(r"\nAnswer:\s*$", "", text)
        if benchmark == "MMLU":
            text = re.sub(r"^The following are multiple choice questions \(with answers\) about [^\n]+\n", "", text)
        elif benchmark == "MedMCQA":
            text = re.sub(r"^Question:\s*", "", text)
        # Select the last A/B/C/D markers. Author initials can otherwise be
        # mistaken for choices in a question's passage (e.g. A. J. P. Taylor).
        markers = list(re.finditer(r"(?:^|\n)[ \t]*([A-D])\.[ \t]*", text))
        last = markers[-4:]
        if len(last) == 4 and [marker.group(1) for marker in last] == list("ABCD"):
            stem = re.sub(r"\nChoices:\s*$", "", text[:last[0].start()])
            choices = [_normalise_text(text[marker.end():last[i + 1].start() if i < 3 else len(text)])
                       for i, marker in enumerate(last)]
            if all(choices):
                payload = json.dumps([_normalise_text(stem), sorted(choices)], ensure_ascii=False)
                rule = f"{benchmark.lower()}_stem_sorted_complete_choices_v1"
                key_namespace = "complete_mcq_v1"
                stem_payload = _normalise_text(stem)
    elif benchmark == "TruthfulQA":
        pieces = re.split(r"(?:^|\n)Q:\s*", text)
        if len(pieces) > 1 and re.search(r"\nA:\s*$", pieces[-1]):
            payload = _normalise_text(re.sub(r"\nA:\s*$", "", pieces[-1]))
            rule = "truthfulqa_last_question_v1"
            stem_payload = payload
    elif benchmark in ("MathQA", "PIQA", "GSM8K"):
        pieces = re.split(r"(?:^|\n)Question:\s*", text)
        if len(pieces) > 1 and re.search(r"\nAnswer:\s*$", pieces[-1]):
            payload = _normalise_text(re.sub(r"\nAnswer:\s*$", "", pieces[-1]))
            rule = f"{benchmark.lower()}_last_question_no_released_choices_v1"
            stem_payload = payload
    elif benchmark == "SocialIQA":
        pieces = re.split(r"(?:^|\n)Q:\s*", text)
        if len(pieces) > 1 and re.search(r"\nA:\s*$", pieces[-1]):
            payload = _normalise_text(re.sub(r"\nA:\s*$", "", pieces[-1]))
            rule = "socialiqa_last_question_no_released_choices_v1"
            stem_payload = payload
    elif benchmark == "ASDiv":
        payload = _normalise_text(re.sub(r"\nAnswer:\s*$", "", text))
        rule = "asdiv_context_and_question_v1"
        stem_payload = payload
    digest = hashlib.sha256((key_namespace + "\0" + payload).encode("utf-8")).hexdigest()
    stem_digest = None if stem_payload is None else hashlib.sha256(("complete_stem_v2\0" + stem_payload).encode("utf-8")).hexdigest()
    return digest, stem_digest, rule


def canonical_prompt_key(prompt: str, benchmark: str) -> tuple[str, str]:
    """Conservative global key; identical full stems intentionally overgroup.

    Complete choice contents remain available to the audit. They do not prevent
    grouping identical stems whose options differ only in punctuation, nor a
    conservative union of generic identical stems with different option sets.
    """
    exact, stem, rule = _prompt_keys(prompt, benchmark)
    return (stem, rule + "+identical_complete_stem_v2") if stem is not None else (exact, rule)


@dataclass
class PreparedDataset:
    labels: np.ndarray
    embeddings: np.ndarray
    benchmark_ids: np.ndarray
    group_ids: np.ndarray
    model_ids: np.ndarray
    prompt_ids: np.ndarray
    model_names: Sequence[str]
    benchmark_names: Sequence[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.float32)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.benchmark_ids = np.asarray(self.benchmark_ids, dtype=np.int64)
        self.group_ids = np.asarray(self.group_ids, dtype=np.int64)
        self.model_ids = np.asarray(self.model_ids, dtype=np.int64)
        self.prompt_ids = np.asarray(self.prompt_ids, dtype=np.int64)
        self.model_names = tuple(str(value) for value in self.model_names)
        self.benchmark_names = tuple(str(value) for value in self.benchmark_names)
        if self.labels.ndim != 2 or self.embeddings.ndim != 2:
            raise ValueError("labels and embeddings must be matrices")
        m, q = self.labels.shape
        if self.embeddings.shape[0] != q:
            raise ValueError("Question embedding/label count mismatch")
        if len(self.model_ids) != m or len(self.model_names) != m:
            raise ValueError("Model metadata/label count mismatch")
        if any(len(value) != q for value in (self.benchmark_ids, self.group_ids, self.prompt_ids)):
            raise ValueError("Question metadata/label count mismatch")
        if len(np.unique(self.model_ids)) != m or len(np.unique(self.prompt_ids)) != q:
            raise ValueError("Model and prompt IDs must be unique")
        finite = self.labels[np.isfinite(self.labels)]
        if not np.isin(finite, [0.0, 1.0]).all():
            raise ValueError("Labels must be binary or NaN")
        if np.isinf(self.labels).any() or not np.isfinite(self.embeddings).all():
            raise ValueError("Invalid infinite labels or non-finite embeddings")
        if q and (self.benchmark_ids.min() < 0 or self.benchmark_ids.max() >= len(self.benchmark_names)):
            raise ValueError("Benchmark ID outside benchmark_names")

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.labels)

    @property
    def domains(self) -> np.ndarray:
        return self.benchmark_ids

    @property
    def domain_ids(self) -> np.ndarray:
        return self.benchmark_ids

    @property
    def domain_names(self) -> Sequence[str]:
        return self.benchmark_names

    @property
    def n_models(self) -> int:
        return self.labels.shape[0]

    @property
    def n_questions(self) -> int:
        return self.labels.shape[1]

    def subset(self, models: Iterable[int] | None = None,
               items: Iterable[int] | None = None) -> "PreparedDataset":
        rows = np.arange(self.n_models) if models is None else np.asarray(list(models), dtype=np.int64)
        cols = np.arange(self.n_questions) if items is None else np.asarray(list(items), dtype=np.int64)
        selected = np.unique(self.benchmark_ids[cols])
        remap = {int(old): new for new, old in enumerate(selected)}
        benchmarks = np.array([remap[int(old)] for old in self.benchmark_ids[cols]], dtype=np.int64)
        return PreparedDataset(self.labels[np.ix_(rows, cols)], self.embeddings[cols], benchmarks,
                               self.group_ids[cols], self.model_ids[rows], self.prompt_ids[cols],
                               [self.model_names[int(i)] for i in rows],
                               [self.benchmark_names[int(i)] for i in selected], dict(self.metadata))


def load_dataset(path: str | Path = "data/processed", label_policy: str = "strict",
                 model_ids: Sequence[int] | None = None,
                 benchmarks: Sequence[str] | None = None,
                 common_questions: bool = True,
                 deduplicate_groups: bool | None = None) -> PreparedDataset:
    """Read compact arrays; IDs select original dataset IDs, never array ranks.

    ``strict`` defaults to the five sources with least template/filter ambiguity
    and retains questions with exactly one raw row for every requested model.
    The structural criterion does not inspect the held-out label values.
    ``published`` retains the authors' matrix as its explicitly named target.
    ``unambiguous`` is an optional, value-conditioned sensitivity policy only.
    Strict defaults to one global representative per candidate base group: the
    lowest original prompt ID remaining after the structural/model/source rules.
    """
    path = Path(path)
    array_path = path / "embedllm.npz" if path.is_dir() else path
    meta_path = array_path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(array_path, allow_pickle=False) as archive:
        if label_policy == "published":
            labels = archive["published_labels"].astype(np.float32)
        elif label_policy in ("strict", "unambiguous"):
            if not metadata.get("raw_audit_complete", False):
                raise ValueError(f"{label_policy} requires the full raw CSV audit")
            labels = archive[f"{label_policy}_labels"].astype(np.float32)
        else:
            raise ValueError("label_policy must be strict, published, or unambiguous")
        dataset = PreparedDataset(labels, archive["embeddings"], archive["benchmark_ids"],
                                  archive["group_ids"], archive["model_ids"], archive["prompt_ids"],
                                  metadata["model_names"], metadata["benchmark_names"], metadata)
    if model_ids is None:
        rows = np.arange(dataset.n_models)
    else:
        positions = {int(value): index for index, value in enumerate(dataset.model_ids)}
        missing = set(int(value) for value in model_ids) - set(positions)
        if missing:
            raise ValueError(f"Unknown original model IDs: {sorted(missing)}")
        rows = np.array([positions[int(value)] for value in model_ids], dtype=np.int64)
    if len(rows) == 0:
        raise ValueError("At least one model is required")
    allowed = set(STRICT_BENCHMARKS if benchmarks is None and label_policy == "strict"
                  else dataset.benchmark_names if benchmarks is None else benchmarks)
    unknown = allowed - set(dataset.benchmark_names)
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
    keep = np.array([dataset.benchmark_names[int(i)] in allowed for i in dataset.benchmark_ids])
    if common_questions:
        keep &= np.isfinite(dataset.labels[rows]).all(axis=0)
    deduplicate_groups = label_policy == "strict" if deduplicate_groups is None else deduplicate_groups
    pre_dedup = np.flatnonzero(keep)
    dropped_representatives: dict[str, list[int]] = {}
    if deduplicate_groups:
        seen: set[int] = set()
        for column in sorted(pre_dedup, key=lambda index: int(dataset.prompt_ids[index])):
            group = int(dataset.group_ids[column])
            if group in seen:
                keep[column] = False
                name = dataset.benchmark_names[int(dataset.benchmark_ids[column])]
                dropped_representatives.setdefault(name, []).append(int(dataset.prompt_ids[column]))
            else:
                seen.add(group)
    result = dataset.subset(rows, np.flatnonzero(keep))
    result.metadata.update({"label_policy": label_policy, "common_questions": common_questions,
                            "retained_models": result.n_models, "retained_questions": result.n_questions,
                            "retained_benchmarks": list(result.benchmark_names),
                            "deduplicate_groups": bool(deduplicate_groups),
                            "representative_rule": "lowest original prompt_id globally after structural/model/source eligibility" if deduplicate_groups else "all eligible prompt variants",
                            "eligible_questions_before_group_representative": len(pre_dedup),
                            "representative_dropped_prompt_ids_by_benchmark": dropped_representatives,
                            "representative_dropped_counts_by_benchmark": {name: len(ids) for name, ids in dropped_representatives.items()},
                            "retained_prompt_ids": result.prompt_ids.tolist()})
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_dataset(raw_dir: str | Path, output_dir: str | Path,
                    legacy_dir: str | Path | None = None,
                    chunksize: int = 100_000,
                    check: Callable[[], None] | None = None,
                    verify: bool = True,
                    progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Audit every CSV record and write compact arrays and provenance.

    The optional legacy directory supplies verified question_catalog / compact
    tensors for a published-only quick preparation; strict loading remains
    disabled until all three raw CSV files have been audited.
    """
    import pandas as pd
    import torch

    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = Path(legacy_dir) if legacy_dir is not None else None
    check = check or (lambda: None)
    progress = progress or (lambda message: None)
    hashes: dict[str, str] = {}
    for filename in ("model_order.csv", "question_order.csv", "question_embeddings.pth"):
        check()
        hashes[filename] = verify_source_file(raw_dir / filename, check) if verify else sha256_file(raw_dir / filename, check)
    models = pd.read_csv(raw_dir / "model_order.csv").sort_values("model_id")
    questions = pd.read_csv(raw_dir / "question_order.csv").sort_values("prompt_id")
    mids = models.model_id.to_numpy(dtype=np.int64)
    pids = questions.prompt_id.to_numpy(dtype=np.int64)
    if not np.array_equal(mids, np.arange(len(mids))) or not np.array_equal(pids, np.arange(len(pids))):
        raise ValueError("Source IDs must be contiguous before alignment")
    embeddings = torch.load(raw_dir / "question_embeddings.pth", map_location="cpu", weights_only=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != len(pids) or not np.isfinite(embeddings).all():
        raise ValueError("Source question embeddings are invalid")
    shape = (len(mids), len(pids))
    raw_count = np.zeros(shape, dtype=np.uint16)
    label_min = np.full(shape, 255, dtype=np.uint8)
    label_max = np.zeros(shape, dtype=np.uint8)
    category_min = np.full(shape, 32767, dtype=np.int16)
    category_max = np.full(shape, -1, dtype=np.int16)
    original_split = np.full(len(pids), -1, dtype=np.int8)
    prompt_categories: list[set[str]] = [set() for _ in pids]
    category_codes: dict[str, int] = {}
    raw_category_ids: dict[str, set[int]] = {}
    condition_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    split_ids: dict[str, np.ndarray] = {}
    audit_splits: dict[str, Any] = {}
    full_raw = all((raw_dir / f"{split}.csv").is_file() for split in ("train", "val", "test"))
    if full_raw:
        for split_num, split in enumerate(("train", "val", "test")):
            check()
            source = raw_dir / f"{split}.csv"
            hashes[source.name] = verify_source_file(source, check) if verify else sha256_file(source, check)
            rows = 0
            invalid = {"model_name_mismatch": 0, "prompt_text_mismatch": 0}
            split_prompts = np.zeros(len(pids), dtype=bool)
            # All seven columns are read to check identity/text, not only labels.
            for chunk in pd.read_csv(source, chunksize=chunksize, keep_default_na=False):
                check()
                mi = chunk.model_id.to_numpy(dtype=np.int64)
                qi = chunk.prompt_id.to_numpy(dtype=np.int64)
                raw_y = chunk.label.to_numpy(dtype=np.float64)
                if not np.isin(raw_y, [0.0, 1.0]).all():
                    raise ValueError(f"Non-binary raw labels in {source.name}")
                yi = raw_y.astype(np.uint8)
                if (mi < 0).any() or (mi >= shape[0]).any() or (qi < 0).any() or (qi >= shape[1]).any():
                    raise ValueError(f"Out-of-range model/prompt ID in {source.name}")
                if ((original_split[qi] != -1) & (original_split[qi] != split_num)).any():
                    raise ValueError("Source CSV prompt IDs overlap across splits")
                original_split[qi] = split_num
                split_prompts[qi] = True
                categories = chunk.category.astype(str).to_numpy()
                for category in np.unique(categories):
                    benchmark_from_category(category)
                    if category not in category_codes:
                        category_codes[category] = len(category_codes)
                    if len(category_codes) > 128:
                        raise ValueError("Condition key encoding supports at most 128 categories")
                ci = np.array([category_codes[value] for value in categories], dtype=np.int16)
                category_pairs = chunk[["category", "category_id"]].drop_duplicates()
                for category, category_id in category_pairs.itertuples(index=False, name=None):
                    raw_category_ids.setdefault(str(category), set()).add(int(category_id))
                pairs = pd.DataFrame({"prompt_id": qi, "category": categories}).drop_duplicates()
                for prompt_id, category in pairs.itertuples(index=False, name=None):
                    prompt_categories[int(prompt_id)].add(str(category))
                flat = mi * shape[1] + qi
                condition_keys, inverse = np.unique(flat * 128 + ci, return_inverse=True)
                condition_counts = np.bincount(inverse).astype(np.uint32)
                condition_bits = np.zeros(len(condition_keys), dtype=np.uint8)
                np.bitwise_or.at(condition_bits, inverse, np.left_shift(np.uint8(1), yi))
                condition_blocks.append((condition_keys.astype(np.uint32), condition_counts, condition_bits))
                counts = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)
                if (raw_count.astype(np.uint32) + counts > np.iinfo(np.uint16).max).any():
                    raise ValueError("Raw duplicate count exceeds uint16")
                raw_count += counts.astype(np.uint16)
                np.minimum.at(label_min.ravel(), flat, yi)
                np.maximum.at(label_max.ravel(), flat, yi)
                np.minimum.at(category_min.ravel(), flat, ci)
                np.maximum.at(category_max.ravel(), flat, ci)
                expected_names = models.model_name.to_numpy()[mi]
                raw_names = chunk.model_name.astype(str).str.replace("/", "__", regex=False).to_numpy()
                invalid["model_name_mismatch"] += int(np.count_nonzero(raw_names != expected_names))
                expected_prompts = questions.prompt.to_numpy()[qi]
                invalid["prompt_text_mismatch"] += int(np.count_nonzero(chunk.prompt.astype(str).to_numpy() != expected_prompts))
                rows += len(chunk)
                progress(f"{split}: {rows:,} raw records audited")
            if any(invalid.values()):
                raise ValueError(f"CSV identity mismatches in {split}: {invalid}")
            split_ids[split] = np.flatnonzero(split_prompts)
            audit_splits[split] = {"rows": rows, "prompt_ids": int(split_prompts.sum()), **invalid}
    else:
        if legacy_dir is None:
            raise FileNotFoundError("Full raw CSV audit or a legacy provenance directory is required")
        catalog = pd.read_csv(legacy_dir / "question_catalog.csv").sort_values("prompt_id")
        if not np.array_equal(catalog.prompt_id.to_numpy(), pids):
            raise ValueError("Legacy question catalog ID mismatch")
        for index, row in catalog.iterrows():
            prompt_categories[index] = set(str(row.categories).split("|"))
            original_split[index] = ("train", "val", "test").index(str(row.original_split))
        compact_path = legacy_dir / "correctness_compact.npz"
        hashes["legacy/correctness_compact.npz"] = sha256_file(compact_path, check)
        expected_compact = "8ac86dfe93bed22def188876274a98131cfdc2b0e5090e93dbdbdc10bfba4151"
        if hashes["legacy/correctness_compact.npz"] != expected_compact:
            raise ValueError("Legacy tensor copy checksum mismatch")
        for split in ("train", "val", "test"):
            split_ids[split] = np.flatnonzero(original_split == ("train", "val", "test").index(split))
    if (original_split < 0).any():
        raise ValueError("Some prompts are absent from all source splits")
    published = np.full(shape, np.nan, dtype=np.float32)
    for split in ("train", "val", "test"):
        check()
        y_path = raw_dir / f"{split}_y.pth"
        if y_path.is_file():
            hashes[y_path.name] = verify_source_file(y_path, check) if verify else sha256_file(y_path, check)
            # Source y files pickle NumPy arrays; hashes are checked before load.
            labels = np.asarray(torch.load(y_path, map_location="cpu", weights_only=False), dtype=np.float32)
        elif legacy_dir is not None:
            with np.load(legacy_dir / "correctness_compact.npz", allow_pickle=False) as compact:
                if not np.array_equal(compact[f"{split}_prompt_ids"], split_ids[split]):
                    raise ValueError(f"Legacy tensor prompt alignment mismatch for {split}")
                labels = compact[f"{split}_labels"].astype(np.float32)
        else:
            raise FileNotFoundError(y_path)
        if labels.shape != (shape[0], len(split_ids[split])) or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Published tensor shape or values invalid for {split}")
        published[:, split_ids[split]] = labels
    sources = []
    for categories in prompt_categories:
        mapped = {benchmark_from_category(category) for category in categories}
        if len(mapped) != 1:
            raise ValueError(f"Prompt has multiple or no benchmark sources: {categories}")
        sources.append(next(iter(mapped)))
    benchmark_ids = np.array([BENCHMARKS.index(value) for value in sources], dtype=np.int16)
    category_single = np.array([len(value) == 1 for value in prompt_categories])
    singleton = (raw_count == 1) & category_single[None, :]
    unambiguous = (raw_count > 0) & (label_min == label_max) & (category_min == category_max) & category_single[None, :]
    strict_labels = np.where(singleton, label_min.astype(np.float32), np.nan)
    unambiguous_labels = np.where(unambiguous, label_min.astype(np.float32), np.nan)
    exact_keys, stem_keys, original_prompt_keys, group_rules = [], [], [], []
    for index, (prompt, source) in enumerate(zip(questions.prompt, sources)):
        if index % 1000 == 0:
            check()
        exact, stem, rule = _prompt_keys(prompt, source)
        exact_keys.append(exact)
        stem_keys.append(stem)
        original_prompt_keys.append(hashlib.sha256(("original_normalised_prompt_v2\0" + _normalise_text(prompt)).encode("utf-8")).hexdigest())
        group_rules.append(rule)
    # Union all three label-free identities globally, including across sources.
    # Identical complete stems may conservatively overgroup generic questions.
    parent = np.arange(len(pids), dtype=np.int64)
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index
    tokens: dict[str, int] = {}
    for index, keys in enumerate(zip(exact_keys, stem_keys, original_prompt_keys)):
        for key in keys:
            if key is None:
                continue
            if key in tokens:
                left, right = find(index), find(tokens[key])
                if left != right:
                    parent[max(left, right)] = min(left, right)
            else:
                tokens[key] = index
    components: dict[int, list[int]] = {}
    for index in range(len(pids)):
        components.setdefault(find(index), []).append(index)
    component_keys = {root: min(original_prompt_keys[index] for index in members) for root, members in components.items()}
    group_keys = [component_keys[find(index)] for index in range(len(pids))]
    unique_keys = sorted(set(group_keys))
    key_to_id = {key: index for index, key in enumerate(unique_keys)}
    group_ids = np.array([key_to_id[key] for key in group_keys], dtype=np.int32)
    condition_summary: dict[str, Any] = {"unique_model_prompt_category_groups": 0, "same_category_conflicting_groups": 0}
    if condition_blocks:
        condition_keys = np.concatenate([block[0] for block in condition_blocks])
        condition_counts = np.concatenate([block[1] for block in condition_blocks])
        condition_bits = np.concatenate([block[2] for block in condition_blocks])
        order = np.argsort(condition_keys, kind="stable")
        condition_keys, condition_counts, condition_bits = condition_keys[order], condition_counts[order], condition_bits[order]
        starts = np.r_[0, np.flatnonzero(condition_keys[1:] != condition_keys[:-1]) + 1]
        condition_keys = condition_keys[starts]
        condition_counts = np.add.reduceat(condition_counts, starts)
        condition_bits = np.bitwise_or.reduceat(condition_bits, starts)
        condition_models = (condition_keys // 128 // shape[1]).astype(np.int32)
        condition_prompts = (condition_keys // 128 % shape[1]).astype(np.int32)
        condition_categories = (condition_keys % 128).astype(np.int16)
        np.savez_compressed(output_dir / "conditions.npz", model_ids=condition_models,
                            prompt_ids=condition_prompts, category_codes=condition_categories,
                            raw_count=condition_counts, label_bits=condition_bits)
        condition_summary = {"unique_model_prompt_category_groups": len(condition_keys),
                             "same_category_conflicting_groups": int((condition_bits == 3).sum()),
                             "condition_label_bits": "1=only0; 2=only1; 3=both0and1",
                             "file": "conditions.npz", "sha256": sha256_file(output_dir / "conditions.npz", check)}
        del condition_blocks, order
    group_sources: dict[int, set[str]] = {}
    group_source_splits: dict[int, set[int]] = {}
    for index, group_id in enumerate(group_ids):
        group_sources.setdefault(int(group_id), set()).add(sources[index])
        group_source_splits.setdefault(int(group_id), set()).add(int(original_split[index]))
    by_benchmark = {}
    for index, name in enumerate(BENCHMARKS):
        select = benchmark_ids == index
        counts = raw_count[:, select]
        by_benchmark[name] = {"prompts": int(select.sum()), "base_group_candidates": int(len(set(group_ids[select]))),
                              "raw_rows": int(counts.sum()), "missing_cells": int((counts == 0).sum()),
                              "duplicate_cells": int((counts > 1).sum()),
                              "conflicting_cells": int(((label_min[:, select] != label_max[:, select]) & (counts > 0)).sum()),
                              "singleton_common_all_models": int(singleton[:, select].all(axis=0).sum())}
    audit = {"schema_version": 1, "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
             "raw_audit_complete": full_raw, "created_at": _utc_now(), "source_sha256": hashes,
             "models": shape[0], "prompts": shape[1], "embedding_shape": list(embeddings.shape),
             "splits": audit_splits, "by_benchmark": by_benchmark,
             "raw_rows": int(raw_count.sum()), "missing_cells": int((raw_count == 0).sum()),
             "duplicate_cells": int((raw_count > 1).sum()),
             "conflicting_cells": int(((label_min != label_max) & (raw_count > 0)).sum()),
             "published_equals_raw_max_on_observed_cells": bool(np.array_equal(published[raw_count > 0], label_max[raw_count > 0])),
             "published_raw_max_mismatch_cells": int(((published != label_max) & (raw_count > 0)).sum()),
             "category_names": sorted(category_codes),
             "category_code_names": [name for name, code in sorted(category_codes.items(), key=lambda pair: pair[1])],
             "raw_category_id_sets": {name: sorted(values) for name, values in raw_category_ids.items()},
             "condition_audit": condition_summary,
             "model_names": models.model_name.tolist(),
             "benchmark_names": list(BENCHMARKS), "strict_benchmarks": list(STRICT_BENCHMARKS),
             "target_definitions": {
                 "published": "Authors' released binary tensor; aggregation/missing-query semantics unresolved.",
                 "strict": "Exactly one raw row per selected model/prompt and exactly one task category for the prompt; complete-case intersection is structural, not conditioned on labels.",
                 "unambiguous": "Sensitivity only: agreeing raw labels within one category; exclusion depends on label values."},
             "grouping": {"version": 2, "groups": len(unique_keys), "rules": sorted(set(group_rules)),
                          "union_rule": "global union of exact normalised full prompts, extracted full stem+sorted complete choices, and identical extracted complete stems",
                          "conservative_same_stem_different_content_groups": sum(len({exact_keys[index] for index in members}) > 1 for members in components.values()),
                          "multi_benchmark_groups": sum(len(value) > 1 for value in group_sources.values()),
                          "groups_crossing_original_splits": sum(len(value) > 1 for value in group_source_splits.values()),
                          "limitation": "Candidate grouping from released prompts, not official source item IDs; semantic paraphrases or unrecognised templates may remain unmatched. Identical generic complete stems with different choices can be conservatively overgrouped."},
             "limitations": ["No raw model responses, evaluator revision, scoring-filter identifiers or generation status are published.",
                             "Singleton records are an auditable proxy; they do not recover upstream scoring correctness.",
                             "Published zero labels are not evidence that a missing query was answered incorrectly."]}
    check()
    destination = output_dir / "embedllm.npz"
    temporary = output_dir / "embedllm.partial.npz"
    np.savez_compressed(temporary, published_labels=published, strict_labels=strict_labels,
                        unambiguous_labels=unambiguous_labels, raw_count=raw_count,
                        raw_label_min=label_min, raw_label_max=label_max,
                        raw_category_min=category_min, raw_category_max=category_max,
                        embeddings=embeddings, benchmark_ids=benchmark_ids, group_ids=group_ids,
                        model_ids=mids, prompt_ids=pids, original_split=original_split)
    check()
    temporary.replace(destination)
    audit["prepared_sha256"] = sha256_file(destination, check)
    (output_dir / "embedllm.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt_table = pd.DataFrame({"prompt_id": pids, "benchmark": sources, "group_id": group_ids,
                                 "group_key": group_keys, "group_rule": group_rules,
                                 "exact_content_key": exact_keys, "complete_stem_key": stem_keys,
                                 "categories": ["|".join(sorted(value)) for value in prompt_categories],
                                 "group_benchmarks": ["|".join(sorted(group_sources[int(value)])) for value in group_ids],
                                 "original_split": np.array(["train", "val", "test"])[original_split],
                                 "category_count": [len(value) for value in prompt_categories]})
    prompt_table.to_csv(output_dir / "prompts.csv", index=False)
    (output_dir / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
