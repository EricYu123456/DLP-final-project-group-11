"""Match released MMLU and MedMCQA prompts to pinned source rows.

Run after preparing the EmbedLLM data. The output contains identifiers only,
never question text. HF's dataset server currently serves the pinned versions;
the script checks both the dataset revision and the Parquet bytes before use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "MMLU": {
        "dataset": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "config": "all",
        "split": "test",
        "rows": 14042,
        "parquet_path": "all/test-00000-of-00001.parquet",
        "parquet_hash": "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7",
    },
    "MedMCQA": {
        "dataset": "openlifescienceai/medmcqa",
        "revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        "config": "default",
        "split": "validation",
        "rows": 4183,
        "parquet_path": "data/validation-00000-of-00001.parquet",
        "parquet_hash": "b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021",
    },
}


def get(url: str, *, params: dict | None = None) -> requests.Response:
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=35)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response
        if attempt == 2:
            response.raise_for_status()
    raise AssertionError("unreachable")


def verify_revision(source: dict) -> None:
    dataset = source["dataset"]
    meta = get(f"https://huggingface.co/api/datasets/{dataset}").json()
    if meta.get("sha") != source["revision"]:
        raise RuntimeError(f"The current {dataset} revision changed; refuse to match unpinned viewer rows")
    index = get("https://datasets-server.huggingface.co/parquet", params={"dataset": dataset}).json()
    files = [item for item in index["parquet_files"]
             if item["config"] == source["config"] and item["split"] == source["split"]]
    if len(files) != 1:
        raise RuntimeError(f"Unexpected Parquet layout for {dataset}")
    pinned = f"https://huggingface.co/datasets/{dataset}/resolve/{source['revision']}/{source['parquet_path']}"
    for url in (pinned, files[0]["url"]):
        digest = hashlib.sha256(get(url).content).hexdigest()
        if digest != source["parquet_hash"]:
            raise RuntimeError(f"Parquet SHA-256 mismatch for {dataset}: {url}")


def fetch_rows(source: dict) -> list[dict]:
    verify_revision(source)

    def page(offset: int) -> list[dict]:
        params = {"dataset": source["dataset"], "config": source["config"],
                  "split": source["split"], "offset": offset, "length": 100}
        response = get("https://datasets-server.huggingface.co/rows", params=params).json()
        if response["num_rows_total"] != source["rows"] or response.get("partial"):
            raise RuntimeError(f"Unexpected viewer shape for {source['dataset']}")
        if any(row.get("truncated_cells") for row in response["rows"]):
            raise RuntimeError(f"Truncated viewer row for {source['dataset']}")
        return response["rows"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        pages = executor.map(page, range(0, source["rows"], 100))
        rows = [item for page_items in pages for item in page_items]
    if len(rows) != source["rows"] or len({r["row_idx"] for r in rows}) != len(rows):
        raise RuntimeError(f"Missing or repeated viewer rows for {source['dataset']}")
    return rows


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def parse_prompt(text: str, benchmark: str) -> tuple[str, tuple[str, ...]]:
    if benchmark == "MMLU":
        text = re.sub(r"^The following are multiple choice questions \(with answers\) about [^\n]+\.\n", "", text, count=1)
    else:
        text = re.sub(r"^Question:\s*", "", text, count=1)
    text = re.sub(r"\nAnswer:\s*$", "", text)
    text = re.sub(r"\nChoices:\s*\n", "\n", text)
    matches = list(re.finditer(r"(?m)^([A-D])\.\s*", text))[-4:]
    if len(matches) != 4 or [match.group(1) for match in matches] != list("ABCD"):
        raise ValueError(f"Cannot parse four {benchmark} options")
    stem = norm(text[:matches[0].start()])
    choices = tuple(norm(text[match.end():matches[i + 1].start() if i + 1 < 4 else len(text)])
                    for i, match in enumerate(matches))
    return stem, choices


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "metadata")
    args = parser.parse_args()
    questions = pd.read_csv(args.raw_dir / "question_order.csv")
    catalog = pd.read_csv(args.processed_dir / "prompts.csv")
    prompts = questions.merge(catalog, on="prompt_id", validate="one_to_one")

    source = SOURCES["MMLU"]
    mmlu = fetch_rows(source)
    index: dict[tuple, list[int]] = defaultdict(list)
    for item in mmlu:
        row = item["row"]
        key = (row["subject"], norm(row["question"]), tuple(norm(choice) for choice in row["choices"]))
        index[key].append(item["row_idx"])
    mapping = []
    for item in prompts[prompts["benchmark"] == "MMLU"].itertuples(index=False):
        stem, choices = parse_prompt(item.prompt, "MMLU")
        subjects = [name.removeprefix("mmlu_") for name in item.categories.split("|") if name.startswith("mmlu_")]
        hits = [(subject, row) for subject in subjects for row in index.get((subject, stem, choices), [])]
        if not hits:
            raise ValueError(f"No upstream MMLU row for prompt ID {item.prompt_id}")
        mapping.extend({"prompt_id": item.prompt_id, "source_dataset": source["dataset"],
                        "source_revision": source["revision"], "source_config": source["config"],
                        "source_split": source["split"], "source_subject": subject,
                        "source_row_index": row, "match_method": "subject_normalized_question_ordered_four_options"}
                       for subject, row in hits)
    if len(mapping) != 14042 or len({row["source_row_index"] for row in mapping}) != 14042:
        raise ValueError("MMLU mapping is incomplete or non-unique at source-row level")
    save_rows(args.out_dir / "mmlu_source_rows.csv", mapping)
    print(f"MMLU: {prompts[prompts['benchmark'] == 'MMLU'].shape[0]:,} prompts matched to {len(mapping):,} source rows")

    source = SOURCES["MedMCQA"]
    med = fetch_rows(source)
    index_med: dict[tuple, list[str]] = defaultdict(list)
    for item in med:
        row = item["row"]
        key = (norm(row["question"]), tuple(sorted(norm(row["op" + c]) for c in "abcd")))
        index_med[key].append(row["id"])
    mapping = []
    for item in prompts[prompts["benchmark"] == "MedMCQA"].itertuples(index=False):
        stem, choices = parse_prompt(item.prompt, "MedMCQA")
        hits = index_med.get((stem, tuple(sorted(choices))), [])
        if len(hits) != 1:
            raise ValueError(f"Expected one upstream MedMCQA row for prompt ID {item.prompt_id}")
        mapping.append({"prompt_id": item.prompt_id, "source_id": hits[0], "source_dataset": source["dataset"],
                        "source_revision": source["revision"], "source_split": source["split"],
                        "match_method": "normalized_question_and_sorted_four_options"})
    if len(mapping) != 4183 or len({row["source_id"] for row in mapping}) != 4183:
        raise ValueError("MedMCQA mapping is not one-to-one")
    save_rows(args.out_dir / "medmcqa_source_ids.csv", mapping)
    print(f"MedMCQA: {len(mapping):,} prompts matched one-to-one to source IDs")


if __name__ == "__main__":
    main()
