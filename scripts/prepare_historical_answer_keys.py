"""Recover verified option labels for historical training prompts with existing source-row mappings."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluate_capability_router import ROOT, input_context
from embed_new_model import read_prompts
from research.verify_source_items import SOURCES, norm, parse_prompt


def verified_options(ids, texts):
    pos = {int(v): i for i, v in enumerate(ids)}
    answers = np.full(len(ids), -1, dtype=np.int16)
    for dataset, mapping in (("MMLU", "mmlu_source_rows.csv"), ("MedMCQA", "medmcqa_source_ids.csv")):
        source = SOURCES[dataset]
        file = hf_hub_download(
            source["dataset"], source["parquet_path"], revision=source["revision"], repo_type="dataset"
        )
        rows = pd.read_csv(ROOT / "data/metadata" / mapping)
        rows = rows[rows.prompt_id.isin(pos)]
        if dataset == "MMLU":
            chosen = (
                pq.read_table(file, columns=["choices", "answer"])
                .take(rows.source_row_index.to_numpy())
                .to_pylist()
            )
        else:
            columns = ["id", "opa", "opb", "opc", "opd", "cop"]
            allowed = rows.source_id.tolist()
            data = pq.read_table(file, columns=columns, filters=[("id", "in", allowed)]).to_pylist()
            lookup = {v["id"]: v for v in data}
            chosen = [lookup[i] for i in allowed]
        keys = {}
        for pid, source_row in zip(rows.prompt_id, chosen):
            if dataset == "MMLU":
                original = source_row["choices"]
                correct = source_row["answer"]
            else:
                original = [source_row["op" + i] for i in "abcd"]
                correct = source_row["cop"]
            if correct is None or not 0 <= correct < 4:
                continue
            _, displayed = parse_prompt(texts[int(pid)], dataset)
            original = tuple(map(norm, original))
            if displayed == original:
                key = correct
            elif displayed.count(original[correct]) == 1:
                key = displayed.index(original[correct])
            else:
                continue
            keys.setdefault(pid, set()).add(key)
        for pid, values in keys.items():
            if len(values) == 1:
                answers[pos[pid]] = next(iter(values))
    return answers


def prepare(args):
    d, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        ids = d.prompt_ids[f["training"]]
    texts = read_prompts(ids, ROOT / "data/raw/question_order.csv")
    answers = verified_options(ids, texts)
    print(int((answers >= 0).sum()), "verified historical option keys", flush=True)
    np.savez_compressed(args.out / "historical_answer_keys.npz", prompt_ids=ids, option=answers)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    prepare(p.parse_args())
