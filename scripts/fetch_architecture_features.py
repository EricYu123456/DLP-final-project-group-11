"""Collect public model configuration fields for a label-free architectural prior."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from evaluate_capability_router import ROOT
from run_experiment import atomic_csv


def describe(row):
    name = row.hf_model_id or row.model_name.replace("__", "/")
    base = f"https://huggingface.co/{name}"
    entry = dict(
        name=name,
        model_type="",
        architecture="",
        parameters=0,
        layers=0,
        width=0,
        heads=0,
        kv_heads=0,
        experts=0,
        intermediate=0,
        context=0,
        vocab=0,
        tags="",
        parents=str(row.parent_models),
        family=str(row.family),
    )
    try:
        r = requests.get(f"https://huggingface.co/api/models/{name}", timeout=14)
        if r.ok:
            card = r.json()
            entry["parameters"] = (card.get("safetensors") or {}).get("total") or 0
            cfg = card.get("config") or {}
            entry["model_type"] = cfg.get("model_type") or ""
            entry["architecture"] = " ".join(cfg.get("architectures") or [])
            tags = card.get("tags") or []
            entry["tags"] = " ".join(
                t
                for t in tags
                if t.lower()
                in {
                    "chat",
                    "instruct",
                    "instruction-tuned",
                    "base_model",
                    "dpo",
                    "sft",
                    "chatml",
                    "moe",
                    "merge",
                    "medical",
                    "math",
                    "code",
                    "coding",
                    "conversational",
                    "text2text-generation",
                    "causal-lm",
                    "mergekit",
                    "gguf",
                    "gptq",
                    "awq",
                    "multilingual",
                }
            )
        revision = row.source_revision if row.source_revision else "main"
        r = requests.get(f"{base}/resolve/{revision}/config.json", timeout=14)
        if r.ok:
            cfg = r.json()
            entry["model_type"] = cfg.get("model_type") or entry["model_type"]
            entry["architecture"] = " ".join(cfg.get("architectures") or []) or entry["architecture"]
            fields = dict(
                layers=("num_hidden_layers", "n_layer"),
                width=("hidden_size", "n_embd"),
                heads=("num_attention_heads", "n_head"),
                kv_heads=("num_key_value_heads",),
                experts=("num_local_experts", "num_experts"),
                intermediate=("intermediate_size",),
                context=("max_position_embeddings", "n_positions"),
                vocab=("vocab_size",),
            )
            for key, aliases in fields.items():
                entry[key] = next((cfg[k] for k in aliases if isinstance(cfg.get(k), (float, int))), 0)
    except (requests.RequestException, ValueError):
        pass
    return entry


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=ROOT / "data/metadata/model_families.csv")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/capability_routing/release_confirmation/architectures.csv",
    )
    args = p.parse_args()
    with ThreadPoolExecutor(max_workers=5) as executor:
        rows = list(executor.map(describe, pd.read_csv(args.table).fillna("").itertuples()))
    atomic_csv(args.output, rows)
    print(
        f"{sum(bool(r['model_type']) for r in rows)} of {len(rows)} public architectures identified.",
        flush=True,
    )
