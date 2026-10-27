"""Build compact matrices, audit raw records, and freeze group-safe splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldstart.data import load_dataset, prepare_dataset
from coldstart.runtime import Deadline, register_job
from coldstart.splits import SplitManifest, make_question_split, make_random_model_split, save_split, validate_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "tmp" / "research" / "agent_a")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--label-policy", choices=("strict", "published", "unambiguous"), default="strict")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--model-seed", type=int, default=2718)
    parser.add_argument("--deadline", default=None, help="Optional ISO-8601 stop time, including timezone")
    parser.add_argument("--no-split", action="store_true")
    args = parser.parse_args()
    deadline = Deadline(args.deadline)
    with register_job(deadline=args.deadline, metadata={"task": "full_raw_data_audit"}):
        audit = prepare_dataset(args.raw_dir, args.output_dir, args.legacy_dir, args.chunksize,
                                check=deadline.check, progress=lambda message: print(message, flush=True))
        if not args.no_split:
            dataset = load_dataset(args.output_dir, label_policy=args.label_policy)
            items = make_question_split(dataset, seed=args.seed)
            models = make_random_model_split(dataset, seed=args.model_seed)
            manifest = SplitManifest(models["train"], models["val"], models["test"],
                                     items["train"], items["val"], items["test"], seed=args.seed,
                                     metadata={"model_seed": args.model_seed, "label_policy": args.label_policy})
            validate_split(dataset, manifest)
            save_split(args.output_dir / f"split_{args.label_policy}_random.json", dataset, manifest)
        print(json.dumps({"raw_audit_complete": audit["raw_audit_complete"],
                          "models": audit["models"], "prompts": audit["prompts"],
                          "raw_rows": audit["raw_rows"], "duplicate_cells": audit["duplicate_cells"],
                          "conflicting_cells": audit["conflicting_cells"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
