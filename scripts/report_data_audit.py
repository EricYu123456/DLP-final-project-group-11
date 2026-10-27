"""Write small, versionable data-quality reports without prompts or labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from coldstart.data import EXPECTED_FILES, SOURCE_REPOSITORY, SOURCE_REVISION, STRICT_BENCHMARKS, load_dataset
from coldstart.runtime import Deadline, register_job
from coldstart.splits import dataset_fingerprint, load_family_table, make_split, save_split


def write_reports(processed: Path, report_dir: Path, family_path: Path, check=lambda: None) -> dict:
    metadata = json.loads((processed / "embedllm.json").read_text(encoding="utf-8"))
    if not metadata["raw_audit_complete"]:
        raise ValueError("Versionable final audit reports require all source CSVs")
    families = load_family_table(family_path)
    model_ids = sorted(families)
    primary = load_dataset(processed, label_policy="strict", model_ids=model_ids, deduplicate_groups=True)
    before_representative = load_dataset(processed, label_policy="strict", model_ids=model_ids, deduplicate_groups=False)
    all_models = load_dataset(processed, label_policy="strict", deduplicate_groups=False)
    prompts = pd.read_csv(processed / "prompts.csv").sort_values("prompt_id")
    with np.load(processed / "embedllm.npz", allow_pickle=False) as arrays:
        raw_count = arrays["raw_count"]
        category_single = prompts.category_count.to_numpy() == 1
        singleton = (raw_count == 1) & category_single[None, :]
        source_ids = arrays["benchmark_ids"]
        raw_min, raw_max = arrays["raw_label_min"], arrays["raw_label_max"]
        source_model_ids = arrays["model_ids"]
    selected_rows = np.flatnonzero(np.isin(source_model_ids, model_ids))
    primary_sources = set(STRICT_BENCHMARKS)
    source_names = metadata["benchmark_names"]
    five = np.array([source_names[int(value)] in primary_sources for value in source_ids])
    structural = singleton[:, five]
    varying = np.any(structural != structural[0], axis=0)
    domain_records = []
    for benchmark_id, benchmark in enumerate(source_names):
        check()
        selected = source_ids == benchmark_id
        counts = raw_count[:, selected]
        stats = metadata["by_benchmark"][benchmark]
        domain_records.append({"benchmark": benchmark, **stats,
                               "eligible_primary_source": benchmark in primary_sources,
                               "singleton_common_audited_models": int(singleton[selected_rows][:, selected].all(axis=0).sum()),
                               "singleton_pattern_varies_across_models_questions": int(np.any(singleton[:, selected] != singleton[0, selected], axis=0).sum()),
                               "primary_before_group_representative": int(np.sum(before_representative.benchmark_ids == before_representative.benchmark_names.index(benchmark))) if benchmark in before_representative.benchmark_names else 0,
                               "primary_representatives": int(np.sum(primary.benchmark_ids == primary.benchmark_names.index(benchmark))) if benchmark in primary.benchmark_names else 0,
                               "representative_dropped": primary.metadata["representative_dropped_counts_by_benchmark"].get(benchmark, 0)})
    model_records = [{"model_id": int(source_model_ids[row]), "model_name": metadata["model_names"][row],
                      "audited_family_inclusion": int(source_model_ids[row]) in families,
                      "main_five_singleton_questions": int(singleton[row, five].sum()),
                      "main_five_non_singleton_questions": int(np.sum(five) - singleton[row, five].sum())}
                     for row in range(len(source_model_ids))]
    summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
               "source_sha256": metadata["source_sha256"], "prepared_sha256": metadata["prepared_sha256"],
               "raw_audit_complete": True, "models": metadata["models"], "prompt_ids": metadata["prompts"],
               "raw_rows": metadata["raw_rows"], "model_prompt_cells": int(raw_count.size),
               "raw_missing_cells": metadata["missing_cells"], "duplicate_cells": metadata["duplicate_cells"],
               "conflicting_cells": metadata["conflicting_cells"], "condition_audit": metadata["condition_audit"],
               "published_equals_raw_max_on_all_cells": metadata["published_equals_raw_max_on_observed_cells"],
               "published_raw_max_mismatch_cells": metadata["published_raw_max_mismatch_cells"],
               "splits": metadata["splits"], "grouping": metadata["grouping"],
               "strict_five": {"benchmarks": list(STRICT_BENCHMARKS), "audited_models": len(model_ids),
                               "total_prompt_ids_before_structure": int(five.sum()),
                               "singleton_common_all112": all_models.n_questions,
                               "singleton_common_audited_models": before_representative.n_questions,
                               "singleton_pattern_identical_all112": bool(not varying.any()),
                               "questions_with_model_varying_singleton_pattern": int(varying.sum()),
                               "raw_conflicting_cells_before_structure": int(((raw_min[:, five] != raw_max[:, five]) & (raw_count[:, five] > 0)).sum()),
                               "representative_questions": primary.n_questions,
                               "representative_removed": before_representative.n_questions - primary.n_questions,
                               "representative_removed_by_benchmark": primary.metadata["representative_dropped_counts_by_benchmark"],
                               "dataset_fingerprint": dataset_fingerprint(primary),
                               "selection_is_value_independent": "Raw row counts, task categories, declared source/family inclusion and question text only; never correctness values or method performance."},
               "target_definitions": metadata["target_definitions"], "limitations": metadata["limitations"] +
               ["Retrospective structural inclusion is known for all released models; future model collection must enforce the same recording protocol.",
                "Full raw support does not recover generation failure logs, metric/filter identity, or official upstream correctness.",
                "Identical generic stems can be overgrouped; representative selection changes the target to a candidate-base-item representative subset."]}
    selection = {"schema_version": 1, "label_policy": "strict", "dataset_fingerprint": dataset_fingerprint(primary),
                 "source_revision": SOURCE_REVISION, "source_model_ids": primary.model_ids.tolist(),
                 "source_prompt_ids": primary.prompt_ids.tolist(),
                 "benchmarks": list(primary.benchmark_names), "representative_rule": primary.metadata["representative_rule"],
                 "dropped_prompt_ids_by_benchmark": primary.metadata["representative_dropped_prompt_ids_by_benchmark"],
                 "information_protocol": "strict_unseen_item", "grouping_version": metadata["grouping"]["version"]}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "primary_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(domain_records).to_csv(report_dir / "domains.csv", index=False)
    pd.DataFrame(model_records).to_csv(report_dir / "model_structure.csv", index=False)
    manifest = make_split(primary)
    manifest.metadata.update({"label_policy": "strict", "deduplicate_groups": True,
                              "primary_selection_fingerprint": dataset_fingerprint(primary)})
    save_split(report_dir / "split_strict_random.json", primary, manifest)
    source_manifest = {"schema_version": 1, "repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
                       "url": f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}/tree/{SOURCE_REVISION}",
                       "files": EXPECTED_FILES,
                       "redistribution_note": "Repository stores source checksums, IDs and aggregate audit counts; downloaded prompt text and labels remain local."}
    (ROOT / "data" / "metadata" / "dataset_source.json").write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "data_audit")
    parser.add_argument("--family-table", type=Path, default=ROOT / "data" / "metadata" / "model_families.csv")
    parser.add_argument("--deadline", default=None, help="Optional ISO-8601 stop time, including timezone")
    args = parser.parse_args()
    deadline = Deadline(args.deadline)
    with register_job(deadline=args.deadline, metadata={"task": "sanitized_data_audit_report"}):
        report = write_reports(args.processed_dir, args.output_dir, args.family_table, deadline.check)
        print(json.dumps(report["strict_five"], indent=2), flush=True)


if __name__ == "__main__":
    main()
