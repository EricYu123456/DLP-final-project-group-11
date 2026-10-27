"""Run the predeclared validation-only architecture/regularization search."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from coldstart.data import load_dataset
from coldstart.runtime import Deadline, register_job
from coldstart.splits import dataset_fingerprint, load_family_table, load_split, make_split, save_split
from run_experiment import (
    VARIANTS, atomic_csv, atomic_json, checkpoint_path, config_hash, log,
    scientific_source_hashes, sha256, train_models, tune,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/protocol.json")
    parser.add_argument("--output", default="reports/development")
    parser.add_argument("--artifacts", default="artifacts/development")
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--deadline", help="Optional timezone-aware ISO deadline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    deadline = Deadline(args.deadline)
    deadline.check()
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_config = base["data"]
    model_ids = data_config.get("model_ids")
    if model_ids is None and data_config.get("model_scope") == "audited_lineage_included_only":
        model_ids = sorted(load_family_table(data_config.get("family_table", data_config.get("model_family_table", "data/metadata/model_families.csv"))))
    dataset = load_dataset(data_config["prepared_path"], label_policy=data_config["label_policy"],
        model_ids=model_ids, benchmarks=data_config.get("candidate_benchmarks"),
        deduplicate_groups=data_config.get("deduplicate_groups"))
    prepared = Path(data_config["prepared_path"])
    prepared = prepared / "embedllm.npz" if prepared.is_dir() else prepared
    actual_hash = sha256(prepared)
    if dataset.metadata.get("prepared_sha256") not in (None, actual_hash):
        raise ValueError("Prepared dataset checksum mismatch")
    dataset.metadata["verified_prepared_sha256"] = actual_hash
    output, artifacts = Path(args.output), Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    if args.split:
        split = load_split(args.split, dataset)
    else:
        split = make_split(dataset, item_seed=base["split"].get("item_seed", 1729),
            model_seed=base["split"].get("model_seed", 2718),
            item_ratios=base["split"].get("item_ratios", [0.7, 0.15, 0.15]),
            model_ratios=base["split"].get("model_ratios", [0.7, 0.15, 0.15]),
            protocol=base["experiment"]["regime"])
    variants = list(VARIANTS)
    plan = {
        "status": "pre_test_validation_only", "seed": args.seed,
        "hidden_dim": [32, 128], "learning_rate": [0.001, 0.002], "latent_l2": [0.0001, 0.002],
        "epochs": 100, "patience_evaluated_checkpoints": 10, "warmup_epochs": 0,
        "checkpoint_selection": "minimum heldout-model macro MAE across prior strengths [0.3,1,3] at balanced-random k100; epoch1 and every2epochs",
        "shared_config_selection": "mean validation macro MAE across all three variants and budgets50/100/200, after each variant's prior tuning",
        "tie_break": "smaller hidden layer, then stronger latent regularization, then smaller learning rate",
        "test_labels_used": False, "dataset_fingerprint": dataset_fingerprint(dataset),
        "prepared_data_sha256": actual_hash, "source_sha256": scientific_source_hashes(),
        "grid_script_sha256": sha256(Path(__file__).resolve()),
        "command": sys.argv,
    }
    atomic_json(output / "search_plan.json", plan)
    records = []
    candidates = []
    started = time.perf_counter()
    with register_job(deadline=args.deadline, metadata={"phase": "validation_only_development_grid", "output": str(output.resolve())}):
        for hidden, learning_rate, latent_l2 in itertools.product([32, 128], [0.001, 0.002], [0.0001, 0.002]):
            deadline.check()
            identifier = f"h{hidden}_lr{learning_rate:g}_l2{latent_l2:g}"
            config = deepcopy(base)
            config["experiment"]["name"] = base["experiment"]["name"] + "_development_" + identifier
            config["seeds"] = [args.seed]
            config["adaptation"]["prior_strengths"] = base.get("development_search", {}).get(
                "prior_strengths_during_search", [0.3, 1.0, 3.0])
            config["model"].update(hidden_dim=hidden, learning_rate=learning_rate, latent_l2=latent_l2,
                epochs=100, patience=10, warmup_epochs=0, selection_metric="coldstart_macro_mae",
                selection_interval=2, selection_budget=100, selection_prior_strengths=[0.3, 1.0, 3.0])
            config["tuning"]["encoder_stopping_models"] = "val_models"
            config["tuning"]["encoder_stopping_metric"] = "coldstart_macro_mae_min_prior_strength"
            run_output = artifacts / identifier
            run_output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / "configs" / f"{identifier}.json", config)
            atomic_json(run_output / "effective_config.json", config)
            save_split(run_output / "split.json", dataset, split)
            log(f"development config={identifier}")
            trained = train_models(dataset, split, config, run_output, variants, deadline, args.device, args.resume)
            selection = tune(dataset, split, config, run_output, variants, deadline, args.device)
            scores = []
            for variant in variants:
                strength = selection["prior_strengths"][variant]
                score = selection["validation_summary"]["priors"][variant][str(strength)]
                scores.append(score)
                metadata = trained[f"{variant}/{args.seed}"]
                records.append({"configuration": identifier, "variant": variant,
                    "hidden_dim": hidden, "learning_rate": learning_rate, "latent_l2": latent_l2,
                    "seed": args.seed, "validation_macro_mae": score, "selected_prior_strength": strength,
                    "best_epoch": metadata["best_epoch"], "best_selection_value": metadata["best_selection_value"],
                    "best_validation_bce": metadata["best_validation_bce"],
                    "checkpoint_sha256": sha256(checkpoint_path(run_output, variant, args.seed)),
                    "config_sha256": config_hash(config)})
            candidate = {"configuration": identifier, "hidden_dim": hidden,
                "learning_rate": learning_rate, "latent_l2": latent_l2,
                "shared_validation_macro_mae": float(np.mean(scores)), "variant_scores": dict(zip(variants, scores)),
                "config_sha256": config_hash(config), "artifacts": str(run_output.resolve())}
            candidates.append(candidate)
            atomic_csv(output / "validation_grid.csv", records)
            atomic_json(output / "configuration_scores.json", candidates)
            log(f"development {identifier} shared_validation_macro_mae={candidate['shared_validation_macro_mae']:.6f}")
        ranked = sorted(candidates, key=lambda item: (item["shared_validation_macro_mae"], item["hidden_dim"], -item["latent_l2"], item["learning_rate"]))
        selected = ranked[0]
        final_config = deepcopy(base)
        final_config["model"].update(hidden_dim=selected["hidden_dim"], learning_rate=selected["learning_rate"],
            latent_l2=selected["latent_l2"], epochs=100, patience=10, warmup_epochs=0,
            selection_metric="coldstart_macro_mae", selection_interval=2,
            selection_budget=100, selection_prior_strengths=[0.3, 1.0, 3.0])
        final_config["tuning"]["encoder_stopping_models"] = "val_models"
        final_config["tuning"]["encoder_stopping_metric"] = "coldstart_macro_mae_min_prior_strength"
        final_config["tuning"]["shared_model_config_selection"] = "reports/development/selection.json"
        final_config["experiment"]["status"] = "development_selected_pre_test"
        decision = {"selected": selected, "ranked": ranked,
            "per_variant_sensitivity": {variant: min((row for row in records if row["variant"] == variant),
                key=lambda row: row["validation_macro_mae"]) for variant in variants},
            "selection_rule": plan["shared_config_selection"], "test_labels_used": False,
            "elapsed_seconds": time.perf_counter() - started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "dataset_fingerprint": dataset_fingerprint(dataset), "source_sha256": scientific_source_hashes()}
        atomic_json(output / "selection.json", decision)
        atomic_json(output / "selected_config.json", final_config)
        log(f"DEVELOPMENT SELECTED {selected['configuration']} shared_val_MAE={selected['shared_validation_macro_mae']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
