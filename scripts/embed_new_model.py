"""Plan probes and infer a reusable embedding for an external language model.

No LLM API is called. A user supplies binary correctness for the released
prompts; compatibility with the unavailable original scoring setup is unverified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from coldstart.adaptation import adapt_map, predict_from_latent
from coldstart.data import load_dataset
from coldstart.models import load_decoder
from coldstart.probes import STRATEGIES, select_probe_order
from coldstart.runtime import Deadline, atomic_replace, register_job
from coldstart.splits import load_family_table, load_split
from run_experiment import VARIANTS, atomic_json, checkpoint_path, config_hash, sha256, verify_lock


def load_run(run: Path, variant: str, seed: int | None):
    config = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    seed = config["seeds"][0] if seed is None else seed
    data_path = config["data"]["prepared_path"]
    requested_models = config["data"].get("model_ids")
    if requested_models is None and config["data"].get("model_scope") == "audited_lineage_included_only":
        requested_models = sorted(load_family_table(config["data"].get("family_table", config["data"].get("model_family_table", "data/metadata/model_families.csv"))))
    dataset = load_dataset(data_path, label_policy=config["data"]["label_policy"], model_ids=requested_models,
                           benchmarks=config["data"].get("candidate_benchmarks"),
                           deduplicate_groups=config["data"].get("deduplicate_groups"))
    source = Path(data_path) / "embedllm.npz" if Path(data_path).is_dir() else Path(data_path)
    dataset.metadata["verified_prepared_sha256"] = sha256(source)
    if dataset.metadata.get("prepared_sha256") not in (None, dataset.metadata["verified_prepared_sha256"]):
        raise ValueError("Prepared dataset checksum differs from metadata")
    lock = json.loads((run / "test_lock.json").read_text(encoding="utf-8"))
    verify_lock(dataset, config, run, lock["variants"])
    if variant not in lock["variants"] or seed not in config["seeds"]:
        raise ValueError("Requested variant/seed is absent from the validated run")
    split = load_split(run / "split.json", dataset)
    return dataset, split, config, lock, int(seed)


def read_prompts(prompt_ids: np.ndarray, source: Path) -> dict[int, str]:
    wanted = set(int(value) for value in prompt_ids)
    result = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt_id = int(row["prompt_id"])
            if prompt_id in wanted:
                result[prompt_id] = row["prompt"]
                if len(result) == len(wanted):
                    break
    if set(result) != wanted:
        raise ValueError("Some selected prompts are missing from question_order.csv")
    return result


def plan(args: argparse.Namespace, deadline: Deadline) -> dict:
    run = Path(args.run).resolve()
    dataset, split, config, lock, seed = load_run(run, args.variant, args.seed)
    indices = select_probe_order(dataset.embeddings, dataset.benchmark_ids, split.probe_items,
        args.budget, args.strategy, seed, device="cpu", check_deadline=deadline.check)
    prompt_ids = dataset.prompt_ids[indices]
    prompts = read_prompts(prompt_ids, Path(args.question_source))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_id", "benchmark", "prompt", "response"])
        writer.writeheader()
        for index in indices:
            writer.writerow({"prompt_id": int(dataset.prompt_ids[index]),
                "benchmark": dataset.benchmark_names[int(dataset.benchmark_ids[index])],
                "prompt": prompts[int(dataset.prompt_ids[index])], "response": ""})
    atomic_replace(temporary, output)
    metadata = {"run": str(run), "variant": args.variant, "seed": seed,
                "strategy": args.strategy, "budget": args.budget,
                "prompt_ids": prompt_ids.tolist(), "response_csv": str(output),
                "response_format": "0 for incorrect, 1 for correct; blank/NaN consumes a probe but adds no label",
                "checkpoint_sha256": sha256(checkpoint_path(run, args.variant, seed)),
                "config_sha256": config_hash(config)}
    atomic_json(output.with_suffix(".plan.json"), metadata)
    return metadata


def read_responses(path: Path, eligible_prompt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prompt_ids, responses = [], []
    eligible = set(int(value) for value in eligible_prompt_ids)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "prompt_id" not in reader.fieldnames:
            raise ValueError("Response CSV requires prompt_id and response (or label) columns")
        response_key = "response" if "response" in reader.fieldnames else "label"
        if response_key not in reader.fieldnames:
            raise ValueError("Response CSV requires a response or label column")
        for row in reader:
            prompt_id = int(row["prompt_id"])
            if prompt_id not in eligible:
                raise ValueError(f"Prompt {prompt_id} is outside the validated support pool")
            if prompt_id in prompt_ids:
                raise ValueError(f"Repeated response for prompt {prompt_id}")
            text = row[response_key].strip()
            response = float(text) if text else float("nan")
            if np.isinf(response) or (np.isfinite(response) and response not in (0.0, 1.0)):
                raise ValueError("Responses must be 0, 1, blank, or NaN")
            prompt_ids.append(prompt_id)
            responses.append(response)
    values = np.asarray(responses, dtype=np.float64)
    if not len(values) or not np.any(np.isfinite(values)):
        raise ValueError("Supply at least one observed binary response")
    return np.asarray(prompt_ids, dtype=np.int64), values


def infer(args: argparse.Namespace, deadline: Deadline) -> dict:
    started = time.perf_counter()
    run = Path(args.run).resolve()
    dataset, split, config, lock, seed = load_run(run, args.variant, args.seed)
    prompt_ids, responses = read_responses(Path(args.responses), dataset.prompt_ids[split.probe_items])
    positions = {int(value): position for position, value in enumerate(dataset.prompt_ids)}
    selected = np.array([positions[int(value)] for value in prompt_ids], dtype=np.int64)
    checkpoint = checkpoint_path(run, args.variant, seed)
    decoder, _ = load_decoder(checkpoint, "cpu")
    mean, base_precision = decoder.latent_prior(
        variance_floor=config["model"].get("prior_scale_floor", 0.2),
        variance_shrinkage=config["model"].get("prior_variance_shrinkage", 0.2))
    strength = lock["selection"]["prior_strengths"][args.variant]
    precision = strength * base_precision
    support_offset, support_design = decoder.export_features(dataset.embeddings[selected],
        dataset.benchmark_ids[selected], selected if decoder.item_bias is not None else None)
    fit = adapt_map(support_design, support_offset, responses, mean, precision,
        max_iterations=config["adaptation"].get("max_iterations", 25),
        tolerance=config["adaptation"].get("tolerance", 1e-7), check_deadline=deadline.check)
    # Query text is encoded only after all support responses and MAP are fixed.
    target = split.test_items
    target_offset, target_design = decoder.export_features(dataset.embeddings[target],
        dataset.benchmark_ids[target], target if decoder.item_bias is not None else None)
    probabilities = predict_from_latent(fit.latent, target_offset, target_design)
    scores = []
    for benchmark, name in enumerate(dataset.benchmark_names):
        group = dataset.benchmark_ids[target] == benchmark
        scores.append({"benchmark": name, "predicted_score": float(probabilities[group].mean()),
                       "questions": int(group.sum())})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    embedding_path = output / "embedding.npz"
    np.savez_compressed(embedding_path, latent=fit.latent, covariance=fit.covariance,
        prior_mean=mean, prior_precision=precision, prompt_ids=prompt_ids, responses=responses,
        checkpoint_sha256=np.array(sha256(checkpoint)), model_name=np.array(args.model_name),
        variant=np.array(args.variant), seed=np.array(seed),
        target_prompt_ids=dataset.prompt_ids[target], probabilities=probabilities)
    metadata = {"model_name": args.model_name, "variant": args.variant,
        "latent_dimension": int(len(fit.latent)), "seed": seed,
        "acquired_responses": len(responses), "observed_responses": int(np.isfinite(responses).sum()),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "run": str(run), "config_sha256": config_hash(config), "prior_strength": strength,
        "embedding": str(embedding_path), "fit": fit.metadata(), "benchmark_predictions": scores,
        "score_scope": "Predicted average correctness on the run's held-out benchmark question subset; not a new full official benchmark evaluation.",
        "elapsed_seconds": time.perf_counter() - started}
    atomic_json(output / "prediction.json", metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", help="Optional timezone-aware ISO deadline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "infer"):
        command = subparsers.add_parser(name)
        command.add_argument("--run", required=True, help="Validated experiment output directory")
        command.add_argument("--variant", choices=list(VARIANTS), default="structured8")
        command.add_argument("--seed", type=int)
        command.add_argument("--output", required=True)
        if name == "plan":
            command.add_argument("--strategy", choices=STRATEGIES, default="balanced_random")
            command.add_argument("--budget", type=int, default=50)
            command.add_argument("--question-source", default="data/raw/question_order.csv")
        else:
            command.add_argument("--responses", required=True)
            command.add_argument("--model-name", required=True)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    deadline = Deadline(args.deadline)
    deadline.check()
    with register_job(deadline=args.deadline, metadata={"command": args.command}):
        result = plan(args, deadline) if args.command == "plan" else infer(args, deadline)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
