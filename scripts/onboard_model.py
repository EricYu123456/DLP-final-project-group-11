"""Request the next onboarding prompt or stop and save a model embedding."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from route_query import LEARNING_SELECTION, ROOT, VARIANT
from embed_new_model import read_prompts
from run_experiment import sha256
from run_sequential_routing import B, input_context, onboarding_context, probe_steps, read_json


def read_history(path: Path | None) -> list[tuple[int, int]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"prompt_id", "response"} <= set(reader.fieldnames or ()):
            raise ValueError("History requires prompt_id and response columns.")
        history = []
        for row in reader:
            if (row["response"] or "").strip() not in {"0", "1"}:
                raise ValueError(f"Supply a binary 0 or 1 response on history line {reader.line_num}.")
            history.append((int(row["prompt_id"]), int(row["response"])))
        return history


def replay(steps, candidate_ids, history):
    position = 0
    try:
        local = next(steps)
        for position, (prompt_id, response) in enumerate(history, start=1):
            expected = int(candidate_ids[local])
            if prompt_id != expected:
                raise ValueError(f"History response {position} is for prompt {prompt_id}, expected {expected}.")
            local = steps.send(response)
    except StopIteration as done:
        if position != len(history):
            raise ValueError(f"The policy stopped after response {position}. Remove later responses.")
        return None, done.value
    return local, None


def onboard(args):
    history = read_history(args.responses)
    dataset, split, config, _, _, _ = input_context(args.run)
    learned = read_json(args.learning_selection)
    selection = read_json(args.selection)
    protocol = selection["protocol"]
    if args.seed not in protocol["seeds"] or protocol["checkpoint_arm"] != B:
        raise ValueError("Requested seed or checkpoint arm is absent from the sequential selection.")
    if (protocol["validation_model_ids"] != dataset.model_ids[split.val_models].tolist()
            or protocol["test_model_ids"] != dataset.model_ids[split.test_models].tolist()):
        raise ValueError("The sequential selection belongs to a different run split.")
    positions = {int(prompt): int(index) for prompt, index in zip(dataset.prompt_ids[split.train_items], split.train_items)}
    reference = np.array([positions[prompt] for prompt in protocol["reference_prompt_ids"]])
    candidate_ids = np.array(protocol["candidate_prompt_ids"])
    candidates = np.array([positions[int(prompt)] for prompt in candidate_ids])
    incumbent = int(np.flatnonzero(dataset.model_ids == protocol["incumbent_model_id"])[0])
    if incumbent not in split.train_models:
        raise ValueError("The incumbent must be part of the established roster.")
    checkpoint = args.checkpoints / f"{B}_{args.seed}.pt"
    _, incumbent_latent, context = onboarding_context(dataset, split, config, learned, incumbent, reference,
                                                      candidates, args.seed, checkpoint)
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    rng = np.random.SeedSequence([args.seed, args.episode_id])
    margin, tolerance = selection["admission_margin"], selection["evpi_tolerance"]
    steps = probe_steps(context["random_order"], context["probe_domains"], context["probe_design"],
                        context["probe_offsets"], context["reference"], context["reference_incumbent"],
                        context["prior"], margin, options, rng, tolerance)
    probe, stopped = replay(steps, candidate_ids, history)
    common = {"model_name": args.model_name, "episode_id": args.episode_id, "seed": args.seed,
              "responses_received": len(history)}
    if stopped is None:
        prompt_id = int(candidate_ids[probe])
        prompt = read_prompts(np.array([prompt_id]), args.question_source)[prompt_id]
        benchmark = dataset.benchmark_names[context["probe_domains"][probe]]
        return {**common, "status": "next_probe", "prompt_id": prompt_id, "benchmark": benchmark, "prompt": prompt}
    latents, trace, selected, reason = stopped
    args.embedding.parent.mkdir(parents=True, exist_ok=True)
    shared = {"checkpoint_sha256": np.array(sha256(checkpoint)), "seed": np.array(args.seed),
              "variant": np.array(VARIANT)}
    with args.embedding.open("wb") as handle:
        np.savez_compressed(handle, latent=latents[-1], model_name=np.array(args.model_name),
                            prompt_ids=candidate_ids[selected], **shared)
    incumbent_embedding = args.embedding.with_name(args.embedding.stem + ".incumbent.npz")
    with incumbent_embedding.open("wb") as handle:
        np.savez_compressed(handle, latent=incumbent_latent, model_name=np.array(dataset.model_names[incumbent]),
                            prompt_ids=dataset.prompt_ids[split.train_items], **shared)
    return {**common, "status": "stop", "reason": reason, "reference_evpi": trace[-1][0],
            "embedding": str(args.embedding), "checkpoint": str(checkpoint),
            "incumbent_model_id": protocol["incumbent_model_id"], "incumbent_embedding": str(incumbent_embedding),
            "minimum_gain": margin}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Start with an empty history. Append each returned prompt_id and its binary correctness "
        "to the CSV in order, then rerun with the same episode ID and seed. The command never calls a model."))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--episode-id", type=int, required=True, help="Stable nonnegative ID for the shortlist sequence.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--responses", type=Path, help="CSV containing chronological prompt_id,response pairs.")
    parser.add_argument("--embedding", type=Path, required=True,
                        help="Newcomer .npz path. On stop, the incumbent .npz is saved beside it.")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--selection", type=Path, default=ROOT / "reports/routing/sequential/validation_selection.json")
    parser.add_argument("--learning-selection", type=Path, default=LEARNING_SELECTION)
    parser.add_argument("--question-source", type=Path, default=ROOT / "data/raw/question_order.csv")
    args = parser.parse_args(argv)
    if args.episode_id < 0:
        parser.error("--episode-id must be nonnegative.")
    try:
        with threadpool_limits(limits=1):
            result = onboard(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
