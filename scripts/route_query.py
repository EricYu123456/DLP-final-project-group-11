"""Fit a candidate embedding or select a model using a frozen decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldstart.adaptation import adapt_map, sigmoid
from coldstart.models import load_decoder
from embed_new_model import load_run, read_responses
from run_experiment import checkpoint_path, sha256


VARIANT = "neural_lowrank"
LEARNING_SELECTION = ROOT / "reports" / "routing" / "learning" / "selection.json"


def choose_models(scores: np.ndarray, minimum_gain: float | None = None) -> np.ndarray:
    """Pick independently for each query, with earlier rows winning ties.

    Scores have shape [candidates] for one query or [candidates, queries].
    A two-candidate probability margin treats the first row as the incumbent.
    """
    if minimum_gain is None:
        return scores.argmax(axis=0)
    if scores.shape[0] != 2:
        raise ValueError("A probability margin requires one incumbent and one newcomer.")
    return (scores[1] - scores[0] > minimum_gain).astype(np.int64)


def interaction_scores(latents: np.ndarray, query_design: np.ndarray) -> np.ndarray:
    """Rank fitted model vectors by their learned interactions with a query."""
    return np.asarray(latents, dtype=np.float64) @ np.asarray(query_design, dtype=np.float64).T


def routing_features(decoder, vectors: np.ndarray, benchmark: int | None = None):
    """Use the mean learned benchmark bias when a prompt has no benchmark label."""
    domains = np.full(len(vectors), 0 if benchmark is None else benchmark, dtype=np.int64)
    offsets, design = decoder.export_features(vectors, domains)
    if benchmark is None:
        bias = decoder.benchmark_bias.detach().cpu().numpy().astype(np.float64)
        offsets = offsets.astype(np.float64) + (bias.mean() - bias[0])
    return offsets, design


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts" / "main_v2")
    parser.add_argument("--seed", type=int, help="Legacy decoder seed (default: 11).")
    parser.add_argument("--checkpoint", type=Path, help="Frozen decoder file. Uses its recorded seed.")
    parser.add_argument("--learning-selection", type=Path, default=LEARNING_SELECTION,
                        help="Selection manifest for an older routing-learning checkpoint.")


def decoder_context(args: argparse.Namespace):
    seed = 11 if args.seed is None else args.seed
    path = args.checkpoint or checkpoint_path(args.run, VARIANT, seed)
    decoder, metadata = load_decoder(path)
    if decoder.config.mode != "lowrank" or decoder.item_bias is not None:
        raise ValueError("Routing requires a low-rank decoder for unseen questions.")
    if args.seed is not None and args.seed != metadata["seed"]:
        raise ValueError("--seed differs from the checkpoint's recorded seed.")
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    training_ids = split["train_model_ids"]
    if decoder.config.num_models != len(training_ids):
        raise ValueError("Checkpoint model count differs from the run's training roster.")
    if "training_model_ids" in metadata:
        if (metadata["training_model_ids"] != training_ids
                or metadata["training_prompt_ids"] != split["train_prompt_ids"]
                or metadata["dataset_fingerprint"] != split["dataset_fingerprint"]):
            raise ValueError("Checkpoint metadata differs from the run's training split.")
        prior_strength = metadata.get("prior_strength")
    else:
        selection = json.loads(args.learning_selection.read_text(encoding="utf-8"))
        protocol = selection["protocol"]
        support_ids = set(split["train_prompt_ids"])
        probes = {prompt for order in protocol["probe_prompt_ids"].values() for prompt in order}
        if (metadata not in selection["checkpoints"]
                or protocol["training_model_ids"] != training_ids
                or protocol["validation_model_ids"] != split["val_model_ids"]
                or protocol["full_support_budget"] != len(support_ids)
                or not (probes | set(protocol["embedding_mean_reference_prompt_ids"])) <= support_ids):
            raise ValueError("Checkpoint and run do not match the routing-learning selection.")
        prior_strength = protocol["prior_strength"]
    return path, decoder, metadata["seed"], training_ids, list(split["train_benchmark_counts"]), prior_strength


def fit(args: argparse.Namespace) -> dict:
    path, decoder, seed, training_ids, _, strength = decoder_context(args)
    dataset, split, config, lock, _ = load_run(args.run, VARIANT, seed)
    if training_ids != dataset.model_ids[split.train_models].tolist():
        raise ValueError("Checkpoint row order differs from the validated support run.")
    prompt_ids, responses = read_responses(args.responses, dataset.prompt_ids[split.probe_items])
    positions = {int(prompt): index for index, prompt in enumerate(dataset.prompt_ids)}
    selected = np.array([positions[int(prompt)] for prompt in prompt_ids])
    offsets, design = decoder.export_features(dataset.embeddings[selected], dataset.benchmark_ids[selected])
    mean, precision = decoder.latent_prior(
        variance_floor=config["model"]["prior_scale_floor"],
        variance_shrinkage=config["model"]["prior_variance_shrinkage"])
    strength = lock["selection"]["prior_strengths"][VARIANT] if strength is None else strength
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    result = adapt_map(design, offsets, responses, mean, strength * precision, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, latent=result.latent, model_name=np.array(args.model_name),
                            checkpoint_sha256=np.array(sha256(path)), seed=np.array(seed),
                            prompt_ids=prompt_ids, variant=np.array(VARIANT))
    return {"embedding": str(args.output), "model_name": args.model_name,
            "latent_dimension": len(result.latent), "acquired_responses": len(responses),
            "observed_responses": result.observed_responses, "converged": result.converged}


def fit_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="route_query.py fit", description="Fit a model from labeled probes.")
    add_run_options(parser)
    parser.add_argument("--responses", type=Path, required=True, help="CSV with prompt_id and binary response columns.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output embedding .npz file.")
    args = parser.parse_args(argv)
    try:
        result = fit(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fit"]:
        return fit_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use route_query.py fit --help to fit a candidate.")
    add_run_options(parser)
    parser.add_argument("--trained-model", type=int, action="append", default=[],
                        help="Source model ID from the checkpoint's training set. Repeat for more.")
    parser.add_argument("--candidate-embedding", type=Path, action="append", default=[],
                        help="Embedding fitted to this decoder. Repeat for more.")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--question-vector", type=Path, help="An all-mpnet-base-v2 question vector in .npy format.")
    query.add_argument("--prompt", help="Prompt text. Requires the optional sentence-transformers package.")
    query.add_argument("--stdin", action="store_true", help="Read one prompt per nonempty line and print one JSON choice.")
    parser.add_argument("--benchmark", help="Known benchmark name, used to compute correctness probabilities.")
    parser.add_argument("--minimum-gain", type=float, default=0.0,
                        help="Minimum probability gain required to replace the first candidate. Uses the mean learned "
                             "benchmark bias when the prompt has no benchmark label.")
    args = parser.parse_args(argv)
    if not args.trained_model and not args.candidate_embedding:
        parser.error("Supply at least one trained model or candidate embedding.")
    if not 0 <= args.minimum_gain <= 1:
        parser.error("--minimum-gain must be between 0 and 1.")
    if args.minimum_gain and len(args.trained_model) + len(args.candidate_embedding) != 2:
        parser.error("A positive --minimum-gain requires exactly two candidates.")

    try:
        checkpoint, decoder, _, training_ids, benchmarks, _ = decoder_context(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    training_rows = {model_id: index for index, model_id in enumerate(training_ids)}
    weights = decoder.model_latents.weight.detach().cpu().numpy()
    candidate_latents, candidates = [], []
    for model_id in args.trained_model:
        if model_id not in training_rows:
            parser.error(f"Model {model_id} is outside the decoder's training set.")
        candidates.append({"model_id": model_id})
        candidate_latents.append(weights[training_rows[model_id]])
    if args.candidate_embedding:
        decoder_digest = sha256(checkpoint)
        for path in args.candidate_embedding:
            with np.load(path, allow_pickle=False) as fit:
                if str(fit["checkpoint_sha256"].item()) != decoder_digest:
                    parser.error(f"{path} was fitted to a different decoder.")
                candidate_latents.append(fit["latent"])
                candidates.append({"model_name": str(fit["model_name"].item())})

    domain = None
    if args.benchmark is not None:
        if args.benchmark not in benchmarks:
            parser.error(f"Unknown benchmark {args.benchmark}. Available benchmarks are {', '.join(benchmarks)}.")
        domain = benchmarks.index(args.benchmark)

    def route(vector: np.ndarray) -> dict:
        if vector.shape != (decoder.config.input_dim,):
            parser.error(f"Supply one flat question vector with {decoder.config.input_dim} features.")
        offset, design = routing_features(decoder, vector[None, :], domain)
        scores = interaction_scores(candidate_latents, design[0])
        result_candidates = candidates
        if domain is not None or args.minimum_gain:
            probabilities = sigmoid(scores + offset[0])
            result_candidates = [{**candidate, "predicted_correctness": float(probability)}
                                 for candidate, probability in zip(candidates, probabilities)]
        choice = choose_models(probabilities, args.minimum_gain) if args.minimum_gain else choose_models(scores)
        return {"selected": result_candidates[int(choice)], "candidates": result_candidates}

    if args.question_vector is not None:
        result = route(np.load(args.question_vector, allow_pickle=False))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            parser.error("Text prompts require sentence-transformers. Use --question-vector for saved features.")
        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        if args.stdin:
            for line in sys.stdin:
                prompt = line.rstrip("\r\n")
                if prompt.strip():
                    print(json.dumps(route(encoder.encode(prompt)), ensure_ascii=False), flush=True)
        else:
            print(json.dumps(route(encoder.encode(args.prompt)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
