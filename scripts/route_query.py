"""Select one model for a prompt using a frozen EmbedLLM-style decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldstart.adaptation import sigmoid
from coldstart.models import load_decoder
from run_experiment import checkpoint_path, sha256


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts" / "main_v2")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--trained-model", type=int, action="append", default=[],
                        help="Source model ID from the checkpoint's training set. Repeat for more.")
    parser.add_argument("--candidate-embedding", type=Path, action="append", default=[],
                        help="Embedding from embed_new_model.py fitted to this decoder. Repeat for more.")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--question-vector", type=Path, help="An all-mpnet-base-v2 question vector in .npy format.")
    query.add_argument("--prompt", help="Prompt text. Requires the optional sentence-transformers package.")
    parser.add_argument("--benchmark", help="Known benchmark name, used to compute correctness probabilities.")
    parser.add_argument("--minimum-gain", type=float, default=0.0,
                        help="Minimum probability gain required to replace the first candidate. Requires --benchmark.")
    args = parser.parse_args(argv)
    if not args.trained_model and not args.candidate_embedding:
        parser.error("Supply at least one trained model or candidate embedding.")
    if not 0 <= args.minimum_gain <= 1:
        parser.error("--minimum-gain must be between 0 and 1.")
    if args.minimum_gain and args.benchmark is None:
        parser.error("A positive --minimum-gain requires --benchmark.")

    checkpoint = checkpoint_path(args.run, "neural_lowrank", args.seed)
    decoder, metadata = load_decoder(checkpoint)
    if decoder.config.mode != "lowrank" or decoder.item_bias is not None:
        parser.error("Routing requires a low-rank decoder for unseen questions.")

    training_rows = {model_id: index for index, model_id in enumerate(metadata["training_model_ids"])}
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

    if args.prompt is not None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            parser.error("--prompt requires sentence-transformers. Supply --question-vector to use a saved vector.")
        vector = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(args.prompt)
    else:
        vector = np.load(args.question_vector, allow_pickle=False)
    if vector.shape != (decoder.config.input_dim,):
        parser.error(f"Supply one flat question vector with {decoder.config.input_dim} features.")

    domain = 0
    if args.benchmark is not None:
        split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
        benchmarks = list(split["train_benchmark_counts"])
        if args.benchmark not in benchmarks:
            parser.error(f"Unknown benchmark {args.benchmark}. Available benchmarks are {', '.join(benchmarks)}.")
        domain = benchmarks.index(args.benchmark)
    offset, design = decoder.export_features(vector[None, :], np.array([domain]))
    scores = interaction_scores(candidate_latents, design[0])
    if args.benchmark is not None:
        probabilities = sigmoid(scores + offset[0])
        for candidate, probability in zip(candidates, probabilities):
            candidate["predicted_correctness"] = float(probability)
    if args.minimum_gain:
        choice = choose_models(probabilities, args.minimum_gain)
    else:
        choice = choose_models(scores)
    print(json.dumps({"selected": candidates[int(choice)], "candidates": candidates},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
