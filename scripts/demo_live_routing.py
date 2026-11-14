"""Generate real local answers during onboarding and dispatch held-out prompts once."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from route_query import LEARNING_SELECTION, ROOT, choose_models, decoder_context, interaction_scores, routing_features
from coldstart.adaptation import adapt_map, sigmoid
from pilot_routing_learning import latent_prior
from pilot_smooth_probe import DETAIL_TOLERANCE, smooth_path
from run_experiment import atomic_csv, atomic_json
from run_external_routing import ENCODER, read_json, tokens
from run_sequential_routing import B, WARMUP


MODELS = (
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
)
ARC = "allenai/ai2_arc"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
SEED = 11
SUPPORT = 120
REFERENCE = 32
CANDIDATES = 100
QUERIES = 24
DEMO_CAP = 40
GENERATION_BATCH = 16
MAX_NEW_TOKENS = 8
MARGIN = 0.025
PRIOR_STRENGTH = 0.3
LETTERS = "ABCD"
INSTRUCTION = "Choose the correct option. Reply with the option letter only: A, B, C, or D."
ANSWER = re.compile(r"\A\s*(?:Answer:\s*)?(?:\(([A-D])\)|([A-D])(?=[.)\s]|\Z))", re.IGNORECASE)


def arc_items(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "question", "choices", "answerKey"])
    rows, stems = [], Counter()
    for row in table.to_pylist():
        question, choices = row["question"], row["choices"]
        key = " ".join(tokens(question))
        stems[key] += 1
        if len(choices["label"]) != 4 or len(choices["text"]) != 4 or choices["label"].count(row["answerKey"]) != 1:
            continue
        correct = LETTERS[choices["label"].index(row["answerKey"])]
        options = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, choices["text"]))
        rows.append((row["id"], key, f"{question}\n{options}\n\n{INSTRUCTION}", correct))
    return [item for item in rows if item[1] and stems[item[1]] == 1], set(stems)


def select_items(raw):
    from huggingface_hub import hf_hub_download

    paths = {}
    for split in ("train", "test"):
        path = raw / f"arc-{split}.parquet"
        if not path.exists():
            path = hf_hub_download(ARC, f"ARC-Challenge/{split}-00000-of-00001.parquet",
                                   repo_type="dataset", revision=ARC_REVISION)
        paths[split] = path
    train, train_stems = arc_items(paths["train"])
    test, _ = arc_items(paths["test"])
    test = [item for item in test if item[1] not in train_stems]
    rng = np.random.default_rng(SEED)
    selected = [train[index] for index in rng.permutation(len(train))[:SUPPORT + REFERENCE]]
    target = [test[index] for index in rng.permutation(len(test))[:QUERIES]]
    return selected[:SUPPORT], selected[SUPPORT:], target


def answer(text):
    match = ANSWER.match(text)
    return next(group.upper() for group in match.groups() if group) if match else None


class LocalModel:
    def __init__(self, model_id, revision, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.id, self.device = model_id, torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                        use_safetensors=True, trust_remote_code=False)
        self.model.to(self.device).eval()
        self.generated_requests = 0
        self.generate_batches = 0

    @torch.inference_mode()
    def generate(self, prompts):
        rendered = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                       tokenize=False, add_generation_prompt=True)
                    for prompt in prompts]
        encoded = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        generated = self.model.generate(**encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                        pad_token_id=self.tokenizer.pad_token_id)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency = perf_counter() - start
        self.generated_requests += len(prompts)
        self.generate_batches += 1
        return self.tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True), latency


def run(args):
    from sentence_transformers import SentenceTransformer

    support, reference, target = select_items(args.raw)
    decoder_path, decoder, _, _, _, prior_strength = decoder_context(args)
    if prior_strength != PRIOR_STRENGTH:
        raise ValueError("The demo expects the frozen B prior.")
    config = read_json(args.run / "effective_config.json")
    encoder = SentenceTransformer(ENCODER, device=args.device)
    vectors = encoder.encode([item[2] for item in [*support, *reference]], show_progress_bar=False)
    offsets, design = routing_features(decoder, vectors)
    prior = latent_prior(decoder, config, prior_strength)
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    incumbent, newcomer = [LocalModel(model, revision, args.device) for model, revision in MODELS]
    local_trace, outcomes, profile_time = [], [], 0.0

    def generate_rows(model, items, stage):
        generated, seconds = model.generate([item[2] for item in items])
        if len(items) != len(generated):
            raise RuntimeError("Local generation count differs from input prompts.")
        rows = []
        for item, text in zip(items, generated):
            extracted = answer(text)
            entry = {"stage": stage, "item_id": item[0], "model": model.id, "correct": int(extracted == item[3]),
                     "parsed_option": extracted, "official_key": item[3], "generated_answer": text}
            local_trace.append(entry)
            rows.append(entry)
        return rows, seconds

    support_labels = []
    for offset in range(0, SUPPORT, GENERATION_BATCH):
        rows, seconds = generate_rows(incumbent, support[offset:offset + GENERATION_BATCH], "incumbent_support")
        support_labels.extend(row["correct"] for row in rows)
        profile_time += seconds
    incumbent_fit = adapt_map(design[:SUPPORT], offsets[:SUPPORT], np.array(support_labels), *prior, **options)
    reference_design = design[SUPPORT:].astype(np.float64)
    reference_offsets = offsets[SUPPORT:]
    reference_incumbent = sigmoid(reference_offsets + reference_design @ incumbent_fit.latent)
    probes = []
    newcomer_seconds = 0.0

    def observe(index):
        nonlocal newcomer_seconds
        rows, seconds = generate_rows(newcomer, [support[index]], "newcomer_probe")
        newcomer_seconds += seconds
        probes.append({"item_id": rows[0]["item_id"], "correct": rows[0]["correct"], "generation_seconds": seconds})
        return rows[0]["correct"]

    order = np.random.default_rng(SEED).permutation(CANDIDATES)
    episode = {"seed": SEED, "model_id": 1, "observe": observe, "random_order": order,
               "probe_domains": np.zeros(CANDIDATES, dtype=np.int64), "probe_design": design[:CANDIDATES].astype(np.float64),
               "probe_offsets": offsets[:CANDIDATES], "reference": (reference_design, reference_offsets),
               "reference_incumbent": reference_incumbent, "prior": prior}
    onboarding_start = perf_counter()
    latents, trace = smooth_path(episode, MARGIN, options, tolerance=DETAIL_TOLERANCE, cap=DEMO_CAP)
    newcomer_wall = perf_counter() - onboarding_start
    attempts = len(latents) + WARMUP - 1
    if newcomer.generated_requests != attempts or len({probe["item_id"] for probe in probes}) != attempts:
        raise RuntimeError("Newcomer received duplicate probes or requests beyond the stopping budget.")
    profiles = np.stack([incumbent_fit.latent, latents[-1]])
    support_calls = [model.generated_requests for model in (incumbent, newcomer)]
    for item in target:
        start = perf_counter()
        vector = encoder.encode([item[2]], show_progress_bar=False)
        query_offset, query_design = routing_features(decoder, vector)
        scores = sigmoid(interaction_scores(profiles, query_design[0]) + query_offset[0])
        choice = int(choose_models(scores, MARGIN))
        routing_seconds = perf_counter() - start
        model = (incumbent, newcomer)[choice]
        before = [value.generated_requests for value in (incumbent, newcomer)]
        generated, seconds = generate_rows(model, [item], "query")
        calls = [value.generated_requests - count for value, count in zip((incumbent, newcomer), before)]
        if calls != [int(index == choice) for index in range(2)]:
            raise RuntimeError("A query did not dispatch exactly one selected downstream model.")
        row = generated[0]
        outcomes.append({"item_id": item[0], "chosen_model": model.id, "qwen_score": float(scores[0]),
                         "smollm2_score": float(scores[1]), "parsed_option": row["parsed_option"], "correct": row["correct"],
                         "downstream_requests": sum(calls), "route_seconds_including_encoding": routing_seconds,
                         "local_generation_seconds": seconds})
    query_requests = [value.generated_requests - prev for value, prev in zip((incumbent, newcomer), support_calls)]
    planned = {"source": ARC, "source_revision": ARC_REVISION, "source_subset": "ARC-Challenge",
               "weights": [{"model": model, "revision": revision} for model, revision in MODELS],
               "seed": SEED, "checkpoint": decoder_path.name, "encoder": ENCODER, "unknown_domain": "mean trained benchmark bias",
               "probability_margin": MARGIN, "prior_strength": PRIOR_STRENGTH,
               "scope": "Local integration demo only; public held-out source split, no comparison policy or efficacy claim.",
               "input": "Only the four-choice question and answer instruction enter generation/encoding; reference questions are text only.",
               "scoring": "Greedy continuation, first standalone A–D (A, A., (A), or Answer: A), exact official key. Invalid is incorrect.",
               "support_source": "Official train with unique normalized stems",
               "target_source": "Official test, excluding all matching train stems",
               "support_items": [item[0] for item in support], "reference_items": [item[0] for item in reference],
               "candidate_items": [item[0] for item in support[:CANDIDATES]], "queries": [item[0] for item in target],
               "warmup": WARMUP, "demo_cap": DEMO_CAP, "evpi_tolerance": DETAIL_TOLERANCE,
               "incumbent_support_requests": support_calls[0], "incumbent_support_local_generate_seconds": profile_time,
               "incumbent_support_correct": sum(support_labels),
               "newcomer_support_requests": attempts, "newcomer_support_correct": sum(probe["correct"] for probe in probes),
               "newcomer_support_local_generate_seconds": newcomer_seconds, "onboarding_total_wall_seconds": newcomer_wall,
               "onboarding_remainder_seconds": newcomer_wall - newcomer_seconds,
               "stop_reason": "demo_cap" if attempts == DEMO_CAP else "evpi",
               "evpi_at_stop": float(trace[-1, 0]), "newcomer_probes": probes,
               "sequential_query_requests": sum(query_requests), "requests_per_model": dict(zip((item[0] for item in MODELS), query_requests)),
               "local_generate_calls": {model.id: model.generate_batches for model in (incumbent, newcomer)},
               "runtime": {"device": args.device,
                           "hardware": torch.cuda.get_device_name(args.device)
                           if torch.device(args.device).type == "cuda" else "CPU",
                           "backend": f"PyTorch {torch.__version__}",
                           "timings_exclude": "downloads, checkpoint loading and weight initialization"},
               "query_correct": sum(row["correct"] for row in outcomes), "target_invalid_outputs": sum(row["parsed_option"] is None for row in outcomes),
               "total_routing_seconds": sum(row["route_seconds_including_encoding"] for row in outcomes),
               "total_query_generation_seconds": sum(row["local_generation_seconds"] for row in outcomes)}
    atomic_json(args.output / "demo.json", planned)
    atomic_csv(args.output / "dispatch.csv", outcomes)
    atomic_json(args.local_output / "generation.json", local_trace)
    print(f"Dispatched {sum(query_requests)} prompts: {query_requests}. Newcomer probes: {attempts}.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/live-routing-demo")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / f"artifacts/routing_learning/checkpoints/{B}_{SEED}.pt")
    parser.add_argument("--learning-selection", type=Path, default=LEARNING_SELECTION)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/live_demo")
    parser.add_argument("--local-output", type=Path, default=ROOT / "artifacts/live_routing_demo")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.seed = SEED
    with threadpool_limits(limits=1):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
