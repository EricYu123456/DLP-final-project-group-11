"""Transfer frozen adaptive onboarding to independent xRouteBench tasks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import re
from pathlib import Path
import unicodedata

import numpy as np
from threadpoolctl import threadpool_limits

from route_query import ROOT
from coldstart.adaptation import sigmoid
from coldstart.models import load_decoder
from coldstart.probes import select_probe_order
from pilot_competitive_onboarding import ANCHOR_QUANTILES
from pilot_routing_acquisition import (CANDIDATES_PER_BENCHMARK, REFERENCE_PER_BENCHMARK,
                                       acquisition_banks)
from pilot_routing_learning import B, latent_prior, model_latents
from pilot_smooth_probe import smooth_path
from run_experiment import atomic_csv, atomic_json
from run_external_routing import ENCODER, normalize, paired_queries, read_json, shared_offsets, tokens
from run_routing import add_one_outcomes, benchmark_means
from run_sequential_routing import CAP, SCREENED_PER_STEP, WARMUP, probabilities, trajectory


SOURCE = "ulab-ai/xRouteBench"
REVISION = "ea4b6e1b29d9a734f55f0a637baf326bad6aa681"
TASKS = ("arc_challenge", "boolq", "commonsense_qa", "hellaswag", "openbook_qa")
EXPECTED_ROSTER_SIZE = 18
EXCLUDED = {
    "llama3-70b-instruct": "direct API alias of EmbedLLM source model 50",
    "llama-3-8b-instruct-lite": "INT4 variant of EmbedLLM source model 64",
    "qwen2.5-7b-instruct-turbo": "same backbone as canonical qwen2.5-7b-instruct; keep canonical endpoint",
}
MARGIN = 0.025
TOLERANCE = 0.002
SEEDS = (11, 23, 37)
PRIOR_STRENGTH = 0.3
PROBE_SAVING = 0.10
MIN_MACRO_LOWER_BOUND = -0.005
SMOOTH = "smooth_own_stop"
RANDOM = "random_own_stop"
FIXED = "random_fixed_200"
SMOOTH_CAP = "smooth_fixed_200"
INCUMBENT = "incumbent_only"
SAME_PROBE_FIXED = "smooth_own_stop_fixed_identity"


def parquet(path, columns):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns,
                         filters=[("task_name", "in", list(TASKS))]).to_pylist()


def user_query(prompt):
    prompt = unicodedata.normalize("NFKC", prompt).replace("\r\n", "\n")
    head, separator, content = prompt.partition("[User Query]")
    if separator and head.lstrip().startswith("[System Instruction]"):
        return content.strip()
    return prompt.strip()


def stem(prompt):
    text = re.sub(r"(?i)^\s*(?:#{1,4}\s*)?Question:\s*", "", user_query(prompt))
    text = re.split(r"(?im)^\s*(?:#{1,4}\s*Options\s*:|\([AB]\)\s|\bA[.)]\s)", text, maxsplit=1)[0]
    return text.strip()


def normalized_tokens(text):
    return " ".join(tokens(text))


def identity(task, prompt):
    return task, normalized_tokens(stem(prompt))


def old_stem(prompt, domain):
    if domain == "GPQA":
        text = re.split(r"Question:|What is the correct answer to this question:", prompt)[-1]
        return text.rsplit("Choices:", 1)[0]
    if domain in ("MMLU", "MedMCQA", "LogiQA"):
        if domain == "MMLU":
            prompt = re.sub(r"^The following are multiple choice questions \(with answers\) about [^\n]+\n", "", prompt)
        return stem(prompt).removesuffix("Answer:").removesuffix("Choices:").strip()
    if domain in ("TruthfulQA", "SocialIQA"):
        return re.sub(r"\nA:\s*$", "", re.split(r"(?:^|\n)Q:\s*", prompt)[-1])
    if domain in ("MathQA", "PIQA", "GSM8K"):
        return re.sub(r"\nAnswer:\s*$", "", re.split(r"(?:^|\n)Question:\s*", prompt)[-1])
    return re.sub(r"\nAnswer:\s*$", "", prompt)


def embedllm_overlap_keys(raw_path, catalog_path):
    with catalog_path.open(newline="", encoding="utf-8-sig") as handle:
        domains = {int(row["prompt_id"]): row["benchmark"] for row in csv.DictReader(handle)}
    with raw_path.open(newline="", encoding="utf-8-sig") as handle:
        return {key for row in csv.DictReader(handle)
                for text in (row["prompt"], old_stem(row["prompt"], domains[int(row["prompt_id"])]))
                if (key := normalized_tokens(text))}


def binary_label(value):
    return type(value) in (float, int, bool) and np.isfinite(value) and value in (0, 1)


def partition(rows, roster, prior_overlap, *, labeled, excluded_train=None):
    groups = defaultdict(list)
    for row in rows:
        groups[identity(row["task_name"], row["query"])].append(row)
    audit = {task: Counter() for task in TASKS}
    selected = []
    for key, candidates in sorted(groups.items()):
        task, question = key
        task_audit = audit[task]
        task_audit["raw_groups"] += 1
        task_audit["raw_records"] += len(candidates)
        observed = Counter(row["model_name"] for row in candidates)
        variants = {normalize(user_query(row["query"])) for row in candidates}
        if not question or len(variants) != 1 or any(count > 1 for count in observed.values()):
            task_audit["ambiguous_or_repeated"] += 1
        elif set(observed) != roster:
            task_audit["incomplete_roster"] += 1
        elif labeled and not all(binary_label(row["performance"]) for row in candidates):
            task_audit["invalid_training_label"] += 1
        elif {question, normalized_tokens(user_query(candidates[0]["query"]))} & prior_overlap:
            task_audit["embedllm_overlap"] += 1
        elif excluded_train is not None and question in excluded_train:
            task_audit["xroute_train_overlap"] += 1
        else:
            task_audit["retained"] += 1
            selected.append((key, user_query(candidates[0]["query"]), candidates))
    return selected, audit, {question for _, question in groups}


def anchor_roster(eligible, labels, domains):
    quality = benchmark_means(labels, domains).mean(axis=1)
    order = np.lexsort((np.asarray(eligible), quality))
    anchors = []
    for quantile in ANCHOR_QUANTILES:
        rank = int(np.floor(quantile * (len(order) - 1) + 0.5))
        model = order[rank]
        anchors.append({"model": eligible[model], "quantile": quantile, "support_rank": rank + 1,
                        "train_macro_accuracy": float(quality[model])})
    return anchors


def prepare(args):
    from sentence_transformers import SentenceTransformer

    identity_columns = ["task_name", "query", "model_name"]
    model_rows = parquet_models(args.raw / "models.parquet")
    roster = {row["model_name"] for row in model_rows}
    if len(roster) != EXPECTED_ROSTER_SIZE or not EXCLUDED.keys() <= roster:
        raise ValueError("Official roster differs from the 18-model source.")
    eligible = sorted(roster - EXCLUDED.keys())
    overlap = embedllm_overlap_keys(args.embedllm_prompts, args.embedllm_catalog)
    training, train_audit, old_train = partition(parquet(args.raw / "train.parquet", [*identity_columns, "performance"]),
                                                 roster, overlap, labeled=True)
    # Parquet column projection excludes test labels, responses and gold answers.
    testing, test_audit, _ = partition(parquet(args.raw / "test.parquet", identity_columns),
                                      roster, overlap, labeled=False, excluded_train=old_train)
    train_domains = np.array([TASKS.index(key[0]) for key, _, _ in training])
    test_domains = np.array([TASKS.index(key[0]) for key, _, _ in testing])
    labels = np.empty((len(eligible), len(training)), dtype=np.float32)
    mapping = {model: index for index, model in enumerate(eligible)}
    for index, (_, _, rows) in enumerate(training):
        for row in rows:
            if row["model_name"] in mapping:
                labels[mapping[row["model_name"]], index] = row["performance"]
    reference, candidates = acquisition_banks(np.arange(len(training)), train_domains)
    if not (np.all(np.bincount(train_domains[reference], minlength=len(TASKS)) == REFERENCE_PER_BENCHMARK)
            and np.all(np.bincount(train_domains[candidates], minlength=len(TASKS)) == CANDIDATES_PER_BENCHMARK)
            and len(set(test_domains)) == len(TASKS)):
        raise ValueError("Complete support or target coverage is insufficient for the prespecified experiment.")
    anchors = anchor_roster(eligible, labels, train_domains)
    protocol = {"source": SOURCE, "revision": REVISION, "tasks": TASKS,
                "source_roster": model_rows, "eligible_models": eligible, "excluded": EXCLUDED,
                "anchors": anchors, "train_counts": train_audit, "test_identity_counts": test_audit,
                "overlap": "identical normalized question stems and stripped EmbedLLM raw prompts; all xRoute train stems excluded from test",
                "duplicate_policy": "discard any normalized group with repeated model rows or differing rendered user queries",
                "family_scope": "exact listed aliases excluded; broader ancestry and families can overlap",
                "test_preparation_columns": identity_columns, "checkpoint_arm": B, "seeds": SEEDS,
                "unknown_task_offset": "learned difficulty plus mean source benchmark bias",
                "encoder": ENCODER, "feature_input": "NFKC-normalized released User Query preserving question, context and choices",
                "reference_per_task": REFERENCE_PER_BENCHMARK, "candidate_per_task": CANDIDATES_PER_BENCHMARK,
                "screened_per_step": SCREENED_PER_STEP,
                "margin": MARGIN, "tolerance": TOLERANCE, "warmup": WARMUP, "cap": CAP,
                "prior_strength": PRIOR_STRENGTH,
                "criterion": {"primary_anchors": list(ANCHOR_QUANTILES[:-1]),
                              "minimum_relative_probe_saving_vs_random_own_stop": PROBE_SAVING,
                              "minimum_conditional_macro_ci_lower_bound_vs_random_200": MIN_MACRO_LOWER_BOUND},
                "inference_scope": "conditional paired query intervals given fixed anchors, newcomers and draws; no external tuning"}
    print(json.dumps({"train": train_audit, "test_identity_only": test_audit, "anchors": anchors}, indent=2), flush=True)
    encoder = SentenceTransformer(ENCODER, device=args.device)
    texts = [prompt for _, prompt, _ in [*training, *testing]]
    pieces = encoder.tokenizer(texts, truncation=False, add_special_tokens=True, verbose=False)["input_ids"]
    limits = np.array([len(row) for row in pieces])
    protocol["max_encoder_tokens"] = encoder.max_seq_length
    protocol["prompts_exceeding_encoder_limit"] = int((limits > encoder.max_seq_length).sum())
    vectors = encoder.encode(texts, batch_size=32, show_progress_bar=False)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.features, train_vectors=vectors[:len(training)], test_vectors=vectors[len(training):],
                        train_domains=train_domains, test_domains=test_domains, training_labels=labels,
                        test_stems=np.asarray([key[1] for key, _, _ in testing]), reference=reference, candidates=candidates)
    atomic_json(args.output / "protocol.json", protocol)


def parquet_models(path):
    import pyarrow.parquet as pq

    return sorted(pq.read_table(path, columns=["model_name", "api_model_id", "service"]).to_pylist(),
                  key=lambda row: row["model_name"])


def load_test_labels(path, roster, eligible, keys):
    positions = {key: index for index, key in enumerate(keys)}
    models = {name: index for index, name in enumerate(eligible)}
    labels = np.full((len(eligible), len(keys)), np.nan, dtype=np.float32)
    seen = [set() for _ in keys]
    for row in parquet(path, ["task_name", "query", "model_name", "performance"]):
        position = positions.get(identity(row["task_name"], row["query"]))
        if position is None:
            continue
        model, value = row["model_name"], row["performance"]
        if model in seen[position] or model not in roster or not binary_label(value):
            raise ValueError(f"Nonbinary or duplicate target cell in task {row['task_name']} for {model}.")
        seen[position].add(model)
        if model in models:
            labels[models[model], position] = value
    if any(found != roster for found in seen) or not np.isfinite(labels).all():
        raise ValueError("The frozen target bank does not contain all 18 binary model rows per question.")
    return labels


def random_path(episode, prior, options, stop):
    calls = []

    def observe(probe):
        calls.append(probe)
        return episode["observe"](probe)

    latents, trace, chosen, _ = trajectory(episode["random_order"], episode["probe_domains"],
                                           episode["probe_design"], episode["probe_offsets"], observe,
                                           episode["reference"], episode["reference_incumbent"],
                                           prior, MARGIN, options, episode["seed"], stop, selection="random")
    if calls != chosen:
        raise RuntimeError("Random arm observed a different number or order of probes.")
    return latents, np.asarray(trace)


def evaluate_arrival(episode, incumbent_support, options):
    observed = []
    oracle = episode["observe"]

    def observe(probe):
        response = oracle(probe)
        observed.append((probe, response))
        return response

    smooth = smooth_path({**episode, "observe": observe}, MARGIN, options, tolerance=TOLERANCE)
    attempts = len(smooth[0]) + WARMUP - 1
    probes, responses = map(np.asarray, zip(*observed))
    if len(probes) != attempts or len(np.unique(probes)) != attempts:
        raise RuntimeError("Smooth arm read beyond its stopping point.")
    comparison = np.stack([incumbent_support[probes], responses])
    fixed_switch = np.diff(benchmark_means(comparison, episode["probe_domains"][probes]).mean(axis=1))[0] > 0
    paths = {SMOOTH: smooth, RANDOM: random_path(episode, episode["prior"], options, TOLERANCE),
             FIXED: random_path(episode, episode["prior"], options, 0.0),
             SMOOTH_CAP: smooth if attempts == CAP else smooth_path(episode, MARGIN, options)}
    results = {INCUMBENT: (np.zeros(len(episode["incumbent"]), dtype=bool), 0, 0.0),
               SAME_PROBE_FIXED: (np.full(len(episode["incumbent"]), fixed_switch), attempts, None)}
    for method, (latents, trace) in paths.items():
        choices = probabilities(latents[-1], episode) - episode["incumbent"] > MARGIN
        results[method] = (choices, len(latents) + WARMUP - 1, trace[-1, 2])
    return results


def seeded_context(config, learning, checkpoint, arrays, seed):
    decoder, metadata = load_decoder(checkpoint)
    if not any(all(metadata[key] == value for key, value in row.items()) for row in learning["checkpoints"]):
        raise ValueError(f"Checkpoint differs from the frozen B selection: {checkpoint}.")
    prior = latent_prior(decoder, config, PRIOR_STRENGTH)
    train = shared_offsets(decoder, arrays["train_vectors"])
    offsets, design = shared_offsets(decoder, arrays["test_vectors"])
    candidates = arrays["candidates"]
    domains = arrays["train_domains"][candidates]
    order = select_probe_order(train[1][candidates], domains, np.arange(len(candidates)),
                               len(candidates), "balanced_random", seed)
    return train, {"seed": seed, "design": design, "offsets": offsets, "prior": prior,
                   "probe_design": train[1][candidates].astype(np.float64), "probe_offsets": train[0][candidates],
                   "random_order": order, "probe_domains": domains}


def score(args):
    protocol = read_json(args.output / "protocol.json")
    if protocol["revision"] != REVISION or protocol["margin"] != MARGIN or protocol["tolerance"] != TOLERANCE:
        raise ValueError("The prepared source or stopping settings differ from the frozen protocol.")
    with np.load(args.features, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    roster, models = {row["model_name"] for row in protocol["source_roster"]}, protocol["eligible_models"]
    domains = arrays["test_domains"]
    keys = [(TASKS[task], text) for task, text in zip(domains, arrays["test_stems"])]
    target = load_test_labels(args.raw / "test.parquet", roster, models, keys)
    train_labels, train_domains = arrays["training_labels"], arrays["train_domains"]
    candidates, reference = arrays["candidates"], arrays["reference"]
    config, learning = read_json(args.run / "effective_config.json"), read_json(args.learning_selection)
    options = {key: config["adaptation"][key] for key in ("max_iterations", "tolerance")}
    contexts = {seed: seeded_context(config, learning, args.checkpoints / f"{B}_{seed}.pt", arrays, seed)
                for seed in SEEDS}
    summaries, by_seed, by_arrival, by_task, paired, all_effects = [], [], [], [], [], {}
    for anchor in protocol["anchors"]:
        position = models.index(anchor["model"])
        arrivals = [index for index in range(len(models)) if index != position]
        truth = target[np.r_[position, arrivals]]
        choices, budgets, elapsed = defaultdict(list), defaultdict(list), defaultdict(list)
        for seed, (train, context) in contexts.items():
            prior = context["prior"]
            incumbent_latent = model_latents(train, train_labels[position:position + 1], train_domains, prior, config)[0]
            reserved = (train[1][reference].astype(np.float64), train[0][reference])
            incumbent_reference = sigmoid(reserved[1] + reserved[0] @ incumbent_latent)
            incumbent_query = probabilities(incumbent_latent, context)
            trial_group = defaultdict(list)
            for row in arrivals:
                def observe(index, model=row):
                    return train_labels[model, candidates[index]]
                episode = {**context, "model_id": row, "observe": observe,
                           "reference": reserved, "reference_incumbent": incumbent_reference, "incumbent": incumbent_query}
                for method, (use_newcomer, budget, time) in evaluate_arrival(episode, train_labels[position, candidates], options).items():
                    actual = np.where(use_newcomer, target[row], target[position])
                    delta = actual - target[position]
                    by_arrival.append({**anchor, "seed": seed, "newcomer": models[row], "method": method,
                                       "probe_attempts": budget, "local_cpu_elapsed_seconds": time,
                                       "macro_accuracy": benchmark_means(actual, domains).mean(),
                                       "micro_accuracy": actual.mean(dtype=np.float64), "newcomer_selected": int(use_newcomer.sum()),
                                       "rescues": int((delta > 0).sum()), "harms": int((delta < 0).sum()),
                                       "opportunities": int((target[row] > target[position]).sum())})
                    trial_group[method].append(use_newcomer)
                    budgets[method].append(budget)
                    if time is not None:
                        elapsed[method].append(time)
            for method, decision in trial_group.items():
                choices[method].append(decision)
        outcomes = {method: add_one_outcomes(np.asarray(decision), truth, domains) for method, decision in choices.items()}
        all_effects[anchor["quantile"]] = {}
        for method, result in outcomes.items():
            actual = result["correct"]
            seed_means = actual.mean(axis=1, dtype=np.float64)
            all_effects[anchor["quantile"]][method] = seed_means
            macro = benchmark_means(seed_means, domains)
            rescues, harms = int(result["rescue"].sum()), int(result["harm"].sum())
            opportunity = int(result["opportunity"].sum()) * len(SEEDS)
            row = {**anchor, "method": method, "arrivals": len(arrivals) * len(SEEDS), "n_queries": len(domains),
                   "macro_accuracy": macro.mean(), "micro_accuracy": actual.mean(dtype=np.float64),
                   "mean_probe_attempts": np.mean(budgets[method]), "median_probe_attempts": np.median(budgets[method]),
                   "total_probe_attempts": sum(budgets[method]), "cap_count": budgets[method].count(CAP),
                   "mean_local_cpu_elapsed_seconds": np.mean(elapsed[method]) if elapsed[method] else None,
                   "newcomer_share": result["selected"].mean(), "rescues": rescues, "harms": harms,
                   "opportunities": opportunity, "opportunity_capture": rescues / opportunity if opportunity else None}
            summaries.append(row)
            for i, seed in enumerate(SEEDS):
                by_seed.append({**anchor, "seed": seed, "method": method, "macro_accuracy": macro[i].mean(),
                                "micro_accuracy": actual[i].mean(dtype=np.float64)})
            for i, task in enumerate(TASKS):
                by_task.append({**anchor, "task": task, "method": method, "items": int((domains == i).sum()),
                                "accuracy": macro[:, i].mean()})
        print(f"Scored fixed xRouteBench incumbent {anchor['model']}.", flush=True)
    competitive = [all_effects[q] for q in ANCHOR_QUANTILES[:-1]]
    all_effects["competitive_average"] = {method: np.mean([group[method] for group in competitive], axis=0)
                                          for method in competitive[0]}
    for scope, methods in all_effects.items():
        for comparator in (RANDOM, FIXED, SMOOTH_CAP, INCUMBENT, SAME_PROBE_FIXED):
            paired.append({"anchor_quantile": scope, "method": SMOOTH, "comparator": comparator,
                           **paired_queries(methods[SMOOTH], methods[comparator], domains)})
    primary = {row["comparator"]: row for row in paired if row["anchor_quantile"] == "competitive_average"}
    competitive_summary = [row for row in summaries if row["quantile"] in ANCHOR_QUANTILES[:-1]]
    smooth_k, random_k = [np.mean([row["mean_probe_attempts"] for row in competitive_summary if row["method"] == name])
                          for name in (SMOOTH, RANDOM)]
    savings = 1 - smooth_k / random_k
    lower = primary[FIXED]["macro_ci_low"]
    for name, rows in (("summary", summaries), ("paired", paired), ("by_seed", by_seed),
                       ("by_arrival", by_arrival), ("by_task", by_task)):
        atomic_csv(args.output / f"test_{name}.csv", rows)
    atomic_json(args.output / "test_criterion.json", {
        "smooth_mean_probes": smooth_k, "random_own_stop_mean_probes": random_k, "relative_saving": savings,
        "macro_lower_bound_vs_random_fixed_200": lower,
        "met": bool(savings >= PROBE_SAVING and lower >= MIN_MACRO_LOWER_BOUND),
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "score"))
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/xroutebench")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--learning-selection", type=Path, default=ROOT / "reports/routing/learning/selection.json")
    parser.add_argument("--embedllm-prompts", type=Path, default=ROOT / "data/raw/question_order.csv")
    parser.add_argument("--embedllm-catalog", type=Path, default=ROOT / "data/processed/prompts.csv")
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/xroute_onboarding/features.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/xroute_onboarding")
    parser.add_argument("--device", default="cuda", help="Optional encoder device during preparation.")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        (prepare if args.stage == "prepare" else score)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
