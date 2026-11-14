"""Transfer frozen routing-learning checkpoints to the open LLMRouterBench roster."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import unicodedata

import numpy as np
from threadpoolctl import threadpool_limits

from route_query import ROOT, choose_models, interaction_scores, routing_features
from coldstart.adaptation import sigmoid
from coldstart.models import MeanBaselines, load_decoder, question_knn
from coldstart.probes import select_probe_order
from pilot_routing_learning import A, B, model_latents, latent_prior
from run_experiment import atomic_csv, atomic_json
from run_routing import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, VARIANT, benchmark_means, pool_outcomes


BENCHMARKS = {"mmlupro": "MMLU-Pro", "bbh": "BBH", "korbench": "KORBench"}
SEEDS = (11, 23, 37)
BUDGETS = (100, 200)
ENCODER = "sentence-transformers/all-mpnet-base-v2"
REFERENCE_SEED = 314159
REFERENCE_PER_BENCHMARK = 64
SUPPORT_FIXED = "support_best_fixed"
DOMAIN = "smoothed_domain"
KNN = "question_knn"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def tokens(text):
    return re.findall(r"[^\W_]+|[<>+=−±*/%]", normalize(text).casefold())


def question_prefix(prompt, origin):
    origin = origin.strip()
    index = prompt.rfind(origin)
    if index < 0:
        raise ValueError("A released question stem is absent from its rendered prompt.")
    return prompt[:index].strip(), prompt[index + len(origin):].strip()


def remove_demonstrations(benchmark, origin, prompt):
    prefix, tail = question_prefix(prompt, origin)
    if benchmark == "mmlupro":
        if not (prefix.startswith("Answer the following ") and "Answer: $LETTER" in prefix
                and tail.endswith("Let's think step by step.")):
            raise ValueError("Unrecognized MMLU-Pro zero-shot template.")
        return prompt, ""
    if benchmark == "bbh":
        separator = "Here are 3 examples to help you understand how to answer the question:"
        if prefix.count(separator) != 1 or not prefix.endswith("Question:") or tail:
            raise ValueError("Unrecognized BBH three-shot template.")
        examples = prefix.split(separator, 1)[1]
        return origin, examples.removesuffix("Question:").strip()
    if benchmark == "korbench":
        markers = list(re.finditer(r"(?m)^[ \t]*### (Question|Answer)[ \t]*:[ \t]*$", prefix))
        if ([marker.group(1) for marker in markers] != ["Question", "Answer"] * 3 + ["Question"]
                or tail != "### Answer:" or prefix[markers[-1].end():].strip()):
            raise ValueError("Unrecognized KORBench three-shot template.")
        rule = prefix[:markers[0].start()]
        if len(re.findall(r"(?m)^### (?:Cipher|Story|Logic|Operation|Puzzle) Rule:[ \t]*$", rule)) != 1:
            raise ValueError("KORBench prompt must include one rule heading.")
        examples = prefix[markers[0].start():markers[-1].start()]
        return f"{origin.strip()}\n\n{rule.strip()}", examples
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_archive(archive_path, audit, include_scores):
    partition = audit["data_split"]["benchmark_splits"]
    names = audit["roster"]
    wanted = {}
    for benchmark in BENCHMARKS:
        choices = (audit["archive_members"][benchmark] if "archive_members" in audit
                   else {model: row["archive_member"] for model, row in
                         audit["benchmark_diagnostics"][benchmark]["model_file_summary"].items()})
        for index, model in enumerate(names if include_scores else names[:1]):
            wanted[choices[model]] = (benchmark, index, model)
    item_indices = [(benchmark, index) for benchmark in BENCHMARKS
                    for fold in ("support", "query") for index in partition[benchmark][fold]]
    positions = {item: position for position, item in enumerate(item_indices)}
    labels = np.full((len(names), len(item_indices)), np.nan, dtype=np.float32) if include_scores else None
    prompts = [""] * len(item_indices)
    origins = [""] * len(item_indices)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            benchmark, model_index, expected_model = wanted.pop(member.name)
            if not member.isfile() or PurePosixPath(member.name).suffix != ".json":
                raise ValueError(f"Expected a regular JSON file: {member.name}")
            with archive.extractfile(member) as stream:
                result = json.load(stream)
            if result["model_name"] != expected_model:
                raise ValueError(f"Model name does not match the structural audit: {member.name}")
            for record in result["records"]:
                position = positions.get((benchmark, record["index"]))
                if position is None:
                    continue
                if include_scores:
                    score = record["score"]
                    if (type(score) not in (int, float, bool) or not math.isfinite(score)
                            or score not in (0, 1) or np.isfinite(labels[model_index, position])):
                        raise ValueError(f"Invalid binary record in {member.name} at {record['index']}.")
                    labels[model_index, position] = score
                else:
                    if prompts[position]:
                        raise ValueError(f"Duplicate selected source index in {member.name}.")
                    prompts[position] = record["prompt"]
                    origins[position] = record["origin_query"]
            if not wanted:
                break
    if wanted or (include_scores and not np.isfinite(labels).all()):
        raise ValueError("The release does not contain every preselected model and question record.")
    if not include_scores and (not all(prompts) or not all(origins)):
        raise ValueError("The release lacks some preselected prompt text.")
    return item_indices, labels, prompts, origins


def target_mentions(demonstrations, target_origins):
    """Count whole detected target-stem sequences in supplied example blocks."""
    lookup = {}
    short_lengths = set()
    for origin in target_origins:
        words = tokens(origin)
        if not words:
            continue
        prefix = tuple(words[:min(8, len(words))])
        short_lengths.add(len(prefix))
        lookup.setdefault(prefix, []).append(tuple(words))
    matched = set()
    for block in set(demonstrations):
        words = tokens(block)
        for offset in range(len(words)):
            for length in short_lengths:
                prefix = tuple(words[offset:offset + length])
                for phrase in lookup.get(prefix, ()):
                    if tuple(words[offset:offset + len(phrase)]) == phrase:
                        matched.add(phrase)
    return len(matched)


def token_statistics(encoder, final, rendered, origins):
    def measure(texts):
        encoded = encoder.tokenizer(texts, truncation=False, add_special_tokens=True, verbose=False)
        counts = np.asarray([len(ids) for ids in encoded["input_ids"]])
        return {"max_tokens": int(counts.max()), "median_tokens": float(np.median(counts)),
                "above_encoder_token_limit": int((counts > encoder.max_seq_length).sum())}

    full_prefixes = [question_prefix(prompt, origin)[0] for prompt, origin in zip(rendered, origins)]
    return {"released_prompt": measure(rendered), "encoder_input": measure(final),
            "final_question_alone": measure(origins),
            "released_prefix_before_final_question": measure(full_prefixes)}


def prepare(args, audit):
    from sentence_transformers import SentenceTransformer

    items, _, rendered, origins = load_archive(args.archive, audit, include_scores=False)
    encoder = SentenceTransformer(ENCODER, device=args.device)
    words = {}
    statistics = {}
    for benchmark in BENCHMARKS:
        group = [i for i, item in enumerate(items) if item[0] == benchmark]
        pairs = [(origins[i], rendered[i]) for i in group]
        clean, examples = zip(*(remove_demonstrations(benchmark, origin, prompt) for origin, prompt in pairs))
        before = [rendered[i] for i in group]
        origin = [origins[i] for i in group]
        statistics[benchmark] = {
            "retained_prompts": len(group), "distinct_release_example_blocks": len({x for x in examples if x}),
            "retained_stems_mentioned_in_original_example_blocks": target_mentions(examples, origins),
            **token_statistics(encoder, clean, before, origin),
        }
        words.update(zip(group, clean))

    texts = [words[i] for i in range(len(items))]
    vectors = encoder.encode(texts, batch_size=32, show_progress_bar=False)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    with args.features.open("wb") as handle:
        np.savez_compressed(handle, embeddings=vectors, benchmark=np.asarray([item[0] for item in items]),
                            index=np.asarray([item[1] for item in items]))
    atomic_json(args.output / "prompt_audit.json",
                {"encoder": ENCODER, "max_seq_length": encoder.max_seq_length,
                 "feature_input": {"mmlupro": "released zero-shot prompt and choices",
                                   "bbh": "original final question, which includes any answer choices",
                                   "korbench": "final question first, followed by task instructions and rule; no worked examples"},
                 "benchmark_counts": statistics})


def shared_offsets(decoder, vectors):
    return routing_features(decoder, vectors)


def paired_queries(left, right, domains):
    effect = left - right
    by_seed = benchmark_means(effect, domains).mean(axis=1)
    by_query = effect.mean(axis=0, dtype=np.float64)
    samples = np.zeros(BOOTSTRAP_DRAWS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = np.unique(domains)
    for domain in unique:
        unit = by_query[domains == domain]
        draws = rng.choice(unit, size=(BOOTSTRAP_DRAWS, len(unit)))
        samples += draws.mean(axis=1) / len(unique)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {"macro_gain": by_seed.mean(), "micro_gain": by_query.mean(),
            "macro_ci_low": lo, "macro_ci_high": hi, "positive_seed_count": int((by_seed > 0).sum())}


def score(args, audit):
    items, labels, _, _ = load_archive(args.archive, audit, include_scores=True)
    names = audit["roster"]
    with np.load(args.features, allow_pickle=False) as data:
        vectors = data["embeddings"]
        if items != list(zip(data["benchmark"].tolist(), data["index"].tolist())):
            raise ValueError("External features and frozen source indices differ.")
    with threadpool_limits(limits=1):
        evaluate(args, audit, items, vectors, labels, names)


def evaluate(args, audit, items, vectors, labels, names):
    partition = audit["data_split"]["benchmark_splits"]
    domains = np.array([list(BENCHMARKS).index(benchmark) for benchmark, _ in items])
    support_ids = {benchmark: set(folds["support"]) for benchmark, folds in partition.items()}
    support = np.array([i for i, (benchmark, index) in enumerate(items) if index in support_ids[benchmark]])
    query = np.setdiff1d(np.arange(len(items)), support)
    orders = {seed: select_probe_order(vectors, domains, support, max(BUDGETS), "balanced_random", seed) for seed in SEEDS}
    unused = np.setdiff1d(support, np.concatenate(list(orders.values())))
    available_per_benchmark = np.bincount(domains[unused], minlength=len(BENCHMARKS))
    reference_size = min(REFERENCE_PER_BENCHMARK, int(available_per_benchmark.min()))
    if not reference_size:
        raise ValueError("The disjoint reference bank needs support items in every benchmark.")
    reference = select_probe_order(vectors, domains, unused, reference_size * len(BENCHMARKS),
                                   "balanced_random", REFERENCE_SEED)
    if not np.all(np.bincount(domains[reference]) == reference_size):
        raise ValueError("The disjoint reference bank must give equal weight to each benchmark.")

    config = read_json(args.run / "effective_config.json")
    historical = read_json(args.run / "test_lock.json")["selection"]
    learning = read_json(args.learning_selection)
    strength = historical["prior_strengths"][VARIANT]
    if learning["protocol"]["prior_strength"] != strength:
        raise ValueError("The learning study and the original run used different prior strengths.")
    smoothing = historical["mean_smoothing"]["domain_probe_mean"]
    knn = historical["knn"]
    baseline = MeanBaselines(len(BENCHMARKS), smoothing)
    query_domains, truth = domains[query], labels[:, query]
    by_seed, by_benchmark, routed, counts = [], [], {}, []
    for seed in SEEDS:
        frozen_features = {}
        for arm in (A, B):
            decoder, metadata = load_decoder(args.checkpoints / f"{arm}_{seed}.pt")
            if not any(all(metadata[key] == value for key, value in row.items())
                       for row in learning["checkpoints"]):
                raise ValueError(f"Checkpoint does not match the original selection: {arm}/{seed}")
            prior = latent_prior(decoder, config, strength)
            offsets, design = shared_offsets(decoder, vectors)
            frozen_features[arm] = offsets, design, prior

        for budget in BUDGETS:
            selected = orders[seed][:budget]
            responses = labels[:, selected]
            selected_domains = domains[selected]
            support_model = int(benchmark_means(responses, selected_domains).mean(axis=1).argmax())
            choices = {SUPPORT_FIXED: np.full(len(query), support_model),
                       DOMAIN: choose_models(baseline.probe_mean(responses, selected_domains, query_domains)),
                       KNN: choose_models(question_knn(vectors[selected], responses, vectors[query], **knn))}
            for arm, (offsets, design, prior) in frozen_features.items():
                fitted = model_latents((offsets[selected], design[selected]), responses,
                                       selected_domains, prior, config)
                probabilities = sigmoid(interaction_scores(fitted, design[reference]) + offsets[reference])
                fixed_model = int(benchmark_means(probabilities, domains[reference]).mean(axis=1).argmax())
                choices[arm] = choose_models(interaction_scores(fitted, design[query]))
                choices[arm + "_embedding_mean_fixed"] = np.full(len(query), fixed_model)

            for method, chosen in choices.items():
                outcome = pool_outcomes(chosen[None, :], truth, query_domains)
                correct = outcome["correct"][0]
                routed.setdefault((budget, method), []).append(correct)
                accuracy = benchmark_means(correct, query_domains)
                oracle = benchmark_means(outcome["oracle"], query_domains)
                shuffled = benchmark_means(outcome["shuffled"][0], query_domains)
                most_selected = np.bincount(chosen, minlength=len(names))
                identity = {"seed": seed, "budget_per_candidate": budget, "method": method,
                            "n_candidates": len(names), "total_probe_responses": len(names) * budget}
                by_seed.append({**identity, "n_queries": len(query), "macro_accuracy": accuracy.mean(),
                                "micro_accuracy": correct.mean(dtype=np.float64), "macro_oracle": oracle.mean(),
                                "macro_regret": (oracle - accuracy).mean(),
                                "macro_gain_over_shuffle": (accuracy - shuffled).mean(),
                                "most_selected_model": names[most_selected.argmax()],
                                "most_selected_share": most_selected.max() / len(query)})
                for domain, benchmark in enumerate(BENCHMARKS.values()):
                    by_benchmark.append({**identity, "benchmark": benchmark,
                                         "n_queries": int((query_domains == domain).sum()),
                                         "accuracy": accuracy[domain], "oracle": oracle[domain]})
            for comparator in (A, SUPPORT_FIXED, B + "_embedding_mean_fixed", A + "_embedding_mean_fixed"):
                left, right = [routed[(budget, key)][-1] for key in (B, comparator)]
                counts.append({"seed": seed, "budget_per_candidate": budget, "comparator": comparator,
                               "n_queries": len(query), "rescues": int((left > right).sum()),
                               "harms": int((left < right).sum()),
                               "opportunities": int((truth.max(axis=0) > right).sum())})

    summary, paired = [], []
    for (budget, method), values in routed.items():
        matrix = np.stack(values)
        oracle = benchmark_means(truth.max(axis=0), query_domains).mean()
        macro = benchmark_means(matrix, query_domains).mean()
        summary.append({"budget_per_candidate": budget, "method": method, "n_queries": len(query),
                        "n_candidates": len(names), "total_probe_responses": len(names) * budget,
                        "macro_accuracy": macro, "micro_accuracy": matrix.mean(dtype=np.float64),
                        "macro_oracle": oracle, "macro_regret": oracle - macro})
        if method != B:
            comparison = paired_queries(np.stack(routed[(budget, B)]), matrix, query_domains)
            paired.append({"budget_per_candidate": budget, "method": B, "comparator": method, **comparison})
    for name, rows in (("summary", summary), ("by_seed", by_seed), ("by_benchmark", by_benchmark),
                       ("paired", paired), ("rescues_harms_by_seed", counts)):
        atomic_csv(args.output / f"{name}.csv", rows)
    archive_identity = {key: audit[key] for key in ("source", "source_revision", "archive_bytes", "roster",
                                                    "literal_embedllm_model_aliases", "stem_match", "data_split")}
    archive_identity["benchmark_diagnostics"] = {benchmark: {field: value for field, value in row.items()
                                                             if field != "model_file_summary"}
                                                 for benchmark, row in audit["benchmark_diagnostics"].items()}
    archive_identity["archive_members"] = audit.get("archive_members")
    if archive_identity["archive_members"] is None:
        archive_identity["archive_members"] = {
            benchmark: {model: row["archive_member"] for model, row in
                        audit["benchmark_diagnostics"][benchmark]["model_file_summary"].items()}
            for benchmark in BENCHMARKS}
    atomic_json(args.output / "protocol.json", {
        **archive_identity, "performance_scope": "20 official lightweight candidates, external release and model IDs; family overlap allowed",
        "checkpoint_selection_source": "existing EmbedLLM validation; no external fine-tuning or selection",
        "unknown_benchmark_offset": "frozen learned difficulty head plus unweighted mean of all five trained benchmark biases",
        "prior_strength": strength, "seed_order": SEEDS, "budgets": BUDGETS,
        "probe_selection": "shared per seed, nested balanced random within support pool",
        "probe_ids": {str(seed): [items[index] for index in order] for seed, order in orders.items()},
        "embedding_mean_reference_ids": [items[index] for index in reference],
        "reference_policy": {"per_benchmark": reference_size, "cap_per_benchmark": REFERENCE_PER_BENCHMARK,
                             "rule": "largest equal quota after removing all seed probe prefixes; text only"},
        "fixed_policy": "same-budget best observed benchmark-macro accuracy, ties first official model",
        "baseline_settings": {"domain_smoothing": smoothing, "question_knn": knn, "default_unknown_prior": 0.5},
        "interval": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                     "unit": "paired queries within benchmarks after seed averaging; conditional on fixed roster, support and checkpoints"},
        "generation": "Recorded correctness only; no API calls or inference cost estimates.",
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "score"))
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/llmrouterbench/bench-release.tar.gz")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/routing/external_transfer/protocol.json",
                        help="Frozen structural manifest (or the raw pre-score structural audit).")
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/external_routing/features.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/external_transfer")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "artifacts/routing_learning/checkpoints")
    parser.add_argument("--learning-selection", type=Path, default=ROOT / "reports/routing/learning/selection.json")
    parser.add_argument("--device", default="cuda", help="All-MPNet encoding device.")
    args = parser.parse_args(argv)
    audit = read_json(args.audit)
    if args.archive.stat().st_size != audit["archive_bytes"]:
        raise ValueError("Archive byte count differs from the audited release.")
    if args.phase == "prepare":
        prepare(args, audit)
    else:
        score(args, audit)
    print(f"Completed external routing {args.phase}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
