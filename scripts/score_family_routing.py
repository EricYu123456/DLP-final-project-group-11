"""Replay routing from the locked EmbedLLM family score-prediction archives."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_routing import ROOT, benchmark_means, pool_outcomes
from coldstart.data import load_dataset
from coldstart.splits import load_family_table
from run_experiment import atomic_csv, atomic_json
from run_external_routing import paired_queries, read_json


FAMILIES = ("Llama", "Mistral", "Yi")
EXCLUDED, RELATED = "excluded", "related_allowed"
ARMS = (EXCLUDED, RELATED)
SEEDS = (11, 23, 37)
BUDGETS = (50, 100, 200)
PRIMARY = 200
LEARNED = "coldstart_neural_lowrank"
GLOBAL = "global_probe_mean"
DOMAIN = "domain_probe_mean"
KNN = "question_knn"
FIXED = "empirical_support_best_fixed"
ARCHIVED = (LEARNED, GLOBAL, DOMAIN, KNN)
COMPARATORS = ((RELATED, LEARNED), (EXCLUDED, FIXED), (EXCLUDED, DOMAIN),
               (EXCLUDED, KNN), (EXCLUDED, GLOBAL))


def archive_choices(folder, seed, budget, method, model_ids, prompt_ids, keep):
    path = folder / f"seed{seed}_balanced_random_k{budget}_{method}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if (not np.array_equal(archive["model_ids"], model_ids)
                or not np.array_equal(archive["prompt_ids"], prompt_ids)):
            raise ValueError(f"Prediction axes differ from the split in {path}.")
        probabilities = archive["probabilities"][:, keep]
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Missing routing probabilities in {path}.")
    # The original IDs, rather than positional rank in the archive, break ties.
    order = np.argsort(model_ids)
    return order[probabilities[order].argmax(axis=0)]


def support_fixed(dataset, model_rows, models, probe, prompt_pos, seed, budget, size):
    local = np.array([[prompt_pos[p] for p in probe[seed, int(model)][:budget]] for model in models])
    observed = dataset.labels[model_rows[:, None], local]
    # Each complete balanced prefix has the same number of probes per benchmark.
    # Mean accuracy within each benchmark also handles masked probe labels.
    probe_domains = dataset.benchmark_ids[local]
    accuracy = np.array([np.mean([np.nanmean(y[d == group]) for group in np.unique(d)])
                         for y, d in zip(observed, probe_domains)])
    if not np.isfinite(accuracy).all():
        raise ValueError("An empirical fixed candidate is missing all probes in a benchmark.")
    best = np.lexsort((models, -accuracy))[0]
    return np.full(size, best, dtype=np.int64)


def score_family(run, family, dataset, model_pos, prompt_pos):
    splits = {arm: read_json(run / family / arm / "split.json") for arm in ARMS}
    split = splits[EXCLUDED]
    models = np.array(split["test_model_ids"])
    prompts = np.array(split["test_prompt_ids"])
    probes = {}
    for arm, view in splits.items():
        for field in ("train_prompt_ids", "val_prompt_ids", "test_prompt_ids", "test_model_ids"):
            if view[field] != split[field]:
                raise ValueError(f"{family} has mismatched archived support or query axes.")
        rows = read_json(run / family / arm / "test_probes.json")
        probes[arm] = {(r["seed"], r["model_id"]): r["prompt_ids"] for r in rows}
    if probes[EXCLUDED] != probes[RELATED]:
        raise ValueError(f"{family} has different probe prefixes between arms.")
    n_models = len(models)
    model_rows = np.array([model_pos[model] for model in models])
    query_rows = np.array([prompt_pos[prompt] for prompt in prompts])
    truth = dataset.labels[np.ix_(model_rows, query_rows)]
    keep = np.isfinite(truth).all(axis=0)
    truth = truth[:, keep]
    domains = dataset.benchmark_ids[query_rows[keep]]
    if not np.isin(truth, (0, 1)).all() or len(np.unique(domains)) != 5:
        raise ValueError(f"{family} needs binary common outcomes for all five benchmarks.")

    records, seed_rows, primary = [], [], {}
    for budget in BUDGETS:
        fixed = np.stack([support_fixed(dataset, model_rows, models, probes[EXCLUDED], prompt_pos,
                                        seed, budget, len(domains)) for seed in SEEDS])
        for arm in ARMS:
            folder = run / family / arm / "test_predictions"
            all_choices = {method: np.stack([archive_choices(folder, seed, budget, method, models, prompts, keep)
                                             for seed in SEEDS]) for method in ARCHIVED}
            all_choices[FIXED] = fixed
            for method, chosen in all_choices.items():
                result = pool_outcomes(chosen, truth, domains)
                correct, oracle, shuffled = (result[key] for key in ("correct", "oracle", "shuffled"))
                identity = {"family": family, "arm": arm, "probes_per_model": budget, "method": method}
                macro = benchmark_means(correct, domains).mean(axis=1)
                micro = correct.mean(axis=1, dtype=np.float64)
                models_used = [len(np.unique(row)) for row in chosen]
                records.append({**identity, "target_models": n_models, "total_probes": n_models * budget,
                                "queries": len(domains), "macro_accuracy": macro.mean(), "micro_accuracy": micro.mean(),
                                "hindsight_oracle_macro": benchmark_means(oracle, domains).mean(),
                                "hindsight_oracle_micro": oracle.mean(dtype=np.float64),
                                "same_usage_shuffle_macro": benchmark_means(shuffled, domains).mean(),
                                "mean_models_selected": np.mean(models_used)})
                if budget == PRIMARY:
                    primary[arm, method] = correct
                    seed_rows.extend({**identity, "seed": seed, "macro_accuracy": q, "micro_accuracy": r,
                                      "models_selected": used} for seed, q, r, used in zip(SEEDS, macro, micro, models_used))
    paired = [{"family": family, "arm": EXCLUDED, "method": LEARNED, "comparator_arm": arm,
               "comparator": method, "probes_per_model": PRIMARY, "queries": len(domains),
               **paired_queries(primary[EXCLUDED, LEARNED], primary[arm, method], domains)}
              for arm, method in COMPARATORS]
    train_sets = [set(splits[arm]["train_model_ids"]) for arm in ARMS]
    identity = {"models": models, "probe_pool_questions": len(split["train_prompt_ids"]),
                "available_queries": len(prompts), "scored_common_queries": len(domains),
                "queries_by_benchmark": {dataset.benchmark_names[int(d)]: int((domains == d).sum())
                                         for d in np.unique(domains)},
                "training_models_per_arm": [len(ids) for ids in train_sets],
                "shared_training_models": len(train_sets[0] & train_sets[1]),
                "relatives_in_allowed_training": splits[RELATED]["metadata"]["related_count"]}
    return records, paired, seed_rows, identity, prompts[keep]


def run(archive, output):
    config = read_json(archive / "config.json")["data"]
    data = load_dataset(ROOT / config["prepared_path"], label_policy=config["label_policy"],
                        model_ids=sorted(load_family_table(ROOT / config["family_table"])),
                        benchmarks=config["candidate_benchmarks"], deduplicate_groups=config["deduplicate_groups"])
    positions = [{int(value): row for row, value in enumerate(ids)} for ids in (data.model_ids, data.prompt_ids)]
    summary, paired, by_seed, identities, question_ids = [], [], [], {}, []
    for family in FAMILIES:
        records, contrasts, seeds, identity, queries = score_family(archive, family, data, *positions)
        summary.extend(records)
        paired.extend(contrasts)
        by_seed.extend(seeds)
        identities[family] = identity
        question_ids.append(queries)

    same_queries = all(np.array_equal(question_ids[0], queries) for queries in question_ids[1:])
    if same_queries:
        family_mean = []
        for arm, method in COMPARATORS:
            rows = [row for row in paired if row["comparator_arm"] == arm and row["comparator"] == method]
            family_mean.append({**rows[0], "family": "equal_family_descriptive",
                                "macro_gain": np.mean([row["macro_gain"] for row in rows]),
                                "micro_gain": np.mean([row["micro_gain"] for row in rows]),
                                "macro_ci_low": "", "macro_ci_high": "", "positive_seed_count": ""})
        paired.extend(family_mean)
    atomic_csv(output / "summary.csv", summary)
    atomic_csv(output / "paired.csv", paired)
    atomic_csv(output / "by_seed.csv", by_seed)
    atomic_json(output / "protocol.json", {
        "phase": "Retrospective routing on EmbedLLM family tests already used for the score-prediction study.",
        "archive": archive.relative_to(ROOT) if archive.is_relative_to(ROOT) else archive,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": identities, "arm_order": ARMS, "budgets": BUDGETS, "primary_budget": PRIMARY, "seeds": SEEDS,
        "routing": "All candidates in one held family. Select one model per query by archived probability. Source ID breaks ties.",
        "comparators": "Archived smoothed global and benchmark means and KNN reuse the same probes and selected settings. The direct fixed choice uses equal-benchmark observed probe accuracy.",
        "training_composition": "The related-allowed and excluded arms swap training slots at equal roster size. The contrast can reflect replacements as well as ancestry.",
        "query_truth": "Prepared strict labels from the original view. Score only queries with complete binary outcomes across the same candidate pool.",
        "quality": "Macro equally weights the five benchmarks. Micro weights queries. Within-benchmark shuffle retains the observed selection shares.",
        "intervals": "Conditional paired bootstrap within benchmark, after averaging the same three seeds.",
        "equal_family_descriptive": same_queries,
    })
    print(f"Replayed archived routing for {len(FAMILIES)} held families at budgets {BUDGETS}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/family_lowrank_secondary_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/routing/family_holdout")
    args = parser.parse_args()
    run(args.archive, args.output)


if __name__ == "__main__":
    main()
