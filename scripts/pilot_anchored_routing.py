"""Cross-fit an admission margin around the support-selected fixed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_routing import MARGINS, ROOT, benchmark_means
from coldstart.data import load_dataset
from coldstart.splits import load_family_table, load_split
from run_experiment import atomic_csv, atomic_json


FOLD_SEED = 20261115
DRAW_SEED = 8675309
DRAW_COUNT = 2000


def query_folds(domains, groups):
    if len(np.unique(groups)) != len(groups):
        raise ValueError("This cross-fit requires one representative per question group.")
    folds = np.empty(len(domains), dtype=np.int64)
    for domain in np.unique(domains):
        rng = np.random.default_rng(np.random.SeedSequence([FOLD_SEED, int(domain)]))
        queries = rng.permutation(np.flatnonzero(domains == domain))
        folds[queries] = np.arange(len(queries)) % 2
    return folds


def conditional_interval(delta, domains, folds):
    """Resample paired queries within each benchmark and fold with thresholds fixed."""
    rng = np.random.default_rng(DRAW_SEED)
    by_query = delta.mean(axis=0)
    domains_present = np.unique(domains)
    bootstrap = np.zeros(DRAW_COUNT)
    for domain in domains_present:
        domain_mask = domains == domain
        for fold in (0, 1):
            subsample = by_query[domain_mask & (folds == fold)]
            draws = rng.choice(subsample, size=(DRAW_COUNT, len(subsample)))
            bootstrap += draws.sum(axis=1) / (domain_mask.sum() * len(domains_present))
    return np.quantile(bootstrap, [0.025, 0.975])


def summarize(identity, choices, anchors, truth, domains):
    query_positions = np.arange(truth.shape[1])
    predictions = truth[choices, query_positions]
    fixed = truth[anchors]
    delta = predictions - fixed
    opportunities = truth.max(axis=0) > fixed
    rescue = delta > 0
    harm = delta < 0
    return {
        **identity, "queries": len(domains),
        "macro_accuracy": benchmark_means(predictions, domains).mean(),
        "micro_accuracy": predictions.mean(dtype=np.float64),
        "macro_gain_over_fixed": benchmark_means(delta, domains).mean(),
        "micro_gain_over_fixed": delta.mean(dtype=np.float64),
        "challenger_share": (choices != anchors[:, None]).mean(),
        "rescues_per_seed": rescue.sum(axis=1).mean(),
        "harms_per_seed": harm.sum(axis=1).mean(),
        "opportunities_per_seed": opportunities.sum(axis=1).mean(),
        "opportunity_capture": rescue.sum() / opportunities.sum(),
    }


def validated_probabilities(run_path, seed, budget, model_ids, prompt_ids):
    name = f"seed{seed}_balanced_random_k{budget}_coldstart_neural_lowrank.npz"
    with np.load(run_path / "validation_predictions" / name, allow_pickle=False) as saved:
        if not (np.array_equal(saved["model_ids"], model_ids) and np.array_equal(saved["prompt_ids"], prompt_ids)):
            raise ValueError(f"Model or query IDs do not align in {name}")
        return saved["probabilities"].astype(np.float64)


def run(run_path: Path, output: Path) -> None:
    config = json.loads((run_path / "effective_config.json").read_text(encoding="utf-8"))
    data = config["data"]
    suite = load_dataset(ROOT / data["prepared_path"], label_policy=data["label_policy"],
                         model_ids=sorted(load_family_table(ROOT / data["family_table"])),
                         benchmarks=data["candidate_benchmarks"], deduplicate_groups=data["deduplicate_groups"])
    split = load_split(run_path / "split.json", suite)
    validation_probes = json.loads((run_path / "validation_probes.json").read_text(encoding="utf-8"))
    candidates, queries = split.val_models, split.val_items
    model_ids, prompt_ids = suite.model_ids[candidates], suite.prompt_ids[queries]
    domains = suite.benchmark_ids[queries]
    folds = query_folds(domains, suite.group_ids[queries])
    truth = suite.labels[np.ix_(candidates, queries)]
    support_by_id = dict(zip(suite.prompt_ids[split.train_items], split.train_items))
    seeds, budgets = config["seeds"], config["budgets"]
    support = {}
    for seed in seeds:
        orders = [row["prompt_ids"] for row in validation_probes
                  if row["seed"] == seed and row["strategy"] == "balanced_random"]
        if len(orders) != len(candidates) or any(order != orders[0] for order in orders):
            raise ValueError(f"Expected one shared probe order among candidates at seed {seed}.")
        support[seed] = np.array([support_by_id[prompt] for prompt in orders[0]])

    summary, fold_summary, seed_summary = [], [], []
    for budget in budgets:
        probabilities = []
        anchors = []
        for seed in seeds:
            selected = support[seed][:budget]
            scores = benchmark_means(suite.labels[np.ix_(candidates, selected)],
                                     suite.benchmark_ids[selected]).mean(axis=1)
            anchors.append(np.lexsort((model_ids, -scores))[0])
            probabilities.append(validated_probabilities(run_path, seed, budget, model_ids, prompt_ids))
        probabilities = np.stack(probabilities)
        anchors = np.asarray(anchors)
        seed_index = np.arange(len(seeds))
        query_index = np.arange(len(queries))
        strongest = np.argmax(probabilities, axis=1)
        edge = probabilities[seed_index[:, None], strongest, query_index] - probabilities[seed_index, anchors]
        possible = truth[strongest, query_index] - truth[anchors]
        fixed = np.broadcast_to(anchors[:, None], strongest.shape)
        gated = fixed.copy()
        for fold in (0, 1):
            evaluate = folds == fold
            calibrate = ~evaluate
            scores = [benchmark_means(possible[:, calibrate] * (edge[:, calibrate] > margin),
                                      domains[calibrate]).mean() for margin in MARGINS]
            chosen_index = max(range(len(MARGINS)), key=lambda i: (scores[i], MARGINS[i]))
            selected_margin = MARGINS[chosen_index]
            gated[:, evaluate] = np.where(edge[:, evaluate] > selected_margin,
                                          strongest[:, evaluate], fixed[:, evaluate])
            fold_summary.append(summarize({"budget": budget, "eval_fold": fold,
                                          "margin": selected_margin, "calibration_macro_gain": scores[chosen_index],
                                          "anchor_model_ids": ",".join(map(str, model_ids[anchors]))},
                                          gated[:, evaluate], anchors, truth[:, evaluate], domains[evaluate]))
        for method, choices in (("fixed_support", fixed), ("embedding_argmax", strongest),
                                ("cross_fit_margin", gated)):
            row = summarize({"budget": budget, "method": method}, choices, anchors, truth, domains)
            if method == "cross_fit_margin":
                delta = truth[choices, query_index] - truth[anchors]
                row["conditional_ci_low"], row["conditional_ci_high"] = conditional_interval(delta, domains, folds)
            summary.append(row)
        for index, seed in enumerate(seeds):
            seed_summary.append(summarize({"budget": budget, "seed": seed,
                                           "anchor_model_id": int(model_ids[anchors[index]])},
                                          gated[index:index + 1], anchors[index:index + 1], truth, domains))

    atomic_csv(output / "summary.csv", summary)
    atomic_csv(output / "folds.csv", fold_summary)
    atomic_csv(output / "by_seed.csv", seed_summary)
    atomic_json(output / "protocol.json", {
        "phase": "exploratory validation only, including validation reuse from the prior decoder selection",
        "seed_order": seeds, "budgets": budgets, "fold_seed": FOLD_SEED,
        "calibration": "choose one macro-gain maximizing margin per budget on the opposite question fold, ties larger",
        "interval": "95% percentile, resample paired queries within benchmark and fold after seed averaging, thresholds fixed",
        "interval_draws": DRAW_COUNT, "interval_seed": DRAW_SEED, "margins": MARGINS,
        "validation_model_ids": model_ids,
        "validation_prompt_ids_by_fold": {str(i): prompt_ids[folds == i] for i in (0, 1)},
    })
    print(f"Saved cross-fit validation results to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts" / "main_v2",
                        help="Run containing saved balanced-random validation predictions.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "routing" / "anchored_pilot")
    args = parser.parse_args(argv)
    run(args.run, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
