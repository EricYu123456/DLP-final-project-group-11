"""Use a CPU text decision when it can replace the frozen 120-question portfolio on development rosters."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from confirmation_models import ConfirmationModels
from evaluate_capability_router import (
    ROOT,
    benchmark_means,
    input_context,
    metrics,
    paired_gain,
    test_rosters,
)
from panel_text_kernel import score_margin
from run_experiment import atomic_csv, atomic_json
from select_confirmation_router import SEEDS, dev_scores, dominant_pools, orders


def text_scores(out, roster, ids, prompts, budget):
    all_scores = []
    for seed in SEEDS:
        with np.load(out / "strict_panels" / f"{roster}_panel-text-{budget}_{seed}.npz") as f:
            pos = {int(v): i for i, v in enumerate(f["model_ids"])}
            if not np.array_equal(prompts, f["prompt_ids"]):
                raise ValueError("The cheap and full decoders require the same query order.")
            all_scores.append(f["scores"][[pos[int(v)] for v in ids]])
    return np.stack(all_scores)


def choices(cheap, full, threshold):
    if cheap.shape[1] == 1:
        return cheap.argmax(1), np.zeros_like(cheap[:, 0], dtype=bool)
    margins = np.stack([score_margin(v) for v in cheap])
    heavy = margins < threshold
    return np.where(heavy, full.argmax(1), cheap.argmax(1)), heavy


def validation(args, d, s, split, method, cfg, budget):
    q = split["development"]
    dom, truth = d.domains[q], d.labels[np.ix_(s.val_models, q)]
    y = d.labels[np.ix_(s.val_models, split["training"])]
    runner = ConfirmationModels(args, d, s, split, q, "development", False)
    runner.target_names = np.asarray(d.model_names)[s.val_models]
    bank = orders(d, s, split, s.train_models)
    full = np.average(
        [dev_scores(args, runner, v, y, bank.orders) for v in cfg["configs"]], axis=0, weights=cfg["weights"]
    )
    if cfg["allowance"] or cfg["specialization"]:
        from coldstart.switch_calibration import paired_uncertainty, rate_challengers

        for k, seed in enumerate(SEEDS):
            u = paired_uncertainty(runner.h, y, bank.orders["balanced", seed][:budget], runner.dt)
            full[k] = rate_challengers(full[k], u, cfg["allowance"], specialization=cfg["specialization"])
    cheap = text_scores(args.out, "development_13", d.model_ids[s.val_models], d.prompt_ids[q], budget)
    teacher = metrics(full.argmax(1), truth, dom)
    pools = dominant_pools(truth, dom)
    margins = np.concatenate([score_margin(v) for v in cheap])
    thresholds = np.unique(np.r_[0.0, np.quantile(margins, np.linspace(0.05, 0.95, 19)), 2.0])
    table = []
    for t in thresholds:
        routes, heavy = choices(cheap, full, t)
        gains = []
        for members in pools:
            local, _ = choices(cheap[:, members], full[:, members], t)
            base = full[:, members].argmax(1)
            local_truth = truth[members]
            diff = local_truth[local, np.arange(len(q))] - local_truth[base, np.arange(len(q))]
            gains.append(0.6 * benchmark_means(diff, dom).mean() + 0.4 * diff.mean())
        table.append(
            dict(
                threshold=float(t),
                full_decoder_share=heavy.mean(),
                subpool_gain=np.mean(gains),
                **metrics(routes, truth, dom),
            )
        )
    eligible = [
        r
        for r in table
        if r["macro"] >= teacher["macro"] - 0.0025
        and r["micro"] >= teacher["micro"] - 0.0025
        and r["top_share"] <= teacher["top_share"] + 0.02
        and r["subpool_gain"] >= -0.0025
    ]
    winner = min(eligible, key=lambda r: (r["full_decoder_share"], -r["objective"]))
    atomic_csv(args.report / "conditional_cost_validation.csv", table)
    atomic_json(
        args.report / "conditional_cost_selection.json",
        dict(
            method="gated-conditional",
            full_method=method,
            probes=budget,
            teacher=teacher,
            development=winner,
            subrosters=len(pools),
        ),
    )
    print("Frozen conditional cost point", winner, flush=True)


def confirmation(args, d, s, split, method, budget, prepared, family):
    selected = json.loads((args.report / "conditional_cost_selection.json").read_text(encoding="utf-8"))
    threshold = selected["development"]["threshold"]
    q = np.r_[split["confirmation"], split["previous_test"]]
    boundary = len(split["confirmation"])
    with (args.report / "selected_feature_cost.csv").open(encoding="utf-8", newline="") as stream:
        helpers = [row["helper"] for row in csv.DictReader(stream) if row["phase"] == "evaluation"]
    helper_count = np.zeros(len(q), int)
    for helper in helpers:
        if helper.endswith("-gated"):
            with np.load(args.out / f"evidence_{helper}_evaluation.npz") as f:
                if not np.array_equal(f["prompt_ids"], d.prompt_ids[q]):
                    raise ValueError("The helper cost and confirmation questions differ.")
                helper_count += f["probabilities"][:, 0, 6].astype(int)
        else:
            helper_count += 1
    _, rosters = test_rosters(d, s, prepared, family)
    rows, pairs = [], []
    for roster, unordered, labels in rosters:
        order = np.argsort(unordered)
        ids, truth = unordered[order], labels[np.ix_(order, q)]
        with np.load(args.out / f"{roster}_{method}_trials.npz") as f:
            if not np.array_equal(f["model_ids"], ids) or not np.array_equal(
                f["prompt_ids"], d.prompt_ids[q]
            ):
                raise ValueError("The teacher's roster or prompts differ from the confirmation data.")
            full = f["scores"]
        cheap = text_scores(args.out, roster, ids, d.prompt_ids[q], budget)
        route, heavy = choices(cheap, full, threshold)
        with np.load(args.out / f"choices_75_{roster}.npz") as f:
            references = {
                k: f[k]
                for k in (
                    method,
                    "previous-8D-refit",
                    "EmbedLLM-known",
                    "EmbedLLM-noisy",
                    f"EmbedLLM-balanced-{budget}",
                    "best-fixed-hindsight",
                )
                if k in f
            }
        pool = joblib.load(args.out / f"{roster}_{method}.joblib")
        fast = joblib.load(args.out / f"{roster}_panel-text-{budget}.joblib")
        if not np.array_equal(pool["probe_ids"], fast["probe_ids"]):
            raise ValueError("A two-stage router must share its entire onboarding panel.")
        pool.update(method="gated-conditional", fast_panel=fast, accept_margin=threshold)
        joblib.dump(pool, args.out / f"{roster}_gated-conditional.joblib", compress=2)
        combined = np.where(heavy[0, None, :], full[0], cheap[0])
        np.savez_compressed(
            args.out / f"{roster}_gated-conditional_scores.npz",
            scores=combined,
            prompt_ids=d.prompt_ids[q],
            large_decoder_used=heavy[0],
        )
        for source, indices in (
            ("confirmation", np.arange(boundary)),
            ("previous-test", np.arange(boundary, len(q))),
        ):
            target, dom = truth[:, indices], d.domains[q[indices]]
            hit = target[route[:, indices], np.arange(len(indices))]
            rows.append(
                dict(
                    split=source,
                    roster=roster,
                    method="gated-conditional",
                    probes=budget,
                    full_decoder_share=heavy[:, indices].mean(),
                    full_mean_helpers=helper_count[indices].mean(),
                    gated_mean_helpers=(heavy[:, indices] * helper_count[indices]).mean(),
                    **metrics(route[:, indices], target, dom),
                )
            )
            for name, choices_ref in references.items():
                other = target[choices_ref[:, indices], np.arange(len(indices))]
                delta = hit - other
                pairs.append(
                    dict(
                        split=source,
                        roster=roster,
                        reference=name,
                        **paired_gain(hit, other, dom),
                        rescues=(delta > 0).sum(1).mean(),
                        harms=(delta < 0).sum(1).mean(),
                    )
                )
    atomic_csv(args.report / "conditional_cost_results.csv", rows)
    atomic_csv(args.report / "conditional_cost_paired.csv", pairs)
    print(rows, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("validation", "confirmation"), required=True)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    args = p.parse_args()
    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    lock = json.loads((args.report / "selection.json").read_text(encoding="utf-8"))
    method = lock["default_method"]
    cfg = lock[method]
    budget = cfg["configs"][0]["probe_count"]
    with threadpool_limits(limits=3):
        if args.phase == "validation":
            validation(args, d, s, split, method, cfg, budget)
        else:
            confirmation(args, d, s, split, method, budget, prepared, family)
