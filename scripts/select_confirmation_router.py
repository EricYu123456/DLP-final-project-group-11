"""Choose a common-panel portfolio on development LLMs before opening confirmation outcomes."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from confirmation_models import ConfirmationModels
from evaluate_capability_router import (
    ROOT,
    benchmark_means,
    input_context,
    known_choices,
    metrics,
    original_cold,
)
from coldstart.capability import representative_order
from coldstart.switch_calibration import paired_uncertainty, rate_challengers
from run_experiment import atomic_csv, atomic_json


SEEDS = (11, 23, 37)


def orders(d, s, split, ref):
    tr = split["training"]
    pos = {int(v): i for i, v in enumerate(tr)}
    h = d.labels[np.ix_(ref, tr)]
    out = {}
    for seed in SEEDS:
        out["balanced", seed] = np.asarray([pos[int(x)] for x in split[f"balanced_{seed}"]])
        out["representative", seed] = representative_order(h, d.domains[tr], 200, seed)
    return SimpleNamespace(items=tr, orders=out, seeds=SEEDS)


def baseline_lock(args, d, s, split):
    dev = split["development"]
    bank = orders(d, s, split, s.train_models)
    y = d.labels[s.val_models]
    truth, dom = y[:, dev], d.domains[dev]
    baselines = {}
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native, noisy = known_choices(f, d, d.model_ids[s.val_models], dev, SEEDS)
    for name, choice in (("native", native), ("native-noisy", noisy)):
        baselines[name] = metrics(choice, truth, dom)
    with np.load(args.out / "embedllm_historical.npz") as f:
        for policy in ("balanced", "representative"):
            for n in (120, 200) if policy == "balanced" else (200,):
                fits = []
                for strength in (0.1, 1.0, 10.0):
                    choice = original_cold(f, d, bank, y, dev, policy, strength, n)
                    fits.append(
                        dict(strength=strength, policy=policy, budget=n, **metrics(choice, truth, dom))
                    )
                baselines[f"cold-{policy}-{n}"] = max(fits, key=lambda r: r["objective"])
    atomic_json(args.report / "references.json", baselines)
    return baselines


def candidate_settings(args, targets, budget, alternatives=False):
    cfgs = {}
    for file in sorted(args.report.glob("*_selection.json")):
        contents = json.loads(file.read_text(encoding="utf-8"))
        if "best" not in contents or "native" not in contents:
            continue
        cfg = contents["best"]
        if cfg.get("fold", "dev") != "dev":
            continue
        if cfg.get("algorithm") == "reference":
            # This companion experiment names a peer using the target roster's probe responses.
            continue
        table = pd.read_csv(file.with_name(file.name.replace("_selection.json", "_validation.csv")))
        policy = table.policy != "balanced" if alternatives else table.policy == "balanced"
        table = table[policy & (table.probe_count == budget) & table.robust_objective.notna()]
        if table.empty:
            continue
        rows = table.astype(object).where(pd.notna(table), None).to_dict("records")
        kind = cfg.get("algorithm", "regional")

        def rank(r):
            return (
                r["macro"] > targets[0] and r["micro"] > targets[1] and r["top_share"] < 0.92,
                r["robust_objective"],
            )

        row = max(rows, key=rank)
        row["algorithm"] = kind
        # Keep the best validation readout for each type of capability model and the CPU-only alternative.
        for key in (
            [kind, "cpu"]
            if row.get("readout", row["name"]) == "text"
            and not row.get("proxies")
            and not row.get("axes")
            and kind != "distilled"
            and not alternatives
            else [kind]
        ):
            if key not in cfgs or rank(row) > rank(cfgs[key]):
                cfgs[key] = row
    return cfgs


def dominant_pools(truth, dom):
    distinct = {}
    rng = np.random.default_rng(3011)
    rates = benchmark_means(truth, dom)
    for anchor in np.argsort(-truth.mean(1))[:5]:
        weaker = np.flatnonzero(
            ((rates <= rates[anchor] + 0.01).sum(1) >= 4) & (truth.mean(1) < truth[anchor].mean() - 0.04)
        )
        for size in (4, 7, 10):
            if len(weaker) < size - 1:
                continue
            for _ in range(3):
                others = tuple(sorted(rng.choice(weaker, size - 1, replace=False).tolist()))
                distinct[(int(anchor), *others)] = 1
    return [np.asarray(k) for k in distinct]


def portfolio_results(scores, truth, dom, pools, uncertainty, allowance, specialization=0.0):
    choices = np.stack(
        [
            rate_challengers(v, p, allowance, specialization=specialization).argmax(0)
            for v, p in zip(scores, uncertainty)
        ]
    )
    row = metrics(choices, truth, dom)
    rates = benchmark_means(truth, dom)
    differences, diversion, concentration = [], [], []
    for pool in pools:
        use = pool[
            np.stack(
                [
                    rate_challengers(v, p, allowance, pool, specialization).argmax(0)
                    for v, p in zip(scores, uncertainty)
                ]
            )
        ]
        diff = truth[use, np.arange(len(dom))] - truth[pool[0]]
        differences.append(0.6 * benchmark_means(diff, dom).mean() + 0.4 * diff.mean())
        diversion.append((use != pool[0]).mean())
        concentration.append(np.mean([np.bincount(r).max() / len(dom) for r in use]))
    within = truth[choices, np.arange(len(dom))] - rates[choices, dom]
    row["dominant_pool_gain"] = float(np.mean(differences)) if differences else 0.0
    row["dominant_pool_switch_share"] = float(np.mean(diversion)) if diversion else 0.0
    row["dominant_top_quartile"] = (
        float(np.quantile(concentration, 0.75)) if concentration else row["top_share"]
    )
    row["within_source_gain"] = float(benchmark_means(within, dom).mean())
    row["selection_score"] = (
        row["objective"] + 1.4 * row["dominant_pool_gain"] + 0.4 * row["within_source_gain"]
    )
    return row


def dev_scores(args, runner, cfg, y, order):
    if cfg["algorithm"] == "joint":
        file = (
            args.out
            / cfg["name"]
            / f"dev_{int(cfg['probe_count'])}_{cfg['real_weight']}_{int(cfg['steps'])}.npz"
        )
        with np.load(file) as f:
            return f["scores"]
    if cfg["algorithm"] == "tabular":
        from coldstart.tabular_capability import query_skills

        pair_cfg = dict(name=cfg["pairs"], algorithm="paired", readout=cfg["readout"], axes=cfg.get("axes"))
        p = runner.component(pair_cfg)
        with np.load(args.out / cfg["name"] / "targets.npz") as f:
            xx = query_skills(p["xx"], p["head"]["dom"], f)
            xx[:, len(f["center"]) :] *= cfg["signal"]
        scores = []
        for s in SEEDS:
            with np.load(args.out / cfg["name"] / f"dev_{s}_{int(cfg['probe_count'])}.npz") as f:
                z = (1 - cfg["tabular_mix"]) * f["prior"] + cfg["tabular_mix"] * f["fit"]
                scores.append(z @ xx.T)
        return np.stack(scores)
    return np.stack(
        [runner.score(cfg, y, runner.panel(cfg, y, seed, order["balanced", seed]), seed)[0] for seed in SEEDS]
    )


def freeze(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    ref_file = args.report / "references.json"
    refs = json.loads(ref_file.read_text()) if ref_file.exists() else baseline_lock(args, d, s, split)
    cold = max(
        (r for k, r in refs.items() if k.startswith("cold-") and r["budget"] == 200),
        key=lambda r: r["objective"],
    )
    targets = [max(refs["native"][k], cold[k]) for k in ("macro", "micro")]
    order = orders(d, s, split, s.train_models).orders
    dev = split["development"]
    runner = ConfirmationModels(args, d, s, split, dev, "development", False)
    runner.target_names = np.asarray(d.model_names)[s.val_models]
    y = d.labels[np.ix_(s.val_models, split["training"])]
    truth, dom = d.labels[np.ix_(s.val_models, dev)], d.domains[dev]
    pools = dominant_pools(truth, dom)
    records = []
    settings = {}
    outputs = {}
    uncertainties = {}

    def key(r):
        return (
            r["macro"] > targets[0] and r["micro"] > targets[1] and r["top_share"] < 0.92,
            r["dominant_pool_gain"] > 0 and r["dominant_pool_switch_share"] >= 0.05,
            r["dominant_pool_gain"] >= 0,
            r["selection_score"],
        )

    for n in (120, 200):
        cfgs = candidate_settings(args, targets, n)
        scores = {name: dev_scores(args, runner, cfg, y, order) for name, cfg in cfgs.items()}
        outputs[n] = scores
        settings[n] = cfgs
        uncertainty = [
            paired_uncertainty(runner.h, y, order["balanced", seed][:n], runner.dt) for seed in SEEDS
        ]
        uncertainties[n, "balanced"] = uncertainty
        variants = [(name,) for name in cfgs]
        kinds = [k for k in cfgs if k != "cpu"]
        variants += list(combinations(kinds, 2)) + list(combinations(kinds, 3))
        if len(kinds) > 3:
            variants.append(tuple(kinds))
        for combo in variants:
            for weights in (
                [(1,)]
                if len(combo) == 1
                else ([(1, 1), (2, 1), (1, 2)] if len(combo) == 2 else [tuple([1] * len(combo))])
            ):
                pred = np.average([scores[k] for k in combo], axis=0, weights=weights)
                for allowance in (0.0, 0.4, 0.85):
                    row = dict(
                        components=list(combo),
                        weights=list(weights),
                        questions=n,
                        allowance=allowance,
                        specialization=0.0,
                        **portfolio_results(pred, truth, dom, pools, uncertainty, allowance),
                    )
                    records.append(row)
        for kind, cfg in candidate_settings(args, targets, n, alternatives=True).items():
            label = f"{kind}-{cfg['policy']}"
            settings[n][label] = cfg
            pred = dev_scores(args, runner, cfg, y, order)
            outputs[n][label] = pred
            special = [
                paired_uncertainty(
                    runner.h, y, runner.panel(cfg, y, seed, order["balanced", seed]), runner.dt
                )
                for seed in SEEDS
            ]
            uncertainties[n, label] = special
            for allowance in (0.0, 0.4, 0.85):
                records.append(
                    dict(
                        components=[label],
                        weights=[1],
                        questions=n,
                        allowance=allowance,
                        specialization=0.0,
                        **portfolio_results(pred, truth, dom, pools, special, allowance),
                    )
                )
    primary = max(records, key=key)
    economy = max((r for r in records if r["components"] == ["cpu"]), key=key)
    # Select the second operating point against the released router and the best fixed model
    # on dominance-matched subrosters. Only the probes set each actual candidate's general ability.
    shortlist = {}
    for r in records:
        if r["allowance"] or (len(r["components"]) > 1 and not key(r)[0]):
            continue
        k = r["questions"], tuple(r["components"])
        if k not in shortlist or key(r) > key(shortlist[k]):
            shortlist[k] = r
    singles = [r for r in shortlist.values() if len(r["components"]) == 1]
    varied = (
        singles
        + sorted((r for r in shortlist.values() if len(r["components"]) > 1), key=key, reverse=True)[:15]
    )
    for r in varied:
        n, combo = r["questions"], r["components"]
        pred = np.average([outputs[n][c] for c in combo], axis=0, weights=r["weights"])
        special = uncertainties.get((n, combo[0]), uncertainties[n, "balanced"])
        for weight in (0.25, 0.5, 0.70):
            for allowance in (0.0, 0.65, 1.2):
                records.append(
                    dict(
                        components=combo,
                        weights=r["weights"],
                        questions=n,
                        allowance=allowance,
                        specialization=weight,
                        **portfolio_results(pred, truth, dom, pools, special, allowance, weight),
                    )
                )

    def switch_key(r):
        original = refs["native-noisy"]
        qualifies = r["micro"] > original["micro"] + 0.005 and r["macro"] > original["macro"] + 0.005
        competitive = r["micro"] >= targets[1] - 0.005 and r["macro"] >= targets[0] - 0.005
        return (
            qualifies,
            competitive,
            r["dominant_top_quartile"] <= 0.90 and r["dominant_pool_gain"] >= -0.003,
            r["dominant_top_quartile"] <= 0.93,
            r["within_source_gain"] > 0,
            r["selection_score"]
            + 0.4 * r["within_source_gain"]
            - 0.10 * max(0, r["dominant_top_quartile"] - 0.88),
        )

    more_switch = max(records, key=switch_key)
    from panel_stopping import validate_stopping

    stopped = validate_stopping(
        runner, settings[economy["questions"]]["cpu"], y, truth, dom, order, SEEDS, economy["questions"], ROOT
    )

    def packed(r):
        return dict(
            validation=r,
            configs=[settings[r["questions"]][c] for c in r["components"]],
            weights=r["weights"],
            allowance=r["allowance"],
            specialization=r["specialization"],
        )

    suffix = "_interim" if args.preview else ""
    cascade = json.loads((args.report / "cascade.json").read_text(encoding="utf-8"))
    cpu_curve = pd.read_csv(args.report / "cascade_validation.csv").iloc[-1].to_dict()
    cpu_cascade = dict(
        configs=[dict(cascade["configs"][0], threshold=1.0)], weights=[1], validation=cpu_curve
    )
    fast = cascade["validation"]
    use_cascade = (
        fast["heavy_query_fraction"] < 1
        and fast["macro"] >= more_switch["macro"] - 0.0025
        and fast["micro"] >= more_switch["micro"] - 0.0025
        and fast["top_share"] <= more_switch["top_share"] + 0.02
    )
    budget = json.loads((args.report / "cascade_budget.json").read_text(encoding="utf-8"))
    short = budget["validation"]
    use_short = (
        (
            short["probes"] < cascade["configs"][0]["probe_count"]
            or short["heavy_query_fraction"] < fast["heavy_query_fraction"]
        )
        and short["macro"] >= more_switch["macro"] - 0.0025
        and short["micro"] >= more_switch["micro"] - 0.0025
        and short["top_share"] <= more_switch["top_share"] + 0.02
    )
    atomic_csv(args.report / f"portfolio_validation{suffix}.csv", records)
    atomic_json(
        args.report / f"selection{suffix}.json",
        dict(
            primary=packed(primary),
            conditional=packed(more_switch),
            economical=packed(economy),
            adaptive=stopped,
            cascade=cascade,
            **{"cascade-cpu": cpu_cascade, "cascade-budget": budget},
            default_method="cascade-budget" if use_short else ("cascade" if use_cascade else "conditional"),
            default_rule="Prefer the cascade with lower onboarding or decoder cost, then the original cascade, when development macro and micro are each within 0.25 points of the conditional policy and concentration is within 2 points.",
            strong_cold_policy=cold,
            dominant_development_pools=len(pools),
            targets=targets,
            rule="Primary selection prioritizes gains in both accuracy metrics over the two stronger controls and top share below 92%, followed by gains in dominant validation subrosters. The conditional rule prioritizes beating the published default by at least 0.5 points on both metrics. It next seeks accuracy within 0.5 points of the stronger controls, a 75th-percentile subroster concentration at most 90%, and average subroster loss at most 0.3 points. Query routing uses measured scores and requires a shared panel for all components.",
            primary_reference_models=75,
            ablation_reference_models=62,
        ),
    )
    print("Frozen primary", primary, "CPU", economy, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    p.add_argument("--references-only", action="store_true")
    p.add_argument(
        "--preview",
        action="store_true",
        help="Compare completed model families on development before locking confirmation.",
    )
    args = p.parse_args()
    with threadpool_limits(limits=3):
        if args.references_only:
            dataset, split, *_ = input_context(args.run)
            with np.load(args.out / "questions.npz") as f:
                baseline_lock(args, dataset, split, f)
        else:
            freeze(args)
