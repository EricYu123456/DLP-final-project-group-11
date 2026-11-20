"""Learn when a strong historical model fails, then measure each newcomer's conditional strengths on its panel."""

import argparse
import json
from itertools import product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from coldstart.reference_outcomes import (
    embed,
    features,
    fit_state_readouts,
    probability,
    scores,
    state_effects,
    state_embedding,
    state_scores,
    train_verifier,
)
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, proxy_data, robustness


def train(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {v: i for i, v in enumerate(tr)}
        orders = {seed: np.asarray([pos[v] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    dt, dd = d.domains[tr], d.domains[dev]
    support = proxy_data(args.out, d.prompt_ids[tr], args.proxies, "development")
    validation = proxy_data(args.out, d.prompt_ids[dev], args.proxies, "development")
    y = d.labels[np.ix_(s.val_models, tr)]
    truth = d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(orders))[0], truth, dd)
    dest = args.out / args.name
    dest.mkdir(exist_ok=True)
    rows, costs = [], []
    for label, cohort in (("dev", s.train_models), ("refit", np.r_[s.train_models, s.val_models])):
        base = args.out / args.readout / ("refit75" if label == "refit" else "")
        with np.load(base / "predictions.npz") as f:
            oof, peer, source = f["historical"], f["dev"], f["dev_source"]
        h = d.labels[np.ix_(cohort, tr)]
        anchor = int(np.flatnonzero(d.model_ids[cohort] == 39).item())
        begin = perf_counter()
        train_x = features(support, oof, dt, len(np.unique(dt)))
        if args.from_verifier:
            parent = args.out / args.from_verifier
            pack = joblib.load(parent / f"{label}.joblib")
            previous = json.loads((parent / "cost.json").read_text(encoding="utf-8"))["training"]
            brier = next(row["oof_brier"] for row in previous if row["reference_models"] == len(cohort))
            pack["state_readouts"] = fit_state_readouts(train_x, dt, h, pack)
        else:
            pack, held = train_verifier(train_x, dt, h, anchor)
            brier = float(np.mean((held - h[anchor]) ** 2))
        joblib.dump(pack, dest / f"{label}.joblib", compress=2)
        if not args.from_verifier:
            np.savez_compressed(dest / f"{label}_oof.npz", chance=held)
        costs.append(
            dict(
                reference_models=len(cohort),
                peer_model=str(d.model_names[cohort[anchor]]),
                training_seconds=perf_counter() - begin,
                oof_brier=brier,
                text_head_brier=float(np.mean((oof[:, anchor] - h[anchor]) ** 2)),
            )
        )
        if label == "refit":
            continue
        query_x = features(validation, peer, source, len(pack["means"]))
        chance = probability(pack, query_x)
        if args.from_verifier:
            effects = state_effects(pack["state_readouts"], query_x)
        for n, near, shrink in product((120, 200), (0.2, 0.75), (9, 35)):
            embedder = state_embedding if args.from_verifier else embed
            embeds = [embedder(pack, h, y, idx[:n], dt, near, shrink) for idx in orders.values()]
            settings = (
                product((0.55, 1.05, 1.65), (0.7, 1.0, 1.35))
                if args.from_verifier
                else product((0.8, 1.3, 2.0), (0.35, 1.0))
            )
            for signal, weight in settings:
                cfg = dict(
                    name=args.name,
                    algorithm="reference",
                    readout=args.readout,
                    proxies=" ".join(args.proxies),
                    policy="balanced",
                    probe_count=n,
                    nearest=near,
                    shrink=shrink,
                    signal=signal,
                    peer=weight,
                )
                if args.from_verifier:
                    cfg.update(conditional_states=True, calibration=weight)
                    pred = np.stack(
                        [state_scores(pack, z, chance, peer, source, effects, signal, weight) for z in embeds]
                    )
                else:
                    pred = np.stack([scores(pack, z, chance, peer, source, cfg) for z in embeds])
                row = dict(**cfg, **metrics(pred.argmax(1), truth, dd))
                row["subpool_gain"], row["within_domain_gain"] = robustness(pred, truth, dd, rosters)
                row["robust_objective"] = row["objective"] + 0.5 * (
                    row["subpool_gain"] + row["within_domain_gain"]
                )
                rows.append(row)

    def rank(row):
        return row["macro"] > native["macro"] and row["micro"] > native["micro"] and row[
            "top_share"
        ] < 0.92, row["robust_objective"]

    best = max(rows, key=rank)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in rows if r["probe_count"] == 120), key=rank), native=native),
    )
    atomic_json(dest / "cost.json", dict(training=costs))
    print(args.name, best, costs, flush=True)


def compare(args):
    import pandas as pd
    from confirmation_models import ConfirmationModels
    from evaluate_capability_router import test_rosters
    from evaluate_routing_confirmation import predictions, summarize
    from select_confirmation_router import orders

    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    q = np.r_[split["confirmation"], split["previous_test"]]
    cut = np.arange(len(q)) < len(split["confirmation"])
    lock = json.loads((args.report / "selection.json").read_text(encoding="utf-8"))
    chosen = {}
    for method in ("reference-modern", "reference-live", "reference-joint-modern", "reference-joint-live"):
        valid = pd.read_csv(args.report / f"{method}_validation.csv")
        rows = valid[valid.probe_count == 200].to_dict("records")
        cfg = max(
            rows,
            key=lambda x: (
                x["macro"] > lock["targets"][0] and x["micro"] > lock["targets"][1] and x["top_share"] < 0.92,
                x["robust_objective"],
            ),
        )
        chosen[method] = dict(configs=[cfg], weights=[1])
    live = chosen["reference-live"]["configs"][0]
    chosen["reference-panel"] = dict(configs=[dict(live, panel_calibration=True)], weights=[1])
    atomic_json(
        args.report / "reference_profiles.json",
        dict(
            **chosen,
            default_method="reference-live",
            selection="Each estimator is fixed on its development roster at 200 common questions before evaluating the separate historical-peer hypothesis.",
        ),
    )
    opt = SimpleNamespace(**vars(args), prefix="", split="both")
    runner = ConfirmationModels(opt, d, s, split, q, "evaluation", True)
    bank = orders(d, s, split, runner.ref)
    names, rosters = test_rosters(d, s, prepared, family)
    records = ([], [], [], [])
    costs = []
    for roster, unordered, labels in rosters:
        order = unordered.argsort()
        ids, y = unordered[order], labels[order]
        pred, cost, _, _ = predictions(opt, runner, chosen, bank, y[:, split["training"]], roster, ids, names)
        costs.extend(cost)
        with np.load(args.out / f"choices_75_{roster}.npz") as f:
            if not np.array_equal(f["model_ids"], ids) or not np.array_equal(
                f["prompt_ids"], d.prompt_ids[q]
            ):
                raise ValueError(
                    "Reference choices and the confirmation experiment differ in models or questions."
                )
            ref = lock["strong_cold_policy"]
            cold = f"EmbedLLM-{ref['policy']}-{ref['budget']}"
            for key in (
                "EmbedLLM-known",
                "EmbedLLM-noisy",
                cold,
                "EmbedLLM-balanced-200",
                "previous-8D-refit",
                "probe-best-fixed",
            ):
                pred[key] = f[key]
        summarize(
            roster,
            75,
            cold,
            [("confirmation", cut), ("previous-test", ~cut)],
            ids,
            y[:, q],
            d.domains[q],
            d,
            pred,
            records,
            chosen,
        )
    for title, data in zip(("results", "seeds", "paired", "by_benchmark", "costs"), (*records, costs)):
        atomic_csv(args.report / f"reference_{title}.csv", data)
    print(
        pd.DataFrame(records[0]).query("method.str.startswith('reference-')").to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("train", "compare"), default="train")
    p.add_argument("--name")
    p.add_argument("--proxies", nargs="+")
    p.add_argument(
        "--from-verifier",
        help="Reuse the verifier and learn conditional historical rescue and harm readouts.",
    )
    p.add_argument("--readout", default="general")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        args = p.parse_args()
        if args.phase == "train" and (not args.name or not args.proxies):
            p.error("Training needs --name and --proxies.")
        (train if args.phase == "train" else compare)(args)
