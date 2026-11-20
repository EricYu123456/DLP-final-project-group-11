"""Optimize the fraction of questions sent through a larger capability decoder using development data."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from coldstart.cascade import approximate, decision, fit_student
from coldstart.conditional_text import text_features
from coldstart.latent_skills import ItemTypeGate, gate_scores, measure_types, route_types
from evaluate_capability_router import ROOT, input_context, metrics
from panel_capabilities import panel_text
from run_experiment import atomic_csv, atomic_json


def train(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        where = {int(v): i for i, v in enumerate(tr)}
        panels = [np.array([where[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)]
    # This is the development-selected 120-question teacher. Each refit keeps its original measured embeddings.
    cfg = dict(
        name="latent-general",
        algorithm="types",
        readout="general",
        coordinates="joint-general",
        policy="balanced",
        probe_count=120,
        classes=12,
        steps=470,
        shrink=22,
        neighbor=0.0,
        type_mix=0.5,
        signal=1.7,
    )
    dest = args.out / "budget-cascade"
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / cfg["coordinates"] / "development_features.npz") as f:
        heavy = f["x"]
    rows = np.r_[tr, dev]
    text = panel_text(d.prompt_ids[rows], ROOT / "data/raw")
    begin = perf_counter()
    curves = []
    for population, folder in ((s.train_models, "dev"), (np.r_[s.train_models, s.val_models], "refit")):
        sub = Path("refit75") if folder == "refit" else Path()
        pf = joblib.load(args.out / "text" / sub / "readout.joblib")
        features = text_features(pf["pack"], d.embeddings[rows], text)
        with np.load(args.out / "text" / sub / "predictions.npz") as f:
            cheap, cp, domdev = f["historical"], f["dev"], f["dev_source"]
        with np.load(args.out / "general" / sub / "predictions.npz") as f:
            teacher, tp, tdev = f["historical"], f["dev"], f["dev_source"]
        h = d.labels[np.ix_(population, tr)]
        dt = d.domains[tr]
        base = args.out / "latent-general" / folder
        gate = ItemTypeGate(heavy.shape[1], int(dt.max()) + 1, cfg["classes"])
        gate.load_state_dict(torch.load(base / "gate_12_470.pt", map_location="cpu", weights_only=True))
        posterior = gate_scores(gate.eval(), heavy[: len(tr)], dt)
        student = fit_student(d.embeddings[tr], features[: len(tr)], dt, cheap, teacher, posterior)
        joblib.dump(student, dest / f"{folder}.joblib", compress=2)
        if folder == "refit":
            continue
        with np.load(base / "types_12.npz") as f:
            means, rates, res = f["means"], f["rates"], f["posterior"]
        y = d.labels[np.ix_(s.val_models, tr)]
        actual = d.labels[np.ix_(s.val_models, dev)]
        hp = gate_scores(gate, heavy[len(tr) :], tdev)
        pred, probs = approximate(d.embeddings[dev], features[len(tr) :], cp, domdev, student)
        configurations = []
        for idx in panels:
            z = measure_types(h, y, idx[: cfg["probe_count"]], dt, rates, res, cfg["shrink"], cfg["neighbor"])
            quick, risk = decision(z, pred, probs, domdev, means, student, cfg)
            full = route_types(z, tp, hp, tdev, means, cfg)
            configurations.append((quick, full, risk))
        for threshold in (-1.0, 0.04, 0.08, 0.13, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 1.0):
            use = [risk > threshold for _, _, risk in configurations]
            choices = np.stack(
                [
                    np.where(v, full.argmax(0), quick.argmax(0))
                    for (quick, full, _), v in zip(configurations, use)
                ]
            )
            curves.append(
                dict(
                    threshold=threshold,
                    heavy_query_fraction=float(np.mean(use)),
                    **metrics(choices, actual, d.domains[dev]),
                )
            )
    original = curves[0]
    eligible = [
        r
        for r in curves
        if r["macro"] >= original["macro"] - 0.0025
        and r["micro"] >= original["micro"] - 0.0025
        and r["top_share"] <= original["top_share"] + 0.02
    ]
    chosen = min(eligible, key=lambda r: (r["heavy_query_fraction"], -r["objective"]))
    c = dict(**cfg, teacher=cfg["name"], threshold=chosen["threshold"])
    c.update(name="budget-cascade", algorithm="cascade", readout="text")
    atomic_json(
        args.report / "cascade.json",
        dict(
            configs=[c],
            weights=[1],
            validation=chosen,
            teacher_validation=original,
            rule="Minimize large-decoder use within 0.25 points of its development macro and micro accuracy and 2 points of its concentration. All choices use the same 120-item embeddings.",
            shared_student_seconds=perf_counter() - begin,
        ),
    )
    atomic_csv(args.report / "cascade_validation.csv", curves)
    print("Cascade development", json.dumps(chosen), flush=True)


def reduce_probes(args):
    from coldstart.cascade import teacher_config
    from confirmation_models import ConfirmationModels

    base = json.loads((args.report / "cascade.json").read_text(encoding="utf-8"))
    cfg = base["configs"][0]
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    tr, dev = split["training"], split["development"]
    ix = {v: i for i, v in enumerate(tr)}
    orders = [np.array([ix[v] for v in split[f"balanced_{seed}"]]) for seed in (11, 23, 37)]
    runner = ConfirmationModels(args, d, s, split, dev, "development", False)
    quick = runner.component(cfg)
    full = runner.component(teacher_config(cfg))
    bank = quick["pack"]
    gt = d.labels[np.ix_(s.val_models, dev)]
    rows = []
    for n in (20, 30, 40, 60, 80, 100, 120):
        states = []
        for order in orders:
            panel = order[:n]
            # Slice the response table before estimating the candidate's model coordinates.
            grades = np.full((len(s.val_models), len(tr)), np.nan, dtype=np.float32)
            grades[:, panel] = d.labels[np.ix_(s.val_models, tr[panel])]
            z = measure_types(
                runner.h,
                grades,
                panel,
                runner.dt,
                bank["rates"],
                bank["posterior"],
                cfg["shrink"],
                cfg["neighbor"],
            )
            fast, risk = decision(
                z,
                quick["peer"],
                quick["posterior"],
                quick["head"]["dom"],
                bank["means"],
                quick["student"],
                cfg,
            )
            exact = route_types(
                z, full["head"]["p"], full["posterior"], full["head"]["dom"], bank["means"], cfg
            )
            states.append((fast.argmax(0), exact.argmax(0), risk))
        for th in (0.04, 0.08, 0.13, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 1.0):
            masks = [risk > th for _, _, risk in states]
            choices = np.stack(
                [np.where(mask, heavy, light) for mask, (light, heavy, _) in zip(masks, states)]
            )
            rows.append(
                dict(
                    probes=n,
                    threshold=th,
                    heavy_query_fraction=np.mean(masks),
                    **metrics(choices, gt, d.domains[dev]),
                )
            )
    baseline = base["validation"]
    valid = [
        r
        for r in rows
        if r["macro"] >= baseline["macro"] - 0.0025
        and r["micro"] >= baseline["micro"] - 0.0025
        and r["top_share"] <= baseline["top_share"] + 0.02
        and r["heavy_query_fraction"] <= baseline["heavy_query_fraction"] + 0.00001
    ]
    best = min(valid, key=lambda r: (r["probes"], r["heavy_query_fraction"], -r["objective"]))
    selected = dict(
        base,
        configs=[dict(cfg, probe_count=best["probes"], threshold=best["threshold"])],
        validation=best,
        rule="Minimize candidate probes, then large-decoder use, within 0.25 points of the 120-probe cascade on both development accuracy metrics and within 2 points of its concentration, using no more heavy queries.",
    )
    atomic_json(args.report / "cascade_budget.json", selected)
    atomic_csv(args.report / "cascade_budget_validation.csv", rows)
    print("Cascade probe budget", best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("train", "reduce-probes"), default="train")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(2)
    with threadpool_limits(limits=3):
        args = p.parse_args()
        (train if args.phase == "train" else reduce_probes)(args)
