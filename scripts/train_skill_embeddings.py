"""Fit model embeddings from measured skills, paired probe outcomes and a historical cohort prior."""

import argparse
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from train_conditional_router import (
    ROOT,
    features_for,
    input_context,
    metrics,
    robustness,
    development_rosters,
)
from coldstart.skill_embedding import (
    fit_skill_profile,
    fit_skill_space,
    skill_coordinates,
    skill_scores,
    skill_strata,
    stratified_cells,
)
from run_experiment import atomic_csv, atomic_json


def skill_measurements(args, d, rows, readout, phase):
    phi = features_for(args, d, rows, readout["pack"], readout["proxies"], phase)
    if args.axes:
        with np.load(args.out / f"axes_{args.axes}_{phase}.npz") as f:
            order = f["prompt_ids"].argsort()
            ix = order[np.searchsorted(f["prompt_ids"][order], d.prompt_ids[rows])]
            extra = f["probabilities"][ix].reshape(len(ix), -1)
        phi = np.column_stack((phi, extra))
    return phi


def blend_relative(scores, estimate):
    return (0.55 * scores + 0.45 * (scores.mean(1, keepdims=True) + estimate)).clip(0, 1)


def train(args):
    d, s, *_ = input_context(args.run)
    head = args.out / args.readout
    path = args.out / args.name
    path.mkdir(exist_ok=True)
    with np.load(args.out / "questions.npz") as split:
        tr, dev = split["training"], split["development"]
        pos = {int(x): i for i, x in enumerate(tr)}
        balanced = {seed: np.array([pos[int(x)] for x in split[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    with np.load(head / "predictions.npz") as f:
        oof, p, src = f["historical"], f["dev"], f["dev_source"]
    readout = joblib.load(head / "readout.joblib")
    base = skill_measurements(args, d, np.r_[tr, dev], readout, "development")
    dt = d.domains[tr]
    tr_d = np.eye(int(dt.max()) + 1)[dt]
    dv = np.eye(tr_d.shape[1])[src]
    h = d.labels[np.ix_(s.train_models, tr)]
    truth, y = d.labels[np.ix_(s.val_models, dev)], d.labels[np.ix_(s.val_models, tr)]
    pack = fit_skill_space(base[: len(tr)], oof, tr_d, h)
    x, hr = skill_coordinates(pack, base[: len(tr)], oof, tr_d)
    xx, vr = skill_coordinates(pack, base[len(tr) :], p, dv)
    hist, scores = blend_relative(oof, hr), blend_relative(p, vr)
    cells = skill_strata(x, h, dt, 22)
    means = np.stack([h[:, dt == j].mean(1) for j in range(tr_d.shape[1])])
    np.savez_compressed(
        path / "skills.npz",
        **pack,
        cells=cells,
        means=means,
        readout=args.readout,
        axes_model=args.axes or "",
    )
    np.savez_compressed(path / "training_coordinates.npz", x=x, historical=hist)
    dom = d.domains[dev]
    rosters = development_rosters(truth, dom)
    native = json.loads((args.report / f"{args.readout}_selection.json").read_text())["native"]
    rows, finalists = [], []

    def key(row):
        win = row["macro"] > native["macro"] and row["micro"] > native["micro"] and row["top_share"] < 0.92
        return win, row.get("robust_objective", row["objective"])

    for policy, budget in product(("balanced", "skill-stratified"), (120, 200)):
        use = cells if policy == "skill-stratified" else dt
        order = {
            seed: (arr[:budget] if policy == "balanced" else stratified_cells(cells, dt, seed, budget))
            for seed, arr in balanced.items()
        }
        for strength, observed, shrink in product((0.2, 0.8), (0.55, 1.0), (0.25, 1.0)):
            z = [
                fit_skill_profile(h, y, ix, dt, x, hist, use, strength, observed, shrink)
                for ix in order.values()
            ]
            grid = []
            for sig, residual, discount in product((0.55, 1.2, 2.1), (0.5, 1.1), (0.0, 0.20)):
                cfg = dict(
                    name=args.name,
                    algorithm="skills",
                    readout=args.readout,
                    axes=args.axes,
                    policy=policy,
                    probe_count=budget,
                    strength=strength,
                    observed=observed,
                    rank_shrink=shrink,
                    signal=sig,
                    residual_scale=residual,
                    global_discount=discount,
                )
                pred = np.stack([skill_scores(pf, xx, scores, dv, means, cfg) for pf in z])
                row = dict(**cfg, **metrics(pred.argmax(1), truth, dom))
                rows.append(row)
                grid.append((row, pred))
            for row, pred in sorted(grid, key=lambda item: key(item[0]), reverse=True)[:2]:
                row["subpool_gain"], row["within_domain_gain"] = robustness(pred, truth, dom, rosters)
                row["robust_objective"] = (
                    row["objective"] + 0.5 * row["subpool_gain"] + 0.5 * row["within_domain_gain"]
                )
                finalists.append(row)
    best = max(finalists, key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in finalists if r["probe_count"] == 120), key=key), native=native),
    )
    print(args.name, best, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--readout", required=True)
    parser.add_argument("--axes")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        train(parser.parse_args())
