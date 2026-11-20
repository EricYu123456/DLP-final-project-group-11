"""Learn candidate embeddings for rescuing a strong incumbent's errors from 200 joint observations."""

import argparse
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from train_conditional_router import (
    ROOT,
    development_rosters,
    input_context,
    known_choices,
    metrics,
    robustness,
)
from train_skill_embeddings import skill_measurements
from coldstart.paired_distillation import (
    behavioral_basis,
    behavioral_design,
    distill,
    infer_teacher,
    learn_pair_space,
    lookup_pair_space,
    pair_scores,
)
from coldstart.skill_embedding import skill_strata, stratified_cells
from run_experiment import atomic_csv, atomic_json


def model_fold(d, s, tr, fold):
    if fold == "dev":
        return np.asarray(s.train_models), np.asarray(s.val_models)
    known = np.r_[s.train_models, s.val_models]
    if fold == "refit":
        return known, np.asarray(s.test_models)
    overall = d.labels[np.ix_(known, tr)].mean(1)
    rng = np.random.default_rng(1811)
    which = np.zeros(len(known), int)
    for start in range(0, len(known), 5):
        loc = np.argsort(overall)[start : start + 5]
        which[rng.permutation(loc)] = np.arange(len(loc))
    part = which == int(fold)
    return known[~part], known[part]


def choose_pair_panel(policy, n, seed, ids, cells, dom):
    if policy == "balanced":
        idx = ids[seed][:n]
        strata = dom
    else:
        idx = stratified_cells(cells, dom, seed, n)
        strata = cells
    mass = np.bincount(strata) / len(strata)
    wt = mass[strata[idx]] / np.bincount(strata[idx], minlength=len(mass))[strata[idx]]
    return idx, wt / wt.sum()


def train(args):
    d, s, *_ = input_context(args.run)
    head = args.out / args.readout
    dest = args.out / args.name / args.fold
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "questions.npz") as split:
        tr, dev = split["training"], split["development"]
        pos = {int(x): i for i, x in enumerate(tr)}
        ids = {seed: np.array([pos[int(x)] for x in split[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    ref, candidates = model_fold(d, s, tr, args.fold)
    dt, dom = d.domains[tr], d.domains[dev]
    h = d.labels[np.ix_(ref, tr)]
    source = joblib.load(head / "readout.joblib")
    with np.load(head / "predictions.npz") as f:
        predicted_dom = f["dev_source"]
    raw = skill_measurements(args, d, np.r_[tr, dev], source, "development")
    ps = dest / "pair_space.npz"
    cache = dest / "pair_controls.npz"
    if ps.exists() and cache.exists():
        with np.load(ps) as f:
            space = dict(f)
        with np.load(cache) as f:
            x, cells = f["x"], f["cells"]
    else:
        space, x = learn_pair_space(raw[: len(tr)], dt, h)
        cells = skill_strata(x, h, dt, 22)
        np.savez_compressed(ps, **space)
        np.savez_compressed(cache, x=x, cells=cells)
    if args.fold == "refit":
        return
    xx = lookup_pair_space(raw[len(tr) :], predicted_dom, space)
    y, truth = d.labels[np.ix_(candidates, tr)], d.labels[np.ix_(candidates, dev)]
    rosters = development_rosters(truth, dom)
    coefficient_basis = behavioral_basis(h, dt)
    behavior = behavioral_design(h, dt, coefficient_basis)
    with np.load(args.out / "embedllm_known_ids.npz") as original:
        nc = known_choices(original, d, d.model_ids[candidates], dev, tuple(ids))[0]
    native = metrics(nc, truth, dom)
    rows, finalists = [], []
    for policy, budget in product(("balanced", "skill-stratified"), (120, 200)):
        panels = {seed: choose_pair_panel(policy, budget, seed, ids, cells, dt) for seed in ids}
        for prior in (8.0, 28.0):
            teacher = {
                seed: infer_teacher(h, y, ix, dt, behavior, coefficient_basis, prior)
                for seed, (ix, _) in panels.items()
            }
            for pseudo, real, penalty in ((700, 1.0, 3), (1600, 1.0, 6), (500, 2.0, 8)):
                z = [
                    distill(x, teacher[seed], y, ix, dt, weights, pseudo, real, penalty)
                    for seed, (ix, weights) in panels.items()
                ]
                opts = []
                for temp in (0.7, 1.4):
                    gains = np.stack([pair_scores(pf, xx, temp) for pf in z])
                    row = dict(
                        name=args.name,
                        algorithm="paired",
                        readout=args.readout,
                        axes=args.axes,
                        fold=args.fold,
                        policy=policy,
                        probe_count=budget,
                        teacher_prior=prior,
                        pseudo_count=pseudo,
                        probe_weight=real,
                        strength=penalty,
                        temperature=temp,
                        **metrics(gains.argmax(1), truth, dom),
                    )
                    rows.append(row)
                    opts.append((row, gains))
                row, gain = max(opts, key=lambda item: item[0]["objective"])
                row["subpool_gain"], row["within_domain_gain"] = robustness(gain, truth, dom, rosters)
                row["robust_objective"] = (
                    row["objective"] + 0.5 * row["subpool_gain"] + 0.5 * row["within_domain_gain"]
                )
                finalists.append(row)

    def key(r):
        return (
            r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92,
            r["robust_objective"],
        )

    best = max(finalists, key=key)
    atomic_csv(args.report / f"{args.name}_{args.fold}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_{args.fold}_selection.json",
        dict(best=best, short=max((r for r in finalists if r["probe_count"] == 120), key=key), native=native),
    )
    print(args.name, args.fold, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--axes")
    p.add_argument("--fold", choices=("dev", "0", "1", "2", "3", "4", "refit"), default="dev")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        train(p.parse_args())
