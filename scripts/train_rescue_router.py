"""Meta-train a rescue-versus-harm calibrator on distinct historical LLMs and question groups."""

import argparse
from pathlib import Path
from time import perf_counter

from catboost import CatBoostClassifier, Pool
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from train_pair_embeddings import choose_pair_panel
from train_conditional_router import (
    ROOT,
    development_rosters,
    input_context,
    known_choices,
    metrics,
    robustness,
)
from train_skill_embeddings import skill_measurements
from coldstart.paired_distillation import learn_pair_space, lookup_pair_space
from coldstart.rescue_router import cohort_context, measured_pool, pair_inputs, pair_profile, rescue_scores
from coldstart.skill_embedding import skill_strata
from run_experiment import atomic_csv, atomic_json


def setup(raw, dt, h):
    space, x = learn_pair_space(raw, dt, h)
    return space, cohort_context(h, dt, x), skill_strata(x, h, dt, 22)


def meta_rows(raw, hist, dt, ids):
    rng = np.random.default_rng(2209)
    fold = np.zeros(len(hist), int)
    sorted_rows = np.argsort(hist.mean(1))
    for i in range(0, len(fold), 5):
        block = sorted_rows[i : i + 5]
        fold[rng.permutation(block)] = np.arange(len(block))
    xs, ys, prior = [], [], []
    for j in range(5):
        y, h = hist[fold == j], hist[fold != j]
        _, ctx, cells = setup(raw, dt, h)
        for seed, n, policy in (
            (11, 200, "balanced"),
            (23, 200, "skill-stratified"),
            (37, 120, "balanced"),
            (11, 120, "skill-stratified"),
        ):
            ix, _ = choose_pair_panel(policy, n, seed, ids, cells, dt)
            pf, teacher = measured_pool(ctx, y, ix)
            query = np.concatenate(
                [
                    rng.choice(pool, min(len(pool), 300), replace=False)
                    for k in np.unique(dt)
                    if len(pool := np.setdiff1d(np.flatnonzero(dt == k), ix))
                ]
            )
            strong = np.argsort(-pf["aggregate"])
            anchors = np.unique(np.r_[strong[:3], strong[len(strong) // 2], pf["source"].argmax(0)])
            for anchor in anchors:
                pair = pair_profile(ctx, pf, teacher, y, ix, anchor)
                for m in range(len(y)):
                    if m == anchor:
                        continue
                    keep = query[y[m, query] != y[anchor, query]]
                    if len(keep):
                        x, score, _ = pair_inputs(ctx, pf, pair, ctx["design"][keep], dt[keep], m)
                        xs.append(x)
                        ys.append(y[m, keep])
                        prior.append(score)
        print(f"Completed disjoint historical-model fold {j + 1}/5", flush=True)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(prior)


def run(args):
    d, s, *_ = input_context(args.run)
    tr_dir = args.out / args.name
    tr_dir.mkdir(parents=True, exist_ok=True)
    head = args.out / args.readout
    readout = joblib.load(head / "readout.joblib")
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {int(x): i for i, x in enumerate(tr)}
        ids = {seed: np.array([pos[int(x)] for x in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    with np.load(head / "predictions.npz") as f:
        source = f["dev_source"]
    raw = skill_measurements(args, d, np.r_[tr, dev], readout, "development")
    dt, dom = d.domains[tr], d.domains[dev]
    ref = np.r_[s.train_models, s.val_models] if args.refit else s.train_models
    historical = d.labels[np.ix_(ref, tr)]
    meta = tr_dir / ("refit.cbm" if args.refit else "development.cbm")
    if not meta.exists():
        start = perf_counter()
        x, y, baseline = meta_rows(raw[: len(tr)], historical, dt, ids)
        model = CatBoostClassifier(
            iterations=620,
            depth=5,
            learning_rate=0.040,
            l2_leaf_reg=8,
            thread_count=6,
            loss_function="Logloss",
            random_seed=2209,
            verbose=200,
            allow_writing_files=False,
        )
        model.fit(Pool(x, y, baseline=baseline))
        model.save_model(str(meta))
        atomic_json(
            tr_dir / ("refit_time.json" if args.refit else "development_time.json"),
            dict(seconds=perf_counter() - start, training_examples=len(y), reference_models=len(ref)),
        )
    else:
        model = CatBoostClassifier().load_model(str(meta))
    space, ctx, cells = setup(raw[: len(tr)], dt, historical)
    prefix = "refit" if args.refit else "development"
    np.savez_compressed(tr_dir / f"{prefix}_space.npz", **space)
    np.savez_compressed(
        tr_dir / f"{prefix}_cohort.npz", design=ctx["design"], cells=cells, ref=d.model_ids[ref]
    )
    if args.refit:
        return
    xx = lookup_pair_space(raw[len(tr) :], source, space)
    truth, y = d.labels[np.ix_(s.val_models, dev)], d.labels[np.ix_(s.val_models, tr)]
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(ids))[0], truth, dom)
    rosters = development_rosters(truth, dom)
    rows = []
    for n in (120, 200):
        for policy in ("balanced", "skill-stratified"):
            inputs = []
            for seed in ids:
                ix, _ = choose_pair_panel(policy, n, seed, ids, cells, dt)
                pf, te = measured_pool(ctx, y, ix)
                pair = pair_profile(ctx, pf, te, y, ix, pf["aggregate"].argmax())
                inputs.append((pf, pair))
            for trees in (160, 360, 620):
                for alpha in (0.30, 0.70, 1.10):
                    score = np.stack(
                        [rescue_scores(ctx, pf, pair, xx, source, model, alpha, trees) for pf, pair in inputs]
                    )
                    row = dict(
                        name=args.name,
                        algorithm="rescue",
                        readout=args.readout,
                        axes=args.axes,
                        policy=policy,
                        probe_count=n,
                        trees=trees,
                        alpha=alpha,
                        **metrics(score.argmax(1), truth, dom),
                    )
                    row["subpool_gain"], row["within_domain_gain"] = robustness(score, truth, dom, rosters)
                    row["robust_objective"] = row["objective"] + 0.5 * (
                        row["subpool_gain"] + row["within_domain_gain"]
                    )
                    rows.append(row)

    def key(row):
        return (
            row["macro"] > native["macro"] and row["micro"] > native["micro"] and row["top_share"] < 0.92,
            row["robust_objective"],
        )

    best = max(rows, key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in rows if r["probe_count"] == 120), key=key), native=native),
    )
    print(args.name, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--axes")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=3):
        run(p.parse_args())
