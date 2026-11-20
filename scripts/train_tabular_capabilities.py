"""Cross-check a tabular foundation model's ability estimates using each 200-question panel."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from train_conditional_router import (
    ROOT,
    development_rosters,
    input_context,
    known_choices,
    metrics,
    robustness,
)
from train_pair_embeddings import choose_pair_panel
from train_skill_embeddings import skill_measurements
from coldstart.paired_distillation import lookup_pair_space
from coldstart.tabular_capability import fit_tabular, query_skills, skill_targets
from run_experiment import atomic_csv, atomic_json
import joblib


def run(args):
    d, s, *_ = input_context(args.run)
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    pair_dir = args.out / args.pairs / ("refit" if args.refit else "dev")
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {int(i): j for j, i in enumerate(tr)}
        ids = {seed: np.array([pos[int(i)] for i in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    with np.load(pair_dir / "pair_controls.npz") as f:
        x, cells = f["x"], f["cells"]
    with np.load(pair_dir / "pair_space.npz") as f:
        space = dict(f)
    with np.load(args.out / args.readout / "predictions.npz") as f:
        ds = f["dev_source"]
    ref = np.r_[s.train_models, s.val_models] if args.refit else s.train_models
    h = d.labels[np.ix_(ref, tr)]
    dt, dom = d.domains[tr], d.domains[dev]
    pack = skill_targets(h, x, dt)
    np.savez_compressed(folder / ("refit_targets.npz" if args.refit else "targets.npz"), **pack)
    if args.refit:
        return
    readout = joblib.load(args.out / args.readout / "readout.joblib")
    raw = skill_measurements(args, d, dev, readout, "development")
    xx = query_skills(lookup_pair_space(raw, ds, space), ds, pack)
    full = d.labels[np.ix_(s.val_models, tr)]
    truth = d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dom)
    with np.load(args.out / "embedllm_known_ids.npz") as orig:
        native = metrics(known_choices(orig, d, d.model_ids[s.val_models], dev, tuple(ids))[0], truth, dom)
    records = []
    for budget in (200, 120):
        estimates = []
        for seed in ids:
            cache = folder / f"dev_{seed}_{budget}.npz"
            if cache.exists():
                with np.load(cache) as f:
                    est = f["prior"], f["fit"]
            else:
                ix, _ = choose_pair_panel("balanced", budget, seed, ids, cells, dt)
                start = perf_counter()
                est = fit_tabular(h, full, ix, dt, pack, args.device)
                np.savez(cache, prior=est[0], fit=est[1], seconds=perf_counter() - start)
                print(args.name, budget, seed, round(perf_counter() - start, 1), flush=True)
            estimates.append(est)
        for mix in (0.15, 0.45, 0.75, 1.0):
            for amplify in (0.7, 1.3, 2.2):
                design = xx.copy()
                design[:, int(dt.max()) + 1 :] *= amplify
                score = np.stack([((1 - mix) * left + mix * right) @ design.T for left, right in estimates])
                row = dict(
                    name=args.name,
                    algorithm="tabular",
                    readout=args.readout,
                    pairs=args.pairs,
                    axes=args.axes,
                    policy="balanced",
                    probe_count=budget,
                    tabular_mix=mix,
                    signal=amplify,
                    **metrics(score.argmax(1), truth, dom),
                )
                row["subpool_gain"], row["within_domain_gain"] = robustness(score, truth, dom, rosters)
                row["robust_objective"] = row["objective"] + 0.5 * (
                    row["subpool_gain"] + row["within_domain_gain"]
                )
                records.append(row)

    def key(r):
        return (
            r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92,
            r["robust_objective"],
        )

    best = max(records, key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", records)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in records if r["probe_count"] == 120), key=key), native=native),
    )
    print(best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="tabular-general")
    p.add_argument("--pairs", default="pairs-general")
    p.add_argument("--readout", default="general")
    p.add_argument("--axes")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(4)
    with threadpool_limits(limits=3):
        run(p.parse_args())
