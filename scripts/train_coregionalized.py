"""Train an exact multi-output question kernel with empirical and model-metadata priors."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from coldstart.coregionalized import descriptive, embed, fit_shared, panel_cross, query, scores
from evaluate_capability_router import ROOT, input_context, known_choices, metrics
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import development_rosters, robustness
from train_skill_embeddings import skill_measurements


def model_metadata(args, names):
    return descriptive(
        names,
        ROOT / "data/metadata/model_families.csv",
        args.out.parent / "release_confirmation/architectures.csv",
    )


def run(args):
    d, s, *_ = input_context(args.run)
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        ix = {int(v): i for i, v in enumerate(tr)}
        panels = {seed: np.array([ix[int(v)] for v in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
    dt, dd = d.domains[tr], d.domains[dev]
    readout = joblib.load(args.out / args.readout / "readout.joblib")
    raw = skill_measurements(args, d, np.r_[tr, dev], readout, "development")
    with np.load(args.out / args.readout / "predictions.npz") as f:
        source = f["dev_source"]
    refs = [s.train_models, np.r_[s.train_models, s.val_models]]
    h, extended = [d.labels[np.ix_(sids, tr)] for sids in refs]
    if not (dest / "refit.joblib").is_file():
        start = perf_counter()
        meta = [model_metadata(args, np.asarray(d.model_names)[r]) for r in refs]
        shared = fit_shared(
            d.embeddings[tr],
            raw[: len(tr)],
            dt,
            [h, extended],
            meta,
            lambda j, n: print(args.name, j, n, round(perf_counter() - start, 1), flush=True),
        )
        for model, prefix in zip(shared, ("development", "refit")):
            joblib.dump(model, dest / f"{prefix}.joblib", compress=2)
        atomic_json(
            dest / "cost.json",
            dict(
                joint_cpu_seconds=perf_counter() - start, questions=len(tr), references=[len(r) for r in refs]
            ),
        )
    else:
        shared = [joblib.load(dest / "development.joblib")]
    model = shared[0]
    hist, cross = query(model, d.embeddings[dev], raw[len(tr) :], source)
    names = np.asarray(d.model_names)
    meta = model_metadata(args, np.r_[names[s.train_models], names[s.val_models]])
    y, truth = d.labels[np.ix_(s.val_models, tr)], d.labels[np.ix_(s.val_models, dev)]
    rosters = development_rosters(truth, dd)
    with np.load(args.out / "embedllm_known_ids.npz") as f:
        native = metrics(known_choices(f, d, d.model_ids[s.val_models], dev, tuple(panels))[0], truth, dd)
    rows = []
    for n, mix in product((120, 200), (0.0, 0.25, 0.54)):
        fits = [embed(model, h, y, idx[:n], dt, meta, mix, seed) for seed, idx in panels.items()]
        links = [panel_cross(cross, dt, idx[:n], len(dev)) for idx in panels.values()]
        for signal, residual in product((0.65, 1.25, 2.2, 3.4), (0.0, 0.5, 1.15)):
            cfg = dict(
                name=args.name,
                algorithm="coregionalized",
                readout=args.readout,
                axes=args.axes,
                policy="balanced",
                probe_count=n,
                metadata=mix,
                signal=signal,
                residual=residual,
            )
            pred = np.stack([scores(p, hist, k, source, cfg) for p, k in zip(fits, links)])
            row = dict(**cfg, **metrics(pred.argmax(1), truth, dd))
            row["subpool_gain"], row["within_domain_gain"] = robustness(pred, truth, dd, rosters)
            row["robust_objective"] = row["objective"] + 0.5 * (
                row["subpool_gain"] + row["within_domain_gain"]
            )
            rows.append(row)

    def key(r):
        return r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92, r[
            "robust_objective"
        ]

    best = max(rows, key=key)
    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(best=best, short=max((r for r in rows if r["probe_count"] == 120), key=key), native=native),
    )
    print(best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--axes")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    with threadpool_limits(limits=4):
        run(p.parse_args())
