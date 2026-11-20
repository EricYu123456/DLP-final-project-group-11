"""Train a joint 77-dimensional historical/newcomer model embedding with a text-conditioned decoder."""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
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
from coldstart.conditional_capability import paired_profile
from coldstart.joint_pool import fit_joint
from coldstart.paired_distillation import behavioral_basis, behavioral_design, infer_teacher
from run_experiment import atomic_csv, atomic_json


def frozen_vectors(args, s, positions):
    bank = np.r_[s.train_items, s.val_items]
    order = {int(v): i for i, v in enumerate(bank)}
    with np.load(args.out.parent / "views_expert/raw.npz") as f:
        existing = f["data"]
    if all(int(i) in order for i in positions):
        return existing[[order[int(i)] for i in positions]]
    with np.load(args.out / "previous_raw_expert.npz") as f:
        other = {int(iv): j for j, iv in enumerate(f["items"])}
        values = f["data"]
    return np.stack(
        [existing[order[int(i)]] if int(i) in order else values[other[int(i)]] for i in positions]
    )


def joint_coordinates(raw, frozen, pack):
    semantic = (frozen[:, :-60] - pack["sem_mean"]) / pack["sem_scale"]
    low = semantic @ pack["axes"].T / pack["pc_scale"]
    other = np.column_stack((raw, frozen[:, -60:]))
    return np.column_stack(
        (low.clip(-5, 5), ((other - pack["other_mean"]) / pack["other_scale"]).clip(-5, 5))
    ).astype(np.float32)


def prepare(args, d, s, tr, dev, source, dest):
    feature_file = dest / "coordinate_pack.npz"
    data_file = dest / "development_features.npz"
    if data_file.exists() and feature_file.exists():
        with np.load(data_file) as f:
            return f["x"]
    positions = np.r_[tr, dev]
    raw = skill_measurements(args, d, positions, source, "development")
    frozen = frozen_vectors(args, s, positions)
    sem = frozen[: len(tr), :-60]
    center, scale = sem.mean(0), sem.std(0).clip(0.04)
    norm = (sem - center) / scale
    _, _, axes = randomized_svd(norm, 144, n_iter=3, random_state=2401)
    pc_scale = (norm @ axes.T).std(0).clip(0.15)
    other = np.column_stack((raw[: len(tr)], frozen[: len(tr), -60:]))
    pack = dict(
        sem_mean=center,
        sem_scale=scale,
        axes=axes,
        pc_scale=pc_scale,
        other_mean=other.mean(0),
        other_scale=other.std(0).clip(0.08),
    )
    x = joint_coordinates(raw, frozen, pack)
    np.savez_compressed(feature_file, **pack)
    np.savez_compressed(data_file, x=x)
    return x


def fit_on_panel(x, membership, h, full, ix, seed, stages, callback, weight):
    dom = membership.argmax(1)
    b = behavioral_basis(h, dom)
    design = behavioral_design(h, dom, b)
    teacher = infer_teacher(h, full, ix, dom, design, b, 12.0)
    pop = np.bincount(dom)
    w = []
    for row in full:
        seen = ix[np.isfinite(row[ix])]
        w.append(paired_profile(h[:, seen], row[None, seen], dom[seen], pop, np.arange(len(pop)))[0][0])
    return fit_joint(
        x, membership, h, full, teacher, ix, np.stack(w), seed, stages, callback, real_weight=weight
    )


def run(args):
    d, s, *_ = input_context(args.run)
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    head = args.out / args.readout
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        pos = {int(x): i for i, x in enumerate(tr)}
        orders = {seed: np.asarray([pos[int(x)] for x in f[f"balanced_{seed}"]]) for seed in (11, 23, 37)}
        previous = f["previous_test"]
    if args.cache_previous:
        from coldstart.question_views import FrozenQuestionViews
        from train_capability_router import prompt_texts

        texts = prompt_texts(ROOT / "data/raw/question_order.csv", d.prompt_ids[previous])
        encoder = FrozenQuestionViews("expert")
        parts = [encoder.encode(texts[i : i + 256], batch=12) for i in range(0, len(texts), 256)]
        np.savez_compressed(args.out / "previous_raw_expert.npz", items=previous, data=np.concatenate(parts))
        return
    source = joblib.load(head / "readout.joblib")
    features = prepare(args, d, s, tr, dev, source, dest)
    if args.prepare_only:
        return
    with np.load(head / "predictions.npz") as f:
        ds = f["dev_source"]
    dt, dom = d.domains[tr], d.domains[dev]
    one = np.eye(int(dt.max()) + 1, dtype=np.float32)
    h = d.labels[np.ix_(s.train_models, tr)]
    full = d.labels[np.ix_(s.val_models, tr)]
    truth = d.labels[np.ix_(s.val_models, dev)]
    xd = torch.as_tensor(features[len(tr) :], device="cuda")
    dv = torch.as_tensor(one[ds], device="cuda")
    rosters = development_rosters(truth, dom)
    with np.load(args.out / "embedllm_known_ids.npz") as original:
        native = metrics(
            known_choices(original, d, d.model_ids[s.val_models], dev, tuple(orders))[0], truth, dom
        )
    stages = (140, 400, 950, 2000, 3800)
    rows = []
    for n, weight in ((200, 0.16), (200, 0.38), (120, 0.16)):
        choices = {k: [] for k in stages}
        times = {k: [] for k in stages}
        for seed, idx in orders.items():
            start = perf_counter()

            def record(step, model):
                _, logits = model(xd, dv)
                choices[step].append(logits.sigmoid().cpu().numpy().T)
                times[step].append(perf_counter() - start)

            fit_on_panel(features[: len(tr)], one[dt], h, full, idx[:n], seed, stages, record, weight)
        for step in stages:
            scores = np.stack(choices[step])
            np.savez_compressed(dest / f"dev_{n}_{weight}_{step}.npz", scores=scores)
            result = dict(
                name=args.name,
                algorithm="joint",
                readout=args.readout,
                axes=args.axes,
                policy="balanced",
                probe_count=n,
                steps=step,
                real_weight=weight,
                seconds=float(np.mean(times[step])),
                **metrics(scores.argmax(1), truth, dom),
            )
            result["subpool_gain"], result["within_domain_gain"] = robustness(scores, truth, dom, rosters)
            result["robust_objective"] = result["objective"] + 0.5 * (
                result["subpool_gain"] + result["within_domain_gain"]
            )
            rows.append(result)
        print(n, weight, max(rows[-len(stages) :], key=lambda r: r["robust_objective"]), flush=True)

    def key(r):
        return (
            r["macro"] > native["macro"] and r["micro"] > native["micro"] and r["top_share"] < 0.92,
            r["robust_objective"],
        )

    atomic_csv(args.report / f"{args.name}_validation.csv", rows)
    atomic_json(
        args.report / f"{args.name}_selection.json",
        dict(
            best=max(rows, key=key),
            short=max((r for r in rows if r["probe_count"] == 120), key=key),
            native=native,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--readout", required=True)
    p.add_argument("--axes")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--cache-previous", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    torch.set_num_threads(2)
    with threadpool_limits(limits=3):
        run(p.parse_args())
