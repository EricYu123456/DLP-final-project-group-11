"""Predict historical response patterns from frozen question semantics and answer-option evidence."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from pilot_routing_learning import ROOT, input_context
from coldstart.capability import CapabilityBasis
from coldstart.question_views import FrozenQuestionViews
from train_capability_router import VramMeter, prompt_texts
from run_experiment import atomic_json


def rows_by_domain(domains, n):
    rng = np.random.default_rng(421)
    return np.concatenate(
        [rng.choice(np.flatnonzero(domains == j), n, replace=False) for j in np.unique(domains)]
    )


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-7)


def view_features(raw, pack):
    semantic = (raw[:, :-60] - pack["center"]) / pack["scale"]
    emb = normalize(semantic)
    pc = semantic @ pack["projection"].T / np.sqrt(semantic.shape[1])
    k = np.exp(4 * (emb @ pack["anchors"].T - 1))
    opt = np.tanh((raw[:, -60:] - pack["opt_center"]) / (2 * pack["opt_scale"]))
    # The first two views expose two sets of letter probabilities for each presentation.
    probs = opt[:, np.r_[12:20, 32:40, 52:60]]
    r, c = np.triu_indices(probs.shape[1], 1)
    interaction = probs[:, r] * probs[:, c]
    basic = np.column_stack((np.ones(len(raw), dtype=np.float32), pc, k, opt))
    return basic, np.column_stack((opt, interaction)), np.column_stack((pc, opt, probs))


def ridge(x, y, weights, penalty):
    xtw = x.T * weights.astype(np.float32)
    gram = xtw @ x
    gram.flat[:: len(gram) + 1] += penalty
    return np.linalg.solve(gram, xtw @ y).astype(np.float32)


def final_features(basic, options, membership):
    return np.column_stack(
        (basic, options, (membership[:, :, None] * options[:, None, :]).reshape(len(basic), -1))
    )


def readout(pack, raw):
    basic, options, tree = view_features(raw, pack)
    dom = (basic @ pack["domain_fit"]).clip(0, 1)
    dom /= np.maximum(dom.sum(1, keepdims=True), 1e-7)
    x = final_features(basic, options, dom)
    return (x @ pack["fit"]).clip(0, 1), dom, tree


def load_readouts(folder, heads):
    with np.load(folder / "view_fit.npz") as f:
        pack = dict(f)
    if "tree" not in heads:
        return lambda raw: {"ridge": readout(pack, raw)[:2]}
    from catboost import CatBoostRegressor

    boost = CatBoostRegressor()
    boost.load_model(str(folder / "response.cbm"))
    with np.load(folder / "tree_basis.npz") as f:
        scale, basis = f["scale"], f["basis"]

    def predict(raw):
        scores, membership, features = readout(pack, raw)
        pred = boost.predict(features, thread_count=2) * scale
        tree_scores = (pred[:, :1] + pred[:, 1:] @ basis.T).clip(0, 1)
        return {"ridge": (scores, membership), "tree": (tree_scores, membership)}

    return predict


def fit_view(raw, history, domains):
    sem = raw[:, :-60]
    center, scale = sem.mean(0), sem.std(0).clip(0.01)
    transformed = (sem - center) / scale
    _, _, projection = randomized_svd(transformed, 128, n_iter=3, random_state=421)
    anchors = normalize(transformed[rows_by_domain(domains, 96)])
    pack = dict(
        center=center,
        scale=scale,
        anchors=anchors,
        projection=projection,
        opt_center=raw[:, -60:].mean(0),
        opt_scale=raw[:, -60:].std(0).clip(0.05),
    )
    basic, options, tree = view_features(raw, pack)
    true_domain = np.eye(domains.max() + 1, dtype=np.float32)[domains]
    weights = np.sqrt(len(domains) / (np.bincount(domains)[domains] * true_domain.shape[1]))
    pack["domain_fit"] = ridge(basic, true_domain, weights, 1.0)
    pack["fit"] = ridge(final_features(basic, options, true_domain), history.T, weights, 10.0)
    return pack, tree, weights


def main(args):
    start = perf_counter()
    d, s, *_ = input_context(args.run)
    bank, val = np.asarray(s.train_items), np.asarray(s.val_items)
    dest = args.output / f"views_{args.size}"
    dest.mkdir(parents=True, exist_ok=True)
    raw_path = dest / "raw.npz"
    if not raw_path.exists():
        meter = VramMeter(8 if args.size == "small" else 16)
        try:
            featurizer = FrozenQuestionViews(args.size)
            for_batch = 64 if args.size == "small" else 32
            raw = []
            lookup = prompt_texts(ROOT / "data/raw/question_order.csv", d.prompt_ids[np.r_[bank, val]])
            for index in range(0, len(lookup), 800):
                raw.append(featurizer.encode(lookup[index : index + 800], for_batch))
                print(
                    f"{args.size} {min(len(lookup), index + 800)}/{len(lookup)} questions, "
                    f"{perf_counter() - start:.0f}s",
                    flush=True,
                )
            raw = np.concatenate(raw)
            np.savez(raw_path, data=raw)
            extraction = dict(
                seconds=perf_counter() - start,
                board_peak_mib=meter.peak,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
            atomic_json(dest / "extraction.json", extraction)
            del featurizer
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            meter.close()
    else:
        with np.load(raw_path) as f:
            raw = f["data"]
    predictors = perf_counter()
    h = d.labels[np.ix_(s.train_models, bank)]
    domains = d.domains[bank]
    pack, tree_x, weights = fit_view(raw[: len(bank)], h, domains)
    np.savez(dest / "view_fit.npz", **pack)
    p, membership, val_x = readout(pack, raw[len(bank) :])
    np.savez(
        dest / "ridge_validation.npz", historical_scores=p, domains=membership, prompt_ids=d.prompt_ids[val]
    )
    ridge_seconds = perf_counter() - predictors
    from catboost import CatBoostRegressor

    basis = CapabilityBasis.fit(h, domains).axes[:, :16].astype(np.float32) * np.sqrt(len(h))
    difficulty = h.mean(0, keepdims=True)
    coords = np.column_stack((difficulty.T, (h - difficulty).T @ basis))
    scale = coords.std(0).clip(0.05)
    boost = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.055,
        loss_function="MultiRMSE",
        thread_count=4,
        verbose=False,
        random_seed=421,
        allow_writing_files=False,
    )
    boost.fit(tree_x, coords / scale, sample_weight=weights)
    boost.save_model(str(dest / "response.cbm"))
    np.savez(dest / "tree_basis.npz", scale=scale, basis=basis)
    pred = boost.predict(val_x) * scale
    probs = pred[:, :1] + pred[:, 1:] @ basis.T
    np.savez(
        dest / "tree_validation.npz",
        historical_scores=probs.clip(0, 1),
        domains=membership,
        prompt_ids=d.prompt_ids[val],
    )
    atomic_json(
        dest / "fit_times.json", dict(ridge_seconds=ridge_seconds, all_seconds=perf_counter() - predictors)
    )
    print(f"Frozen view readouts trained in {perf_counter() - predictors:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--size", choices=("small", "large", "expert"), required=True)
    with threadpool_limits(limits=4):
        main(p.parse_args())
