"""Train a pairwise query ranker on tournaments of simulated 200-answer newcomers."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from catboost import CatBoostRanker
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from evaluate_capability_router import (
    KernelReadout,
    ProbeBank,
    ROOT,
    benchmark_means,
    get_kernel,
    input_context,
    load_features,
    metrics,
)
from coldstart.peer_ranker import peer_profiles, rank_features
from run_experiment import atomic_csv, atomic_json


def episodes(d, s, bank, rng):
    train = d.embeddings[bank.items]
    center = train.mean(0)
    _, _, projection = randomized_svd(train - center, 16, n_iter=3, random_state=421)
    semantics = (train - center) @ projection.T
    x, y = [], []
    # The score features for a training query must come from a readout that never saw its labels.
    folds = np.empty(len(train), dtype=int)
    for domain in np.unique(bank.domains):
        ix = rng.permutation(np.flatnonzero(bank.domains == domain))
        folds[ix] = np.arange(len(ix)) % 3
    all_models = np.arange(len(bank.history))
    for fold in range(3):
        old, new = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        readout = KernelReadout.fit(train[old], bank.history[:, old], bank.domains[old])
        oof, memberships = readout.predict(train[new])
        means = benchmark_means(bank.history[:, old], bank.domains[old]).T.astype(np.float32)
        pools = [old[bank.domains[old] == j] for j in range(len(means))]
        q_pools = [np.flatnonzero(bank.domains[new] == j) for j in range(len(means))]
        for trial in range(45):
            target = rng.choice(all_models, 13, replace=False)
            refs = np.setdiff1d(all_models, target)
            if trial % 3 == 1:
                # More competitive rosters teach distinctions among the stronger candidates.
                top = np.argsort(means.mean(0))[-32:]
                target = rng.choice(top, 13, replace=False)
                refs = np.setdiff1d(all_models, target)
            support = np.concatenate([rng.choice(pool, 40, replace=False) for pool in pools])
            chosen = np.concatenate([rng.choice(pool, 36, replace=False) for pool in q_pools])
            q = new[chosen]
            z = peer_profiles(
                bank.history[np.ix_(refs, support)].T,
                means[:, refs],
                bank.domains[support],
                bank.history[np.ix_(target, support)],
            )
            x.append(
                rank_features(z, oof[np.ix_(chosen, refs)], memberships[chosen], means[:, refs], semantics[q])
            )
            y.append(bank.history[np.ix_(target, q)].T.ravel())
    return np.vstack(x), np.concatenate(y), center, projection


def main(args):
    start = perf_counter()
    d, s, c, *_ = input_context(args.run)
    bank = ProbeBank(d, s, c["seeds"])
    dest = args.weights / "peers"
    dest.mkdir(exist_ok=True, parents=True)
    x, y, center, projection = episodes(d, s, bank, np.random.default_rng(427))
    np.savez(dest / "semantic_projection.npz", center=center, projection=projection)
    model = CatBoostRanker(
        loss_function="PairLogit",
        iterations=500,
        depth=5,
        learning_rate=0.045,
        l2_leaf_reg=12,
        random_seed=427,
        thread_count=4,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(x, y, group_id=np.repeat(np.arange(len(y) // 13), 13))
    model.save_model(str(dest / "ranker.cbm"))
    seconds = perf_counter() - start
    kernel = get_kernel(args, d, s)
    val = np.asarray(s.val_items)
    truth = d.labels[np.ix_(s.val_models, val)]
    mean = bank.domain_means.astype(np.float32)
    sem = (d.embeddings[val] - center) @ projection.T
    rows = []
    for encoders in (["kernel"], ["kernel", "gpu8/minute_8_1"]):
        p, m = load_features(encoders, "validation", args, d, val, kernel)
        for policy in ("balanced", "representative", "informative"):
            choices = {t: [] for t in (50, 150, 300, 500)}
            for seed in c["seeds"]:
                selected = bank.orders[policy, seed]
                z = peer_profiles(
                    bank.history[:, selected].T,
                    mean,
                    bank.domains[selected],
                    d.labels[np.ix_(s.val_models, bank.items[selected])],
                )
                features = rank_features(z, p, m, mean, sem)
                for trees in choices:
                    pred = model.predict(features, ntree_end=trees, thread_count=4).reshape(len(val), -1)
                    choices[trees].append(pred.argmax(1))
            for trees, route in choices.items():
                rows.append(
                    dict(
                        encoders="+".join(encoders),
                        policy=policy,
                        trees=trees,
                        **metrics(np.stack(route), truth, d.domains[val]),
                    )
                )
    atomic_csv(dest / "validation.csv", rows)
    atomic_json(dest / "runtime.json", dict(seconds=seconds, examples=len(y), groups=len(y) // 13))
    print(max(rows, key=lambda r: r["objective"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--weights", type=Path, default=ROOT / "artifacts/capability_routing")
    with threadpool_limits(limits=3):
        main(p.parse_args())
