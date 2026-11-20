"""Transfer the frozen panel-only learner to complete external rosters and refit upstream TextMF controls."""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, benchmark_means, metrics, paired_gain
from fit_roster_panel import fit_pool, predict
from run_experiment import atomic_csv
from select_confirmation_router import SEEDS


def llmrouterbench():
    from run_external_routing import BENCHMARKS, load_archive, remove_demonstrations

    protocol = json.loads(
        (ROOT / "reports/routing/external_transfer/protocol.json").read_text(encoding="utf-8")
    )
    archive = ROOT / "data/raw/llmrouterbench/bench-release.tar.gz"
    items, _, rendered, origins = load_archive(archive, protocol, False)
    texts = np.asarray(
        [
            remove_demonstrations(b, origin, prompt)[0]
            for (b, _), origin, prompt in zip(items, origins, rendered)
        ]
    )
    with np.load(ROOT / "artifacts/external_routing/features.npz") as f:
        if items != list(zip(f["benchmark"], f["index"])):
            raise ValueError("External texts and precomputed question vectors differ in order.")
        x = f["embeddings"]
    tasks = list(BENCHMARKS)
    dom = np.asarray([tasks.index(b) for b, _ in items])
    split = protocol["data_split"]["benchmark_splits"]
    support = {(b, i) for b in tasks for i in split[b]["support"]}
    tr = np.asarray([item in support for item in items])
    index = {item: j for j, item in enumerate(item for item in items if item in support)}
    orders = {
        seed: np.asarray([index[tuple(item)] for item in protocol["probe_ids"][str(seed)]][:200])
        for seed in SEEDS
    }
    _, grades, _, _ = load_archive(archive, protocol, True)
    return (
        protocol["roster"],
        tasks,
        x[tr],
        texts[tr],
        dom[tr],
        grades[:, tr],
        x[~tr],
        texts[~tr],
        dom[~tr],
        grades[:, ~tr],
        orders,
    )


def xroutebench():
    from run_xroute_onboarding import TASKS, embedllm_overlap_keys, load_test_labels, parquet, partition

    raw = ROOT / "data/raw/xroutebench"
    protocol = json.loads(
        (ROOT / "reports/routing/xroute_onboarding/protocol.json").read_text(encoding="utf-8")
    )
    roster = {v["model_name"] for v in protocol["source_roster"]}
    names = protocol["eligible_models"]
    cols = ["task_name", "query", "model_name"]
    overlap = embedllm_overlap_keys(ROOT / "data/raw/question_order.csv", ROOT / "data/processed/prompts.csv")
    train, _, used = partition(
        parquet(raw / "train.parquet", [*cols, "performance"]), roster, overlap, labeled=True
    )
    tested, _, _ = partition(
        parquet(raw / "test.parquet", cols), roster, overlap, labeled=False, excluded_train=used
    )
    with np.load(ROOT / "artifacts/xroute_onboarding/features.npz") as f:
        x, xx, dt, dd, h = [
            f[k]
            for k in ("train_vectors", "test_vectors", "train_domains", "test_domains", "training_labels")
        ]
        if not np.array_equal(f["test_stems"], [key[1] for key, _, _ in tested]) or not np.array_equal(
            dt, [TASKS.index(key[0]) for key, _, _ in train]
        ):
            raise ValueError("External task inputs differ from the audited question order.")
    key = [key for key, _, _ in tested]
    truth = load_test_labels(raw / "test.parquet", roster, names, key)
    orders = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        groups = [rng.permutation(np.flatnonzero(dt == j))[:40] for j in range(len(TASKS))]
        orders[seed] = np.stack(groups, axis=1).ravel()
    return (
        names,
        TASKS,
        x,
        np.asarray([p for _, p, _ in train]),
        dt,
        h,
        xx,
        np.asarray([p for _, p, _ in tested]),
        dd,
        truth,
        orders,
    )


def native_checkpoints(upstream, x, y, xx, seeds, epochs, order=None, device="cpu"):
    # Supply only the observed training cells. Frozen query vectors let the public forward route unseen items.
    m, n = y.shape
    vectors = torch.from_numpy(np.vstack((x, xx)).astype(np.float32))
    torch.manual_seed(42)
    net = upstream.TextMF(vectors, 232, 0.05, m, len(vectors)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)
    # The native loader keeps input order across epochs. Interleave models and tasks once on each new corpus.
    if order is None:
        order = np.random.default_rng(42).permutation(m * n)
    models = torch.from_numpy((order % m).astype(np.int64)).to(device)
    items = torch.from_numpy((order // m).astype(np.int64)).to(device)
    labels = torch.from_numpy(y.T.ravel()[order].astype(np.int64)).to(device)
    for epoch in range(1, max(epochs) + 1):
        for j in range(0, len(order), 2048):
            batch = slice(j, j + 2048)
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(net(models[batch], items[batch]), labels[batch]).backward()
            optimizer.step()
        if epoch not in epochs:
            continue
        result = []
        net.eval()
        cuda_devices = [torch.cuda.current_device()] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            for seed in (None, *seeds):
                if seed is not None:
                    torch.manual_seed(seed)
                choices = []
                for start in range(0, len(xx), 64):
                    qs = torch.arange(n + start, n + min(start + 64, len(xx)), device=device)
                    logits = net(
                        torch.arange(m, device=device).repeat(len(qs)), qs.repeat_interleave(m), test_mode=seed is None
                    )
                    choices.extend((logits[:, 1] - logits[:, 0]).reshape(-1, m).argmax(1).tolist())
                result.append(choices)
        net.train()
        yield epoch, np.asarray(result)


def native_routes(upstream, x, y, xx, seeds, order=None, epochs=50):
    return next(native_checkpoints(upstream, x, y, xx, seeds, (epochs,), order))[1]


def run(args):
    dataset = args.dataset
    reader = llmrouterbench if dataset == "llmrouterbench" else xroutebench
    names, tasks, x, text, dt, history, xx, query, dd, truth, orders = reader()
    if args.text_only:
        records = []
        for seed, panel in orders.items():
            for n in (120, 200):
                idx = panel[:n]
                pool, fit = fit_pool(
                    None, dt[idx], history[:, idx], text[idx], idx, names, seed, text_only=True
                )
                begin = perf_counter()
                route = predict(pool, None, query).argmax(0)[None]
                records.append(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        method=pool["method"],
                        questions=len(dd),
                        fit_seconds=fit["seconds"],
                        raw_query_seconds=perf_counter() - begin,
                        **metrics(route, truth, dd),
                    )
                )
                if seed == SEEDS[0]:
                    joblib.dump(pool, args.out / f"{dataset}_{pool['method']}.joblib", compress=2)
        atomic_csv(args.report / f"{dataset}_text_seeds.csv", records)
        return
    spec = importlib.util.spec_from_file_location(
        "external_upstream_textmf", ROOT / "artifacts/capability_routing/release112/upstream_mf.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    started = perf_counter()
    full = native_routes(upstream, x, history, xx, SEEDS)
    timing = [
        dict(method="TextMF-full", seed=42, fit_and_score_seconds=perf_counter() - started, probes=len(dt))
    ]
    scored = {
        "TextMF-full-deterministic": np.repeat(full[:1], len(SEEDS), axis=0),
        "TextMF-full-noisy": full[1:],
    }
    for j, seed in enumerate(SEEDS):
        idx = orders[seed]
        if len(idx) != 200 or len(np.unique(idx)) != 200:
            raise ValueError("Expected 200 distinct common support questions.")
        started = perf_counter()
        native = native_routes(upstream, x[idx], history[:, idx], xx, [seed])
        timing.append(
            dict(
                method="TextMF-same-200",
                seed=seed,
                fit_and_score_seconds=perf_counter() - started,
                probes=len(idx),
            )
        )
        for label, row in zip(("TextMF-same-deterministic", "TextMF-same-noisy"), native):
            scored.setdefault(label, []).append(row)
        best = int(benchmark_means(history[:, idx], dt[idx]).mean(1).argmax())
        scored.setdefault("panel-best-fixed", []).append(np.full(len(query), best))
        for n in (120, 200):
            use = idx[:n]
            start = perf_counter()
            pool, fit = fit_pool(x[use], dt[use], history[:, use], text[use], use, names, seed)
            pred = predict(pool, xx, query)
            timing.append(
                dict(
                    method=f"panel-{n}",
                    seed=seed,
                    probes=n,
                    fit_and_score_seconds=perf_counter() - start,
                    fit_seconds=fit["seconds"],
                    embedding_width=fit["embedding_width"],
                )
            )
            scored.setdefault(f"panel-{n}", []).append(pred.argmax(0))
            if j == 0:
                joblib.dump(pool, args.out / f"{dataset}_panel-{n}.joblib", compress=2)
        print(dataset, seed, "completed", flush=True)
    scored = {name: np.asarray(rows) for name, rows in scored.items()}
    hit = {name: truth[rows, np.arange(len(query))] for name, rows in scored.items()}
    summary, pairs, seeds = [], [], []
    for name, rows in scored.items():
        summary.append(
            dict(
                dataset=dataset,
                method=name,
                candidates=len(names),
                training_questions=len(dt),
                test_questions=len(dd),
                **metrics(rows, truth, dd),
            )
        )
        for i, seed in enumerate(SEEDS):
            seeds.append(dict(method=name, seed=seed, **metrics(rows[i : i + 1], truth, dd)))
        for method in ("panel-120", "panel-200"):
            if method != name:
                delta = hit[method] - hit[name]
                pairs.append(
                    dict(
                        method=method,
                        reference=name,
                        **paired_gain(hit[method], hit[name], dd),
                        rescues=(delta > 0).sum(1).mean(),
                        harms=(delta < 0).sum(1).mean(),
                    )
                )
    for key, rows in (("summary", summary), ("paired", pairs), ("seeds", seeds), ("cost", timing)):
        atomic_csv(args.report / f"{dataset}_{key}.csv", rows)
    print(
        dataset,
        [(row["method"], round(row["micro"], 4), round(row["top_share"], 4)) for row in summary],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("llmrouterbench", "xroutebench"), required=True)
    p.add_argument(
        "--text-only",
        action="store_true",
        help="Evaluate the later raw-text decoder on the same external panels.",
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/external_confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/external_confirmation")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
