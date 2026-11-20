"""Refit the author's TextMF on the same 22,400 published grades as the fitted CPU text router."""

import argparse
import importlib.util
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from confirm_external_panels import native_checkpoints, native_routes
from confirm_release_panel import paired_groups
from evaluate_capability_router import metrics
from panel_capabilities import RELEASE_SEEDS, panel_text
from panel_text_kernel import text_kernel_scores
from route_query import ROOT
from run_embedllm_release import chunks
from run_experiment import atomic_csv


def final_rows(path, ids):
    rows = pd.concat((chunk[chunk.prompt_id.isin(ids)] for chunk in chunks(path)), ignore_index=True)
    return rows.drop_duplicates(["model_id", "prompt_id"], keep="last")


def training_order(item, model, prompt_ids, mode):
    if mode == "published":
        return item, model
    if mode == "model-major":
        order = np.lexsort((prompt_ids[item], model))
    else:
        # Start from the same cell numbering for every file layout, then interleave once.
        ordered = np.lexsort((model, item))
        order = ordered[np.random.default_rng(42).permutation(len(item))]
    return item[order], model[order]


def tabular_choices(release, seed, parts, xx, texts):
    from coldstart.conditional_text import lexical
    from panel_tabular_skills import prepare_estimators, tabular_scores

    if seed == RELEASE_SEEDS[0]:
        choices = []
        for name, pids in parts.items():
            with np.load(release / f"tabular-panel_{name}_scores.npz") as f:
                if not np.array_equal(pids, f["prompt_ids"]):
                    raise ValueError("The stored neural and text router questions differ.")
                choices.append(f["scores"].argmax(0))
        return np.concatenate(choices)
    pack = joblib.load(release / f"tabular-panel_{seed}.joblib")
    estimators = prepare_estimators(pack)
    return tabular_scores(pack, xx, texts, lexical(texts, []), estimators).argmax(0)


def select_epoch(upstream, panel, group, item, model, seed, device="cpu"):
    x, y, dom = panel["x"], panel["y"], panel["domains"]
    epochs = (50, 200, 800)
    draws = {k: np.empty(len(x), dtype=int) for k in epochs}
    fold = np.empty(len(x), dtype=int)
    rng = np.random.default_rng(seed + 814)
    for source in np.unique(dom):
        in_source = dom == source
        groups = rng.permutation(np.unique(group[in_source]))
        for k in range(3):
            fold[in_source & np.isin(group, groups[k::3])] = k
    for k in range(3):
        keep = fold != k
        cell = keep[item]
        position = np.cumsum(keep) - 1
        order = position[item[cell]] * len(y) + model[cell]
        for epoch, pred in native_checkpoints(upstream, x[keep], y[:, keep], x[~keep], [], epochs, order, device):
            draws[epoch][~keep] = pred[0]
        print("Within-panel TextMF", seed, "fold", k + 1, flush=True)
    results = [dict(seed=seed, epochs=k, **metrics(route[None], y, dom)) for k, route in draws.items()]
    selected = max(results, key=lambda r: (r["objective"], -r["epochs"]))["epochs"]
    return selected, results


def run(decoders, tune):
    base = ROOT / "artifacts/capability_routing"
    release = base / "release_confirmation"
    raw = ROOT / "data/raw"
    report = ROOT / "reports/routing/capability/release_confirmation"
    spec = importlib.util.spec_from_file_location("official_equal_budget", base / "release112/upstream_mf.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    with np.load(release / "reserve.npz") as f:
        parts = {"reserve": f["prompt_ids"], "test": f["test"]}
    queries = np.concatenate(list(parts.values()))
    texts = panel_text(queries, raw)
    with np.load(ROOT / "data/processed/embedllm.npz") as f:
        xx, dom, group = (f[key][queries] for key in ("embeddings", "benchmark_ids", "group_ids"))
        all_groups = f["group_ids"]
    panels = {}
    for seed in RELEASE_SEEDS:
        with np.load(base / f"release112/panel_{seed}.npz") as f:
            panels[seed] = {k: f[k] for k in ("prompt_ids", "x", "y", "domains")}
    ids = np.concatenate([p["prompt_ids"] for p in panels.values()])
    train = final_rows(raw / "train.csv", np.r_[ids, parts["reserve"]])
    test = final_rows(raw / "test.csv", parts["test"])
    truth = np.column_stack(
        [
            df[df.prompt_id.isin(pids)]
            .pivot(index="model_id", columns="prompt_id", values="label")[pids]
            .to_numpy()
            for df, pids in zip((train, test), parts.values())
        ]
    )
    scores = (
        {"tabular-panel": [], "text-kernel": []}
        if decoders
        else {"TextMF-same-200-deterministic": [], "TextMF-same-200-noisy": [], "text-kernel": []}
    )
    costs, validation = [], []
    for seed, panel in panels.items():
        pids, x, y = panel["prompt_ids"], panel["x"], panel["y"]
        selected = train[train.prompt_id.isin(pids)]
        lookup = {int(v): i for i, v in enumerate(pids)}
        item = selected.prompt_id.map(lookup).to_numpy()
        model = selected.model_id.to_numpy()
        if len(item) != y.size or not np.array_equal(y[model, item], selected.label):
            raise ValueError("The two learners must receive exactly the same 200 graded questions per LLM.")
        before = perf_counter()
        epochs = 50
        if decoders:
            scores["tabular-panel"].append(tabular_choices(release, seed, parts, xx, texts))
        else:
            if tune:
                epochs, results = select_epoch(upstream, panel, all_groups[pids], item, model, seed)
                validation.extend(results)
            det, noisy = native_routes(upstream, x, y, xx, [seed], order=item * len(y) + model, epochs=epochs)
            scores["TextMF-same-200-deterministic"].append(det)
            scores["TextMF-same-200-noisy"].append(noisy)
        costs.append(
            dict(
                seed=seed,
                candidates=len(y),
                questions=len(pids),
                grades=y.size,
                **({"selected_epochs": epochs} if tune else {}),
                **{
                    ("stored_or_replayed_seconds" if decoders else "fit_and_score_seconds"): perf_counter()
                    - before
                },
            )
        )
        pack = joblib.load(release / f"text-kernel_{seed}.joblib")
        scores["text-kernel"].append(text_kernel_scores(pack, texts).argmax(0))
        print("Matched panel", seed, costs[-1], flush=True)
    summary, paired = [], []
    for phase, pids in parts.items():
        select = np.isin(queries, pids)
        hits = {}
        for name, choices in scores.items():
            routes = np.asarray(choices)[:, select]
            summary.append(dict(split=phase, method=name, **metrics(routes, truth[:, select], dom[select])))
            hits[name] = truth[:, select][routes, np.arange(len(pids))]
        for name in scores.keys() - {"text-kernel"}:
            delta = hits["text-kernel"] - hits[name]
            paired.append(
                dict(
                    split=phase,
                    reference=name,
                    **paired_groups(hits["text-kernel"], hits[name], dom[select], group[select]),
                    rescues=(delta > 0).sum(1).mean(),
                    harms=(delta < 0).sum(1).mean(),
                )
            )
    for name, rows in (("summary", summary), ("paired", paired), ("cost", costs)):
        prefix = "decoder_comparison" if decoders else ("equal_200_tuned" if tune else "equal_200")
        atomic_csv(report / f"{prefix}_{name}.csv", rows)
    if tune:
        atomic_csv(report / "equal_200_tuned_validation.csv", validation)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--decoders", action="store_true", help="Compare the two fitted 18D query decoders directly."
    )
    group.add_argument(
        "--tune-epochs",
        action="store_true",
        help="Select the native training duration using only the 200 questions.",
    )
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        args = parser.parse_args()
        run(args.decoders, args.tune_epochs)
