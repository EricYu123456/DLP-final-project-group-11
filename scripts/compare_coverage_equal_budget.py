"""Refit the published TextMF on each independently sampled clean 200-question coverage panel."""

import argparse
import importlib.util
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from compare_release_equal_budget import final_rows, select_epoch, training_order
from confirm_external_panels import native_checkpoints
from cover_release_panels import BASE, DATA, OUT, REPORT, SEEDS
from replicate_release_panels import grades, two_way_interval
from run_experiment import atomic_csv


def run(device, order="published"):
    spec = importlib.util.spec_from_file_location(
        "upstream_coveragemf", BASE / "release112/upstream_mf.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    with (
        np.load(OUT / "panels.npz") as panels,
        np.load(OUT / "routes.npz") as routes,
        np.load(DATA / "processed/embedllm.npz") as raw,
    ):
        chosen, ref, q, cut = (
            panels["coverage"],
            routes["coverage"],
            routes["prompts"],
            int(routes["fresh_count"]),
        )
        x, dom, groups = raw["embeddings"], raw["benchmark_ids"], raw["group_ids"]
    rows = final_rows(DATA / "raw/train.csv", np.unique(chosen))
    directory = OUT / ("matched-native" if order == "published" else f"matched-native-{order}")
    directory.mkdir(exist_ok=True)
    metadata, routes, validation = [], [], []
    for seed, ids in zip(SEEDS, chosen):
        file = directory / f"{seed}.npz"
        if not file.exists():
            local = rows[rows.prompt_id.isin(ids)]
            y = local.pivot(index="model_id", columns="prompt_id", values="label")[
                ids
            ].to_numpy(np.float32)
            item = local.prompt_id.map(
                {int(v): i for i, v in enumerate(ids)}
            ).to_numpy()
            model = local.model_id.to_numpy()
            if (
                y.shape != (112, 200)
                or len(local) != y.size
                or not np.array_equal(y[model, item], local.label.to_numpy())
            ):
                raise ValueError(
                    "The original and new routers must use the same observed grades."
                )
            item, model = training_order(item, model, ids, order)
            start = perf_counter()
            param, scores = select_epoch(
                upstream,
                dict(x=x[ids], domains=dom[ids], y=y),
                groups[ids],
                item,
                model,
                seed,
                device,
            )
            _, pred = next(
                native_checkpoints(
                    upstream,
                    x[ids],
                    y,
                    x[q],
                    [seed],
                    (param,),
                    item * len(y) + model,
                    device,
                )
            )
            np.savez_compressed(
                file,
                probe_ids=ids,
                query=q,
                chosen_epochs=param,
                seconds=perf_counter() - start,
                choices=pred,
                fold_scores=[
                    [v[k] for k in ("epochs", "micro", "macro", "objective")]
                    for v in scores
                ],
            )
        with np.load(file) as f:
            if not np.array_equal(f["probe_ids"], ids) or not np.array_equal(
                f["query"], q
            ):
                raise ValueError("The resumed TextMF run uses a different panel.")
            metadata.append(
                dict(
                    seed=seed,
                    epochs=int(f["chosen_epochs"]),
                    seconds=float(f["seconds"]),
                    device=device,
                )
            )
            routes.append(f["choices"])
            validation.extend(
                dict(seed=seed, epochs=int(e), micro=m, macro=a, objective=o)
                for e, m, a, o in f["fold_scores"]
            )
        print("Matched original", metadata[-1], flush=True)
    results = []
    routes = np.stack(routes)
    for stage, ix in (("fresh", np.arange(cut)), ("official", np.arange(cut, len(q)))):
        ids = q[ix]
        truth = grades("train.csv" if stage == "fresh" else "test.csv", ids)
        hit = truth[ref[:, ix], np.arange(len(ix))]
        for i, name in enumerate(("deterministic", "noisy")):
            native = truth[routes[:, i, :][:, ix], np.arange(len(ix))]
            gap = hit - native
            results.append(
                dict(
                    split=stage,
                    reference=name,
                    questions=len(ids),
                    panel_count=len(hit),
                    accuracy=hit.mean(),
                    original=native.mean(),
                    gain=gap.mean(),
                    panel_wins=(gap.mean(1) > 0).sum(),
                    **two_way_interval(gap, dom[ids], groups[ids]),
                )
            )
    prefix = "equal_200" if order == "published" else f"equal_200_{order}"
    atomic_csv(REPORT / f"{prefix}_costs.csv", metadata)
    atomic_csv(REPORT / f"{prefix}_validation.csv", validation)
    atomic_csv(REPORT / f"{prefix}_paired.csv", results)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--order", choices=("published", "interleaved"), default="published")
    torch.set_num_threads(2)
    args = parser.parse_args()
    run(args.device, args.order)
