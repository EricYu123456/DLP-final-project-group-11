"""Compare two cold-start routers using identical grades in the original 13- and 37-model pools."""

import argparse
import importlib.util
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from compare_release_equal_budget import final_rows, select_epoch, training_order
from confirm_external_panels import native_checkpoints
from evaluate_capability_router import input_context, known_choices, metrics, test_rosters
from fit_roster_panel import fit_pool, predict
from panel_capabilities import panel_text
from replicate_nominal_profiles import DATA, FOLDER, OUT, REPORT, ROOT, SEEDS
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv
from threadpoolctl import threadpool_limits

FITS = FOLDER / "equal-200"


def run(roster, phase, device, order_mode="published"):
    d, split, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    names, rosters = test_rosters(d, split, prepared, family)
    unordered, truth = next((m, y) for name, m, y in rosters if name == roster)
    order = unordered.argsort()
    mids, truth = unordered[order], truth[order]
    with np.load(FOLDER / "study.npz") as study:
        panels = study["panels"]
        stages = {
            "combined-withheld": np.r_[study["novel"], study["active"], study["coverage"]],
            "confirmation": study["confirmation"],
        }
    queries = np.concatenate(list(stages.values()))
    query_ids = d.prompt_ids[queries]
    directory = FITS / roster
    directory.mkdir(exist_ok=True, parents=True)
    native_name = "native" if order_mode == "published" else f"native-{order_mode}"
    if phase == "score":
        routes = {}
        costs = []
        for seed, panel in zip(SEEDS, panels):
            detail = {"seed": seed, "candidates": len(mids), "grades": len(mids) * len(panel)}
            for approach, keys in (
                (native_name, ("deterministic", "noisy")),
                ("panel", ("roster-200", "fixed")),
            ):
                with np.load(directory / f"{approach}_{seed}.npz") as f:
                    if (
                        not np.array_equal(f["panel"], d.prompt_ids[panel])
                        or not np.array_equal(f["queries"], query_ids)
                        or not np.array_equal(mids, f["model_ids"])
                    ):
                        raise ValueError(
                            "The paired comparison requires identical candidates and questions."
                        )
                    for k in keys:
                        routes.setdefault(k, []).append(f[k])
                    detail[approach + "_seconds"] = float(f["seconds"])
                    if approach == native_name:
                        detail.update(
                            epochs=int(f["epochs"]),
                            source_label_differences=int(f["source_label_differences"]),
                        )
            costs.append(detail)
        results, paired = [], []
        with np.load(FOLDER / "original.npz") as withheld, np.load(OUT / "embedllm_known_ids.npz") as saved:
            weights, bias = withheld["weight"][mids], withheld["bias"][mids]
            known = dict(saved)
        for stage, raw in stages.items():
            ix = np.flatnonzero(np.isin(queries, raw) & np.isfinite(truth[:, queries]).all(0))
            y = truth[:, queries[ix]]
            group, dom = d.group_ids[queries[ix]], d.domains[queries[ix]]
            hits = {}
            for method, value in routes.items():
                route = np.stack(value)[:, ix]
                hits[method] = y[route, np.arange(len(ix))]
                results.append(
                    dict(split=stage, method=method, questions=len(ix), **metrics(route, y, dom))
                )
            if stage == "confirmation":
                complete, noisy = known_choices(known, d, mids, queries[ix], SEEDS)
            else:
                logits = weights @ d.embeddings[queries[ix]].T + bias[:, None]
                complete = np.tile(logits.argmax(0), (len(SEEDS), 1))
                deviation = 0.05 * np.linalg.norm(weights, axis=1, keepdims=True)
                noisy = np.stack(
                    [
                        (logits + deviation * np.random.default_rng(s).normal(size=logits.shape)).argmax(0)
                        for s in SEEDS
                    ]
                )
            for method, route in (("full-deterministic", complete), ("full-noisy", noisy)):
                hits[method] = y[route, np.arange(len(ix))]
                results.append(
                    dict(split=stage, method=method, questions=len(ix), **metrics(route, y, dom))
                )
            for reference in ("deterministic", "noisy", "fixed", "full-deterministic", "full-noisy"):
                gap = hits["roster-200"] - hits[reference]
                paired.append(
                    dict(
                        split=stage,
                        reference=reference,
                        gain=gap.mean(),
                        wins=int((gap.mean(1) > 0).sum()),
                        rescues=(gap > 0).sum(1).mean(),
                        harms=(gap < 0).sum(1).mean(),
                        **two_way_interval(gap, dom, group),
                    )
                )
        for key, data in (("results", results), ("paired", paired), ("costs", costs)):
            suffix = "" if order_mode == "published" else f"_{order_mode}"
            atomic_csv(REPORT / f"equal_200_{roster}{suffix}_{key}.csv", data)
        return

    if phase == "native":
        spec = importlib.util.spec_from_file_location(
            "upstream_roster_mf", OUT.parent / "release112/upstream_mf.py"
        )
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        support_ids = np.unique(d.prompt_ids[panels])
        # The project's five-task split regroups train, val and test from the public ten-task split.
        rows = pd.concat(
            [final_rows(DATA / "raw" / f"{part}.csv", support_ids) for part in ("train", "val", "test")],
            ignore_index=True,
        )
        rows = rows[rows.model_id.isin(mids)]
        model_index = {int(mid): i for i, mid in enumerate(mids)}
    else:
        texts = panel_text(query_ids, DATA / "raw")
        bank = np.unique(panels)
        support = panel_text(d.prompt_ids[bank], DATA / "raw")
    for seed, panel in zip(SEEDS, panels):
        file = directory / f"{native_name if phase == 'native' else phase}_{seed}.npz"
        if file.exists():
            continue
        ids, y = d.prompt_ids[panel], truth[:, panel]
        result = {"panel": ids, "queries": query_ids, "model_ids": mids}
        if not np.isfinite(y).all():
            raise ValueError("Each candidate needs exactly one observed grade on each of the 200 prompts.")
        begin = perf_counter()
        if phase == "native":
            selected = rows[rows.prompt_id.isin(ids)]
            item = selected.prompt_id.map({int(v): i for i, v in enumerate(ids)}).to_numpy()
            model = selected.model_id.map(model_index).to_numpy()
            if len(item) != y.size:
                raise ValueError("The original public order is missing some of the measured panel cells.")
            differences = int((y[model, item] != selected.label.to_numpy()).sum())
            item, model = training_order(item, model, ids, order_mode)
            epochs, _ = select_epoch(
                upstream,
                {"x": d.embeddings[panel], "domains": d.domains[panel], "y": y},
                d.group_ids[panel],
                item,
                model,
                seed,
                device,
            )
            _, routes = next(
                native_checkpoints(
                    upstream,
                    d.embeddings[panel],
                    y,
                    d.embeddings[queries],
                    [seed],
                    (epochs,),
                    item * len(y) + model,
                    device,
                )
            )
            result.update(
                deterministic=routes[0],
                noisy=routes[1],
                epochs=epochs,
                source_label_differences=differences,
            )
        else:
            pool, _ = fit_pool(
                None,
                d.domains[panel],
                y,
                support[np.searchsorted(bank, panel)],
                ids,
                [names[k] for k in mids],
                seed,
                text_only=True,
            )
            matrix = predict(pool, None, texts)
            result["roster-200"] = matrix.argmax(0)
            result["fixed"] = np.full(len(queries), y.mean(1).argmax())
            if seed == SEEDS[0]:
                pool.update(method="roster-200", model_ids=mids)
                joblib.dump(pool, OUT / f"{roster}_roster-200.joblib", compress=2)
                np.savez_compressed(
                    OUT / f"{roster}_roster-200_scores.npz", prompt_ids=query_ids, scores=matrix
                )
                skill = pool["skill"]
                np.savez_compressed(
                    OUT / f"{roster}_roster-200_embeddings.npz",
                    model_names=pool["model_names"],
                    model_ids=mids,
                    probe_ids=ids,
                    embeddings=np.column_stack((skill["z"], skill["axes"])),
                )
        result["seconds"] = perf_counter() - begin
        np.savez_compressed(file, **result)
        print(roster, phase, seed, round(result["seconds"], 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", choices=("audited_13", "expanded_37"), required=True)
    parser.add_argument("--phase", choices=("panel", "native", "score"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--order", choices=("published", "model-major", "interleaved"), default="published")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.roster, args.phase, args.device, args.order)
