"""Compare the 120-question incumbent and larger embedding budgets on separate development models."""

import argparse
import json
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from confirmation_models import ConfirmationModels
from distill_conditional_router import teacher_for_pool
from ensemble_nominal_profiles import (
    METHODS,
    combine,
    ensemble_folder,
    fit_ensemble,
    member_scores,
    train,
)
from evaluate_capability_router import input_context, metrics
from evaluate_panel_budget import nested_order
from replicate_nominal_profiles import FOLDER, OUT, REPORT, ROOT, SEEDS
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv, atomic_json
from threadpoolctl import threadpool_limits
from train_conditional_router import proxy_data

from coldstart.latent_answers import answer_arrays
from coldstart.nominal_response import query


def validate():
    d, split, *_ = input_context(ROOT / "artifacts/main_v2")
    folder = ensemble_folder(OUT, True) / "development"
    base = joblib.load(folder / "fit.joblib")
    members = [joblib.load(folder / f"ensemble_member_{j}.joblib") for j in range(9)]
    nets = [m["net"] for m in members]
    with np.load(OUT / "questions.npz") as f, np.load(FOLDER / "study.npz") as study:
        manifest, panels = dict(f), study["panels"]
    tr, dev = manifest["training"], manifest["development"]
    mids = split.val_models
    names = np.asarray(d.model_names)[mids]
    truth = d.labels[np.ix_(mids, dev)]
    cfg = base["config"]
    with np.load(OUT / "text/predictions.npz") as f:
        peer, domain = f["dev"], f["dev_source"]
    inputs = answer_arrays(proxy_data(OUT, d.prompt_ids[dev], cfg["proxies"].split(), "development"))
    items = [query(net, inputs, domain) for net in nets]
    choice = json.loads((REPORT / "selection.json").read_text(encoding="utf-8"))["conditional"]
    args = SimpleNamespace(out=OUT, report=REPORT, run=ROOT / "artifacts/main_v2")
    teacher = ConfirmationModels(args, d, split, manifest, dev, "development", False)
    teacher.target_names = names
    pos = {int(v): i for i, v in enumerate(tr)}
    routes = {k: [] for k in ("conditional", *METHODS)}
    for seed, panel in zip(SEEDS, panels):
        idx = np.asarray([pos[int(v)] for v in panel])
        prefix = np.sort(nested_order(d.domains[panel], seed)[:120])
        y = np.full((len(mids), len(tr)), np.nan, np.float32)
        y[:, idx[prefix]] = d.labels[np.ix_(mids, panel[prefix])]
        routes["conditional"].append(teacher_for_pool(teacher, choice, y, idx[prefix], seed).argmax(0))
        y[:, idx] = d.labels[np.ix_(mids, panel)]
        pool = fit_ensemble(
            base, members, y, idx, names, d.prompt_ids[panel], seed, "nominal-ensemble", device="cpu"
        )
        mat = member_scores(pool["profiles"], nets, items, domain, peer, base["means"], cfg)
        for method, (count, penalty) in METHODS.items():
            routes[method].append(combine(mat[:count], pool["anchor"], penalty).argmax(0))
    rows, hit = [], {}
    for method, values in routes.items():
        route = np.stack(values)
        scores = metrics(route, truth, d.domains[dev])
        hit[method] = truth[route, np.arange(len(dev))]
        rows.append(
            dict(
                method=method,
                probes=120 if method == "conditional" else 200,
                members=METHODS.get(method, (0, 0))[0],
                **scores,
            )
        )
    current = rows[0]
    eligible = [
        r
        for r in rows[1:]
        if r["macro"] > current["macro"]
        and r["micro"] > current["micro"]
        and r["top_share"] <= current["top_share"] + 0.02
    ]
    chosen = min(eligible, key=lambda r: (r["members"], -r["objective"])) if eligible else current
    paired = []
    for method in METHODS:
        gap = hit[method] - hit["conditional"]
        paired.append(
            dict(method=method, gain=gap.mean(), **two_way_interval(gap, d.domains[dev], d.group_ids[dev]))
        )
    atomic_csv(REPORT / "ensemble_development_results.csv", rows)
    atomic_csv(REPORT / "ensemble_development_paired.csv", paired)
    atomic_json(
        REPORT / "ensemble_deployment_selection.json",
        {
            "references": base["grades"].shape[0],
            "development_models": len(mids),
            "development_questions": len(dev),
            "panels": len(panels),
            "selected_method": chosen["method"] + "-full" if chosen in eligible else "conditional",
            "validation": chosen,
        },
    )
    print(chosen, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "select"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase == "train":
            train(args.device, full=True, development=True)
        else:
            validate()
