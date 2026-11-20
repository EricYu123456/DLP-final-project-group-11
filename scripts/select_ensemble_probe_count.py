"""Select a shorter probe budget on development models for the full-history ensemble."""

import argparse
import json

import joblib
import numpy as np
import torch
from ensemble_nominal_profiles import (
    combine,
    ensemble_folder,
    fit_ensemble,
    inference_models,
    member_scores,
)
from evaluate_capability_router import (
    input_context,
    known_choices,
    metrics,
    test_rosters,
)
from evaluate_panel_budget import nested_order
from panel_capabilities import panel_text
from replicate_nominal_profiles import DATA, FOLDER, OUT, REPORT, ROOT, SEEDS
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv, atomic_json
from threadpoolctl import threadpool_limits
from train_conditional_router import proxy_data

from coldstart.conditional_text import text_features
from coldstart.latent_answers import answer_arrays
from coldstart.nominal_response import query

SIZES = (80, 120, 160, 200)
LOCK = REPORT / "ensemble_probe_budget_selection.json"


def run(development):
    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    folder = ensemble_folder(OUT, True)
    with np.load(OUT / "questions.npz") as mf, np.load(FOLDER / "study.npz") as st:
        tr, panels = mf["training"], st["panels"]
        stages = (
            {"development": mf["development"]}
            if development
            else {
                "confirmation": mf["confirmation"],
                "previous-test": mf["previous_test"],
            }
        )
    q = np.concatenate(list(stages.values()))
    if development:
        sizes = SIZES
        folder = folder / "development"
        names = dict(zip(d.model_ids, d.model_names))
        rosters = [
            ("development_13", d.model_ids[s.val_models], d.labels[s.val_models])
        ]
        with np.load(OUT / "text/predictions.npz") as pred:
            peer, dom = pred["dev"], pred["dev_source"]
    else:
        chosen = json.loads(LOCK.read_text(encoding="utf-8"))["questions"]
        sizes = tuple(sorted({chosen, 200}))
        names, rosters = test_rosters(d, s, prepared, family)
        _, ref = inference_models(OUT, True)
        h = ref["head"]
        peer, m = h["readout"].predict(
            d.embeddings[q],
            text_features(
                h["pack"], d.embeddings[q], panel_text(d.prompt_ids[q], DATA / "raw")
            ),
        )
        dom = m.argmax(1)
    members = [joblib.load(folder / f"ensemble_member_{j}.joblib") for j in range(9)]
    base = joblib.load(folder / "fit.joblib")
    cfg = base["config"]
    nets = [m["net"] for m in members]
    raw = answer_arrays(
        proxy_data(
            OUT,
            d.prompt_ids[q],
            cfg["proxies"].split(),
            "development" if development else "evaluation",
        )
    )
    items = [query(net, raw, dom) for net in nets]
    pos = {int(v): i for i, v in enumerate(tr)}
    rows, pairs = [], []
    for roster, unordered, truth in rosters:
        order = unordered.argsort()
        mids, truth = unordered[order], truth[order]
        routes = {n: [] for n in sizes}
        for seed, panel in zip(SEEDS, panels):
            prefix = nested_order(d.domains[panel], seed)
            for n in sizes:
                sampled = panel[np.sort(prefix[:n])]
                idx = np.asarray([pos[int(v)] for v in sampled])
                y = np.full((len(mids), len(tr)), np.nan, np.float32)
                y[:, idx] = truth[:, sampled]
                pool = fit_ensemble(
                    base,
                    members,
                    y,
                    idx,
                    [names[k] for k in mids],
                    d.prompt_ids[sampled],
                    seed,
                    "nominal-ensemble-cautious-full",
                    device="cpu",
                )
                matrix = combine(
                    member_scores(
                        pool["profiles"], nets, items, dom, peer, base["means"], cfg
                    ),
                    pool["anchor"],
                    pool["risk_penalty"],
                )
                routes[n].append(matrix.argmax(0))
                if not development and seed == SEEDS[0] and n == chosen:
                    joblib.dump(
                        pool, OUT / f"{roster}_ensemble-budget.joblib", compress=2
                    )
                    np.savez_compressed(
                        OUT / f"{roster}_ensemble-budget_scores.npz",
                        prompt_ids=d.prompt_ids[q],
                        scores=matrix,
                    )
        routes = {n: np.stack(v) for n, v in routes.items()}
        for stage, study in stages.items():
            ix = np.flatnonzero(np.isin(q, study) & np.isfinite(truth[:, q]).all(0))
            y, source = truth[:, q[ix]], d.domains[q[ix]]
            hit = {n: y[v[:, ix], np.arange(len(ix))] for n, v in routes.items()}
            for n, v in routes.items():
                rows.append(
                    dict(
                        roster=roster,
                        split=stage,
                        questions=n,
                        **metrics(v[:, ix], y, source),
                    )
                )
            if not development:
                with np.load(OUT / "embedllm_known_ids.npz") as f:
                    original = known_choices(f, d, mids, q[ix], SEEDS)[0]
                for ref, h in (
                    ("same-200", hit[200]),
                    ("EmbedLLM-known", y[original, np.arange(len(ix))]),
                ):
                    gap = hit[chosen] - h
                    pairs.append(
                        dict(
                            roster=roster,
                            split=stage,
                            questions=chosen,
                            reference=ref,
                            gain=gap.mean(),
                            wins=int((gap.mean(1) > 0).sum()),
                            **two_way_interval(gap, source, d.group_ids[q[ix]]),
                        )
                    )
    if development:
        full = next(v for v in rows if v["questions"] == 200)
        selected = min(
            v["questions"]
            for v in rows
            if v["micro"] >= full["micro"] - 0.0025
            and v["macro"] >= full["macro"] - 0.0025
            and v["top_share"] <= full["top_share"] + 0.02
        )
        atomic_csv(REPORT / "ensemble_probe_budget_validation.csv", rows)
        atomic_json(
            LOCK,
            {
                "questions": selected,
                "method": "nominal-ensemble-cautious-full",
                "panels": len(panels),
                "development_models": len(truth),
                "development_questions": len(q),
            },
        )
        print("Selected", selected, "questions", flush=True)
    else:
        atomic_csv(REPORT / "ensemble_probe_budget_results.csv", rows)
        atomic_csv(REPORT / "ensemble_probe_budget_paired.csv", pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("select", "score"), required=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(p.parse_args().phase == "select")
