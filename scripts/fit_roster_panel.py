"""Fit a roster from its own common panel alone and predict new questions with the same CPU skill estimator as the public release."""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from coldstart.conditional_text import lexical
from coldstart.model_covariates import model_descriptions
from route_query import ROOT
from panel_capabilities import panel_text
from panel_tabular_skills import select_skills, tabular_scores


def fit_pool(x, dom, y, texts, ids, names, seed, device="cpu", text_only=False):
    t = perf_counter()
    if len(names) == 1:
        z = np.stack([y[:, dom == j].mean(1) for j in np.unique(dom)], axis=1)
        skill, fit = dict(z=z, axes=np.empty((1, 0)), texts=texts, domains=dom), {}
    else:
        meta, size = model_descriptions(names, ROOT / "data/metadata/model_families.csv")
        if text_only:
            from panel_text_kernel import fit_text_kernel

            skill, fit = fit_text_kernel(dom, y, texts, meta, size, seed)
        else:
            skill, fit = select_skills(x, dom, y, texts, lexical(texts, []), meta, size, seed, device=device)
    pool = dict(
        algorithm="text-kernel" if text_only else "tabular-panel",
        method=f"panel-text-{len(ids)}" if text_only else f"panel-{len(ids)}",
        skill=skill,
        model_names=np.asarray(names),
        probe_ids=np.asarray(ids),
        seed=seed,
    )
    return pool, dict(
        **fit,
        seconds=perf_counter() - t,
        questions=len(ids),
        embedding_width=skill["z"].shape[1] + skill["axes"].shape[1],
    )


def predict(pool, x, texts, estimators=None):
    if len(pool["model_names"]) == 1:
        from panel_capabilities import template_membership

        skill = pool["skill"]
        return skill["z"] @ template_membership(skill["texts"], texts, skill["domains"]).T
    if pool["algorithm"] == "text-kernel":
        from panel_text_kernel import text_kernel_scores

        return text_kernel_scores(pool["skill"], texts)
    return tabular_scores(pool["skill"], x, np.asarray(texts), lexical(texts, []), estimators)


def run(args):
    from evaluate_capability_router import input_context, test_rosters
    from run_experiment import atomic_json

    d, s, _, _, prepared, family = input_context(args.run)
    if args.development:
        names = dict(zip(d.model_ids, d.model_names))
        rosters = [("development_13", d.model_ids[s.val_models], d.labels[s.val_models])]
    else:
        names, rosters = test_rosters(d, s, prepared, family)
    with np.load(args.out / "questions.npz") as f:
        panels = {seed: f[f"balanced_{seed}"][:200] for seed in (11, 23, 37)}
        query = np.r_[f["confirmation"], f["previous_test"]]
        if args.development:
            query = f["development"]
    raw = ROOT / "data/raw"
    texts = panel_text(d.prompt_ids[query], raw)
    folder = args.out / "strict_panels"
    folder.mkdir(exist_ok=True)
    costs, validation = [], []
    for roster, unordered, labels in rosters:
        order = np.argsort(unordered)
        mids = unordered[order]
        roster_names = [names[m] for m in mids]
        for seed, full in panels.items():
            support = panel_text(d.prompt_ids[full], raw)
            for n in (120, 200):
                pos = full[:n]
                # Every candidate outcome supplied to the learner is in this one shared prefix.
                grades = labels[np.ix_(order, pos)]
                if not np.isfinite(grades).all():
                    raise ValueError(f"The common panel for {roster} has missing candidate grades.")
                pool, detail = fit_pool(
                    d.embeddings[pos],
                    d.domains[pos],
                    grades,
                    support[:n],
                    d.prompt_ids[pos],
                    roster_names,
                    seed,
                    text_only=args.text_only,
                )
                begin = perf_counter()
                score = predict(pool, d.embeddings[query], texts)
                costs.append(
                    dict(roster=roster, seed=seed, **detail, cached_query_seconds=perf_counter() - begin)
                )
                file = f"{roster}_{pool['method']}"
                np.savez_compressed(
                    folder / f"{file}_{seed}.npz",
                    scores=score.astype(np.float32),
                    prompt_ids=d.prompt_ids[query],
                    model_ids=mids,
                )
                if args.development:
                    from evaluate_capability_router import metrics

                    validation.append(
                        dict(
                            method=pool["method"],
                            seed=seed,
                            **metrics(score.argmax(0)[None], labels[np.ix_(order, query)], d.domains[query]),
                        )
                    )
                elif seed == 11:
                    pool["model_ids"] = mids
                    joblib.dump(pool, args.out / f"{file}.joblib", compress=2)
                    np.savez_compressed(
                        args.out / f"{file}_scores.npz",
                        scores=score.astype(np.float32),
                        prompt_ids=d.prompt_ids[query],
                    )
                print(roster, seed, n, round(detail["seconds"], 1), "fit seconds", flush=True)
    if args.development:
        from run_experiment import atomic_csv

        filename = "text_kernel_development.csv" if args.text_only else "strict_panel_development.csv"
        atomic_csv(args.report / filename, validation)
        return
    atomic_json(
        args.report / ("text_kernel_panels.json" if args.text_only else "strict_panels.json"),
        dict(
            training=costs,
            rule="Each three-fold fit uses only this roster's measured prefix and public model ancestry.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    p.add_argument("--text-only", action="store_true", help="Use the CPU word, character and format kernel.")
    p.add_argument(
        "--development", action="store_true", help="Measure the low-cost profile on development models."
    )
    with threadpool_limits(limits=3):
        run(p.parse_args())
