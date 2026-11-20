"""Apply the fixed within-panel posterior stopping rule to the five-task 13- and 37-model rosters."""

import argparse
import json
from time import perf_counter

import joblib
import numpy as np
from compare_roster_equal_budget import FITS
from evaluate_capability_router import input_context, metrics, test_rosters
from evaluate_panel_budget import nested_order
from fit_roster_panel import fit_pool, predict
from panel_capabilities import panel_text
from posterior_release_budget import fit_posterior
from replicate_nominal_profiles import DATA, FOLDER, OUT, REPORT, ROOT, SEEDS
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv
from serve_confirmation_router import measurements, write_probes
from threadpoolctl import threadpool_limits

from coldstart.model_covariates import model_descriptions

LIMITS = {"roster-stop": 0.005, "roster-stop-economical": 0.01}
STAGES = (80, 120, 160, 200)


def design(d, out):
    with np.load(out / "nominal-replication/study.npz") as f:
        panels, tr = f["panels"], f["train"]
    allowed = tr[~np.isin(d.group_ids[tr], np.unique(d.group_ids[panels]))]
    _, first = np.unique(d.group_ids[allowed], return_index=True)
    ungraded = np.random.default_rng(62924).choice(allowed[first], 300, replace=False)
    return panels, panel_text(d.prompt_ids[ungraded], DATA / "raw")


def checkpoint(names, ids, domain, graded, texts, ungraded, meta, size, seed):
    if len(names) == 1:
        pool, _ = fit_pool(None, domain, graded, texts, ids, names, seed, text_only=True)
        return pool, 0.0
    skill, gain = fit_posterior(domain, graded, texts, meta, size, seed, ungraded)
    return {
        "algorithm": "text-kernel",
        "skill": skill,
        "model_names": np.asarray(names),
        "probe_ids": ids,
        "seed": seed,
    }, gain


def trace(names, ids, domain, grades, texts, ungraded, seed):
    meta, size = model_descriptions(names, DATA / "metadata/model_families.csv")
    order = nested_order(domain, seed)
    for n in STAGES:
        take = np.sort(order[:n])
        if not np.isfinite(grades[:, take]).all():
            yield take, None, None
            return
        pool, gain = checkpoint(
            names, ids[take], domain[take], grades[:, take], texts[take], ungraded, meta, size, seed
        )
        yield take, pool, gain


def onboard(args):
    d, *_ = input_context(args.run)
    panels, ungraded = design(d, args.out)
    rows = panels[SEEDS.index(args.seed)]
    ids, domain = d.prompt_ids[rows], d.domains[rows]
    names, y = measurements(args.responses, args.model_name, ids)
    texts = panel_text(ids, DATA / "raw")
    with threadpool_limits(limits=2):
        for take, pool, gain in trace(names, ids, domain, y, texts, ungraded, args.seed):
            if pool is None or gain < LIMITS[args.method] or len(take) == 200:
                break
    if np.isfinite(y[:, np.setdiff1d(np.arange(len(ids)), take)]).any():
        raise ValueError("The graded CSV includes later probes than this sequential policy requested.")
    reason = "awaiting_responses" if pool is None else ("cap" if len(take) == 200 else "estimated-gain")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "plan":
        write_probes(
            args.output,
            names,
            ids[take],
            [d.benchmark_names[j] for j in domain[take]],
            texts[take],
            y[:, take],
        )
    else:
        if pool is None:
            raise ValueError("Grade the current prefix before fitting the pool.")
        pool["method"] = args.method
        joblib.dump(pool, args.output, compress=2)
    print(
        json.dumps(
            {
                "questions_per_model": len(take),
                "models": len(names),
                "file": str(args.output),
                "stop_reason": reason,
                "estimated_gain": gain,
            }
        )
    )


def fit(roster):
    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    model_names, rosters = test_rosters(d, s, prepared, family)
    unordered, truth = next((m, v) for r, m, v in rosters if r == roster)
    order = unordered.argsort()
    names, truth = [model_names[m] for m in unordered[order]], truth[order]
    panels, ungraded = design(d, OUT)
    with np.load(FOLDER / "routes.npz") as f:
        q = f["query"]
    text = panel_text(d.prompt_ids[q], DATA / "raw")
    bank = np.unique(panels)
    support = panel_text(d.prompt_ids[bank], DATA / "raw")
    choices = {key: [] for key in (*LIMITS, "roster-posterior-200")}
    steps = []
    for seed, rows in zip(SEEDS, panels):
        pending = set(LIMITS)
        start = perf_counter()
        for take, pool, gain in trace(
            names,
            d.prompt_ids[rows],
            d.domains[rows],
            truth[:, rows],
            support[np.searchsorted(bank, rows)],
            ungraded,
            seed,
        ):
            methods = [key for key in sorted(pending) if gain < LIMITS[key] or len(take) == 200]
            pending.difference_update(methods)
            if len(take) == 200:
                methods.append("roster-posterior-200")
            steps.append(
                {
                    "seed": seed,
                    "probes": len(take),
                    "gain": gain,
                    "methods": " ".join(methods),
                    "elapsed_seconds": perf_counter() - start,
                }
            )
            if methods:
                mat = predict(pool, None, text)
                for method in methods:
                    choices[method].append(mat.argmax(0))
                    if seed == SEEDS[0]:
                        pool["method"] = method
                        joblib.dump(pool, OUT / f"{roster}_{method}.joblib", compress=2)
                        np.savez_compressed(
                            OUT / f"{roster}_{method}_scores.npz", prompt_ids=d.prompt_ids[q], scores=mat
                        )
        print(roster, seed, [(r["probes"], r["methods"]) for r in steps[-4:] if r["methods"]], flush=True)
    np.savez_compressed(FOLDER / f"{roster}_stopped_routes.npz", q=q, **choices)
    atomic_csv(REPORT / f"{roster}_stopping_trace.csv", steps)


def score(roster):
    import pandas as pd

    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    _, rosters = test_rosters(d, s, prepared, family)
    model, truth = next((m, v) for r, m, v in rosters if r == roster)
    truth = truth[model.argsort()]
    with np.load(FOLDER / f"{roster}_stopped_routes.npz") as f, np.load(FOLDER / "study.npz") as st:
        q = f["q"]
        routes = {m: f[m] for m in (*LIMITS, "roster-posterior-200")}
        stages = {
            "confirmation": st["confirmation"],
            "combined-withheld": np.r_[st["novel"], st["active"], st["coverage"]],
        }
    for prefix, method in (("panel", "roster-200"), ("native", "deterministic")):
        items = []
        for seed in SEEDS:
            with np.load(FITS / roster / f"{prefix}_{seed}.npz") as f:
                if not np.array_equal(f["queries"], d.prompt_ids[q]):
                    raise ValueError("Both methods must use the same held-out questions.")
                items.append(f[method])
        routes[method] = np.stack(items)
    trace = pd.read_csv(REPORT / f"{roster}_stopping_trace.csv").fillna({"methods": ""})
    rows, pairs = [], []
    for stage, queries in stages.items():
        ix = np.flatnonzero(np.isin(q, queries) & np.isfinite(truth[:, q]).all(0))
        y, dom = truth[:, q[ix]], d.domains[q[ix]]
        hit = {}
        for method, v in routes.items():
            selected = trace[trace.methods.str.split().apply(lambda names: method in names)]
            hit[method] = y[v[:, ix], np.arange(len(ix))]
            rows.append(
                dict(
                    split=stage,
                    method=method,
                    mean_probes=selected.probes.mean() if len(selected) else 200,
                    minimum=int(selected.probes.min()) if len(selected) else 200,
                    maximum=int(selected.probes.max()) if len(selected) else 200,
                    **metrics(v[:, ix], y, dom),
                )
            )
        for method in (*LIMITS, "roster-posterior-200"):
            for ref in ("roster-200", "deterministic"):
                gap = hit[method] - hit[ref]
                pairs.append(
                    dict(
                        split=stage,
                        method=method,
                        reference=ref,
                        gain=gap.mean(),
                        wins=int((gap.mean(1) > 0).sum()),
                        **two_way_interval(gap, dom, d.group_ids[q[ix]]),
                    )
                )
    atomic_csv(REPORT / f"{roster}_stopping_results.csv", rows)
    atomic_csv(REPORT / f"{roster}_stopping_paired.csv", pairs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("fit", "score"), required=True)
    p.add_argument("--roster", choices=("audited_13", "expanded_37"), required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        {"fit": fit, "score": score}[args.phase](args.roster)
