"""Learn when two small auxiliary models can reproduce the full routing decision for a newly fitted roster."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from coldstart.conditional_text import consensus_features, text_features
from confirmation_models import ConfirmationModels
from distill_conditional_router import (
    load_basis,
    project,
    teacher_for_pool,
    training_inputs,
)
from evaluate_capability_router import (
    ROOT,
    input_context,
    metrics,
    paired_gain,
    test_rosters,
)
from panel_capabilities import panel_text
from panel_text_kernel import score_margin
from run_experiment import atomic_csv
from select_confirmation_router import SEEDS
from train_conditional_router import proxy_data

SMALL = ("smol", "math")
BUDGETS = {"small-helper": 0.005, "small-helper-economical": 0.015}
SEMANTIC_BUDGETS = {"semantic-conditional": 0.005, "semantic-economical": 0.015}


def inputs(views, evidence):
    return np.column_stack((*views, consensus_features(evidence))).astype(np.float32)


@lru_cache(maxsize=1)
def semantic_head(out):
    return joblib.load(out / "text/refit75/readout.joblib")


def semantic_inputs(out, vectors, texts, views):
    head = semantic_head(out)
    signals = text_features(head["pack"], vectors, texts)
    peer, member = head["readout"].predict(vectors, signals)
    return np.column_stack((*views, signals, member, peer)).astype(np.float32)


def training_data(args, d, split, semantic=False):
    q = np.r_[split["training"], split["development"]]
    views = training_inputs(args.out, d, split)
    if semantic:
        return semantic_inputs(
            args.out,
            d.embeddings[q],
            panel_text(d.prompt_ids[q], ROOT / "data/raw"),
            views,
        )
    return inputs(views, proxy_data(args.out, d.prompt_ids[q], SMALL, "development"))


def stage_one(pack, x):
    raw = pack["regression"].predict(x)
    reg = (raw[:, :-1] + 5 * raw[:, -1:]).T
    votes = np.zeros_like(reg)
    if pack["classifier"] is None:
        votes[pack["classes"][0]] = 1
    else:
        votes[pack["classes"]] = pack["classifier"].predict_proba(x, thread_count=3).T
    scores = reg + 0.065 * (votes - votes.mean(0))
    chosen = scores.argmax(0)
    ordered = np.arange(len(x))
    # Surface cues, small-model agreement, and the two independent rankings predict fidelity.
    cues = np.column_stack(
        (
            votes[chosen, ordered],
            score_margin(scores),
            score_margin(reg),
            score_margin(votes),
            reg[chosen, ordered],
            chosen == votes.argmax(0),
            np.eye(len(reg))[chosen],
            x[:, :20],
            x[:, 96:107],
            x[:, -27:],
        )
    )
    return scores, cues.astype(np.float32)


def predicted_agreement(pack, cues):
    return (
        np.full(len(cues), pack["constant"])
        if pack["risk"] is None
        else pack["risk"].predict_proba(cues, thread_count=3)[:, 1]
    )


def fit_student(teacher, x, count, groups, semantic=False):
    truth = teacher.argmax(0)
    target = teacher[:, :count].T
    mean = target.mean(1, keepdims=True)
    reg = ExtraTreesRegressor(
        n_estimators=96,
        max_features=0.6,
        min_samples_leaf=7,
        max_depth=17,
        n_jobs=3,
        random_state=1061,
    )
    reg.fit(x[:count], np.column_stack((target - mean, 0.2 * mean)))
    classes = np.unique(truth[:count])
    clf = None
    if len(classes) > 1:
        clf = CatBoostClassifier(
            iterations=265,
            depth=5,
            learning_rate=0.08,
            l2_leaf_reg=7,
            thread_count=3,
            loss_function="MultiClass",
            allow_writing_files=False,
            verbose=False,
            random_seed=1061,
        )
        clf.fit(x[:count], truth[:count])
    pack = dict(regression=reg, classifier=clf, classes=classes)
    scores, cues = stage_one(pack, x[count:])
    ok = scores.argmax(0) == truth[count:]
    rng = np.random.default_rng(6738)
    keys = rng.permutation(np.unique(groups))
    fit = np.isin(groups, keys[::2])
    risk = None
    if len(np.unique(ok[fit])) > 1:
        risk = CatBoostClassifier(
            iterations=150,
            depth=3,
            learning_rate=0.055,
            l2_leaf_reg=10,
            thread_count=3,
            allow_writing_files=False,
            verbose=False,
            random_seed=1061,
        )
        risk.fit(cues[fit], ok[fit].astype(int))
    pack.update(risk=risk, constant=float(ok[fit].mean()))
    certainty = predicted_agreement(pack, cues[~fit])
    correct = ok[~fit]
    thresholds = {}
    diagnostics = {}
    budgets = SEMANTIC_BUDGETS if semantic else BUDGETS
    for name, budget in budgets.items():
        threshold = np.inf
        for t in np.unique(np.quantile(certainty, np.linspace(0, 1, 21))):
            mistakes = (~correct & (certainty >= t)).sum()
            bound = beta.ppf(1 - 0.05 / 21, 1 + mistakes, len(correct) - mistakes)
            if bound <= budget:
                threshold = float(t)
                break
        thresholds[name] = threshold
        chosen = certainty >= threshold
        diagnostics[name] = dict(
            calibration_queries=len(correct),
            early_share=float(chosen.mean()),
            changed_share=float((chosen & ~correct).mean()),
        )
    pack["thresholds"] = thresholds
    return pack, diagnostics


def decision(pack, x, name):
    scores, cues = stage_one(pack, x)
    return scores, predicted_agreement(pack, cues) < pack["thresholds"][name]


def run(args):
    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    tr, dev = split["training"], split["development"]
    q = np.r_[split["confirmation"], split["previous_test"]]
    budgets = SEMANTIC_BUDGETS if args.semantic else BUDGETS
    all_x = training_data(args, d, split, args.semantic)
    texts = panel_text(d.prompt_ids[q], ROOT / "data/raw")
    views = project(load_basis(args.out), texts)
    x = (
        semantic_inputs(args.out, d.embeddings[q], texts, views)
        if args.semantic
        else inputs(views, proxy_data(args.out, d.prompt_ids[q], SMALL, "evaluation"))
    )
    teacher = ConfirmationModels(args, d, s, split, np.r_[tr, dev], "development", True)
    fixed = json.loads((args.report / "selection.json").read_text(encoding="utf-8"))[
        "conditional"
    ]
    names, rosters = test_rosters(d, s, prepared, family)
    pos = {int(v): i for i, v in enumerate(tr)}
    helper_count = np.full(len(q), 6)
    for key in ("moe-gated", "bagel-gated"):
        part = proxy_data(args.out, d.prompt_ids[q], [key], "evaluation")[0][0]
        helper_count += part[:, 0, 6].astype(int)
    rows, pairs, costs = [], [], []
    for roster, unordered, all_y in rosters:
        order = unordered.argsort()
        ids = unordered[order]
        teacher.target_names = np.array([names[i] for i in ids])
        with np.load(args.out / f"{roster}_conditional_trials.npz") as f:
            if not np.array_equal(f["model_ids"], ids) or not np.array_equal(
                f["prompt_ids"], d.prompt_ids[q]
            ):
                raise ValueError(
                    "Full teacher and small helper evaluation pools differ."
                )
            full = f["scores"]
        routes = {k: [] for k in budgets}
        gates = {k: [] for k in budgets}
        for j, seed in enumerate(SEEDS):
            before = perf_counter()
            idx = np.array([pos[int(v)] for v in split[f"balanced_{seed}"][:120]])
            y = np.full((len(ids), len(tr)), np.nan, np.float32)
            y[:, idx] = all_y[np.ix_(order, tr[idx])]
            target = teacher_for_pool(teacher, fixed, y, idx, seed)
            pack, cal = fit_student(
                target, all_x, len(tr), d.group_ids[dev], args.semantic
            )
            seconds = perf_counter() - before
            for name in budgets:
                cheap, heavy = decision(pack, x, name)
                mat = np.where(heavy[None], full[j], cheap)
                routes[name].append(mat.argmax(0))
                gates[name].append(heavy)
                costs.append(
                    dict(
                        roster=roster,
                        seed=seed,
                        method=name,
                        fit_and_teacher_seconds=seconds,
                        **cal[name],
                    )
                )
                if j == 0:
                    pool = joblib.load(args.out / f"{roster}_conditional.joblib")
                    pool.update(method=name, helper_student=pack)
                    joblib.dump(pool, args.out / f"{roster}_{name}.joblib", compress=2)
                    np.savez_compressed(
                        args.out / f"{roster}_{name}_scores.npz",
                        scores=mat.astype(np.float32),
                        prompt_ids=d.prompt_ids[q],
                        large_decoder_used=heavy,
                    )
            print("Calibrated small helpers", roster, seed, cal, flush=True)
        with np.load(args.out / f"choices_75_{roster}.npz") as f:
            refs = {
                k: f[k]
                for k in (
                    "conditional",
                    "previous-8D-refit",
                    "EmbedLLM-known",
                    "EmbedLLM-noisy",
                )
            }
        cut = len(split["confirmation"])
        for stage, ix in (
            ("confirmation", np.arange(cut)),
            ("previous-test", np.arange(cut, len(q))),
        ):
            labels = all_y[np.ix_(order, q[ix])]
            dom = d.domains[q[ix]]
            for name in budgets:
                v, heavy = np.array(routes[name])[:, ix], np.array(gates[name])[:, ix]
                hits = labels[v, np.arange(len(ix))]
                rows.append(
                    dict(
                        split=stage,
                        roster=roster,
                        method=name,
                        probes=120,
                        full_decoder_share=heavy.mean(),
                        mean_helpers=(
                            np.where(heavy, helper_count[ix], 0 if args.semantic else 2)
                        ).mean(),
                        teacher_agreement=(v == refs["conditional"][:, ix]).mean(),
                        **metrics(v, labels, dom),
                    )
                )
                for key, control in refs.items():
                    reference = labels[control[:, ix], np.arange(len(ix))]
                    pairs.append(
                        dict(
                            split=stage,
                            roster=roster,
                            method=name,
                            reference=key,
                            **paired_gain(hits, reference, dom),
                        )
                    )
    for key, items in (("costs", costs), ("results", rows), ("paired", pairs)):
        prefix = "semantic" if args.semantic else "small_helper"
        atomic_csv(args.report / f"{prefix}_{key}.csv", items)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation"
    )
    p.add_argument(
        "--report", type=Path, default=ROOT / "reports/routing/capability/confirmation"
    )
    with threadpool_limits(limits=3):
        run(p.parse_args())
