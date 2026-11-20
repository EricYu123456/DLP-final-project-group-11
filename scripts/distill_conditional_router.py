"""Route easy prompts from text, calibrating agreement with the fitted 120-question teacher for each roster."""

import argparse
from functools import lru_cache
import json
from time import perf_counter

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from panel_text_kernel import score_margin, surface
from route_query import ROOT
from coldstart.conditional_text import lexical


def project(basis, texts):
    samples = surface(texts)
    shape = lexical(texts, [])
    return [np.column_stack((view.transform(samples), shape)) for view in basis]


@lru_cache(maxsize=1)
def load_basis(out):
    return joblib.load(out / "conditional-student" / "basis.joblib")


def training_inputs(out, d, split):
    from panel_capabilities import panel_text

    folder = out / "conditional-student"
    folder.mkdir(exist_ok=True)
    ids = d.prompt_ids[np.r_[split["training"], split["development"]]]
    count = len(split["training"])
    file = folder / "training.npz"
    if file.is_file() and (folder / "basis.joblib").is_file():
        with np.load(file) as data:
            if np.array_equal(data["ids"], ids) and int(data["training"]) == count:
                return [data["words"], data["chars"]]
    texts = panel_text(ids, ROOT / "data/raw")
    train = surface(texts[:count])
    basis = []
    for analyzer, ngrams in (("word", (1, 2)), ("char_wb", (3, 4))):
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngrams,
            sublinear_tf=True,
            min_df=2,
            max_features=18000,
            dtype=np.float32,
        )
        basis.append(
            make_pipeline(vectorizer, TruncatedSVD(96, n_iter=4, random_state=319)).fit(
                train
            )
        )
    views = project(basis, texts)
    joblib.dump(basis, folder / "basis.joblib", compress=2)
    np.savez_compressed(file, ids=ids, training=count, words=views[0], chars=views[1])
    load_basis.cache_clear()
    return views


def estimates(student, views):
    mats = []
    for model, features in zip(student["readouts"], views):
        prediction = model.predict(features)
        mats.append((prediction[:, :-1] + 5 * prediction[:, -1:]).T)
    margin = np.minimum(score_margin(mats[0]), score_margin(mats[1]))
    margin[mats[0].argmax(0) != mats[1].argmax(0)] = -np.inf
    return 0.5 * (mats[0] + mats[1]), margin


def early_scores(student, views):
    scores, margin = estimates(student, views)
    return scores, margin < student["threshold"]


def fit_student(teacher, views, count):
    targets = teacher[:, :count].T
    mean = targets.mean(1, keepdims=True)
    targets = np.column_stack((targets - mean, 0.2 * mean))
    linear = make_pipeline(StandardScaler(), Ridge(alpha=40))
    forest = ExtraTreesRegressor(
        n_estimators=64,
        max_features=0.6,
        min_samples_leaf=10,
        max_depth=16,
        n_jobs=3,
        random_state=319,
    )
    student = dict(
        readouts=[
            model.fit(x[:count], targets) for model, x in zip((linear, forest), views)
        ]
    )
    scores, margin = estimates(student, [x[count:] for x in views])
    disagreement = scores.argmax(0) != teacher[:, count:].argmax(0)
    finite = margin[np.isfinite(margin)]
    if teacher.shape[0] == 1:
        threshold = -np.inf
    else:
        thresholds = (
            np.unique(np.quantile(finite, np.linspace(0, 1, 21))) if len(finite) else []
        )
        threshold = np.inf
        for t in thresholds:
            errors = np.sum((margin >= t) & disagreement)
            upper = beta.ppf(1 - 0.05 / 21, 1 + errors, len(margin) - errors)
            if upper <= 0.005:
                threshold = float(t)
                break
    student["threshold"] = threshold
    accepted = margin >= threshold
    return student, dict(
        calibration_questions=len(margin),
        calibration_early_share=accepted.mean(),
        calibration_changed_share=(accepted & disagreement).mean(),
    )


def teacher_for_pool(runner, choice, y, idx, seed):
    from coldstart.switch_calibration import paired_uncertainty, rate_challengers

    mats = [runner.score(config, y, idx, seed)[0] for config in choice["configs"]]
    result = np.average(mats, axis=0, weights=choice["weights"])
    return rate_challengers(
        result,
        paired_uncertainty(runner.h, y, idx, runner.dt),
        choice.get("allowance", 0),
        specialization=choice.get("specialization", 0),
    )


def train_for_pool(runner, views, choice, y, idx, seed):
    teacher = teacher_for_pool(runner, choice, y, idx, seed)
    return fit_student(teacher, views, len(runner.train))


def run(args):
    from confirmation_models import ConfirmationModels
    from evaluate_capability_router import (
        input_context,
        metrics,
        paired_gain,
        test_rosters,
    )
    from panel_capabilities import panel_text
    from run_experiment import atomic_csv
    from select_confirmation_router import SEEDS

    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        split = dict(f)
    tr, dev = split["training"], split["development"]
    runner = ConfirmationModels(args, d, s, split, np.r_[tr, dev], "development", True)
    names, rosters = test_rosters(d, s, prepared, family)
    choice = json.loads((args.report / "selection.json").read_text(encoding="utf-8"))[
        "conditional"
    ]
    start = perf_counter()
    views = training_inputs(args.out, d, split)
    basis_seconds = perf_counter() - start
    q = np.r_[split["confirmation"], split["previous_test"]]
    views_test = project(
        load_basis(args.out), panel_text(d.prompt_ids[q], ROOT / "data/raw")
    )
    position = {int(v): i for i, v in enumerate(tr)}
    count = int(choice["configs"][0]["probe_count"])
    rows, pairs, costs = [], [], []
    for roster, unordered, truth in rosters:
        order = np.argsort(unordered)
        ids = unordered[order]
        runner.target_names = np.array([names[i] for i in ids])
        routes, gates = [], []
        with np.load(args.out / f"{roster}_conditional_trials.npz") as data:
            if not np.array_equal(data["model_ids"], ids) or not np.array_equal(
                data["prompt_ids"], d.prompt_ids[q]
            ):
                raise ValueError("Teacher and student evaluation pools differ.")
            full = data["scores"]
        for k, seed in enumerate(SEEDS):
            before = perf_counter()
            idx = np.array(
                [position[int(i)] for i in split[f"balanced_{seed}"][:count]]
            )
            y = np.full((len(ids), len(tr)), np.nan, dtype=np.float32)
            y[:, idx] = truth[np.ix_(order, tr[idx])]
            student, calibration = train_for_pool(runner, views, choice, y, idx, seed)
            seconds = perf_counter() - before
            cheap, heavy = early_scores(student, views_test)
            mat = np.where(heavy[None], full[k], cheap)
            routes.append(mat.argmax(0))
            gates.append(heavy)
            costs.append(
                dict(
                    roster=roster,
                    seed=seed,
                    basis_setup_or_load_seconds=basis_seconds,
                    teacher_replay_and_fit_seconds=seconds,
                    **calibration,
                )
            )
            if k == 0:
                pool = joblib.load(args.out / f"{roster}_conditional.joblib")
                if not np.array_equal(pool["probe_ids"], d.prompt_ids[tr[idx]]):
                    raise ValueError("Teacher and student onboarding panels differ.")
                pool.update(method="distilled-conditional", student=student)
                joblib.dump(
                    pool,
                    args.out / f"{roster}_distilled-conditional.joblib",
                    compress=2,
                )
                np.savez_compressed(
                    args.out / f"{roster}_distilled-conditional_scores.npz",
                    scores=mat,
                    prompt_ids=d.prompt_ids[q],
                    large_decoder_used=heavy,
                )
            print("Distilled", roster, seed, round(seconds, 2), calibration, flush=True)
        route, heavy = np.asarray(routes), np.asarray(gates)
        with np.load(args.out / f"choices_75_{roster}.npz") as f:
            controls = {
                k: f[k]
                for k in (
                    "conditional",
                    "previous-8D-refit",
                    "EmbedLLM-known",
                    "EmbedLLM-noisy",
                )
            }
        for stage, ix in (
            ("confirmation", np.arange(len(split["confirmation"]))),
            ("previous-test", np.arange(len(split["confirmation"]), len(q))),
        ):
            labels = truth[np.ix_(order, q[ix])]
            source = d.domains[q[ix]]
            hits = labels[route[:, ix], np.arange(len(ix))]
            rows.append(
                dict(
                    split=stage,
                    roster=roster,
                    method="distilled-conditional",
                    probes=count,
                    full_decoder_share=heavy[:, ix].mean(),
                    teacher_agreement=(
                        route[:, ix] == controls["conditional"][:, ix]
                    ).mean(),
                    **metrics(route[:, ix], labels, source),
                )
            )
            for key, baseline in controls.items():
                ref = labels[baseline[:, ix], np.arange(len(ix))]
                pairs.append(
                    dict(
                        split=stage,
                        roster=roster,
                        reference=key,
                        **paired_gain(hits, ref, source),
                    )
                )
    for name, records in (("results", rows), ("paired", pairs), ("costs", costs)):
        atomic_csv(args.report / f"student_{name}.csv", records)
    print(rows, flush=True)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/routing/capability/confirmation"
    )
    with threadpool_limits(limits=3):
        run(parser.parse_args())
