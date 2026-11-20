"""Distill the historical nominal decoder across model embeddings for inexpensive raw-text routing."""

import argparse
from functools import lru_cache
from time import perf_counter

import joblib
import numpy as np
import torch
from evaluate_capability_router import input_context, metrics, test_rosters
from panel_capabilities import panel_text
from panel_text_kernel import score_margin, surface
from replicate_nominal_profiles import DATA, FOLDER, OUT, REPORT, ROOT, SEEDS, package
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv, atomic_json
from scipy.spatial.distance import cdist
from scipy.stats import beta
from serve_confirmation_router import readout_file
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_conditional_router import proxy_data

from coldstart.conditional_text import lexical, text_features
from coldstart.latent_answers import answer_arrays
from coldstart.nominal_response import outputs, query, scores

METHODS = ("nominal-word", "nominal-word-gated")


def features(views, texts):
    samples = surface(texts)
    return np.column_stack([lexical(texts, []), *(v.transform(samples) for v in views)])


def latent_basis(history, domains, global_rank, local_rank):
    rng = np.random.default_rng(6241)
    fractions = rng.uniform(0.2, 0.8, (len(history), 1))
    interpolated = fractions * history + (1 - fractions) * history[rng.permutation(len(history))]
    anchors = np.vstack((history, interpolated)).astype(np.float32)
    spaces = []
    for source in range(domains):
        first = domains + global_rank + local_rank * source
        cols = np.r_[source, domains : domains + global_rank, first : first + local_rank]
        scale = history[:, cols].std(0).clip(0.12)
        positions = anchors[:, cols] / scale
        squared = cdist(positions, positions, "sqeuclidean")
        width = np.median(squared[squared > 0])
        gram = np.exp(-squared / width)
        inverse = np.linalg.solve(gram + 0.015 * np.eye(len(gram)), np.eye(len(gram)))
        spaces.append({"columns": cols, "scale": scale, "positions": positions, "width": width, "inverse": inverse})
    return anchors, spaces


def anchor_weights(embedding, spaces):
    weights = []
    for space in spaces:
        x = embedding[:, space["columns"]] / space["scale"]
        k = np.exp(-cdist(x, space["positions"], "sqeuclidean") / space["width"])
        w = k @ space["inverse"]
        weights.append(w + (1 - w.sum(1, keepdims=True)) / w.shape[1])
    return weights


@lru_cache(maxsize=1)
def load_student(out):
    return joblib.load(out / "nominal-replication/word_student.joblib")


def item_codes(student, texts):
    x = features(student["views"], texts)
    return 0.55 * student["trees"].predict(x) + 0.45 * student["ridge"].predict(x)


def read_scores(pool, student, codes):
    weights = pool["word_weights"]
    count = weights[0].shape[1]
    _, source_logits, peer = np.split(codes, [count, count + len(weights)], axis=1)
    domain = source_logits.argmax(1)
    nominal = np.empty((len(pool["model_names"]), len(codes)))
    for source, w in enumerate(weights):
        ix = domain == source
        nominal[:, ix] = w @ codes[ix, :count].T
    return scores(pool["profiles"][0], nominal, peer, domain, student["means"], pool["configs"][0])


def calibration_data(out):
    return joblib.load(out / "nominal-replication/word_calibration.joblib")


def attach(pool, student, cal, group_ids):
    profile = pool["profiles"][0]
    pool["word_weights"] = anchor_weights(profile["z"], student["spaces"])
    use = ~np.isin(cal["groups"], group_ids)
    q = tuple(x[use] for x in cal["query"])
    domain = cal["domain"][use]
    target = scores(
        profile,
        outputs(cal["net"], profile["z"], q, domain),
        cal["peer"][use],
        domain,
        student["means"],
        pool["configs"][0],
    )
    cheap = read_scores(pool, student, cal["codes"][use])
    margin = score_margin(cheap)
    threshold = np.inf
    if len(cheap) == 1:
        threshold = -np.inf
    else:
        wrong = cheap.argmax(0) != target.argmax(0)
        for t in np.unique(np.quantile(margin, np.linspace(0, 1, 21))):
            errors = int(np.sum(wrong & (margin >= t)))
            if beta.ppf(1 - 0.05 / 21, errors + 1, len(wrong) - errors) <= 0.015:
                threshold = float(t)
                break
    pool["word_threshold"] = threshold


def train():
    before = perf_counter()
    d, *_ = input_context(ROOT / "artifacts/main_v2")
    with np.load(FOLDER / "study.npz") as study:
        tr = study["train"]
    shared = joblib.load(FOLDER / "fit.joblib")
    net, cfg = shared["net"], shared["config"]
    group = d.group_ids[tr]
    calibration = np.isin(group, np.random.default_rng(6241).permutation(np.unique(group))[::5])
    ids = d.prompt_ids[tr]
    texts = panel_text(ids, DATA / "raw")
    h = joblib.load(readout_file(OUT, cfg))["head"]
    peer, member = h["readout"].predict(d.embeddings[tr], text_features(h["pack"], d.embeddings[tr], texts))
    domain = member.argmax(1)
    raw = answer_arrays(proxy_data(OUT, ids, cfg["proxies"].split(), "development"))
    q = query(net, raw, domain)
    anchors, spaces = latent_basis(shared["history"], net.domains, net.global_rank, net.local_rank)
    targets = np.column_stack(
        (outputs(net, anchors, q, domain).T, np.eye(net.domains)[domain], peer)
    ).astype(np.float32)
    views = []
    for analyzer, ngram in (("word", (1, 2)), ("char_wb", (3, 4))):
        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            max_features=16000,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        views.append(
            make_pipeline(vec, TruncatedSVD(80, n_iter=4, random_state=6241)).fit(
                surface(texts[~calibration])
            )
        )
    x = features(views, texts)
    tree = ExtraTreesRegressor(
        n_estimators=72, max_depth=14, min_samples_leaf=9, max_features=0.65, n_jobs=3, random_state=6241
    )
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=55))
    student = {
        "views": views,
        "trees": tree.fit(x[~calibration], targets[~calibration]),
        "ridge": ridge.fit(x[~calibration], targets[~calibration]),
        "spaces": spaces,
        "means": shared["means"],
    }
    joblib.dump(student, FOLDER / "word_student.joblib", compress=2)
    cal = {
        "groups": group[calibration],
        "domain": domain[calibration],
        "peer": peer[calibration],
        "query": tuple(v[calibration] for v in q),
        "net": net,
        "codes": item_codes(student, texts[calibration]),
    }
    joblib.dump(cal, FOLDER / "word_calibration.joblib", compress=2)
    atomic_json(
        REPORT / "nominal_word_training.json",
        {
            "shared_seconds": perf_counter() - before,
            "training_groups": len(np.unique(group[~calibration])),
            "calibration_groups": len(np.unique(group[calibration])),
            "anchors": len(anchors),
            "student_bytes": (FOLDER / "word_student.joblib").stat().st_size,
        },
    )


def evaluate():
    d, split, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    names, rosters = test_rosters(d, split, prepared, family)
    student = load_student(OUT)
    shared, cal = joblib.load(FOLDER / "fit.joblib"), calibration_data(OUT)
    with np.load(FOLDER / "routes.npz") as f, np.load(FOLDER / "study.npz") as st:
        old = dict(f)
        tr, panels, q = st["train"], st["panels"], old["query"]
        stages = {
            "combined-withheld": np.concatenate([st[k] for k in ("novel", "active", "coverage")]),
            "confirmation": st["confirmation"],
        }
    codes = item_codes(student, panel_text(d.prompt_ids[q], DATA / "raw"))
    positions = {int(v): i for i, v in enumerate(tr)}
    rows, pairs, costs = [], [], []
    for roster, unordered, truth in rosters:
        order = unordered.argsort()
        mids = unordered[order]
        choices, gates = [], []
        for seed, panel in zip(SEEDS, panels):
            idx = np.array([positions[int(row)] for row in panel])
            y = np.full((len(mids), len(tr)), np.nan, np.float32)
            y[:, idx] = truth[np.ix_(order, panel)]
            pool = package(shared, y, idx, [names[i] for i in mids], d.prompt_ids[panel], seed)
            begin = perf_counter()
            attach(pool, student, cal, d.group_ids[panel])
            costs.append(
                {
                    "roster": roster,
                    "seed": seed,
                    "profile_projection_and_calibration_seconds": perf_counter() - begin,
                }
            )
            matrix = read_scores(pool, student, codes)
            late = score_margin(matrix) < pool["word_threshold"]
            choices.append(matrix.argmax(0))
            gates.append(late)
            if seed == SEEDS[0]:
                with np.load(OUT / f"{roster}_nominal-200_scores.npz") as f:
                    if not np.array_equal(f["prompt_ids"], d.prompt_ids[q]):
                        raise ValueError("Full teacher and word student evaluated different prompts.")
                    teacher_scores = f["scores"]
                for method in METHODS:
                    pool["method"] = method
                    joblib.dump(pool, OUT / f"{roster}_{method}.joblib", compress=2)
                    np.savez_compressed(
                        OUT / f"{roster}_{method}_scores.npz",
                        prompt_ids=d.prompt_ids[q],
                        scores=matrix if method == METHODS[0] else np.where(late, teacher_scores, matrix),
                    )
        choices, gate, teacher = np.stack(choices), np.stack(gates), old[roster]
        for stage, queries in stages.items():
            ix = np.flatnonzero(np.isin(q, queries) & np.isfinite(truth[np.ix_(order, q)]).all(0))
            actual = truth[np.ix_(order, q[ix])]
            dom = d.domains[q[ix]]
            for method, route, heavy in (
                (METHODS[0], choices, np.zeros_like(gate)),
                (METHODS[1], np.where(gate, teacher, choices), gate),
                ("nominal-200", teacher, np.ones_like(gate)),
            ):
                v = route[:, ix]
                rows.append(
                    dict(
                        roster=roster,
                        split=stage,
                        method=method,
                        probes=200,
                        questions=len(ix),
                        full_decoder_share=heavy[:, ix].mean(),
                        teacher_agreement=(v == teacher[:, ix]).mean(),
                        **metrics(v, actual, dom),
                    )
                )
                if method != "nominal-200":
                    delta = actual[v, np.arange(len(ix))] - actual[teacher[:, ix], np.arange(len(ix))]
                    pairs.append(
                        dict(
                            roster=roster,
                            split=stage,
                            method=method,
                            gain=delta.mean(),
                            wins=int((delta.mean(1) > 0).sum()),
                            **two_way_interval(delta, dom, d.group_ids[q[ix]]),
                        )
                    )
    atomic_csv(REPORT / "nominal_word_results.csv", rows)
    atomic_csv(REPORT / "nominal_word_paired.csv", pairs)
    atomic_csv(REPORT / "nominal_word_costs.csv", costs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("train", "evaluate"), required=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        {"train": train, "evaluate": evaluate}[p.parse_args().phase]()
