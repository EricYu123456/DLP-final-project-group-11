"""Model capability vectors conditioned on calibrated disagreements between independent solvers."""

from itertools import combinations

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import softmax

from .switch_calibration import peer_vectors


def option_features(raw, dom, d):
    v, mask = raw["votes"], raw["mask"]
    n, e, k = v.shape
    highest = v.max(2, keepdims=True)
    top = (v >= highest - 1e-6) & (highest > 1 / mask.sum(1)[:, None, None] + 0.001)
    ctx = np.column_stack((raw["quality"], np.eye(d)[dom], raw["mc"]))
    pos = np.tile(np.eye(k, 4), (n, 1, 1)) * raw["mc"][:, None, None]
    per = np.concatenate(
        (
            v.transpose(0, 2, 1),
            top.transpose(0, 2, 1),
            np.broadcast_to(ctx[:, None], (n, k, ctx.shape[1])),
            pos,
        ),
        axis=2,
    ).astype(np.float32)
    return per, top


def fit_judge(raw, dom, keys):
    d = int(dom.max()) + 1
    x, _ = option_features(raw, dom, d)
    labeled = np.flatnonzero((keys >= 0) & raw["mc"])
    rng = np.random.default_rng(9749)
    order = rng.permutation(labeled)
    posterior = np.zeros(raw["mask"].shape, np.float32)

    def fit(ids):
        target = np.arange(4)[None, :] == keys[ids, None]
        learner = CatBoostClassifier(
            iterations=225,
            depth=5,
            learning_rate=0.055,
            l2_leaf_reg=12,
            verbose=False,
            thread_count=4,
            random_seed=9749,
            allow_writing_files=False,
        )
        learner.fit(x[ids, :4].reshape(-1, x.shape[-1]), target.ravel().astype(np.int8))
        return learner

    folds = np.array_split(order, 3)
    for i, idx in enumerate(folds):
        model = fit(np.concatenate([v for j, v in enumerate(folds) if j != i]))
        posterior[idx] = judge_probs(model, {k: v[idx] for k, v in raw.items()}, dom[idx], d)
    model = fit(labeled)
    other = np.setdiff1d(np.arange(len(keys)), labeled)
    posterior[other] = judge_probs(model, {k: v[other] for k, v in raw.items()}, dom[other], d)
    return model, posterior


def judge_probs(model, raw, dom, d):
    x, _ = option_features(raw, dom, d)
    n, k, _ = x.shape
    score = model.predict(x.reshape(n * k, -1), prediction_type="RawFormulaVal", thread_count=3).reshape(n, k)
    mask = raw["mask"]
    calibrated = softmax(np.where(mask, score, -1e4), axis=1)
    vote = raw["votes"].mean(1)
    p = np.where(raw["mc"][:, None], 0.90 * calibrated + 0.10 * vote, vote)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def contexts(raw, p, dom, d):
    v = raw["votes"]
    n, e, k = v.shape
    _, top = option_features(raw, dom, d)
    support = lambda a: np.einsum("nek,nk->ne", a, p)
    prob, agree = support(v), support(top)
    pairs = (
        np.column_stack([np.sum(top[:, i] * top[:, j] * p, axis=1) for i, j in combinations(range(e), 2)])
        if e > 1
        else np.zeros((n, 0))
    )
    order = p[:, :4] * raw["mc"][:, None]
    common = raw["quality"][:, ::4]
    return np.column_stack((prob, agree, pairs, order, common)), agree


def coordinates(pack, raw, p, dom):
    core, agree = contexts(raw, p, dom, pack["domains"])
    c = (core - pack["center"][dom]) / pack["scale"]
    a = (agree - pack["agree"][dom]) / pack["ascal"]
    local = (np.eye(pack["domains"])[dom, :, None] * a[:, None, :]).reshape(len(a), -1)
    return np.column_stack((c, local)).clip(-5, 5).astype(np.float32)


def historical_design(raw, dom, keys):
    judge, prob = fit_judge(raw, dom, keys)
    actual = prob.copy()
    observed = (keys >= 0) & raw["mc"]
    actual[observed] = np.eye(prob.shape[1])[keys[observed]]
    core, agree = contexts(raw, actual, dom, int(dom.max()) + 1)
    avg = lambda v: np.stack([v[dom == j].mean(0) for j in np.unique(dom)])
    center, acenter = avg(core), avg(agree)
    pack = dict(
        judge=judge,
        domains=int(dom.max()) + 1,
        center=center,
        scale=(core - center[dom]).std(0).clip(0.17),
        agree=acenter,
        ascal=(agree - acenter[dom]).std(0).clip(0.17),
    )
    x = coordinates(pack, raw, actual, dom)
    xp = coordinates(pack, raw, prob, dom)
    return pack, x, xp


def references(h, dom, x, xp):
    means = np.stack([h[:, dom == j].mean(1) for j in np.unique(dom)])
    y = h.T - means[dom]
    y -= y.mean(1, keepdims=True)
    coef = np.linalg.solve(x.T @ x + 0.45 * xp.T @ xp + 95 * np.eye(x.shape[1]), x.T @ y + 0.45 * xp.T @ y).T
    return dict(means=means, coef=coef)


def embed(ref, h, y, dom, idx, x, xp, nearest, ridge, seed):
    w, off = peer_vectors(h, y, idx, dom, nearest)
    fold = np.empty(len(idx), int)
    rng = np.random.default_rng(seed + 1417)
    for j in np.unique(dom[idx]):
        ids = rng.permutation(np.flatnonzero(dom[idx] == j))
        fold[ids] = np.arange(len(ids)) % 4
    residual = np.full((len(y), len(idx)), np.nan)
    for k in range(4):
        use = fold == k
        peers, adj = peer_vectors(h, y, idx[~use], dom, nearest)
        residual[:, use] = y[:, idx[use]] - peers @ h[:, idx[use]] - adj[:, dom[idx[use]]]
    coef = w @ ref["coef"]
    for i, v in enumerate(residual):
        ok = np.isfinite(v)
        a, b = x[idx[ok]], xp[idx[ok]]
        coef[i] += np.linalg.solve(
            a.T @ a + 0.45 * b.T @ b + ridge * np.eye(x.shape[1]), (a.T + 0.45 * b.T) @ v[ok]
        )
    return dict(z=coef, source=w @ ref["means"].T + off, w=w)


def score(pf, xx, peer, dom, means, cfg):
    other = pf["w"] @ (peer - means[dom]).T
    return pf["source"][:, dom] + cfg["signal"] * ((1 - cfg["peer"]) * (pf["z"] @ xx.T) + cfg["peer"] * other)
