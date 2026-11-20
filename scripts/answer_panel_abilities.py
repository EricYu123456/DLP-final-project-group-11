"""Fit 112 model embeddings from paired grades and independent solvers' answers to the same panel."""

from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from coldstart.latent_answers import answer_arrays
from panel_capabilities import template_membership
from panel_tabular_skills import fit_skills, skill_parts


def answer_coordinates(raw):
    v = raw["votes"].transpose(0, 2, 1)
    n, k, e = v.shape
    top = (v == v.max(1, keepdims=True)) & (v > 1 / raw["mask"].sum(1)[:, None, None] + 0.01)
    joint = [top[:, :, i] * top[:, :, j] for i, j in combinations(range(e), 2)]
    order = np.broadcast_to(np.eye(k, 4), (n, k, 4)) * raw["mc"][:, None, None]
    coordinates = np.concatenate((v, top, np.stack(joint, axis=2), order), axis=2)
    return v, coordinates


def answer_prior(raw, weights):
    v = raw["votes"].transpose(0, 2, 1)
    logvote = np.log(v.clip(0.008))
    return softmax(np.where(raw["mask"], logvote @ weights, -1000), axis=1)


def fit_judge(raw, keys, weight):
    n = raw["votes"].shape[1]
    initial = np.ones(n) / n
    ids = np.flatnonzero((keys >= 0) & raw["mc"])
    if not len(ids):
        return initial
    logvote = np.log(raw["votes"][ids].transpose(0, 2, 1).clip(0.008))
    key = np.eye(logvote.shape[1])[keys[ids]]

    def objective(w):
        probs = softmax(np.where(raw["mask"][ids], logvote @ w, -1000), axis=1)
        err = w - initial
        loss = -(np.log(probs.clip(1e-9)) * key).sum(1) @ weight[ids] + 5.5 * (err @ err)
        grad = np.einsum("nk,nke,n->e", probs - key, logvote, weight[ids]) + 11 * err
        return loss, grad

    return minimize(objective, initial, jac=True, bounds=[(0.02, 1.8)] * n, method="L-BFGS-B").x


def contrasts(raw, dom, basis):
    _, coordinates = answer_coordinates(raw)
    centered = (coordinates - basis["centers"][dom, None]) / basis["scale"]
    bins = np.eye(len(basis["centers"]))[dom]
    # Allow affinities with each solver to vary by source, with stronger regularization.
    local = (bins[:, None, :, None] * centered[:, :, : basis["experts"]][:, :, None, :]).reshape(
        len(dom), centered.shape[1], -1
    )
    return np.concatenate((centered, 0.48 * local), axis=2).clip(-5, 5)


def fit_answer_map(raw, dom, y, keys, base, meta, seed, weights=None):
    weights = np.ones(len(dom)) if weights is None else weights / np.mean(weights)
    reliability = fit_judge(raw, keys, weights)
    prior = answer_prior(raw, reliability)
    known = (keys >= 0) & raw["mc"]
    truth = prior.copy()
    truth[known] = np.eye(truth.shape[1])[keys[known]]
    _, coords = answer_coordinates(raw)
    means = np.einsum("nk,nkp->np", truth, coords)
    centers = np.stack(
        [np.average(means[dom == j], axis=0, weights=weights[dom == j]) for j in np.unique(dom)]
    )
    basis = dict(
        centers=centers,
        scale=(means - centers[dom]).std(0).clip(0.15),
        experts=raw["votes"].shape[1],
        reliability=reliability,
    )
    all_x = contrasts(raw, dom, basis)
    # Shared item difficulty cancels from simultaneous paired outcomes.
    target = (y - base[:, dom]).T
    target -= target.mean(1, keepdims=True)
    sim = 0.6 * meta + 0.4 * np.exp(-np.mean(np.abs(y[:, None] - y[None, :]), axis=2) / 0.16)
    eig, vec = np.linalg.eigh((sim + sim.T) * 0.5)
    pool = (vec * (0.60 + 0.40 * eig.clip(0) / (eig.clip(0) + 1.35))) @ vec.T
    for i in range(4):
        mean_x = np.einsum("nk,nkf->nf", truth, all_x)
        fisher = np.einsum("nk,nkf,nkg->fg", truth * weights[:, None], all_x, all_x, optimize=True)
        observed = mean_x.T @ (weights[:, None] * mean_x)
        gram = 0.54 * fisher + 0.46 * observed + 44 * np.eye(len(fisher))
        z = pool @ np.linalg.solve(gram, mean_x.T @ (weights[:, None] * target)).T
        if i == 3:
            break
        pred = np.einsum("nkp,mp->nkm", all_x, z)
        mse = ((pred - target[:, None]) ** 2).mean(2)
        relative_pop = (y.mean(0) - base.mean(0)[dom]).clip(-0.6, 0.6)
        consensus = raw["votes"].mean(1)
        pop_effect = relative_pop[:, None] * (consensus - (consensus * prior).sum(1, keepdims=True))
        inferred = softmax(
            np.where(raw["mask"], np.log(prior.clip(1e-8)) - 16 * mse + 2.9 * pop_effect, -1000), axis=1
        )
        truth[~known] = 0.80 * inferred[~known] + 0.20 * prior[~known]
    basis["z"] = z
    return basis


def components(x, dom, y, text, aux, raw, keys, meta, size, seed, weights):
    pack = fit_skills(x, dom, y, text, aux, meta, size, weights)
    pack["answer_map"] = fit_answer_map(raw, dom, y, keys, pack["z"], meta, seed, weights)
    return pack


def parts(pack, x, text, aux, raw, estimators=None):
    member = template_membership(pack["texts"], text, pack["domains"])
    base, semantic = skill_parts(pack, x, text, aux, member, estimators)
    judge = pack["answer_map"]
    expected = answer_prior(raw, judge["reliability"])
    basis = contrasts(raw, member.argmax(1), judge)
    query = np.einsum("nk,nkf->nf", expected, basis)
    return base, semantic, judge["z"] @ query.T


def answer_scores(pack, x, text, aux, raw, estimators=None):
    base, semantic, answers = parts(pack, x, text, aux, raw, estimators)
    mix, scale = pack["signals"]
    return base + mix * semantic + scale * answers


def fit_answers(x, dom, y, text, aux, evidence, keys, meta, size, seed, weights=None):
    import torch

    torch.set_num_threads(3)
    raw = answer_arrays(evidence)
    folds = np.empty(len(dom), int)
    rng = np.random.default_rng(seed + 2375)
    for j in np.unique(dom):
        rows = rng.permutation(np.flatnonzero(dom == j))
        folds[rows] = np.arange(len(rows)) % 3
    settings = list(product((0.45, 1.0, 1.6), (0.0, 0.55, 1.15, 2.1)))
    loss = np.zeros((len(settings), len(dom)))
    for k in range(3):
        a, b = folds != k, folds == k
        part = lambda ix: {key: v[ix] for key, v in raw.items()}
        pack = components(
            x[a],
            dom[a],
            y[:, a],
            text[a],
            aux[a],
            part(a),
            keys[a],
            meta,
            size,
            seed + k,
            weights[a] if weights is not None else None,
        )
        base, semantic, answer = parts(pack, x[b], text[b], aux[b], part(b))
        mass = 0.7 * softmax(27 * base, axis=0) + 0.3 * softmax(12 * base, axis=0)
        for i, (u, v) in enumerate(settings):
            err = base + u * semantic + v * answer - y[:, b]
            err -= (mass * err).sum(0)
            loss[i, b] = (mass * err**2).sum(0)
    diff = loss - loss[loss.mean(1).argmin()]
    prob = softmax(-diff.mean(1) / (diff.std(1) / np.sqrt(len(dom))).clip(0.00020))
    signals = prob @ np.asarray(settings)
    pack = components(x, dom, y, text, aux, raw, keys, meta, size, seed, weights)
    pack["signals"] = signals
    return pack, dict(
        signals=signals.tolist(),
        verified_panel_keys=int((keys >= 0).sum()),
        answer_dimensions=pack["answer_map"]["z"].shape[1],
    )
