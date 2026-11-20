"""Distill a latent-type router and defer only queries whose candidate ranking is uncertain."""

import numpy as np
from scipy.special import ndtr

from .latent_skills import route_types


def teacher_config(cfg):
    return dict(cfg, name=cfg["teacher"], algorithm="types", readout="general")


def inputs(x, raw, pack):
    radial = np.exp(4.5 * (x @ pack["landmarks"].T - 1))
    lexical = ((raw - pack["center"]) / pack["scale"]).clip(-4, 4)
    return np.column_stack((np.ones(len(x)), lexical / np.sqrt(lexical.shape[1]), 2.4 * radial))


def approximate(x, raw, peer, dom, pack):
    design = inputs(x, raw, pack)
    predicted = np.empty((len(dom), pack["coef"].shape[-1]), dtype=np.float32)
    for j in np.unique(dom):
        use = dom == j
        predicted[use] = design[use] @ pack["coef"][j]
    width = peer.shape[1]
    probs = predicted[:, width:].clip(0.002, 1.0)
    return np.clip(peer + predicted[:, :width], 0, 1), probs / probs.sum(1, keepdims=True)


def fit_student(x, raw, dom, cheap, teacher, posterior, seed=1177):
    rng = np.random.default_rng(seed)
    landmarks = np.concatenate(
        [
            rng.choice(np.flatnonzero(dom == j), min(np.sum(dom == j), 90), replace=False)
            for j in np.unique(dom)
        ]
    )
    pack = dict(landmarks=x[landmarks].copy(), center=raw.mean(0), scale=raw.std(0).clip(0.035))
    design = inputs(x, raw, pack)
    target = np.column_stack((teacher - cheap, posterior))
    coef, covariance = [], []
    for j in np.unique(dom):
        mask = dom == j
        X, Y = design[mask], target[mask]
        folds = np.empty(len(Y), int)
        folds[rng.permutation(len(Y))] = np.arange(len(Y)) % 4
        fitted = np.zeros_like(Y)

        def fit(a):
            prior = np.eye(X.shape[1]) * 0.63
            prior[0, 0] = 0.00001
            return np.linalg.solve(X[a].T @ X[a] + prior, X[a].T @ Y[a])

        for k in range(4):
            a = folds != k
            fitted[~a] = X[~a] @ fit(a)
        coef.append(fit(np.ones(len(Y), bool)))
        delta = Y - fitted
        cov = np.cov(delta, rowvar=False)
        covariance.append(0.94 * cov + 0.06 * np.diag(cov.diagonal()))
    pack.update(coef=np.stack(coef), covariance=np.stack(covariance))
    return pack


def decision(z, peer, posterior, dom, means, pack, cfg):
    scores = route_types(z, peer, posterior, dom, means, cfg)
    risk = np.zeros(len(dom))
    h, d = means.shape[1], len(means)
    skill = z[:, h + d :].reshape(len(z), d, -1)
    mix = cfg["type_mix"]
    for j in np.unique(dom):
        use = dom == j
        load = cfg["signal"] * np.column_stack(((1 - mix) * z[:, :h], mix * skill[:, j]))
        cov = load @ pack["covariance"][j] @ load.T
        v = cov.diagonal()
        sd = np.sqrt(np.maximum(v[:, None] + v[None, :] - 2 * cov, 1e-9))
        value = scores[:, use]
        best = value.argmax(0)
        cols = np.arange(len(best))
        prob = ndtr((value - value[best, cols]) / sd[:, best])
        prob[best, cols] = 0
        risk[use] = prob.max(0)
    return scores, risk
