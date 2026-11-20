"""Measure source abilities using other models as examples and cross-fit across related model groups."""

from itertools import product

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sklearn.utils.extmath import randomized_svd

from panel_capabilities import template_membership
from panel_tabular_skills import fit_skills, learner, skill_parts


def model_coordinates(y, source, domain, meta, size, weights):
    keep = source != domain
    values = y[:, keep]
    labels = source[keep]
    scores = np.column_stack(
        [
            np.average(values[:, labels == j], axis=1, weights=weights[keep][labels == j])
            for j in np.unique(labels)
        ]
    )
    standardized = (scores - scores.mean(0)) / scores.std(0).clip(0.035)
    centered = values - values.mean(0)
    for j in np.unique(labels):
        block = labels == j
        centered[:, block] -= centered[:, block].mean(1, keepdims=True)
    axes, sv, _ = randomized_svd(centered, 9, n_iter=5, random_state=379)
    similarity = 0.7 * meta + 0.3 * size
    eigen, basis = np.linalg.eigh(similarity)
    names = basis[:, -6:] * np.sqrt(eigen[-6:].clip(0))
    features = np.column_stack((scores, standardized.mean(1), axes * sv, names))
    w = weights[keep] / weights[keep].sum()
    differences = np.einsum("ij,j,kj->ik", values, w, 1 - values)
    distance = differences + differences.T
    behavior = np.exp(-distance / 0.17)
    return features.astype(np.float32), 0.68 * behavior + 0.32 * similarity, distance


def grouped_folds(distance, seed):
    clean = np.maximum(distance, 0)
    np.fill_diagonal(clean, 0)
    tree = linkage(squareform(clean, checks=False), method="average")
    groups = fcluster(tree, 22, criterion="maxclust")
    size = np.bincount(groups)
    rng = np.random.default_rng(seed + 5721)
    order = np.argsort(-size - rng.uniform(0, 0.1, len(size)))
    folds = np.zeros(len(groups), dtype=int)
    counts = np.zeros(4)
    for g in order:
        fold = counts.argmin()
        folds[groups == g] = fold
        counts[fold] += size[g]
    return folds


def cross_abilities(y, domains, meta, size, weights, seed):
    weights = np.ones(len(domains)) if weights is None else np.asarray(weights)
    output = []
    reg = learner()
    for j in np.unique(domains):
        use = domains == j
        wt = weights[use] / weights[use].sum()
        target = y[:, use] @ wt
        obs = y[:, use] - y[:, use].mean(0)
        noise = np.cov(obs, aweights=wt) * (wt @ wt)
        x, kernel, distance = model_coordinates(y, domains, j, meta, size, weights)
        folds = grouped_folds(distance, seed + int(j))
        prior = np.empty(len(y))
        for fold in np.unique(folds):
            take = folds == fold
            reg.fit(x[~take], target[~take])
            prior[take] = reg.predict(x[take])
        # Denoise each measured difference using its paired sampling covariance.
        residual = target - prior
        innovation = np.mean((residual - residual.mean()) ** 2)
        signal = max(0.0012, innovation - np.trace(noise) / len(y))
        kernel = kernel - kernel.mean(0) - kernel.mean(1, keepdims=True) + kernel.mean()
        k = signal * (0.52 * np.eye(len(y)) + 0.48 * kernel / kernel.diagonal().mean().clip(0.02))
        s = 0.56 * noise + 0.44 * np.diag(noise.diagonal())
        adjustment = k @ np.linalg.solve(k + s + 0.00004 * np.eye(len(y)), residual)
        output.append(np.stack((prior, adjustment)))
    return np.stack(output, axis=-1)


def fit_one(x, dom, y, text, aux, meta, size, weights, seed):
    pack = fit_skills(x, dom, y, text, aux, meta, size, weights)
    pack["cross_model"] = cross_abilities(y, dom, meta, size, weights, seed)
    return pack


def parts(pack, x, text, aux, estimators=None):
    membership = template_membership(pack["texts"], text, pack["domains"])
    original, conditional = skill_parts(pack, x, text, aux, membership, estimators)
    prior, adjustment = pack["cross_model"] @ membership.T
    return original, conditional, prior, adjustment


def combine(p, params):
    mix, learn, semantic = params
    original, conditional, prior, adjustment = p
    return (1 - mix) * original + mix * (prior + learn * adjustment) + semantic * conditional


def fit_cross_model(x, dom, y, text, aux, meta, size, seed, weights=None):
    import torch

    torch.set_num_threads(3)
    options = list(product((0.25, 0.65, 1.0), (0.4, 0.85, 1.3), (0.35, 0.85, 1.55)))
    rng = np.random.default_rng(seed + 691)
    fold = np.zeros(len(dom), dtype=int)
    for j in np.unique(dom):
        rows = rng.permutation(np.flatnonzero(dom == j))
        fold[rows] = np.arange(len(rows)) % 3
    risks = np.zeros((len(options), len(dom)))
    for k in range(3):
        tr, va = fold != k, fold == k
        fit = fit_one(
            x[tr],
            dom[tr],
            y[:, tr],
            text[tr],
            aux[tr],
            meta,
            size,
            weights[tr] if weights is not None else None,
            seed + k,
        )
        p = parts(fit, x[va], text[va], aux[va])
        center = 0.5 * (p[0] + p[2] + p[3])
        focus = 0.65 * softmax(30 * center, axis=0) + 0.35 * softmax(12 * center, axis=0)
        for i, params in enumerate(options):
            error = combine(p, params) - y[:, va]
            error -= (error * focus).sum(0)
            risks[i, va] = (focus * error**2).sum(0)
    best = risks.mean(1).argmin()
    diff = risks - risks[best]
    prob = softmax(-diff.mean(1) / (diff.std(1) / np.sqrt(len(dom))).clip(0.0002))
    params = (prob @ np.asarray(options)).tolist()
    # The correction's weight is the average product of the two component weights.
    params[1] = float(prob @ np.array([a * b for a, b, _ in options]) / params[0])
    pack = fit_one(x, dom, y, text, aux, meta, size, weights, seed)
    pack["cross_params"] = params
    return pack, dict(model_fold_count=4, question_fold_count=3, mixing=params)


def cross_model_scores(pack, x, text, aux, estimators=None):
    return combine(parts(pack, x, text, aux, estimators), pack["cross_params"])
