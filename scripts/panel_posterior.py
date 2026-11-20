"""Estimate conditional ability embeddings while accounting for correlations between LLMs' probe errors."""

import numpy as np
from scipy.spatial.distance import cdist

from panel_capabilities import normalized, template_membership
from panel_rasch import subtype_matrix


def center(k):
    return k - k.mean(0) - k.mean(1, keepdims=True) + k.mean()


def posterior_abilities(y, domains, meta, size, cfg, weights=None):
    """Use the variance of paired differences in each source, including off-diagonal sampling noise."""
    n = len(y)
    all_domains = np.unique(domains)
    weights = np.ones(len(domains)) if weights is None else np.asarray(weights)
    raw = np.stack(
        [np.average(y[:, domains == j], axis=1, weights=weights[domains == j]) for j in all_domains], axis=1
    )
    out = []
    for j in all_domains:
        keep = domains == j
        others = y[:, ~keep]
        z = np.delete(raw, j, axis=1)
        z = (z - z.mean(0)) / z.std(0).clip(0.04)
        reliability = min(1, others.shape[1] / 140)
        behavior = 0.45 * (z @ z.T / z.shape[1]) + 0.55 * center(
            np.exp(-cdist(others, others, "hamming") / 0.15)
        )
        design = np.column_stack((np.ones(n), z))
        penalty = np.eye(design.shape[1]) * 15
        penalty[0, 0] = 0.001
        # Other models help estimate the relationship between the benchmark and general ability.
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ raw[:, j])
        prior = design @ beta
        residual = raw[:, j] - prior
        panel = y[:, keep] - y[:, keep].mean(0)
        w = weights[keep]
        effective = w.sum() ** 2 / (w @ w)
        noise = np.cov(panel, aweights=w) / effective
        independent = np.diag(np.diag(noise))
        noise = (1 - cfg["noise_corr"]) * independent + cfg["noise_corr"] * noise
        family = center(0.67 * meta + 0.33 * size)
        kernel = (1 - cfg["metadata"]) * reliability * center(behavior) + cfg["metadata"] * family
        scale = max(0.0007, np.mean(residual**2) - np.trace(noise) / n)
        kernel = scale * (0.7 * kernel / (np.trace(kernel) / n + 1e-6) + 0.3 * np.eye(n))
        inferred = prior + kernel @ np.linalg.solve(
            kernel + cfg["noise"] * noise + 1e-5 * np.eye(n), residual
        )
        out.append(inferred)
    return np.stack(out, axis=1)


def question_map(x, texts, aux, support=None):
    subtype = subtype_matrix(texts)
    if support is None:
        support = dict(aux_mean=aux.mean(0), aux_std=aux.std(0).clip(0.08))
    a = np.tanh((aux - support["aux_mean"]) / (2 * support["aux_std"]))
    return np.column_stack((0.68 * normalized(x), 0.37 * subtype, 0.65 * normalized(a))), support


def train(x, domain, y, texts, auxiliary, meta, size, cfg, weights=None):
    z = posterior_abilities(y, domain, meta, size, cfg, weights)
    membership = np.eye(len(z.T))[domain]
    residue = (y - y.mean(0)) - (z - z.mean(0)) @ membership.T
    # Denoise candidate contrasts across related models before learning their question dependence.
    gram = center(meta)
    smoothed = gram @ np.linalg.solve(gram + 1.3 * np.eye(len(y)), residue)
    residue = (1 - cfg["metadata"] * 0.6) * residue + cfg["metadata"] * 0.6 * smoothed
    u, s, _ = np.linalg.svd(residue, full_matrices=False)
    rank = min(20, len(s))
    axes = u[:, :rank]
    phi, pack = question_map(x, texts, auxiliary)
    kernel = phi @ phi.T
    kernel = 0.5 * kernel + 0.5 * np.exp(3 * (kernel - 1))
    kernel *= 0.4 + 0.6 * (membership @ membership.T)
    weight = np.ones(len(x)) if weights is None else weights / np.mean(weights)
    coefficients = np.linalg.solve(kernel + cfg["ridge"] * np.diag(1 / weight), (axes.T @ residue).T)
    return dict(
        vectors=np.column_stack((z, axes)),
        coef=coefficients,
        support=phi,
        domains=domain,
        texts=texts,
        **pack,
    )


def design(pack, x, texts, aux, signal, membership=None):
    if membership is None:
        membership = template_membership(pack["texts"], texts, pack["domains"])
    phi, _ = question_map(x, texts, aux, pack)
    k = phi @ pack["support"].T
    k = 0.5 * k + 0.5 * np.exp(3 * (k - 1))
    k *= 0.4 + 0.6 * membership[:, pack["domains"]]
    return np.column_stack((membership, signal * k @ pack["coef"]))
