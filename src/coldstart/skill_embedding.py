"""Empirical-Bayes LLM embeddings on continuous, text-predictable skill dimensions."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

from .conditional_capability import paired_profile


def feature_map(evidence, hist, memberships, center, scale):
    phi = np.tanh((evidence - center) / (2 * scale)).astype(np.float32)
    centered = hist - hist.mean(1, keepdims=True)
    return np.column_stack(
        (
            np.ones(len(phi), np.float32),
            memberships,
            centered,
            phi,
            (memberships[:, :, None] * phi[:, None, :]).reshape(len(phi), -1),
        )
    ).astype(np.float32)


def fit_skill_space(features, hist, memberships, truth):
    center, scale = features.mean(0), features.std(0).clip(0.06)
    phi = feature_map(features, hist, memberships, center, scale)
    relative = (truth - truth.mean(0)).T
    dom = memberships.argmax(1)
    wt = np.sqrt(len(dom) / (len(np.unique(dom)) * np.bincount(dom)[dom])).astype(np.float32)
    xtw = phi.T * wt
    gram = xtw @ phi
    gram.flat[:: len(gram) + 1] += 12.0
    beta = np.linalg.solve(gram, xtw @ relative).astype(np.float32)
    fitted = phi @ beta
    _, _, model_axes = randomized_svd(fitted * np.sqrt(wt[:, None]), 28, n_iter=4, random_state=1733)
    stand = np.tanh((features - center) / (2 * scale))
    _, _, extra = randomized_svd(stand, 16, n_iter=3, random_state=1733)
    scores = stand @ extra.T
    return dict(
        center=center,
        scale=scale,
        beta=beta,
        model_axes=model_axes,
        extra=extra,
        extra_scale=scores.std(0).clip(0.1),
    )


def skill_coordinates(pack, evidence, hist, memberships):
    phi = feature_map(evidence, hist, memberships, pack["center"], pack["scale"])
    relative = phi @ pack["beta"]
    axes = relative @ pack["model_axes"].T / np.sqrt(relative.shape[1])
    stand = np.tanh((evidence - pack["center"]) / (2 * pack["scale"]))
    xc = (stand @ pack["extra"].T) / pack["extra_scale"]
    # The learned task coordinates carry a cohort prior. The remaining coordinates let a
    # newcomer differ from that cohort on a measured skill, even at fixed benchmark.
    return np.column_stack(
        (
            np.full(len(hist), 0.10),
            0.075 * memberships,
            axes,
            0.020 * xc,
            0.023 * (memberships[:, :, None] * xc[:, None]).reshape(len(xc), -1),
        )
    ), relative


def skill_strata(x, h, dom, n):
    cells = np.empty(len(x), dtype=int)
    count = 0
    for j in np.unique(dom):
        ix = np.flatnonzero(dom == j)
        y = h[:, ix].T
        # Similar historical responses control the sampling variance inside each question-skill stratum.
        difficulty = y.mean(1, keepdims=True)
        _, _, axes = randomized_svd(y - difficulty, 8, n_iter=3, random_state=1733)
        behavioral = np.column_stack((difficulty, (y - difficulty) @ axes.T))
        z = x[ix] / x[ix].std(0).clip(0.02)
        z = np.column_stack((z / np.sqrt(z.shape[1]), 0.55 * behavioral / behavioral.std(0).clip(0.07) / 3))
        k = min(n, max(2, len(ix) // 20))
        labels = KMeans(n_clusters=k, n_init=3, random_state=1733).fit_predict(z)
        cells[ix] = count + labels
        count += k
    return cells


def stratified_cells(cells, domains, seed, n):
    rng = np.random.default_rng(seed)
    pop = np.bincount(cells)
    cell_dom = np.array([domains[np.flatnonzero(cells == j)[0]] for j in range(len(pop))])
    source_mass = 0.48 / len(np.unique(domains)) + 0.52 * np.bincount(domains) / len(domains)
    raw = pop**0.55
    mass = raw * source_mass[cell_dom] / np.bincount(cell_dom, weights=raw)[cell_dom]
    order = [rng.permutation(np.flatnonzero(cells == j)) for j in range(len(pop))]
    counts = np.zeros(len(pop), dtype=int)
    selected = []
    for i in range(n):
        j = int(np.argmax(np.where(counts < pop, mass * (i + 1) - counts, -np.inf)))
        selected.append(order[j][counts[j]])
        counts[j] += 1
    return np.asarray(selected)


def fit_skill_profile(
    history, responses, idx, domains, x, conditional, cells, strength, observed, rank_shrink
):
    if not np.isfinite(responses[:, idx]).all():
        return np.vstack(
            [
                fit_skill_profile(
                    history,
                    row[None],
                    idx[np.isfinite(row[idx])],
                    domains,
                    x,
                    conditional,
                    cells,
                    strength,
                    observed,
                    rank_shrink,
                )
                for row in responses
            ]
        )
    h, y = history[:, idx].astype(np.float64), responses[:, idx]
    pop = np.bincount(cells)
    group_domain = np.array([domains[np.flatnonzero(cells == j)[0]] for j in range(len(pop))])
    w, ipw = paired_profile(h, y, cells[idx], pop, group_domain)
    measured = w @ (observed * h + (1 - observed) * conditional[idx].T)
    residual = y - measured
    phi = x[idx]
    # Incorporate a cohort-level prior on the magnitude of the individual skill corrections.
    penalty = np.eye(phi.shape[1]) * strength
    penalty[1 + len(np.unique(domains)) : 1 + len(np.unique(domains)) + 28] *= rank_shrink
    xtw = phi.T * (len(idx) * ipw)
    delta = np.linalg.solve(xtw @ phi + penalty, xtw @ residual.T).T
    return np.column_stack((w, delta))


def skill_scores(z, x, predicted, membership, hist_means, cfg):
    n = hist_means.shape[1]
    mean = membership @ hist_means
    base = mean + cfg["signal"] * (predicted - mean)
    answer = z[:, :n] @ base.T + cfg["residual_scale"] * z[:, n:] @ x.T
    if cfg["global_discount"]:
        answer -= cfg["global_discount"] * (z[:, :n] @ hist_means.mean(0))[:, None]
    return answer
