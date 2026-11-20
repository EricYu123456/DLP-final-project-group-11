"""Estimate when switching away from a measured incumbent gains correct answers."""

import numpy as np
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression

from .conditional_capability import paired_profile


def peer_vectors(h, y, idx, dt, nearest):
    population = np.bincount(dt)
    w, offset = [], []
    for row in y:
        take = idx[np.isfinite(row[idx])]
        obs = h[:, take]
        source = dt[take]
        target = row[take]
        pw, mass = paired_profile(obs, target[None], source, population, np.arange(len(population)))
        match = softmax(-abs(obs - target) @ mass / 0.029)
        weights = (1 - nearest) * pw[0] + nearest * match
        residual = target - weights @ obs
        adjust = (np.bincount(source, weights=residual, minlength=len(population)) + 9 * residual.mean()) / (
            np.bincount(source, minlength=len(population)) + 9
        )
        w.append(weights)
        offset.append(adjust)
    return np.stack(w), np.stack(offset)


def relative_scores(coefs, head, means, dom):
    diff = coefs @ (head - means[dom]).T
    return diff


def paired_uncertainty(h, y, idx, domain):
    """Estimate strengths and uncertainty of each paired difference using exactly the observed panel."""
    weights, adjust = peer_vectors(h, y, idx, domain, 0.40)
    means = np.stack([h[:, domain == j].mean(1) for j in np.unique(domain)])
    mass = 0.45 / len(means) + 0.55 * np.bincount(domain) / len(domain)
    strength = (weights @ means.T + adjust) @ mass
    # Historical controls remove variance caused by the question being easy for all similar models.
    residual = y[:, idx] - weights @ h[:, idx]
    variance = np.zeros((len(y), len(y)))
    for j, m in enumerate(mass):
        errors = residual[:, domain[idx] == j]
        diff = errors[:, None] - errors
        count = np.isfinite(diff).sum(2).clip(1)
        variance += m * m * np.nanvar(diff, axis=2) / count
    return dict(strength=strength, uncertainty=np.sqrt(variance).clip(0, 0.12))


def rate_challengers(scores, profile, allowance, pool=None, specialization=0.0):
    members = np.arange(len(scores)) if pool is None else np.asarray(pool)
    values = scores[members]
    if not (allowance or specialization):
        return values
    incumbent = members[profile["strength"][members].argmax()]
    general = profile["strength"][members] - profile["strength"][incumbent]
    return (
        values
        - specialization * general[:, None]
        + allowance * profile["uncertainty"][members, incumbent, None]
    )


def tail_panel(h, y, balanced, dt, oof, seed, n):
    warm = min(n, 95)
    initial = balanced[:warm]
    if warm == n:
        return initial
    w, off = peer_vectors(h, y, initial, dt, 0.45)
    means = np.stack([h[:, dt == j].mean(1) for j in np.unique(dt)])
    typical = w @ means.T + off
    mass = 0.5 * np.bincount(dt) / len(dt) + 0.5 / len(means)
    anchor = int((typical @ mass).argmax())
    key = relative_scores(w - w[anchor], oof, means, dt) + typical[:, dt] - typical[anchor, dt]
    key[anchor] = -np.inf
    opportunity = key.max(0)
    cells = np.empty(len(dt), int)
    for j in range(len(means)):
        pos = np.flatnonzero(dt == j)
        cuts = np.quantile(opportunity[pos], [0.45, 0.72, 0.88])
        cells[pos] = 4 * j + np.searchsorted(cuts, opportunity[pos])
    rng = np.random.default_rng(seed + 9029)
    size = np.bincount(cells, minlength=4 * len(means))
    mass = (size / size.sum()) * np.tile([0.38, 0.82, 1.45, 2.15], len(means))
    mass = 0.5 * mass / mass.sum() + 0.5 * (
        np.repeat(1 / len(means), len(mass)) * size / np.bincount(dt).repeat(4)
    )
    mass /= mass.sum()
    withheld = np.ones(len(dt), bool)
    withheld[initial] = False
    pools = [rng.permutation(np.flatnonzero((cells == j) & withheld)) for j in range(len(mass))]
    count = np.zeros(len(mass), int)
    chosen = list(initial)
    for i in range(n - warm):
        j = int(np.argmax(np.where(count < [len(p) for p in pools], mass * (i + 1) - count, -np.inf)))
        chosen.append(pools[j][count[j]])
        count[j] += 1
    return np.array(chosen)


def fit_switches(h, y, idx, dom, oof, nearest, prior):
    w, adjust = peer_vectors(h, y, idx, dom, nearest)
    means = np.stack([h[:, dom == j].mean(1) for j in np.unique(dom)])
    mass = 0.5 * np.bincount(dom) / len(dom) + 0.5 / len(means)
    incumbent = int(((w @ means.T + adjust) @ mass).argmax())
    w = w - w[incumbent]
    adjust = adjust - adjust[incumbent]
    relative = relative_scores(w, oof, means, dom)
    witness = w @ h
    cuts = np.array([0, 0.15, 0.4, 0.65, 0.82, 0.94, 1.0])
    n_bins = len(cuts) - 1
    centers = np.zeros((len(y), len(means), n_bins))
    estimates = np.zeros_like(centers)
    uncertainty = np.zeros_like(centers)
    for m in range(len(y)):
        if m == incumbent:
            continue
        observed = y[m, idx] - y[incumbent, idx]
        residual = observed - witness[m, idx]
        for j in range(len(means)):
            sub = dom == j
            slot = np.flatnonzero((dom[idx] == j) & np.isfinite(residual))
            q = relative[m, sub]
            thresholds = np.quantile(q, cuts[1:-1])
            hist_win = witness[m, sub]
            bins = np.searchsorted(thresholds, q)
            observed_bins = np.searchsorted(thresholds, relative[m, idx[slot]])
            src = adjust[m, j]
            raw = np.empty(n_bins)
            count = np.zeros(n_bins)
            for k in range(n_bins):
                take = bins == k
                used = slot[observed_bins == k]
                centers[m, j, k] = (
                    q[take].mean() if take.any() else np.quantile(q, (cuts[k] + cuts[k + 1]) * 0.5)
                )
                pop = hist_win[take].mean() if take.any() else w[m] @ means[j]
                # Actual reference outcomes across the whole stratum reduce variance of the paired survey.
                correction = (residual[used].sum() + prior * src) / (len(used) + prior)
                raw[k] = pop + correction
                count[k] = max(1, take.sum())
                variation = (np.sum((residual[used] - src) ** 2) + 4 * np.nanvar(residual)) / (len(used) + 4)
                uncertainty[m, j, k] = np.sqrt(variation / (len(used) + prior))
            estimates[m, j] = IsotonicRegression(increasing=True).fit_transform(
                np.arange(n_bins), raw, sample_weight=count
            )
    return dict(
        w=w,
        adjust=adjust,
        means=means,
        anchor=incumbent,
        centers=centers,
        advantages=estimates,
        uncertainty=uncertainty,
    )


def switch_scores(profile, readout, dom, cfg):
    means, weights = profile["means"], profile["w"]
    v = relative_scores(weights, readout, means, dom)
    base = weights @ means.T + profile["adjust"]
    result = base[:, dom] + v
    for m in range(len(weights)):
        if m == profile["anchor"]:
            continue
        for j in np.unique(dom):
            chosen = dom == j
            beneficial = profile["advantages"][m, j] + cfg["optimism"] * profile["uncertainty"][m, j]
            measured = np.interp(v[m, chosen], profile["centers"][m, j], beneficial)
            measured = base[m, j] + cfg["signal"] * (measured - base[m, j])
            result[m, chosen] = (1 - cfg["calibration"]) * result[m, chosen] + cfg["calibration"] * measured
    return result
