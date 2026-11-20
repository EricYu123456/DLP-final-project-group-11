"""Calibrate the expected gain from routing to the best challenger on a shared measured panel."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.extmath import randomized_svd

from .switch_calibration import peer_vectors


QUANTILES = np.array([0, 0.22, 0.45, 0.65, 0.80, 0.90, 0.96, 1.0])


def source_basis(readout, dom):
    center = np.stack([readout[dom == j].mean(0) for j in np.unique(dom)])
    residual = readout - center[dom]
    scale = residual.std(0).clip(0.04)
    _, _, axes = randomized_svd(residual / scale, min(18, readout.shape[1]), random_state=5107, n_iter=4)
    project = axes.T / scale[:, None]
    coords = residual @ project
    project /= coords.std(0).clip(0.18)
    return dict(center=center, project=project)


def design(basis, p, dom):
    q = np.clip((p - basis["center"][dom]) @ basis["project"], -4, 4)
    local = (np.eye(len(basis["center"]))[dom, :, None] * q[:, None, :3]).reshape(len(q), -1)
    return np.column_stack((q, local))


def profile(h, y, idx, dom, p, xx, nearest, ridge):
    w, off = peer_vectors(h, y, idx, dom, nearest)
    means = np.stack([h[:, dom == j].mean(1) for j in np.unique(dom)])
    rates = w @ means.T + off
    residual = y[:, idx] - w @ h[:, idx] - off[:, dom[idx]]
    coeff = []
    for v in residual:
        valid = np.isfinite(v)
        dx = xx[idx[valid]]
        coeff.append(np.linalg.solve(dx.T @ dx + ridge * np.eye(xx.shape[1]), dx.T @ v[valid]))
    return dict(w=w, rates=rates, offset=off, coeff=np.array(coeff))


def relative(pf, center, p, xx, dom):
    return pf["rates"][:, dom] + pf["w"] @ (p - center[dom]).T + pf["coeff"] @ xx.T


def top_advantage(score, anchor):
    rel = score - score[anchor]
    rel[anchor] = -np.inf
    chosen = rel.argmax(0)
    return chosen, rel[chosen, np.arange(rel.shape[1])]


def ranks(adv, dom):
    edges = np.stack([np.quantile(adv[dom == j], QUANTILES[1:-1]) for j in np.unique(dom)])
    return edges, np.sum(adv[:, None] >= edges[dom], axis=1)


def fit_policy(h, y, idx, dom, readout, nearest, ridge, seed):
    basis = source_basis(readout, dom)
    xx = design(basis, readout, dom)
    pf = profile(h, y, idx, dom, readout, xx, nearest, ridge)
    mixture = 0.55 * np.bincount(dom) / len(dom) + 0.45 / len(pf["rates"].T)
    anchor = int((pf["rates"] @ mixture).argmax())
    if len(y) == 1:
        return dict(**pf, basis=basis, anchor=anchor)
    raw = relative(pf, basis["center"], readout, xx, dom)
    _, adv = top_advantage(raw, anchor)
    edges, _ = ranks(adv, dom)
    shape = (len(mixture), len(QUANTILES) - 1)
    hist = np.zeros(shape)
    counts = np.zeros(shape)
    corrections = np.zeros(shape)
    deviations = np.zeros(shape)
    observed = np.zeros(shape)
    rng = np.random.default_rng(seed + 1813)
    fold = np.zeros(len(idx), int)
    for j in np.unique(dom[idx]):
        pos = rng.permutation(np.flatnonzero(dom[idx] == j))
        fold[pos] = np.arange(len(pos)) % 4
    for k in range(4):
        train, hold = idx[fold != k], idx[fold == k]
        measured = profile(h, y, train, dom, readout, xx, nearest, ridge)
        scores = relative(measured, basis["center"], readout, xx, dom)
        chosen, adv = top_advantage(scores, anchor)
        _, group = ranks(adv, dom)
        # Known reference answers are valid control variates even when a query-only decoder
        # selects the competing model. Target outcomes enter only through held-out panel responses.
        teacher = measured["w"] @ h + measured["offset"][:, dom]
        gain = teacher[chosen, np.arange(len(dom))] - teacher[anchor]
        np.add.at(hist, (dom, group), gain)
        np.add.at(counts, (dom, group), 1)
        diff = y[chosen[hold], hold] - y[anchor, hold] - gain[hold]
        valid = np.isfinite(diff)
        key = dom[hold[valid]], group[hold[valid]]
        np.add.at(corrections, key, diff[valid])
        np.add.at(deviations, key, diff[valid] ** 2)
        np.add.at(observed, key, 1)
    expected = hist / counts.clip(1)
    global_rate = corrections.sum() / (observed.sum() + 8)
    source_rate = (corrections.sum(1) + 12 * global_rate) / (observed.sum(1) + 12)
    correct = (corrections + 9 * source_rate[:, None]) / (observed + 9)
    noise = (deviations + 9 * (deviations.sum() + 1) / (observed.sum() + 2)) / (observed + 9)
    uncertainty = np.sqrt(noise / (observed + 9))
    quality = expected + correct
    for j, values in enumerate(quality):
        quality[j] = IsotonicRegression(increasing=True).fit_transform(
            np.arange(len(values)), values, sample_weight=counts[j].clip(1)
        )
    return dict(**pf, basis=basis, anchor=anchor, edges=edges, quality=quality, uncertainty=uncertainty)


def route_policy(pf, p, dom, cfg):
    xx = design(pf["basis"], p, dom)
    raw = relative(pf, pf["basis"]["center"], p, xx, dom)
    a = pf["anchor"]
    if len(raw) == 1:
        return raw
    _, adv = top_advantage(raw, a)
    # Interpolate the measured gain in order to avoid jumps at survey stratum boundaries.
    estimated = np.empty(len(dom))
    for j in np.unique(dom):
        use = dom == j
        cuts = pf["edges"][j]
        mid = 0.5 * (np.r_[cuts[0] - 0.06, cuts] + np.r_[cuts, cuts[-1] + 0.06])
        v = pf["quality"][j] + cfg["allow"] * pf["uncertainty"][j]
        estimated[use] = np.interp(adv[use], mid, v)
    offset = cfg["calibrate"] * (estimated - adv)
    result = raw + offset
    result[a] = raw[a]
    return result
