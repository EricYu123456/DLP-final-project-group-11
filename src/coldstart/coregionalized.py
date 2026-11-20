"""A vector-valued Gaussian-process readout with graded and descriptive model embeddings."""

import numpy as np
from scipy.linalg import eigh
from scipy.special import softmax

from .model_covariates import architecture_kernel, model_descriptions


def normalize(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-6)).astype(np.float32)


def features(x, features, center, scale):
    return normalize(x), normalize(np.tanh((features - center) / scale / 1.7))


def kernel(a, b):
    sem = (a[0] @ b[0].T).clip(-1, 1)
    aux = (a[1] @ b[1].T).clip(-1, 1)
    return 0.28 * sem + 0.25 * np.exp(4 * (sem - 1)) + 0.18 * aux + 0.29 * np.exp(2.7 * (aux - 1))


def descriptive(names, family, architectures):
    meta, size = model_descriptions(names, family)
    k = 0.70 * meta + 0.30 * size
    if architectures.is_file():
        k = 0.62 * k + 0.38 * architecture_kernel(names, architectures)
    return k


def fit_modes(h, meta, values, vectors):
    mean = h.mean(1)
    y = h - mean[:, None]
    y -= y.mean(0)
    covariance = np.cov(y)
    covariance /= np.diag(covariance).mean().clip(0.01)
    model_kernel = 0.66 * covariance + 0.34 * meta
    model_kernel -= model_kernel.mean(0) + model_kernel.mean(1, keepdims=True) - model_kernel.mean()
    _, v = eigh(model_kernel, check_finite=False)
    v = v.astype(np.float32)
    projected = vectors.T @ (y.T @ v)
    # Generalized cross-validation on historical responses selects how much each model
    # direction can actually be predicted from text, including rare specialization directions.
    options = np.array([0.4, 1.5, 5, 15, 45, 150, 650, 5000], dtype=np.float32)
    unused = options[None, :] / (values[:, None] + options[None, :])
    risks = (unused.T**2 @ projected**2) / len(h.T) / unused.mean(0)[:, None] ** 2
    selected = options[risks.argmin(0)]
    coeff = (vectors @ (projected / (values[:, None] + selected))) @ v.T
    return mean, coeff.astype(np.float32), selected


def fit_shared(x, raw, dom, histories, metadata, progress=None):
    models = [dict(groups=[], domains=dom) for _ in histories]
    for j in np.unique(dom):
        idx = np.flatnonzero(dom == j)
        center, scale = raw[idx].mean(0), raw[idx].std(0).clip(0.075)
        support = features(x[idx], raw[idx], center, scale)
        gram = kernel(support, support)
        mean = gram.mean(0)
        grand = float(mean.mean())
        gram -= mean[None, :] + mean[:, None] - grand
        values, vectors = eigh(gram, overwrite_a=True, driver="evd", check_finite=False)
        values = values.clip(0)
        for model, h, k in zip(models, histories, metadata):
            ability, coef, penalty = fit_modes(h[:, idx], k, values, vectors)
            model["groups"].append(
                dict(
                    idx=idx,
                    center=center,
                    scale=scale,
                    support=support,
                    kmean=mean,
                    grand=grand,
                    ability=ability,
                    coef=coef,
                    penalty=penalty,
                )
            )
        if progress:
            progress(j, len(idx))
    return models


def query(model, x, raw, dom):
    h = len(model["groups"][0]["ability"])
    predicted = np.zeros((h, len(x)), np.float32)
    cross = []
    for j, group in enumerate(model["groups"]):
        use = np.flatnonzero(dom == j)
        if len(use):
            k = kernel(features(x[use], raw[use], group["center"], group["scale"]), group["support"])
            k -= k.mean(1, keepdims=True) + group["kmean"] - group["grand"]
            predicted[:, use] = (k @ group["coef"]).T
        else:
            k = np.zeros((0, len(group["idx"])), np.float32)
        cross.append((use, group["idx"], k))
    return predicted, cross


def panel_kernel(model, idx):
    out = np.zeros((len(idx), len(idx)))
    for j, group in enumerate(model["groups"]):
        keep = np.flatnonzero(model["domains"][idx] == j)
        inside = np.searchsorted(group["idx"], idx[keep])
        supports = tuple(v[inside] for v in group["support"])
        mean = group["kmean"][inside]
        out[np.ix_(keep, keep)] = kernel(supports, supports) - mean[:, None] - mean + group["grand"]
    return out


def panel_cross(cross, domains, idx, n):
    out = np.zeros((n, len(idx)), np.float32)
    for j, (use, bank, k) in enumerate(cross):
        take = np.flatnonzero(domains[idx] == j)
        out[np.ix_(use, take)] = k[:, np.searchsorted(bank, idx[take])]
    return out


def model_weights(h, y, idx, dom, meta, mix):
    n = len(h)
    out = []
    d = int(dom.max()) + 1
    for i, row in enumerate(y):
        keep = idx[np.isfinite(row[idx])]
        ds = dom[keep]
        wt = 0.50 / len(ds) + 0.5 / (d * np.bincount(ds, minlength=d)[ds])
        z = np.vstack((h[:, keep], row[keep]))
        p = z @ wt
        joint = (z * wt) @ z.T
        distance = p[:, None] + p - 2 * joint
        centered = z - z[:n].mean(0)
        cov = (centered * wt) @ centered.T
        diag = np.sqrt(np.diag(cov).clip(0.01))
        behavior = 0.55 * np.exp(-distance / 0.14) + 0.45 * cov / diag[:, None] / diag
        new = np.r_[np.arange(n), n + i]
        gram = (1 - mix) * behavior + mix * meta[np.ix_(new, new)]
        avg = gram[:n].mean(0)
        gram -= avg[:, None] + avg[None, :] - avg[:n].mean()
        w = np.linalg.solve(gram[:n, :n] + 0.13 * np.eye(n), gram[n, :n])
        w += (1 - w.sum()) / n
        near = softmax(-distance[-1, :n] / 0.036 + 1.8 * mix * meta[n + i, :n])
        out.append(0.82 * w + 0.18 * near)
    return np.asarray(out)


def embed(model, h, y, idx, dom, meta, mix, seed):
    w = model_weights(h, y, idx, dom, meta, mix)
    error = np.full((len(y), len(idx)), np.nan)
    rng = np.random.default_rng(seed + 9221)
    fold = np.empty(len(idx), int)
    for j in np.unique(dom[idx]):
        take = rng.permutation(np.flatnonzero(dom[idx] == j))
        fold[take] = np.arange(len(take)) % 4
    for k in range(4):
        valid = fold == k
        part = model_weights(h, y, idx[~valid], dom, meta, mix)
        error[:, valid] = y[:, idx[valid]] - part @ h[:, idx[valid]]
    means = np.stack([g["ability"] for g in model["groups"]], axis=1)
    source = []
    average = np.nanmean(error, axis=1)
    for j in range(means.shape[1]):
        v = error[:, dom[idx] == j]
        source.append((np.nansum(v, axis=1) + 8 * average) / (np.isfinite(v).sum(1) + 8))
    off = np.stack(source, axis=1)
    err = error - off[:, dom[idx]]
    kp = panel_kernel(model, idx)
    coeff = np.zeros_like(err)
    for m, row in enumerate(err):
        finite = np.isfinite(row)
        coeff[m, finite] = np.linalg.solve(
            kp[np.ix_(finite, finite)] + 4.0 * np.eye(finite.sum()), row[finite]
        )
    return dict(w=w, source=w @ means + off, coeff=coeff, idx=idx)


def scores(profile, hist, delta, dom, cfg):
    return (
        profile["source"][:, dom]
        + cfg["signal"] * profile["w"] @ hist
        + cfg["residual"] * profile["coeff"] @ delta.T
    )
