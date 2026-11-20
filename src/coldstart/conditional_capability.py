"""Estimate local model strengths from a common, optionally adaptive panel of 200 questions."""

import numpy as np
from sklearn.cluster import KMeans


def make_regions(train_predictions, domains, n_regions, other_predictions=()):
    """Group questions by predicted relative strengths of historical LMs within each benchmark."""
    output = [np.empty(len(train_predictions), dtype=int)] + [
        np.empty(len(p), dtype=int) for p, _ in other_predictions
    ]
    centers, models, offset = [], [], 0
    for j in np.unique(domains):
        tr = domains == j
        raw = train_predictions[tr]
        keep = np.argsort(-raw.mean(0))[: min(36, raw.shape[1])]
        raw = raw[:, keep]
        z = raw - raw.mean(1, keepdims=True)
        mean, scale = z.mean(0), z.std(0).clip(0.018)
        k = min(n_regions, max(2, tr.sum() // 65))
        km = KMeans(n_clusters=k, n_init=5, random_state=1349).fit((z - mean) / scale)
        output[0][tr] = offset + km.labels_
        for dest, (p, dom) in zip(output[1:], other_predictions):
            chosen = dom == j
            if chosen.any():
                a = p[chosen][:, keep]
                dest[chosen] = offset + km.predict(
                    ((a - a.mean(1, keepdims=True) - mean) / scale).astype(z.dtype)
                )
        models.append((keep, mean, scale, km.cluster_centers_.copy(), offset))
        centers.extend([j] * k)
        offset += k
    return output, np.array(centers), models


def assign_regions(predictions, domains, models):
    groups = np.empty(len(domains), dtype=int)
    for j, (keep, mean, scale, centers, offset) in enumerate(models):
        use = domains == j
        if use.any():
            x = predictions[use][:, keep]
            z = (x - x.mean(1, keepdims=True) - mean) / scale
            groups[use] = offset + ((z[:, None] - centers) ** 2).sum(2).argmin(1)
    return groups


def paired_profile(h, y, group, population, group_domain, ridge=0.40):
    """Correct for unequal regional sampling when matching a newcomer's response pattern to peers."""
    count = np.bincount(group, minlength=len(population))
    ipw = population[group] / count[group]
    dom = group_domain[group]
    # Equal source weights agree with the training metric; population weights within source undo adaptive oversampling.
    ipw /= np.bincount(dom, weights=ipw)[dom]
    ipw /= ipw.sum()
    h = h.astype(np.float64)
    all_y = np.vstack((h, y))
    all_yh = (all_y * ipw) @ h.T
    dissimilarity = (all_y @ ipw)[:, None] + (h @ ipw)[None, :] - 2 * all_yh
    relative = all_y - h.mean(0)
    signed = (relative * ipw) @ relative[: len(h)].T
    design = np.eye(int(group_domain.max()) + 1)[dom] * ipw[:, None]
    rates = (all_y @ design) / design.sum(0).clip(1e-8)
    rates = (rates - rates[: len(h)].mean(0)) / rates[: len(h)].std(0).clip(0.035)
    k = np.exp(-dissimilarity / 0.10) + 3 * signed + 0.35 * rates @ rates[: len(h)].T / rates.shape[1]
    k -= k[: len(h)].mean(0) + k.mean(1, keepdims=True) - k[: len(h)].mean()
    embedding = np.linalg.solve(k[: len(h)] + ridge * np.eye(len(h)), k[len(h) :].T).T
    embedding += (1 - embedding.sum(1, keepdims=True)) / len(h)
    return embedding, ipw


def fit_profile(h, y, group, population, group_domain, local_prior, survey=True):
    # A response may be missing for one of the expanded candidates.
    if not np.isfinite(y).all():
        return np.vstack(
            [
                fit_profile(
                    h[:, np.isfinite(row)],
                    row[None, np.isfinite(row)],
                    group[np.isfinite(row)],
                    population,
                    group_domain,
                    local_prior,
                    survey,
                )
                for row in y
            ]
        )
    w, ipw = paired_profile(
        h, y, group, population if survey else np.bincount(group, minlength=len(population)), group_domain
    )
    residual = y - w @ h
    dom = group_domain[group]
    unit = np.eye(int(group_domain.max()) + 1)[dom]
    weights = unit * ipw[:, None]
    mean = residual @ weights / weights.sum(0).clip(1e-8)
    global_mean = (residual @ ipw)[:, None]
    source = 0.75 * mean + 0.25 * global_mean
    unit = np.eye(len(population))[group]
    sums = (residual - source[:, dom]) @ unit
    regional = sums / (unit.sum(0) + local_prior)
    return np.column_stack((w, source, regional))


def regional_prediction(model, pred, region, historical_means, group_domain, signal, local_scale):
    h = historical_means.shape[1]
    d = int(group_domain.max()) + 1
    conditional = historical_means[region] + signal * (pred - historical_means[region])
    offsets = model[:, h : h + d][:, group_domain[region]] + local_scale * model[:, h + d :][:, region]
    return model[:, :h] @ conditional.T + offsets


def choose_panel(history, answers, train_regions, group_domain, means, policy, seed, budget=200):
    rng = np.random.default_rng(seed)
    n = len(group_domain)
    pools = [rng.permutation(np.flatnonzero(train_regions == j)) for j in range(n)]
    sizes = np.array([len(pool) for pool in pools])
    mass = 0.5 * sizes / sizes.sum() + 0.5 * sizes / np.bincount(group_domain, weights=sizes)[
        group_domain
    ] / (group_domain.max() + 1)
    base = np.sqrt(mass)
    counts = np.zeros(n, dtype=int)
    order = []
    for stop in (min(70, budget), min(120, budget), min(160, budget), budget):
        priority = base.copy()
        if order and policy == "adaptive":
            z = fit_profile(
                history[:, order], answers[:, order], train_regions[order], sizes, group_domain, 12
            )
            scores = regional_prediction(z, means, np.arange(n), means, group_domain, 0, 0.7)
            incumbent = (scores @ mass).argmax()
            rival = scores.copy()
            rival[incumbent] = -np.inf
            margin = rival.max(0) - scores[incumbent]
            # Measure specialisms near a possible decision change while keeping coverage across all regions.
            priority *= 0.20 + np.exp(-0.5 * (margin / 0.10) ** 2)
        while len(order) < stop:
            j = int(np.argmax(np.where(counts < sizes, priority / (counts + 1), -np.inf)))
            order.append(pools[j][counts[j]])
            counts[j] += 1
    return np.asarray(order)


def local_fingerprints(h, y, groups, means, population, group_domain, focus, prior):
    """Infer a different historical-peer mixture per skill, sharing information across all 200 answers."""
    if not np.isfinite(y).all():
        return np.concatenate(
            [
                local_fingerprints(
                    h[:, np.isfinite(row)],
                    row[None, np.isfinite(row)],
                    groups[np.isfinite(row)],
                    means,
                    population,
                    group_domain,
                    focus,
                    prior,
                )
                for row in y
            ]
        )
    h = np.asarray(h, dtype=np.float64)
    n, k = len(h), len(population)
    joint = np.vstack((h, y))
    relative = joint - h.mean(0)
    domains = group_domain[groups]
    _, wt = paired_profile(h, y, groups, population, group_domain)
    one = np.eye(int(group_domain.max()) + 1)[domains] * wt[:, None]
    rates = joint @ one / one.sum(0).clip(1e-8)
    rates = (rates - rates[:n].mean(0)) / rates[:n].std(0).clip(0.035)
    domain_kernel = 0.35 * rates @ rates[:n].T / rates.shape[1]
    # Similar tasks can share evidence even when a particular regional cell has few probes.
    patterns = means - means.mean(1, keepdims=True)
    patterns = patterns - np.average(patterns, axis=0, weights=population)
    patterns /= np.linalg.norm(patterns, axis=1, keepdims=True).clip(0.05)
    sim = (patterns @ patterns.T).clip(0, 1) ** 2
    weights, uncertainty = [], []
    for j in range(k):
        similarity = 0.45 * (domains == group_domain[j]) + 0.40 * sim[j, groups] + 0.15 * (groups == j)
        local = wt * (1 + focus * similarity)
        local /= local.sum()
        dis = (joint @ local)[:, None] + (h @ local)[None, :] - 2 * (joint * local) @ h.T
        mat = domain_kernel + 3 * (relative * local) @ relative[:n].T + np.exp(-dis / 0.10)
        mat -= mat[:n].mean(0) + mat.mean(1, keepdims=True) - mat[:n].mean()
        w = np.linalg.solve(mat[:n] + 0.40 * np.eye(n), mat[n:].T).T
        w += (1 - w.sum(1, keepdims=True)) / n
        weights.append(w)
        noise = y - w @ h
        src = domains == group_domain[j]
        ix = groups == j
        general = (noise @ wt)[:, None]
        source = (noise[:, src] @ wt[src])[:, None] / wt[src].sum().clip(1e-9)
        offset = (0.75 * source + 0.25 * general)[:, 0]
        delta = (noise[:, ix] - offset[:, None]).sum(1) / (ix.sum() + prior)
        uncertainty.append(np.column_stack((offset, delta)))
    return np.concatenate((np.stack(weights, axis=1), np.stack(uncertainty, axis=1)), axis=2)


def score_local(z, pred, groups, means, pop, group_domain, cfg):
    n = means.shape[1]
    result = np.empty((len(z), len(groups)))
    baseline = (z[:, :, :n] * means).sum(2) + z[:, :, n] + cfg["local_scale"] * z[:, :, n + 1]
    for j in np.unique(groups):
        ix = groups == j
        result[:, ix] = baseline[:, j, None] + cfg["signal"] * z[:, j, :n] @ (pred[ix] - means[j]).T
    if cfg.get("global_discount"):
        mass = 0.5 * pop / pop.sum() + 0.5 * pop / np.bincount(group_domain, weights=pop)[group_domain] / (
            group_domain.max() + 1
        )
        result -= cfg["global_discount"] * (baseline @ mass)[:, None]
    return result


def calibrate_switches(h, y, idx, oof, train_group, means, pop, group_domain, z, cfg, seed):
    """Use held-out parts of the same probe panel to learn whether each challenger's predicted gains transfer."""
    group = train_group[idx]
    mass = 0.5 * pop / pop.sum() + 0.5 * pop / np.bincount(group_domain, weights=pop)[group_domain] / (
        group_domain.max() + 1
    )
    baseline = score_local(z, means, np.arange(len(means)), means, pop, group_domain, cfg) @ mass
    incumbent = int(baseline.argmax())
    xx, anchor_pred = np.zeros_like(y[:, idx], dtype=np.float64), np.zeros((len(y), len(idx)))
    rng = np.random.default_rng(seed + 92)
    folds = np.zeros(len(idx), int)
    # Balance within sources while preserving every probe for out-of-fold calibration.
    for j in np.unique(group_domain):
        g = rng.permutation(np.flatnonzero(group_domain[group] == j))
        folds[g] = np.arange(len(g)) % 3
    for i in range(3):
        a, b = folds != i, folds == i
        zi = local_fingerprints(
            h[:, idx[a]], y[:, idx[a]], group[a], means, pop, group_domain, cfg["focus"], cfg["prior"]
        )
        avg = score_local(zi, means, np.arange(len(means)), means, pop, group_domain, cfg) @ mass
        pred = score_local(zi, oof[idx[b]], group[b], means, pop, group_domain, cfg) - avg[:, None]
        xx[:, b] = pred - pred[incumbent]
        # Historical models' actual answers are observed on a probe, allowing a paired correction for item difficulty.
        control = np.column_stack(
            [
                zi[:, j, : len(h)] @ h[:, item] + zi[:, j, len(h)] + cfg["local_scale"] * zi[:, j, len(h) + 1]
                for item, j in zip(idx[b], group[b])
            ]
        )
        anchor_pred[:, b] = control - control[incumbent]
    delta = y[:, idx] - y[incumbent, idx]
    importance = mass[group] / np.bincount(group, minlength=len(mass))[group]
    importance = np.broadcast_to(importance, delta.shape).copy()
    good = np.isfinite(delta)
    importance *= good
    importance *= len(idx) / importance.sum(1, keepdims=True).clip(1e-9)
    delta = np.nan_to_num(delta)
    xc = xx - (xx * importance).sum(1, keepdims=True) / len(idx)
    slope = (1.25 + (xc * importance * delta).sum(1)) / (1.25 + (xc**2 * importance).sum(1))
    bias = ((delta - anchor_pred) * importance).sum(1) / (len(idx) + 120)
    return dict(
        baseline=baseline - baseline[incumbent],
        incumbent=incumbent,
        slope=slope.clip(0.15, 2.5),
        bias=bias.clip(-0.08, 0.08),
    )


def calibrated_scores(raw, calibration, amount):
    d = raw - raw[calibration["incumbent"]]
    mean = calibration["baseline"][:, None]
    return d + amount * ((calibration["slope"][:, None] - 1) * (d - mean) + calibration["bias"][:, None])
