"""Distill a newcomer's measured rescue and harm probabilities into two question-only ability vectors."""

import numpy as np
from scipy.special import expit, logit
from sklearn.utils.extmath import randomized_svd


def logistic(design, positive, total, penalty, initial=None, iterations=7):
    """Independent binomial MAP fits with one axis-aligned prior for all candidates."""
    result = np.zeros((len(positive), design.shape[1])) if initial is None else initial.copy()
    prior = np.zeros_like(result) if initial is None else initial
    regularizer = np.diag(penalty)
    for j, (yes, trials) in enumerate(zip(positive, total)):
        row = result[j]
        for _ in range(iterations):
            prob = expit(design @ row)
            gradient = design.T @ (yes - trials * prob) - (row - prior[j]) * penalty
            curvature = (design.T * (trials * prob * (1 - prob))) @ design + regularizer
            change = np.linalg.solve(curvature, gradient)
            row += change / max(1.0, np.max(np.abs(change)) / 1.5)
            if np.max(abs(change)) < 0.0002:
                break
    return result


def behavioral_basis(history, domains):
    h = history.T.astype(np.float64)
    mean = np.stack([h[domains == j].mean(0) for j in np.unique(domains)])
    z = h - mean[domains]
    difficulty = z.mean(1, keepdims=True)
    _, _, axes = randomized_svd(z - difficulty, min(24, h.shape[1] - 1), n_iter=4, random_state=1789)
    skill = (z - difficulty) @ axes.T
    return dict(
        mean=mean, axes=axes, scale=skill.std(0).clip(0.08), difficulty_scale=difficulty.std().clip(0.08)
    )


def behavioral_design(h, dom, pack):
    z = h.T - pack["mean"][dom]
    value = z.mean(1, keepdims=True)
    common = np.eye(len(pack["mean"]))[dom]
    return np.column_stack(
        (
            np.ones(len(z)),
            common,
            value / pack["difficulty_scale"],
            ((z - value) @ pack["axes"].T) / pack["scale"],
        )
    )


def infer_teacher(history, y, ix, dom, data, basis, strength):
    """Calibrate nearest behavioral peers, then learn a compact logistic correction on the same probes."""
    observed = history[:, ix].astype(np.float64)
    y = y[:, ix]
    output = []
    for row in y:
        seen = np.isfinite(row)
        a, target, source = observed[:, seen], row[seen], dom[ix[seen]]
        distance = abs(a - target).mean(1)
        peers = np.argsort(distance)[:5]
        affinity = np.exp(-(distance[peers] - distance[peers[0]]) / 0.025)
        affinity /= affinity.sum()
        peer_answer = affinity @ history[peers].astype(np.float64)
        # A smoothed two-cell calibration gives an interpretable probability even for an exact sibling.
        truth_if = []
        for val in (0, 1):
            w = peer_answer[ix[seen]] if val else 1 - peer_answer[ix[seen]]
            rate = (w @ target + 1) / (w.sum() + 2)
            by_domain = [
                (w[source == j] @ target[source == j] + 8 * rate) / (w[source == j].sum() + 8)
                for j in range(len(basis["mean"]))
            ]
            truth_if.append(np.array(by_domain))
        p0, p1 = [v[dom] for v in truth_if]
        peer_prob = (p0 + peer_answer * (p1 - p0)).clip(0.01, 0.99)
        aug = np.column_stack((data, logit(peer_prob)))
        init = np.zeros((1, aug.shape[1]))
        init[0, -1] = 1
        penalty = np.full(aug.shape[1], strength)
        penalty[0], penalty[-1] = 2, strength * 2
        fit = logistic(aug[ix[seen]], target[None], np.ones((1, seen.sum())), penalty, init)
        output.append(expit(aug @ fit[0]))
    return np.stack(output)


def conditional_events(probs, true_panel, idx, domain, incumbent):
    """Subtract the probability of both models succeeding, accounting for their measured residual agreement."""
    pa = probs[incumbent]
    centered = np.nan_to_num(true_panel[:, idx] - probs[:, idx])
    anchor = centered[incumbent]
    scale = np.sqrt((probs * (1 - probs)) * (pa * (1 - pa)))
    rho = np.zeros((len(probs), int(domain.max()) + 1))
    for j in range(rho.shape[1]):
        sel = domain[idx] == j
        cov = (centered[:, sel] * anchor[sel]).sum(1)
        rho[:, j] = (cov / (scale[:, idx[sel]].sum(1) + 3)).clip(-0.45, 0.85)
    both = np.clip(probs * pa + rho[:, domain] * scale, np.maximum(0, probs + pa - 1), np.minimum(probs, pa))
    return probs - both, pa - both


def distill(design, probabilities, responses, idx, domains, weights, pseudo_count, probe_weight, strength):
    mass = 0.5 / len(domains) + 0.5 / (len(np.unique(domains)) * np.bincount(domains)[domains])
    incumbent = int((probabilities @ mass).argmax())
    rescue, harm = conditional_events(probabilities, responses, idx, domains, incumbent)
    win, discordance = rescue * (pseudo_count * mass), (rescue + harm) * (pseudo_count * mass)
    eff = np.broadcast_to(pseudo_count * mass, win.shape).copy()
    real = responses[:, idx] - responses[incumbent, idx]
    wt = probe_weight * weights * len(idx)
    win[:, idx] += (real > 0) * wt
    discordance[:, idx] += (abs(real) == 1) * wt
    eff[:, idx] += np.isfinite(real) * wt
    penalty = np.full(design.shape[1], strength)
    penalty[0] = 0.35
    skill = logistic(design, win, discordance, penalty)
    frequency = logistic(design, discordance, eff, penalty)
    return dict(incumbent=incumbent, skill=skill, frequency=frequency)


def pair_scores(z, design, temperature=1.0):
    prob = expit(z["skill"] @ design.T / temperature)
    freq = expit(z["frequency"] @ design.T)
    gain = (2 * prob - 1) * freq
    gain[z["incumbent"]] = 0
    return gain


def pair_features(features, domains, pack):
    basic = np.tanh((features - pack["center"]) / (2 * pack["scale"]))
    membership = np.eye(pack["n_domains"], dtype=np.float32)[domains]
    return np.column_stack(
        (
            np.ones(len(basic), np.float32),
            membership,
            basic,
            (membership[:, :, None] * basic[:, None]).reshape(len(basic), -1),
        )
    ).astype(np.float32), basic


def event_ridge(design, target, weight):
    xtw = design.T * weight
    gram = xtw @ design
    gram.flat[:: len(gram) + 1] += 15
    return np.linalg.solve(gram, xtw @ target).astype(np.float32)


def pair_design(factors, basic, domains, pack):
    relative = ((factors - pack["factor_mean"]) / pack["factor_scale"]).clip(-3, 3)
    context = ((basic @ pack["extra"].T) / pack["extra_scale"]).clip(-3, 3)
    membership = np.eye(pack["n_domains"])[domains]
    return np.column_stack((np.ones(len(basic)), membership, relative, 0.6 * context))


def learn_pair_space(raw, domains, history):
    """Learn query factors specifically for rescue/harm events against strong and specialist reference LMs."""
    d = int(domains.max()) + 1
    means = np.stack([history[:, domains == j].mean(1) for j in range(d)])
    peers = np.unique(np.r_[np.argsort(-means.mean(0))[:4], np.argsort(-means, axis=1)[:, :2].ravel()])
    h = history.T.astype(np.float32)
    events = np.column_stack(
        (h, *(h * (1 - h[:, a : a + 1]) for a in peers), *((1 - h) * h[:, a : a + 1] for a in peers))
    )
    events -= np.stack([events[domains == j].mean(0) for j in range(d)])[domains]
    events /= events.std(0).clip(0.12)
    _, _, axes = randomized_svd(events, 26, n_iter=4, random_state=1811)
    target = events @ axes.T
    pack = dict(center=raw.mean(0), scale=raw.std(0).clip(0.06), n_domains=d)
    x, basic = pair_features(raw, domains, pack)
    _, _, pack["extra"] = randomized_svd(basic, 14, n_iter=3, random_state=1811)
    pack["extra_scale"] = (basic @ pack["extra"].T).std(0).clip(0.1)
    weight = np.sqrt(len(domains) / (d * np.bincount(domains)[domains])).astype(np.float32)
    rng = np.random.default_rng(1811)
    fold = np.empty(len(domains), dtype=int)
    for j in range(d):
        order = rng.permutation(np.flatnonzero(domains == j))
        fold[order] = np.arange(len(order)) % 3
    preds = np.zeros_like(target)
    for j in range(3):
        b = fold == j
        preds[b] = x[b] @ event_ridge(x[~b], target[~b], weight[~b])
    pack["decoder"] = event_ridge(x, target, weight)
    pack["factor_mean"], pack["factor_scale"] = preds.mean(0), preds.std(0).clip(0.20)
    return pack, pair_design(preds, basic, domains, pack)


def lookup_pair_space(raw, dom, pack):
    features, basic = pair_features(raw, dom, pack)
    return pair_design(features @ pack["decoder"], basic, dom, pack)
