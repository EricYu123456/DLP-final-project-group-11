"""Select trustworthy specialist switches relative to measured, transferable generalist strengths."""

from itertools import product

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

from ensemble_panel_abilities import components, predictions
from panel_capabilities import template_membership


def anchors(y, dom, meta, base, weights, mass):
    m, d = base.shape
    weights = np.ones(len(dom)) if weights is None else weights
    ranks = rankdata(base, axis=0) / m
    corr = np.corrcoef(ranks.T)
    source_prior = []
    reference = []
    uncertain = []
    for j in range(d):
        other = np.arange(d) != j
        v = np.maximum(0.04, corr[j, other]) ** 3 * np.sqrt(mass[other])
        score = ranks[:, other] @ (v / v.sum())
        strength = IsotonicRegression(out_of_bounds="clip").fit_transform(score, base[:, j])
        # Break isotonic plateaus by the measured rank across similar tasks.
        prior = strength + 0.045 * (score - score.mean())
        domain = dom == j
        yy = y[:, domain]
        wt = weights[domain] / weights[domain].sum()
        x = yy @ wt
        var = ((yy - x[:, None]) ** 2) @ wt
        se = np.sqrt(var * (wt @ wt))
        general = ranks @ np.sqrt(mass) / np.sqrt(mass).sum()
        refs = [np.argmax(prior), np.argmax(general), np.argmax(0.68 * base[:, j] + 0.32 * prior - 0.75 * se)]
        noise = []
        for a in refs:
            delta = yy - yy[a]
            avg = delta @ wt
            variance = ((delta - avg[:, None]) ** 2) @ wt
            # Smooth paired variances using the measurement agreement outside this source.
            all_delta = y[:, ~domain] - y[a, ~domain]
            v0 = np.var(all_delta, axis=1)
            variance = (variance * domain.sum() + 6 * v0) / (domain.sum() + 6)
            noise.append(np.sqrt(np.maximum(0, variance) * (wt @ wt)))
        source_prior.append(prior)
        uncertain.append(noise)
        reference.append(refs)
    return dict(prior=np.array(source_prior).T, deviation=np.asarray(uncertain), anchors=np.array(reference))


def fit_basis(x, dom, y, text, aux, meta, size, weights, seed, mass, draws):
    latent = components(x, dom, y, text, aux, meta, size, weights, seed, draws)
    anchor = anchors(
        y,
        dom,
        meta,
        0.55 * latent["cov"]["vectors"][:, : len(mass)] + 0.45 * latent["typical"],
        weights,
        mass,
    )
    return dict(latent=latent, anchor=anchor)


def prediction_parts(pack, x, text, aux):
    base, semantic, _, irt = predictions(pack["latent"], x, text, aux)
    cov = pack["latent"]["cov"]
    member = template_membership(cov["texts"], text, cov["domains"])
    source = pack["anchor"]["prior"] @ member.T
    penalty = np.einsum("qd,dkm->kmq", member, pack["anchor"]["deviation"])
    return (base + 0.8 * semantic, 0.55 * (base + semantic) + 0.45 * irt[1, 0.65], source, penalty)


def combine(parts, cfg):
    i, k, penalty, pooling = cfg
    return (1 - pooling) * parts[i] + pooling * parts[2] - penalty * parts[3][k]


def fit_anchor(x, dom, y, text, aux, meta, size, seed, mass, weights=None):
    import torch

    torch.set_num_threads(3)
    configs = list(product((0, 1), (0, 1, 2), (0.0, 0.35, 0.75, 1.25, 1.9), (0.0, 0.28, 0.55)))
    scores = np.zeros((len(configs), len(x)))
    rng = np.random.default_rng(seed + 3083)
    fold = np.empty(len(dom), int)
    for j in np.unique(dom):
        ii = rng.permutation(np.flatnonzero(dom == j))
        fold[ii] = np.arange(len(ii)) % 4
    for i in range(4):
        a, b = fold != i, fold == i
        pack = fit_basis(
            x[a],
            dom[a],
            y[:, a],
            text[a],
            aux[a],
            meta,
            size,
            weights[a] if weights is not None else None,
            seed + i,
            mass,
            13,
        )
        parts = prediction_parts(pack, x[b], text[b], aux[b])
        for k, cfg in enumerate(configs):
            pred = combine(parts, cfg)
            # Every measured candidate answers the same held-out panel question. The smooth
            # routing utility gives most weight to close contenders and supports paired comparisons.
            prob = 0.65 * softmax(pred / 0.025, axis=0) + 0.35 * softmax(pred / 0.065, axis=0)
            scores[k, b] = (prob * y[:, b]).sum(0)
    # Shrink small-sample policy selection by averaging winners under a stratified Bayesian bootstrap.
    selected = np.zeros(len(configs))
    task_weight = 0.68 * mass + 0.32 / len(mass)
    for _ in range(120):
        w = rng.exponential(size=len(dom))
        if weights is not None:
            w *= weights
        w *= task_weight[dom] / np.bincount(dom, weights=w)[dom]
        reward = scores @ w
        best = reward.argmax()
        difference = scores - scores[best]
        sd = np.sqrt(((difference - (difference @ w)[:, None]) ** 2) @ (w**2))
        selected += softmax((reward - reward[best]) / sd.clip(0.0015))
    selected /= selected.sum()
    keep = selected > 1e-4
    pack = fit_basis(x, dom, y, text, aux, meta, size, weights, seed, mass, 45)
    pack.update(
        anchor_configs=[c for c, k in zip(configs, keep) if k], weights=selected[keep] / selected[keep].sum()
    )
    return pack, dict(
        within_panel_best=list(configs[selected.argmax()]),
        expected_penalty=float(sum(w * c[2] for w, c in zip(selected, configs))),
        expected_transfer=float(sum(w * c[3] for w, c in zip(selected, configs))),
    )


def anchor_scores(pack, x, text, aux):
    parts = prediction_parts(pack, x, text, aux)
    output = np.zeros_like(parts[0])
    for cfg, w in zip(pack["anchor_configs"], pack["weights"]):
        output += w * combine(parts, cfg)
    return output
