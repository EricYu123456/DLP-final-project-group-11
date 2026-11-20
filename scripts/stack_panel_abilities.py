"""Combine candidate skill embeddings using held-out questions from the very same measured panel."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from ensemble_panel_abilities import components, predictions
from panel_capabilities import normalized, question_kernel, release_fit, template_membership
from panel_tabular_skills import fit_skills, skill_parts


def base_models(x, dom, y, text, aux, meta, size, weights, seed, draws):
    latent = components(x, dom, y, text, aux, meta, size, weights, seed, draws)
    tab = fit_skills(x, dom, y, text, aux, meta, size, weights)
    support, ability, axes, coef = release_fit(x, dom, y, 6.0, "rbf")
    return dict(
        latent=latent, tabular=tab, classic=dict(support=support, ability=ability, axes=axes, coef=coef)
    )


def all_scores(pack, x, text, aux, estimators=None):
    base, conditional, risk, irt = predictions(pack["latent"], x, text, aux)
    d, change = skill_parts(pack["tabular"], x, text, aux, estimators=estimators)
    cv = pack["latent"]["cov"]
    source = template_membership(cv["texts"], text, cv["domains"])
    classic = pack["classic"]
    c = (
        classic["ability"] @ source.T
        + 0.70
        * classic["axes"]
        @ (question_kernel(normalized(x), classic["support"], "rbf") @ classic["coef"]).T
    )
    return np.stack(
        (
            base + 0.45 * conditional,
            base + 1.35 * conditional - risk * 1.10,
            irt[0, 0.65],
            irt[1, 1.6] - 0.45 * risk,
            d + 0.6 * change,
            d + 1.5 * change - 0.45 * risk,
            c,
        )
    ), source


def sufficient(pred, truth):
    average = pred.mean(0)
    contenders = np.argsort(-average, axis=0)[:16]
    quality = np.take_along_axis(truth, contenders, axis=0)
    p = np.take_along_axis(pred, np.broadcast_to(contenders, pred.shape[:1] + contenders.shape), axis=1)
    wt = softmax(24 * np.take_along_axis(average, contenders, axis=0), axis=0)
    p -= np.sum(p * wt, axis=1, keepdims=True)
    quality -= np.sum(quality * wt, axis=0)
    g = np.einsum("kiq,liq,iq->qkl", p, p, wt)
    b = np.einsum("kiq,iq,iq->qk", p, quality, wt)
    return g, b


def mixture(gram, mean, prior, ridge):
    n = len(prior)
    g = gram + ridge * np.eye(n)
    b = mean + ridge * prior
    result = minimize(
        lambda v: (0.5 * v @ g @ v - b @ v, g @ v - b),
        prior,
        method="SLSQP",
        jac=True,
        bounds=[(0, 1)] * n,
        constraints=[dict(type="eq", fun=lambda v: v.sum() - 1, jac=lambda v: np.ones(n))],
        options=dict(ftol=1e-12, maxiter=150),
    )
    return result.x.clip(0) / result.x.clip(0).sum()


def fit_stack(x, dom, y, text, aux, meta, size, seed, weights=None):
    import torch

    torch.set_num_threads(3)
    fold = np.empty(len(dom), int)
    rng = np.random.default_rng(seed + 8113)
    for j in np.unique(dom):
        seen = rng.permutation(np.flatnonzero(dom == j))
        fold[seen] = np.arange(len(seen)) % 4
    out = np.zeros((7, len(y), len(dom)))
    for i in range(4):
        a, b = fold != i, fold == i
        pack = base_models(
            x[a],
            dom[a],
            y[:, a],
            text[a],
            aux[a],
            meta,
            size,
            weights[a] if weights is not None else None,
            seed + i,
            10,
        )
        out[:, :, b] = all_scores(pack, x[b], text[b], aux[b])[0]
    gram, mean = sufficient(out, y)
    uniform = np.full(7, 1 / 7)
    global_mix = mixture(gram.mean(0), mean.mean(0), uniform, 0.00060)
    mixes = []
    for j in np.unique(dom):
        use = dom == j
        strength = 48 / (use.sum() + 48)
        g = (1 - strength) * gram[use].mean(0) + strength * gram.mean(0)
        m = (1 - strength) * mean[use].mean(0) + strength * mean.mean(0)
        mixes.append(mixture(g, m, global_mix, 0.0011))
    pack = base_models(x, dom, y, text, aux, meta, size, weights, seed, 36)
    pack["mixture"] = np.stack(mixes)
    return pack, dict(weights=global_mix.tolist(), per_source_weights=pack["mixture"].tolist(), outer_folds=4)


def stack_scores(pack, x, text, aux, estimators=None):
    values, source = all_scores(pack, x, text, aux, estimators)
    wt = source @ pack["mixture"]
    return np.einsum("qk,kmq->mq", wt, values)
