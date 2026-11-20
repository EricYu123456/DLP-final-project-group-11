"""Fit a 200-probe ensemble of multidimensional item-response and paired-covariance embeddings."""

from itertools import product

import numpy as np

from panel_capabilities import template_membership
from panel_multidim_irt import fit_irt, irt_scores
from panel_posterior import design, posterior_abilities, train


def components(x, dom, y, text, aux, meta, size, weights, seed, draws):
    cfg = dict(noise_corr=0.55, metadata=0.45, noise=1.05, ridge=3.2)
    cov = train(x, dom, y, text, aux, meta, size, cfg, weights)
    boot = []
    rng = np.random.default_rng(seed + 6211)
    for _ in range(draws):
        wt = rng.exponential(size=len(x))
        if weights is not None:
            wt *= weights
        varied = dict(
            cfg,
            metadata=rng.uniform(0.2, 0.72),
            noise_corr=rng.uniform(0.25, 0.8),
            noise=rng.uniform(0.65, 1.5),
        )
        boot.append(posterior_abilities(y, dom, meta, size, varied, wt))
    boot = np.stack(boot)
    typical = np.median(boot, axis=0)
    candidates = np.argsort(-typical, axis=0)[:8]
    centers = np.take_along_axis(
        boot, np.broadcast_to(candidates, (len(boot), *candidates.shape)), axis=1
    ).mean(1, keepdims=True)
    uncertainty = (boot - centers).std(0)
    models = [fit_irt(x, dom, y, text, aux, meta, size, reg, seed, weights) for reg in (1.4, 4.5)]
    return dict(cov=cov, typical=typical, uncertainty=uncertainty, irt=models)


def predictions(pack, x, text, aux):
    cov = pack["cov"]
    mem = template_membership(cov["texts"], text, cov["domains"])
    d = mem.shape[1]
    raw = (0.55 * cov["vectors"][:, :d] + 0.45 * pack["typical"]) @ mem.T
    semantic = design(cov, x, text, aux, 1.0, mem)[:, d:]
    component = cov["vectors"][:, d:] @ semantic.T
    risk = pack["uncertainty"] @ mem.T
    irt = {}
    for i, model in enumerate(pack["irt"]):
        for amp in (0.65, 1.6):
            irt[i, amp] = irt_scores(model, x, text, aux, amp, mem)
    return raw, component, risk, irt


def combine(parts, cfg):
    raw, semantic, risk, irt = parts
    signal, penalty, model, irt_signal, mix = cfg
    return (1 - mix) * (raw + signal * semantic) + mix * irt[model, irt_signal] - penalty * risk


def fit_ensemble(x, dom, y, text, aux, meta, size, seed, weights=None):
    import torch

    torch.set_num_threads(3)
    options = list(product((0.55, 1.5), (0.0, 0.55, 1.15), range(2), (0.65, 1.6), (0.0, 0.45, 0.85)))
    rng = np.random.default_rng(seed + 1213)
    fold = np.empty(len(dom), int)
    for j in np.unique(dom):
        group = rng.permutation(np.flatnonzero(dom == j))
        fold[group] = np.arange(len(group)) % 4
    risk = {k: [] for k in options}
    for i in range(4):
        a, b = fold != i, fold == i
        pack = components(
            x[a],
            dom[a],
            y[:, a],
            text[a],
            aux[a],
            meta,
            size,
            weights[a] if weights is not None else None,
            seed + i,
            12,
        )
        pred = predictions(pack, x[b], text[b], aux[b])
        contenders = np.argsort(-(pred[0] + pred[3][1, 0.65]) / 2, axis=0)[:16]
        left, right = np.triu_indices(len(contenders), 1)
        actual = np.take_along_axis(y[:, b], contenders, axis=0)
        delta = actual[left] - actual[right]
        for cfg in options:
            v = np.take_along_axis(combine(pred, cfg), contenders, axis=0)
            loss = ((v[left] - v[right] - delta) ** 2).mean(0)
            risk[cfg].extend(loss)
    best = min(options, key=lambda k: np.mean(risk[k]))
    baseline = np.array(risk[best])
    eligible, importance = [], []
    for key in options:
        diff = np.asarray(risk[key]) - baseline
        scale = max(0.0008, diff.std(ddof=1) / np.sqrt(len(x)))
        if diff.mean() <= 1.25 * scale:
            eligible.append(key)
            importance.append(np.exp(-diff.mean() / scale))
    pack = components(x, dom, y, text, aux, meta, size, weights, seed, 42)
    pack.update(configs=eligible, weights=np.array(importance) / np.sum(importance))
    return pack, dict(best=list(best), retained=len(eligible), within_panel_pair_risk=float(baseline.mean()))


def ensemble_scores(pack, x, text, aux):
    p = predictions(pack, x, text, aux)
    return np.average([combine(p, cfg) for cfg in pack["configs"]], axis=0, weights=pack["weights"])
