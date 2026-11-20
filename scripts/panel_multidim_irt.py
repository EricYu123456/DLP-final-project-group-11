"""Learn multidimensional ability and prompt discrimination from a single shared response matrix."""

import numpy as np
import torch
from scipy.special import expit, logit
from torch.nn import functional as F

from panel_capabilities import template_membership
from panel_posterior import posterior_abilities, question_map


def _kernel(q, s, member, source):
    dot = q @ s.T
    return (0.35 * dot + 0.65 * np.exp(3.0 * (dot - 1))) * (0.28 + 0.72 * member[:, source])


def fit_irt(x, groups, y, texts, aux, meta, size, strength, seed, weights=None):
    rng = np.random.default_rng(seed + 377)
    n, d = len(y), int(groups.max()) + 1
    phi, context = question_map(x, texts, aux)
    membership = np.eye(d)[groups]
    kernel = _kernel(phi, phi, membership, groups)
    eigen, ev = np.linalg.eigh((kernel + kernel.T) * 0.5)
    choose = np.argsort(eigen)[-min(52, len(x)) :]
    scales = np.sqrt(eigen[choose].clip(1e-5))
    features = ev[:, choose] * scales
    transform = ev[:, choose] / scales
    ability = posterior_abilities(
        y, groups, meta, size, dict(noise_corr=0.55, metadata=0.45, noise=1.05), weights
    )
    logits = logit(ability.clip(0.012, 0.988))
    source_base = logits.mean(0)
    base_prior = logits - source_base
    item_prior = logit(((y.sum(0) + 1) / (n + 2)).clip(0.01, 0.99))
    residual = y - expit(base_prior[:, groups] + item_prior)
    u, _, _ = np.linalg.svd(residual, full_matrices=False)
    rank = min(10, n - 1)
    init = u[:, :rank] * np.sqrt(n)
    # The first ability is a model's broad empirical skill. The remaining ones need not be monotone.
    broad = base_prior.mean(1)
    init[:, 0] = (broad - broad.mean()) / broad.std().clip(0.05)
    init += rng.normal(0, 0.015, init.shape)

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32)

    def parameter(a):
        return torch.nn.Parameter(tensor(a).clone())

    bias = parameter(item_prior)
    baseline = parameter(base_prior)
    z = parameter(init)
    proj = parameter(np.zeros((features.shape[1], rank)))
    residual_q = parameter(np.zeros((len(x), rank)))
    coords = tensor(features)
    initial_z = tensor(init)
    initial_base = tensor(base_prior)
    initial_bias = tensor(item_prior)
    target = tensor(y)
    weight = tensor(np.ones(len(x)) if weights is None else weights / np.mean(weights))
    regularization = np.eye(n) + 0.55 * meta + 0.25 * size
    precision = tensor(np.linalg.inv(regularization))
    optimizer = torch.optim.Adam([bias, baseline, z, proj, residual_q], lr=0.027)
    for _ in range(265):
        model = z - z.mean(0, keepdim=True)
        discr = coords @ proj + residual_q
        prediction = bias + baseline[:, groups] + model @ discr.T
        base_delta = baseline - initial_base
        loss = (F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * weight).sum()
        loss += strength * (2.5 * (base_delta * (precision @ base_delta)).sum() + 6.0 * proj.square().sum())
        loss += strength * (3.5 * residual_q.square().sum() + 1.3 * (z - initial_z).square().sum())
        loss += 0.65 * (bias - initial_bias).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        vec = np.column_stack((baseline.numpy(), (z - z.mean(0, keepdim=True)).numpy()))
        query = transform @ proj.numpy()
        difficulties = np.array([bias.numpy()[groups == j].mean() for j in range(d)])
    return dict(
        vectors=vec,
        coef=query,
        support=phi,
        domains=groups,
        texts=texts,
        source_difficulty=difficulties,
        **context,
    )


def irt_scores(pack, x, texts, aux, signal=1.0, membership=None):
    if membership is None:
        membership = template_membership(pack["texts"], texts, pack["domains"])
    phi, _ = question_map(x, texts, aux, pack)
    k = _kernel(phi, pack["support"], membership, pack["domains"])
    query = np.column_stack((membership, signal * k @ pack["coef"]))
    logits = pack["vectors"] @ query.T + membership @ pack["source_difficulty"]
    return expit(logits)
