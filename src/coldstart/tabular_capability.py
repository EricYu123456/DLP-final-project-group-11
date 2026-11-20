"""Infer low-rank model skill coordinates by regressing across historical LLMs."""

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .conditional_capability import paired_profile


def skill_targets(history, design, domains):
    means = np.stack([history[:, domains == j].mean(1) for j in np.unique(domains)], axis=1)
    residual = history.T - means[:, domains].T
    # Routing uses only differences between models on the same question.
    residual -= residual.mean(1, keepdims=True)
    psi = design[:, 1 + means.shape[1] :]
    center = np.stack([psi[domains == j].mean(0) for j in range(means.shape[1])])
    psi = psi - center[domains]
    coef = np.linalg.solve(psi.T @ psi + 30 * np.eye(psi.shape[1]), psi.T @ residual).T
    _, _, axes = randomized_svd(coef - coef.mean(0), 12, n_iter=4, random_state=2731)
    z = np.column_stack((means, coef @ axes.T))
    return dict(z=z, axes=axes, center=center)


def query_skills(design, domains, pack):
    d = len(pack["center"])
    basis = (design[:, 1 + d :] - pack["center"][domains]) @ pack["axes"].T
    return np.column_stack((np.eye(d)[domains], basis))


def onboarding_features(h, y, dom, represented):
    joint = np.vstack((h, y))
    common = h.mean(0)
    values = []
    for j in np.unique(dom):
        keep = dom == j
        source_rate = np.nanmean(joint[:, keep], axis=1)
        values.append(source_rate)
        for k in np.array_split(np.flatnonzero(keep)[common[keep].argsort()], 3):
            if len(k):
                values.append(
                    (np.nansum(joint[:, k], axis=1) + 3 * source_rate) / (np.isfinite(joint[:, k]).sum(1) + 3)
                )
    _, _, axes = randomized_svd(h - common, 10, n_iter=4, random_state=2731)
    rec = np.nan_to_num(joint - common) @ axes.T / np.sqrt(len(dom))
    return np.column_stack((*values, rec, represented)).astype(np.float32)


def fit_tabular(h, y, idx, domains, pack, device="cpu"):
    from huggingface_hub import hf_hub_download
    from tabicl import TabICLRegressor

    d = domains.max() + 1
    train = h[:, idx]
    target = y[:, idx]
    group = domains[idx]
    pop = np.bincount(domains)
    weights = []
    for i in range(len(h)):
        take = np.arange(len(h)) != i
        w, _ = paired_profile(train[take], train[i : i + 1], group, pop, np.arange(d))
        padded = np.zeros(len(h))
        padded[take] = w[0]
        weights.append(padded)
    for t in target:
        keep = np.isfinite(t)
        w, _ = paired_profile(train[:, keep], t[None, keep], group[keep], pop, np.arange(d))
        weights.append(w[0])
    prior = np.stack(weights) @ pack["z"]
    inputs = onboarding_features(train, target, group, prior[:, :d])
    ckpt = hf_hub_download(
        "jingang/TabICL",
        "tabicl-regressor-v2-20260212.ckpt",
        revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
    )
    learner = TabICLRegressor(
        n_estimators=4, batch_size=4, model_path=ckpt, device=device, n_jobs=1, use_fa3=False
    )
    actual = pack["z"]
    output = []
    for j in range(actual.shape[1]):
        # Predict both the absolute coordinate and an adjustment backed by leave-one-model-out errors.
        learner.fit(inputs[: len(h)], actual[:, j])
        direct = learner.predict(inputs[len(h) :])
        learner.fit(inputs[: len(h)], actual[:, j] - prior[: len(h), j])
        correction = learner.predict(inputs[len(h) :])
        output.append(0.5 * direct + 0.5 * (prior[len(h) :, j] + correction))
    return prior[len(h) :], np.column_stack(output)
