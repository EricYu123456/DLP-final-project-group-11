"""Estimate a new question's latent skill requirements from a panel of 200 or fewer graded questions."""

from functools import lru_cache

import numpy as np
from sklearn.utils.extmath import randomized_svd

from panel_capabilities import normalized, template_membership
from panel_posterior import posterior_abilities
from panel_rasch import subtype_matrix


def compress(source, rank):
    mean = source.mean(0)
    scale = source.std(0).clip(0.08)
    _, _, axes = randomized_svd(
        (source - mean) / scale, min(rank, source.shape[1], len(source) - 1), n_iter=4, random_state=6081
    )
    return mean, scale, axes


def inputs(pack, x, aux, text, member):
    cols = [member, subtype_matrix(text)]
    for k, v in (("semantic", x), ("extra", aux)):
        mean, scale, axes = pack[k]
        cols.append(((v - mean) / scale) @ axes.T)
    return np.column_stack(cols).astype(np.float32)


def ability_factors(dom, y, meta, size, weights=None):
    member = np.eye(int(dom.max()) + 1)[dom]
    z = posterior_abilities(y, dom, meta, size, dict(noise_corr=0.6, metadata=0.46, noise=1.1), weights)
    residue = y - (z @ member.T)
    residue -= residue.mean(0)
    # Shrink noisy directions toward the measured behavior of related models.
    family = meta - 0.5 * (meta.mean(0) + meta.mean(1, keepdims=True))
    kernel = 0.6 * family + 0.4 * (normalized(z - z.mean(0)) @ normalized(z - z.mean(0)).T)
    kernel = kernel + 0.02 * np.eye(len(z))
    ev, vec = np.linalg.eigh(kernel)
    blended = (vec * (0.26 + 0.74 * ev.clip(0) / (ev.clip(0) + 1.1))) @ vec.T @ residue
    weights = np.ones(len(dom)) if weights is None else np.asarray(weights) / np.mean(weights)
    # The embedding retains directions that distinguish plausible winners on the panel.
    importance = 0.45 + 0.55 * np.mean(z >= np.quantile(z, 0.70, axis=0), axis=1)
    u, _, _ = randomized_svd(
        blended * np.sqrt(importance[:, None] * weights[None, :]), 8, n_iter=6, random_state=6081
    )
    axes = u / np.sqrt(importance[:, None])
    target = (u.T @ (blended * np.sqrt(importance[:, None]))).T
    return z, axes, target.astype(np.float32)


def fit_skills(x, dom, y, text, aux, meta, size, weights=None):
    member = np.eye(int(dom.max()) + 1)[dom]
    z, axes, target = ability_factors(dom, y, meta, size, weights)
    pack = dict(z=z, axes=axes, semantic=compress(x, 16), extra=compress(aux, 16), domains=dom, texts=text)
    pack.update(train=inputs(pack, x, aux, text, member), target=target)
    return pack


@lru_cache(maxsize=1)
def pretrained_regressor():
    from huggingface_hub import hf_hub_download
    from tabicl import TabICLRegressor

    class PanelRegressor(TabICLRegressor):
        def _load_model(self):
            # TabICL 2.2 reloads the same frozen checkpoint on every fit. Scalers and KV caches
            # are fitted separately for each target by the upstream fit method.
            if not hasattr(self, "model_"):
                super()._load_model()

    file = hf_hub_download(
        "jingang/TabICL",
        "tabicl-regressor-v2-20260212.ckpt",
        revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
    )
    return PanelRegressor, file


def learner(kv_cache=False, device="cpu", shared=None):
    regressor, file = pretrained_regressor()
    estimator = regressor(
        n_estimators=2,
        batch_size=2,
        model_path=file,
        device=device,
        n_jobs=1,
        use_fa3=False,
        kv_cache=kv_cache,
    )
    if shared is not None:
        estimator.model_ = shared.model_
        estimator.model_config_ = shared.model_config_
        estimator.model_path_ = shared.model_path_
    return estimator


def prepare_estimators(pack, device="cpu"):
    """Cache each response direction's support once for repeated inference on the same roster."""
    estimators = []
    for target in pack["target"].T:
        est = learner(kv_cache="kv", device=device, shared=estimators[0] if estimators else None)
        est.fit(pack["train"], target)
        estimators.append(est)
    return estimators


def skill_parts(pack, x, text, aux, member=None, estimators=None, fold_learner=None):
    if member is None:
        member = template_membership(pack["texts"], text, pack["domains"])
    features = inputs(pack, x, aux, text, member)
    estim = fold_learner
    if estimators is None and estim is None:
        estim = learner(kv_cache="kv" if len(x) > 384 else False)
    targets = []
    for i, v in enumerate(pack["target"].T):
        if estimators is None:
            estim.fit(pack["train"], v)
        else:
            estim = estimators[i]
        targets.append(
            np.concatenate([estim.predict(features[start : start + 384]) for start in range(0, len(x), 384)])
        )
    return pack["z"] @ member.T, pack["axes"] @ np.stack(targets)


def select_skills(x, dom, y, text, aux, meta, size, seed, weights=None, device="cpu"):
    import torch

    torch.set_num_threads(3)
    options = (0.3, 0.65, 1.15, 1.9)
    risks = np.zeros((len(options), len(x)))
    rng = np.random.default_rng(seed + 723)
    fold = np.empty(len(x), int)
    for j in np.unique(dom):
        ix = rng.permutation(np.flatnonzero(dom == j))
        fold[ix] = np.arange(len(ix)) % 3
    estimator = learner(device=device)
    for i in range(3):
        a, b = fold != i, fold == i
        pack = fit_skills(
            x[a], dom[a], y[:, a], text[a], aux[a], meta, size, weights[a] if weights is not None else None
        )
        base, contrast = skill_parts(pack, x[b], text[b], aux[b], fold_learner=estimator)
        top = np.argsort(-base, axis=0)[:15]
        truth = np.take_along_axis(y[:, b], top, axis=0)
        for k, mix in enumerate(options):
            residual = np.take_along_axis(base + mix * contrast, top, axis=0) - truth
            risks[k, b] = ((residual - residual.mean(0)) ** 2).mean(0)
    best = risks.mean(1).argmin()
    delta = risks - risks[best]
    weight = np.exp(-delta.mean(1) / np.maximum(0.00015, delta.std(1) / np.sqrt(len(x))))
    signal = float(np.average(options, weights=weight))
    pack = fit_skills(x, dom, y, text, aux, meta, size, weights)
    pack["signal"] = signal
    return pack, dict(signal=signal, within_panel_risk=float(risks[best].mean()))


def tabular_scores(pack, x, text, aux, estimators=None):
    base, difference = skill_parts(pack, x, text, aux, estimators=estimators)
    return base + pack["signal"] * difference
