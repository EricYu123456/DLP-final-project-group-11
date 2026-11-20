"""Behavioral model embeddings for ranking candidates against historical peers."""

import numpy as np


def peer_profiles(history, means, domains, answers):
    """Infer peer weights and benchmark abilities using paired answers to the same probes."""
    n_domains, n_history = means.shape
    out = []
    for y in answers:
        valid = np.isfinite(y)
        h, d, y = history[valid].astype(np.float64), domains[valid], y[valid]
        disagreement = abs(y[:, None] - h).mean(0)
        near = np.exp(-(disagreement - disagreement.min()) / 0.034)
        near /= near.sum()
        centered = h - h.mean(1, keepdims=True)
        residual = y - h @ near
        delta = np.linalg.solve(
            centered.T @ centered + 0.2 * n_history * np.eye(n_history), centered.T @ residual
        )
        posterior = near + delta
        actual, correction, sensitivity = [], [], []
        for j in range(n_domains):
            sel = d == j
            actual.append(y[sel].mean() if sel.any() else y.mean())
            correction.append(residual[sel].mean() if sel.any() else residual.mean())
            difficulty = h[sel].mean(1)
            difficulty -= difficulty.mean() if len(difficulty) else 0
            sensitivity.append(difficulty @ residual[sel] / (2 + difficulty @ difficulty))
        correction = np.asarray(correction)
        control = means @ near + 0.55 * correction + 0.45 * correction.mean()
        domains_fit = np.stack((means @ posterior, control, actual, sensitivity), axis=1)
        out.append(np.r_[near, posterior, domains_fit.ravel(), disagreement.min(), disagreement.mean()])
    return np.stack(out).astype(np.float32)


def rank_features(profile, scores, membership, means, semantics):
    """Flatten in query-major order so each historical tournament is one ranking group."""
    q, h = scores.shape
    n = len(profile)
    d = len(means)
    near, posterior = profile[:, :h], profile[:, h : 2 * h]
    skills = profile[:, 2 * h : -2].reshape(n, d, 4)
    aligned = np.einsum("qd,mds->qms", membership, skills)
    global_fit = skills.mean(1)
    base = membership @ means
    perturb = scores - base
    conditional = [scores @ near.T, scores @ posterior.T, perturb @ near.T, perturb @ posterior.T]
    quantiles = np.quantile(scores, [0.1, 0.5, 0.9], axis=1).T.astype(np.float32)
    qfeatures = np.column_stack((membership, scores.mean(1), scores.std(1), quantiles, semantics))
    model_info = np.column_stack((global_fit, profile[:, -2:]))
    features = np.concatenate(
        (
            aligned,
            np.stack(conditional, axis=-1),
            np.broadcast_to(model_info, (q, n, model_info.shape[-1])),
            np.broadcast_to(qfeatures[:, None], (q, n, qfeatures.shape[-1])),
        ),
        axis=-1,
    )
    return features.reshape(q * n, -1).astype(np.float32)
