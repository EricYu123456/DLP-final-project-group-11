"""Represent each candidate by its strengths when a similar historical model succeeds or fails."""

from itertools import combinations, product

import numpy as np
from scipy.special import expit, logit, softmax

from .latent_answers import answer_arrays
from .switch_calibration import peer_vectors


def features(evidence, peer, domain, domains):
    raw = answer_arrays(evidence)
    votes = raw["votes"]
    top = votes.argmax(2)
    pair = []
    for a, b in combinations(range(votes.shape[1]), 2):
        pair.extend(
            (
                (votes[:, a] * votes[:, b]).sum(1),
                np.abs(votes[:, a] - votes[:, b]).sum(1),
                top[:, a] == top[:, b],
            )
        )
    availability = [p[:, 0, 6] for p, _ in evidence if p.shape[-1] > 6]
    return np.column_stack(
        (peer, np.eye(domains)[domain], raw["quality"], raw["mc"], *pair, *availability)
    ).astype(np.float32)


def train_verifier(x, domain, history, anchor):
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression

    truth = history[anchor].astype(int)
    rng = np.random.default_rng(4973)
    fold = np.empty(len(domain), int)
    for j in np.unique(domain):
        ids = rng.permutation(np.flatnonzero(domain == j))
        fold[ids] = np.arange(len(ids)) % 4

    def fit(where):
        model = CatBoostClassifier(
            iterations=480,
            depth=5,
            learning_rate=0.045,
            l2_leaf_reg=9,
            thread_count=3,
            random_seed=4973,
            allow_writing_files=False,
            verbose=False,
        )
        return model.fit(x[where], truth[where])

    out = np.zeros(len(domain))
    for k in range(4):
        use = fold == k
        out[use] = fit(~use).predict(x[use], prediction_type="RawFormulaVal").ravel()
    # Calibration uses only out-of-fold historical predictions. Absolute error probability drives switches.
    calibrator = LogisticRegression(C=2, max_iter=120).fit(out[:, None], truth)
    a, b = calibrator.coef_.item(), calibrator.intercept_.item()
    means = np.stack([history[:, domain == j].mean(1) for j in np.unique(domain)])
    cond = np.stack(
        [[history[:, (domain == j) & (truth == hit)].mean(1) for hit in (0, 1)] for j in np.unique(domain)]
    )
    return dict(
        model=fit(np.ones(len(domain), bool)),
        slope=a,
        intercept=b,
        means=means,
        conditional=cond,
        anchor=anchor,
    ), expit(a * out + b)


def probability(pack, x):
    odds = pack["model"].predict(x, prediction_type="RawFormulaVal", thread_count=3).ravel()
    return expit(pack["slope"] * odds + pack["intercept"])


def fit_state_readouts(x, domain, history, pack):
    """Learn which historical peers rescue a question when the reference misses, and when they regress."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.utils.extmath import randomized_svd

    readouts = []
    for state in (0, 1):
        use = history[pack["anchor"]] == state
        relative = history[:, use].T - pack["conditional"][domain[use], state]
        relative -= relative.mean(1, keepdims=True)
        _, _, axes = randomized_svd(relative, min(16, len(history) - 1), random_state=7481)
        target = relative @ axes.T
        scale = target.std(0).clip(0.10)
        tree = ExtraTreesRegressor(
            n_estimators=112,
            max_features=0.68,
            min_samples_leaf=10,
            max_depth=15,
            n_jobs=3,
            random_state=7481 + state,
        ).fit(x[use], target / scale)
        readouts.append((tree, scale[:, None] * axes))
    return readouts


def state_effects(readouts, x):
    return np.stack([model.predict(x) @ axes for model, axes in readouts])


def state_embedding(pack, h, y, idx, domain, nearest, shrink):
    global_w, offset = peer_vectors(h, y, idx, domain, nearest)
    dims = len(pack["means"])
    result = []
    for state in (0, 1):
        use = idx[h[pack["anchor"], idx] == state]
        weights = global_w.copy()
        if len(use):
            for m, row in enumerate(y):
                valid = use[np.isfinite(row[use])]
                if len(valid) >= 8:
                    local, _ = peer_vectors(h, row[None], valid, domain, nearest)
                    weights[m] += len(valid) / (len(valid) + 75) * (local[0] - global_w[m])
        prior = (weights @ pack["conditional"][:, state].T + offset).clip(0.005, 0.995)
        value = prior.copy()
        for j in range(dims):
            yy = y[:, use[domain[use] == j]]
            value[:, j] = (np.nansum(yy, axis=1) + shrink * prior[:, j]) / (np.isfinite(yy).sum(1) + shrink)
        result.append(np.column_stack((weights, value)))
    return np.column_stack(result)


def state_scores(pack, z, chance, peer, domain, effects, signal, calibration):
    r = peer.shape[1]
    p = expit(
        0.85 * logit(chance.clip(0.005, 0.995)) + 0.15 * logit(peer[:, pack["anchor"]].clip(0.005, 0.995))
    )
    mean = pack["means"][domain, pack["anchor"]]
    p = (mean + calibration * (p - mean)).clip(0.0, 1.0)
    result = 0.0
    for state, (block, weight) in enumerate(zip(np.hsplit(z, 2), (1 - p, p))):
        result += weight * (block[:, r:][:, domain] + signal * (block[:, :r] @ effects[state].T))
    return result


def embed(pack, h, y, idx, domain, nearest, shrink):
    w, offset = peer_vectors(h, y, idx, domain, nearest)
    cond, mu = pack["conditional"], pack["means"]
    anchor = h[pack["anchor"], idx]
    pred = []
    for state in (0, 1):
        prior = (w @ cond[:, state].T + offset).clip(0.005, 0.995)
        value = prior.copy()
        for j in range(len(mu)):
            take = idx[(domain[idx] == j) & (anchor == state)]
            yy = y[:, take]
            value[:, j] = (np.nansum(yy, axis=1) + shrink * prior[:, j]) / (np.isfinite(yy).sum(1) + shrink)
        pred.append(value)
    chance = mu[:, pack["anchor"]]
    intercept = 0.5 * (w @ mu.T + offset) + 0.5 * ((1 - chance) * pred[0] + chance * pred[1])
    return np.column_stack((w, intercept, pred[1] - pred[0]))


def scores(pack, z, chance, peer, domain, cfg):
    r, d = peer.shape[1], len(pack["means"])
    anchor = pack["anchor"]
    mu = pack["means"][domain]
    raw = peer[:, anchor].clip(0.005, 0.995)
    # A small amount of smoothing protects source-specific tails of the calibrated verifier.
    probability = expit(0.85 * logit(chance.clip(0.005, 0.995)) + 0.15 * logit(raw))
    historical_slope = (pack["conditional"][:, 1] - pack["conditional"][:, 0])[domain]
    independent = peer - mu - (raw - mu[:, anchor])[:, None] * historical_slope
    conditional = z[:, r + d :][:, domain] * (probability - mu[:, anchor])
    return z[:, r : r + d][:, domain] + cfg["signal"] * (
        conditional + cfg["peer"] * (z[:, :r] @ independent.T)
    )


def calibrate(pack, h, y, idx, domain, oof, chance, cfg, seed):
    """Calibrate switches against the leading candidate using paired differences on withheld panel folds."""
    rng = np.random.default_rng(seed + 9719)
    folds = np.empty(len(idx), int)
    state = 2 * domain[idx] + h[pack["anchor"], idx].astype(int)
    for j in np.unique(state):
        ids = rng.permutation(np.flatnonzero(state == j))
        folds[ids] = (np.arange(len(ids)) + j) % 4
    variants = list(product((0.0, 0.55, 1.0, 1.65, 2.6), (0.35, 1.0)))
    loss = np.zeros((len(variants), len(idx)))
    r, d = h.shape[0], len(pack["means"])
    mass = np.bincount(domain) / len(domain)
    for k in range(4):
        a, b = idx[folds != k], idx[folds == k]
        z = embed(pack, h, y, a, domain, cfg["nearest"], cfg["shrink"])
        ability = z[:, r : r + d]
        leader = int((ability @ mass).argmax())
        mix = softmax(12 * ability[:, domain[b]], axis=0)
        for j, (signal, weight) in enumerate(variants):
            pred = scores(pack, z, chance[b], oof[b], domain[b], dict(signal=signal, peer=weight))
            residual = pred - y[:, b]
            loss[j, folds == k] = np.nansum(mix * (residual - residual[leader]) ** 2, axis=0)
    delta = loss - loss[loss.mean(1).argmin()]
    weight = softmax(-delta.mean(1) / (delta.std(1) / np.sqrt(len(idx))).clip(0.00035))
    signal, interaction = weight @ np.asarray(variants)
    return dict(signal=float(signal), peer=float(interaction))
