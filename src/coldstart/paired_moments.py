"""Represent a model by its conditional agreement with pairs of historical models."""

from itertools import combinations

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import randomized_svd


def joint_responses(h):
    left, right = np.triu_indices(len(h), 1)
    return np.column_stack((h.T, h[left].T * h[right].T)).astype(np.float32), left, right


class MomentReadout:
    """Decode first and second moments of historical binary outcomes using only question-side features."""

    @classmethod
    def fit(cls, x, dom, h):
        model = cls()
        values, left, right = joint_responses(h)
        means = np.stack([values[dom == j].mean(0) for j in np.unique(dom)])
        residual = values - means[dom]
        # Preserve individual response differences even though there are many more product features.
        scale = np.r_[np.full(len(h), 3.5), np.ones(len(left))].astype(np.float32)
        mass = 0.5 + 0.5 * len(dom) / (len(means) * np.bincount(dom)[dom])
        _, _, axes = randomized_svd(
            residual * scale * np.sqrt(mass[:, None]).astype(np.float32), 48, n_iter=4, random_state=6127
        )
        target = (residual * scale) @ axes.T
        target_scale = target.std(0).clip(0.08)
        target /= target_scale
        model.means, model.axes = means, axes / scale * target_scale[:, None]
        model.left, model.right = left, right
        model.center, model.std = x.mean(0), x.std(0).clip(0.15)
        z = np.tanh((x - model.center) / (2 * model.std))
        rng = np.random.default_rng(6127)
        centers = np.concatenate(
            [
                rng.choice(np.flatnonzero(dom == j), min(85, np.sum(dom == j)), replace=False)
                for j in range(len(means))
            ]
        )
        model.anchors = z[centers] / np.linalg.norm(z[centers], axis=1, keepdims=True).clip(1e-6)
        model.svd = randomized_svd(z, 60, n_iter=3, random_state=6127)[2]
        feat = model.features(z, dom)
        dual = feat.T * mass.astype(np.float32)
        gram = dual @ feat
        gram.flat[:: len(gram) + 1] += 8.0
        model.coefficient = np.linalg.solve(gram, dual @ target)
        model.forest = ExtraTreesRegressor(
            n_estimators=112,
            min_samples_leaf=18,
            max_features=0.55,
            max_depth=16,
            n_jobs=5,
            random_state=6127,
        )
        model.forest.fit(np.column_stack((x, dom)), target, sample_weight=np.sqrt(mass))
        return model

    def features(self, z, dom):
        norms = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-6)
        kernel = np.exp(4 * (norms @ self.anchors.T - 1))
        low = z @ self.svd.T
        sources = np.eye(len(self.means), dtype=np.float32)[dom]
        return np.column_stack((z, kernel, (low[:, None, :] * sources[:, :, None]).reshape(len(z), -1)))

    def predict(self, x, dom):
        z = np.tanh((x - self.center) / (2 * self.std))
        linear = self.features(z, dom) @ self.coefficient
        forest = self.forest.predict(np.column_stack((x, dom)))
        return 0.55 * linear + 0.45 * forest


def pair_candidates(observed, target, domains):
    distance = np.abs(observed - target)
    by_source = np.stack([distance[:, domains == j].mean(1) for j in np.unique(domains)])
    overall = by_source.mean(0)
    peers = np.unique(np.r_[overall.argsort()[:7], by_source.argsort(1)[:, :2].ravel()])
    return np.array(list(combinations(peers, 2)), dtype=int)


def fit_pair_table(states, answers, domain, domain_count, pseudo):
    n_pair = len(states)
    marg = np.bincount(domain, weights=answers, minlength=domain_count)
    counts = np.bincount(domain, minlength=domain_count)
    marginal = (marg + 6 * answers.mean()) / (counts + 6)
    global_success = np.stack([(states == k) @ answers for k in range(4)], axis=1)
    global_count = np.stack([(states == k).sum(1) for k in range(4)], axis=1)
    # A modest monotone prior lets an unobserved joint event borrow from the two binary indicators.
    prior = (1.5 * answers.mean() + np.array([0, 1, 1, 2])) / 3.5
    global_rate = (global_success + 5 * prior) / (global_count + 5)
    table = np.empty((domain_count, n_pair, 4))
    for j in range(domain_count):
        mask = domain == j
        local = states[:, mask]
        count = np.stack([(local == k).sum(1) for k in range(4)], axis=1)
        hits = np.stack([(local == k) @ answers[mask] for k in range(4)], axis=1)
        conditional = (global_rate + 0.6 * (marginal[j] - answers.mean())).clip(0.015, 0.985)
        table[j] = (hits + pseudo * conditional) / (count + pseudo)
    return table


def measure(h, y, idx, domain, model, pseudo, sharpness):
    """Cross-validate each two-peer response table on the newcomer's own panel and average the tables."""
    d = len(model.means)
    pair_lookup = np.zeros((len(h), len(h)), int)
    pair_lookup[model.left, model.right] = len(h) + np.arange(len(model.left))
    encoded = []
    for row in y:
        keep = idx[np.isfinite(row[idx])]
        answers = row[keep].astype(np.float64)
        sample = h[:, keep].astype(int)
        dt = domain[keep]
        pairs = pair_candidates(sample, answers, dt)
        a, b = pairs.T
        state = 2 * sample[a] + sample[b]
        loss = np.zeros((len(pairs), len(keep)))
        folds = np.zeros(len(keep), int)
        for j in range(d):
            mask = np.flatnonzero(dt == j)
            folds[mask] = np.arange(len(mask)) % 4
        for f in range(4):
            tr, va = folds != f, folds == f
            table = fit_pair_table(state[:, tr], answers[tr], dt[tr], d, pseudo)
            probability = table[dt[None, va], np.arange(len(pairs))[:, None], state[:, va]].clip(0.015, 0.985)
            target = answers[va]
            loss[:, va] = -(target * np.log(probability) + (1 - target) * np.log1p(-probability))
        table = fit_pair_table(state, answers, dt, d, pseudo)
        score = np.stack([loss[:, dt == j].mean(1) for j in range(d)])
        goodness = -(0.68 * score.mean(0) + 0.32 * score)
        weights = softmax(sharpness * goodness, axis=1)
        tables = []
        for j in range(d):
            v = table[j]
            coef = np.zeros(model.means.shape[1])
            np.add.at(coef, a, weights[j] * (v[:, 2] - v[:, 0]))
            np.add.at(coef, b, weights[j] * (v[:, 1] - v[:, 0]))
            np.add.at(coef, pair_lookup[a, b], weights[j] * (v[:, 3] - v[:, 2] - v[:, 1] + v[:, 0]))
            mean = weights[j] @ v[:, 0] + coef @ model.means[j]
            observed = table[j, np.arange(len(pairs))[:, None], state[:, dt == j]]
            correction = (answers[dt == j] - weights[j] @ observed).sum() / ((dt == j).sum() + 15)
            tables.append(np.r_[mean + correction, model.axes @ coef])
        encoded.append(tables)
    return np.stack(encoded)


def score_moments(embedding, coordinates, domain, signal):
    rows = embedding[:, domain]
    return rows[:, :, 0] + signal * np.einsum("mqk,qk->mq", rows[:, :, 1:], coordinates)
