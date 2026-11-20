"""Partition questions to maximize routed accuracy in historical LLM tournaments."""

import numpy as np
from scipy.special import softmax

from .conditional_capability import paired_profile


def tournaments(h, rng):
    means = h.mean(1)
    order = np.argsort(-means)
    pools = []
    for _ in range(22):
        # Include near ties and pools with an incumbent ahead on aggregate.
        n = int(rng.choice((8, 16, len(h))))
        rows = rng.choice(order[:n], 4, replace=False)
        pools.append(rows)
        anchor = rng.choice(order[:10])
        weaker = np.flatnonzero(means < means[anchor] - 0.04)
        if len(weaker) >= 3:
            pools.append(np.r_[anchor, rng.choice(weaker, 3, replace=False)])
    return np.stack(pools)


def build_tree(x, h, indices, estimation, base, pools, rng, min_leaf):
    nodes = []

    def branch(rows, hold, depth, parent):
        index = len(nodes)
        leaf = (h[:, hold].sum(1) + 14 * parent) / (len(hold) + 14)
        nodes.append((None, None, None, None, leaf))
        if depth >= 5 or len(rows) < 2 * min_leaf:
            return index
        feature = rng.choice(x.shape[1], min(28, x.shape[1]), replace=False)
        sample = x[np.ix_(rows, feature)]
        levels = np.quantile(sample, [0.20, 0.38, 0.55, 0.78], axis=0)
        masks = (sample.T[None, :, :] <= levels[:, :, None]).reshape(-1, len(rows))
        count = masks.sum(1)
        usable = (count >= min_leaf) & (len(rows) - count >= min_leaf)
        if not usable.any():
            return index
        mask = masks[usable].astype(np.float32)
        count = count[usable, None]
        yy = h[:, rows].T
        sums = mask @ yy
        left = (sums + 22 * base) / (count + 22)
        right = (yy.sum(0) - sums + 22 * base) / (len(rows) - count + 22)
        weight = count / len(rows)
        value = (weight * left[:, pools].max(2) + (1 - weight) * right[:, pools].max(2)).mean(1)
        # Preserve a relative-response signal in leaves where all sampled tournaments have the same winner.
        contrast = left - right
        contrast -= contrast.mean(1, keepdims=True)
        score = value + 0.60 * (weight * (1 - weight)).ravel() * np.square(contrast).mean(1)
        best = np.flatnonzero(usable)[score.argmax()]
        j = feature[best % len(feature)]
        cut = levels.reshape(-1)[best]
        train_left = masks[best]
        hold_left = x[hold, j] <= cut
        a = branch(rows[train_left], hold[hold_left], depth + 1, leaf)
        b = branch(rows[~train_left], hold[~hold_left], depth + 1, leaf)
        nodes[index] = (j, cut, a, b, leaf)
        return index

    branch(indices, estimation, 0, base)
    return nodes


class PolicyForest:
    @classmethod
    def fit(cls, x, h, dom, trees=36):
        model = cls()
        rng = np.random.default_rng(4197)
        h = h.astype(np.float32)
        model.means = np.stack([h[:, dom == j].mean(1) for j in np.unique(dom)])
        model.trees = []
        for j, base in enumerate(model.means):
            pool = np.flatnonzero(dom == j)
            groups = []
            for _ in range(trees):
                order = rng.permutation(pool)
                n = int(0.58 * len(pool))
                train, hold = order[:n], order[n:]
                rosters = tournaments(h[:, train], rng)
                groups.append(
                    build_tree(x, h, train, hold, base, rosters, rng, min(34, max(14, len(train) // 28)))
                )
            model.trees.append(groups)
        return model

    def predict(self, x, dom):
        n = len(self.trees[0])
        leaves = np.zeros((len(x), n), np.int32)
        predicted = np.zeros((len(x), self.means.shape[1]), np.float32)
        for j, group in enumerate(self.trees):
            pool = np.flatnonzero(dom == j)
            for t, nodes in enumerate(group):
                stack = [(0, pool)]
                while stack:
                    k, rows = stack.pop()
                    if not len(rows):
                        continue
                    feature, cut, left, right, means = nodes[k]
                    if feature is None:
                        leaves[rows, t] = 128 * j + k
                        predicted[rows] += means / n
                    else:
                        keep = x[rows, feature] <= cut
                        stack.extend(((left, rows[keep]), (right, rows[~keep])))
        return predicted, leaves


def leaf_similarity(left, right):
    result = np.zeros((len(left), len(right)), np.float32)
    for j in range(left.shape[1]):
        result += (left[:, j, None] == right[:, j]) / left.shape[1]
    return result


def fit_forest_profile(h, y, idx, domains, leaves, penalty, nearest):
    pop = np.bincount(domains)
    chosen = leaves[idx]
    kernel = leaf_similarity(chosen, chosen)
    output = []
    for row in y:
        finite = np.isfinite(row[idx])
        subset = idx[finite]
        source = domains[subset]
        hist = h[:, subset]
        answers = row[subset]
        w, mass = paired_profile(hist, answers[None], source, pop, np.arange(len(pop)))
        dist = abs(hist - answers) @ mass
        peer = (1 - nearest) * w[0] + nearest * softmax(-dist / 0.032)
        error = answers - peer @ hist
        by_source = (np.bincount(source, weights=error, minlength=len(pop)) + 9 * error.mean()) / (
            np.bincount(source, minlength=len(pop)) + 9
        )
        weights = np.zeros(len(idx))
        g = kernel[np.ix_(finite, finite)] + penalty * np.eye(len(subset))
        weights[finite] = np.linalg.solve(g, error - by_source[source])
        output.append(np.r_[peer, by_source, weights])
    return np.stack(output)


def forest_scores(z, means, pred, dom, similarity, cfg):
    d, h = means.shape
    peer = z[:, :h]
    avg = means[dom]
    contrast = pred - avg
    return (
        peer @ (avg + cfg["signal"] * contrast).T
        + z[:, h : h + d][:, dom]
        + cfg["residual"] * (z[:, h + d :] @ similarity.T)
    )
