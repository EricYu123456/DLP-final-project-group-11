"""Fit an unseen LLM's ability coordinates from its answers on shared historical probes."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CapabilityBasis:
    axes: np.ndarray
    domain_count: int

    @classmethod
    def fit(cls, history, domains):
        """PCA over historical LLMs, giving each benchmark equal weight."""
        centered = history.astype(np.float64) - history.mean(axis=0, dtype=np.float64)
        weights = 1 / np.bincount(domains)[domains]
        _, vectors = np.linalg.eigh((centered * weights) @ centered.T)
        # Centering across LLMs removes one dimension.
        return cls(vectors[:, :0:-1].copy() / np.sqrt(len(history)), len(np.unique(domains)))

    def design(self, historical_scores, domains, rank, domain_scale):
        """Scores are observed on probes and predicted from text on incoming questions."""
        rows = np.asarray(historical_scores, dtype=np.float64)
        centered = rows - rows.mean(axis=1, keepdims=True)
        domains = np.asarray(domains)
        memberships = np.eye(self.domain_count)[domains] if domains.ndim == 1 else domains
        return np.column_stack((centered @ self.axes[:, :rank], domain_scale * memberships))


def fit_profiles(design, history, responses, ridge, prior=None):
    """Bayesian ridge with a common item-difficulty offset and one embedding per LLM."""
    mean = history.mean(axis=1, dtype=np.float64)
    outcomes = np.atleast_2d(responses) - mean
    if prior is None:
        prior = np.zeros((len(outcomes), design.shape[1]))
    outcomes = outcomes - prior @ design.T
    penalty = ridge * np.eye(design.shape[1])
    if np.isfinite(outcomes).all():
        return prior + np.linalg.solve(design.T @ design + penalty, design.T @ outcomes.T).T
    fitted = []
    for row in outcomes:
        observed = np.isfinite(row)
        x = design[observed]
        fitted.append(np.linalg.solve(x.T @ x + penalty, x.T @ row[observed]))
    return prior + np.stack(fitted)


def neighbor_prior(basis, history, responses, rank, temperature):
    """Center an unseen model's prior on historical LLMs with similar probe answers."""
    if temperature == 0:
        return None
    distance = np.nanmean(np.abs(responses[:, :, None] - history[None, :, :]), axis=1)
    weights = np.exp(-(distance - distance.min(axis=1, keepdims=True)) / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    coords = (weights @ basis.axes[:, :rank]) * len(basis.axes)
    return np.pad(coords, ((0, 0), (0, basis.domain_count)))


def fingerprint_profiles(history, domains, responses, n_domains, bandwidth, ridge, correction):
    """A model embedding with peer-interpolation weights and a paired offset per benchmark."""
    h = np.asarray(history, dtype=np.float64).T
    y = np.atleast_2d(responses).astype(np.float64)
    if not np.isfinite(y).all():
        fits = []
        for row in y:
            seen = np.isfinite(row)
            if not seen.any():
                raise ValueError("A candidate needs at least one observed probe answer.")
            fits.append(
                fingerprint_profiles(
                    h[:, seen].T, domains[seen], row[seen], n_domains, bandwidth, ridge, correction
                )[0]
            )
        return np.stack(fits)
    n, n_probes = h.shape
    all_answers = np.vstack((h, y))
    groups = [domains == j for j in range(n_domains)]
    ability = np.column_stack(
        [all_answers[:, g].mean(1) if g.any() else np.zeros(len(all_answers)) for g in groups]
    )
    ability = (ability - ability[:n].mean(0)) / ability[:n].std(0).clip(0.025)
    disagreement = (all_answers.sum(1, keepdims=True) + h.sum(1) - 2 * all_answers @ h.T) / n_probes
    centered = all_answers - h.mean(0)
    kernel = (
        0.35 * ability @ ability[:n].T / n_domains
        + 3 * centered @ centered[:n].T / n_probes
        + np.exp(-disagreement / bandwidth)
    )
    kernel = kernel - kernel[:n].mean(0) - kernel.mean(1, keepdims=True) + kernel[:n].mean()
    weights = np.linalg.solve(kernel[:n] + ridge * np.eye(n), kernel[n:].T).T
    weights += (1 - weights.sum(1, keepdims=True)) / n
    errors = y - weights @ h
    offsets = np.column_stack([errors[:, g].mean(1) if g.any() else errors.mean(1) for g in groups])
    return np.column_stack((weights, correction * offsets))


def fit_capabilities(basis, history, domains, responses, cfg):
    if cfg.get("algorithm") == "fingerprint":
        return fingerprint_profiles(
            history, domains, responses, basis.domain_count, cfg["bandwidth"], cfg["ridge"], cfg["correction"]
        )
    design = basis.design(history, domains, cfg["rank"], cfg["domain_scale"])
    prior = neighbor_prior(basis, history, responses, cfg["rank"], cfg["temperature"])
    return fit_profiles(design, history, responses, cfg["ridge"], prior)


def capability_query(basis, domain_means, historical_scores, membership, cfg):
    mean = membership @ domain_means
    scores = mean + cfg["signal"] * (historical_scores - mean)
    if cfg.get("algorithm") == "fingerprint":
        return np.column_stack((scores, membership))
    return basis.design(scores, membership, cfg["rank"], cfg["domain_scale"])


def informative_order(basis, history, domains, budget, seed, rank=20):
    """Select distinct probes by their D-optimal information in the historical ability space."""
    design = basis.design(history.T, domains, rank, 0.08)
    inverse = np.diag(np.full(design.shape[1], 1 / 0.14))
    leverage = np.einsum("ij,ij->i", design @ inverse, design)
    rng = np.random.default_rng(seed)
    order = []
    for step in range(budget):
        pool = np.flatnonzero((domains == step % basis.domain_count) & np.isfinite(leverage))
        # Randomize within a small set of equally informative alternatives for repeated probe draws.
        n = min(12, len(pool))
        best = pool[np.argpartition(leverage[pool], -n)[-n:]]
        index = int(rng.choice(best))
        direction = inverse @ design[index]
        divisor = 1 + design[index] @ direction
        leverage -= (design @ direction) ** 2 / divisor
        inverse -= np.outer(direction, direction) / divisor
        leverage[index] = -np.inf
        order.append(index)
    return np.asarray(order)


def representative_order(history, domains, budget, seed):
    """Herd historical response patterns toward each benchmark's population mean."""
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(domains == d) for d in np.unique(domains)]
    centered = [history[:, p].T.astype(np.float64) - history[:, p].mean(axis=1) for p in pools]
    residuals = [np.zeros(len(history)) for _ in pools]
    available = [np.ones(len(p), dtype=bool) for p in pools]
    norm = [(x * x).sum(1) for x in centered]
    order = []
    for step in range(budget):
        d = step % len(pools)
        x = centered[d]
        error = norm[d] + 2 * (x @ residuals[d])
        error[~available[d]] = np.inf
        top = min(8, int(available[d].sum()))
        i = int(rng.choice(np.argpartition(error, top - 1)[:top]))
        available[d][i] = False
        residuals[d] += x[i]
        order.append(pools[d][i])
    return np.asarray(order)


def stratified_order(basis, history, domains, question_vectors, budget, seed):
    """Sample once per equal-size stratum, alternating historical ability and text features."""
    rng = np.random.default_rng(seed)
    per_domain = []
    for domain in range(basis.domain_count):
        pool = np.flatnonzero(domains == domain)
        h = history[:, pool].T
        q = question_vectors[pool]
        strong = np.argsort(h.mean(0))[-20:]
        skills = (h - h.mean(1, keepdims=True)) @ basis.axes[:, :8]
        ability = np.column_stack((h[:, strong].mean(1), h.mean(1), skills))
        text = (q - q.mean(0)) @ rng.normal(size=(q.shape[1], 6))
        features = np.column_stack((ability / (ability.std(0) + 0.01), text / (text.std(0) + 0.01)))

        def sample(indices, count, depth):
            if count == 1:
                return [rng.choice(indices)]
            cols = (
                np.arange(ability.shape[1], features.shape[1]) if depth % 2 else np.arange(ability.shape[1])
            )
            column = cols[features[np.ix_(indices, cols)].var(0).argmax()]
            shuffled = rng.permutation(indices)
            ordered = shuffled[np.argsort(features[shuffled, column], kind="stable")]
            left = count // 2
            cut = round(len(indices) * left / count)
            return sample(ordered[:cut], left, depth + 1) + sample(ordered[cut:], count - left, depth + 1)

        leaves = sample(np.arange(len(pool)), budget // basis.domain_count, 0)
        per_domain.append(pool[rng.permutation(leaves)])
    return np.stack(per_domain).T.ravel()


@dataclass
class KernelReadout:
    anchors: np.ndarray
    coefficients: np.ndarray
    n_models: int

    @staticmethod
    def features(vectors, anchors):
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        similarities = vectors @ anchors.T
        return np.column_stack(
            (np.ones(len(vectors), dtype=np.float32), vectors, np.exp(5 * (similarities - 1)))
        )

    @classmethod
    def fit(cls, vectors, history, domains):
        rng = np.random.default_rng(401)
        selected = np.concatenate(
            [rng.choice(np.flatnonzero(domains == d), 96, replace=False) for d in np.unique(domains)]
        )
        anchors = vectors[selected].copy()
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        x = cls.features(vectors, anchors)
        targets = np.column_stack((history.T, np.eye(domains.max() + 1, dtype=np.float32)[domains]))
        weights = (len(domains) / (len(np.unique(domains)) * np.bincount(domains)[domains])) ** 0.5
        wx = (weights[:, None] * x).T
        normal = wx @ x
        normal.flat[:: len(normal) + 1] += 3
        return cls(anchors, np.linalg.solve(normal, wx @ targets).astype(np.float32), len(history))

    def predict(self, vectors):
        scores = self.features(vectors, self.anchors) @ self.coefficients
        domains = scores[:, self.n_models :].clip(0, 1)
        domains /= np.maximum(domains.sum(axis=1, keepdims=True), 1e-8)
        return scores[:, : self.n_models].clip(0, 1), domains
