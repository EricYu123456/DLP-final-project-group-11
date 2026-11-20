"""Fit and route model-capability embeddings using only a small shared graded panel."""

from functools import lru_cache
from itertools import product
import re

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

from coldstart.character_tfidf import character_tfidf

RELEASE_SEEDS = (11, 23, 37)


def normalized(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-7)


def panel_text(ids, raw):
    from embed_new_model import read_prompts

    found = read_prompts(ids, raw / "question_order.csv")
    return np.asarray([found[int(i)] for i in ids])


def template_views(prompts):
    markers = r"(?:^|\n)\s*(?:[A-E][.)]:?|(?:Question|Answer|Passage|Choices|Q|A):)|\?|\n|\$|\d+[%/]"
    return (
        [p[:230] for p in prompts],
        [p[-120:] for p in prompts],
        [" ".join(re.findall(markers, p, re.I)) for p in prompts],
    )


@lru_cache(maxsize=1)
def fitted_templates(train, domains):
    gram = np.zeros((len(train), len(train)))
    views = []
    for left in template_views(train):
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=6000, sublinear_tf=True)
        support = vec.fit_transform(left)
        gram += (support @ support.T).toarray() / 3
        views.append((vec, support))
    coef = np.linalg.solve(gram + 0.1 * np.eye(len(train)), np.eye(max(domains) + 1)[list(domains)])
    return [(vec, support.T @ coef) for vec, support in views]


@lru_cache(maxsize=1)
def marker_readout(vectorizer):
    # A newly fitted vectorizer has a new set of source weights.
    return {}


def template_membership(train, query, domains):
    """Learn a source readout on the panel once and apply it to each batch of new questions."""
    views = fitted_templates(tuple(train), tuple(domains))
    return template_probabilities(views, query)


def template_probabilities(views, query):
    prefix, suffix, markers = template_views(query)
    logits = sum(
        (character_tfidf(vec, text) @ weights) / 3 for (vec, weights), text in zip(views[:2], (prefix, suffix))
    )
    vec, weights = views[2]
    cache = marker_readout(vec)
    if len(cache) >= 1024:
        marker_readout.cache_clear()
        cache = marker_readout(vec)
    missing = list(dict.fromkeys(key for key in markers if key not in cache))
    if missing:
        cache.update(zip(missing, (character_tfidf(vec, missing) @ weights) / 3))
    logits += np.asarray([cache[key] for key in markers])
    return softmax(12 * logits, axis=1)


def peer_skills(y, domains, penalty):
    """Denoise a benchmark's ability using peer similarity measured on the other benchmarks."""
    from run_routing import benchmark_means

    estimates = []
    for j in np.unique(domains):
        other = domains != j
        h = y[:, other]
        scores = benchmark_means(h, domains[other])
        z = (scores - scores.mean(0)) / scores.std(0).clip(0.03)
        residual = h - h.mean(0)
        k = (
            0.35 * z @ z.T / z.shape[1]
            + 3 * residual @ residual.T / h.shape[1]
            + np.exp(-cdist(h, h, "hamming") / 0.16)
        )
        k = k - k.mean(0) - k.mean(1, keepdims=True) + k.mean()
        raw = y[:, ~other].mean(1)
        estimates.append(raw.mean() + k @ np.linalg.solve(k + penalty * np.eye(len(h)), raw - raw.mean()))
    return np.stack(estimates, axis=1)


def question_kernel(query, support, shape):
    cosine = normalized(query) @ support.T
    return 0.6 * cosine + 0.4 * cosine**2 if shape == "broad" else np.exp(6 * (cosine - 1))


def release_fit(x, groups, y, regularization, shape):
    """Empirical-Bayes domain capabilities plus low-rank semantic residuals, all fit within the panel."""
    from run_routing import benchmark_means

    x = normalized(x)
    d = int(groups.max()) + 1
    membership = np.eye(d)[groups]
    means = benchmark_means(y, groups)
    centered = y - y.mean(0)
    variance = np.column_stack(
        [centered[:, groups == j].var(1, ddof=1) / (groups == j).sum() for j in range(d)]
    )
    covariance = np.cov(means.T) - np.diag(variance.mean(0))
    eig, v = np.linalg.eigh(covariance)
    cov = (v * eig.clip(0.0006)) @ v.T
    mu = means.mean(0)
    abilities = np.stack(
        [
            mu + cov @ np.linalg.solve(cov + np.diag(0.5 * o + 0.5 * variance.mean(0)), m - mu)
            for m, o in zip(means, variance)
        ]
    )
    residual = centered - (abilities - abilities.mean(0)) @ membership.T
    u, _, _ = np.linalg.svd(residual, full_matrices=False)
    axes = u[:, :12]
    kernel = question_kernel(x, x, shape)
    coef = np.linalg.solve(kernel + regularization * np.eye(len(x)), (axes.T @ residual).T)
    return x, abilities, axes, coef


def combined_vectors(x, views, context=None):
    if context is None:
        context = (views.mean(0), views.std(0).clip(0.05))
    standardized = np.clip((views - context[0]) / context[1], -5, 5)
    features = np.column_stack(
        (
            0.7 * normalized(x),
            0.6 * normalized(standardized[:, :-60]),
            np.sqrt(0.15) * normalized(standardized[:, -60:]),
        )
    )
    return features, context


def pack_panel(x, domains, y, texts, ids, names, cfg, views=None):
    context = {}
    if cfg["encoder"] == "views":
        x, (context["view_mean"], context["view_scale"]) = combined_vectors(x, views)
    support, abilities, axes, coef = release_fit(x, domains, y, cfg["ridge"], cfg["kernel"])
    if cfg["peer_mix"]:
        peer = peer_skills(y, domains, cfg["peer_penalty"])
        abilities = (1 - cfg["peer_mix"]) * abilities + cfg["peer_mix"] * peer
    abilities = shrink_abilities(abilities, cfg.get("source_retention", 1.0))
    return dict(
        model_names=names,
        vectors=np.column_stack((abilities, axes)),
        support=support,
        coef=coef,
        signal=cfg["signal"],
        kernel=cfg["kernel"],
        encoder=cfg["encoder"],
        texts=texts,
        domains=domains,
        prompt_ids=ids,
        **context,
    )


def panel_design(router, x, texts, views=None):
    if "method" in router and str(router["method"]) == "rasch":
        from panel_rasch import rasch_design

        return rasch_design(router, x, texts)
    membership = template_membership(router["texts"], texts, router["domains"])
    if not float(router["signal"]):
        residual = np.zeros((len(texts), router["coef"].shape[1]))
        return np.column_stack((membership, residual))
    if str(router["encoder"]) == "views":
        x, _ = combined_vectors(x, views, (router["view_mean"], router["view_scale"]))
    k = question_kernel(x, router["support"], str(router["kernel"]))
    return np.column_stack((membership, float(router["signal"]) * (k @ router["coef"])))


def public_routes(router, x, texts, views=None):
    return (router["vectors"] @ panel_design(router, x, texts, views).T).argmax(0)


def shrink_abilities(abilities, retention):
    common = abilities.mean(1, keepdims=True) + abilities.mean(0) - abilities.mean()
    return retention * abilities + (1 - retention) * common


def calibrated_vectors(router, retention, residual_scale):
    vectors = router["vectors"].copy()
    d = int(router["domains"].max()) + 1
    vectors[:, :d] = shrink_abilities(vectors[:, :d], retention)
    vectors[:, d:] *= residual_scale
    return vectors


def select_panel(x, domains, y, texts, seed, views=None, retentions=(1.0,)):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(domains), int)
    for j in np.unique(domains):
        g = rng.permutation(np.flatnonzero(domains == j))
        fold[g] = np.arange(len(g)) % 4
    candidates = {}
    for i in range(4):
        tr, va = np.flatnonzero(fold != i), np.flatnonzero(fold == i)
        membership = template_membership(texts[tr], texts[va], domains[tr])
        peers = {penalty: peer_skills(y[:, tr], domains[tr], penalty) for penalty in (1.0, 4.0)}
        contenders = None
        alternatives = [("mpnet", x[tr], x[va])]
        if views is not None:
            v = views
            train, context = combined_vectors(x[tr], v[tr])
            validation, _ = combined_vectors(x[va], v[va], context)
            alternatives.append(("views", train, validation))
        for encoder, train, validation in alternatives:
            for reg, shape in product((0.5, 3.0), ("rbf", "broad")):
                support, abilities, axes, coef = release_fit(train, domains[tr], y[:, tr], reg, shape)
                base = abilities @ membership.T
                residual = axes @ (question_kernel(validation, support, shape) @ coef).T
                if contenders is None:
                    contenders = np.argsort(-base, axis=0)[:12]
                    left, right = np.triu_indices(len(contenders), k=1)
                    truth = np.take_along_axis(y[:, va], contenders, axis=0)
                    observed = truth[left] - truth[right]
                for mix, penalty in ((0.0, 1.0), (0.5, 1.0), (0.5, 4.0), (1.0, 4.0)):
                    ability = (1 - mix) * abilities + mix * peers[penalty]
                    for retention in retentions:
                        typical = shrink_abilities(ability, retention) @ membership.T
                        for signal in (0.0, 0.35, 1.0):
                            pred = typical + signal * residual
                            scores = np.take_along_axis(pred, contenders, axis=0)
                            loss = ((scores[left] - scores[right] - observed) ** 2).mean(0)
                            key = encoder, reg, signal, mix, penalty, shape, retention
                            candidates.setdefault(key, []).extend(loss.tolist())
    risks = {key: np.asarray(value) for key, value in candidates.items()}
    minimum = risks[min(risks, key=lambda key: risks[key].mean())]
    eligible = [
        key
        for key, loss in risks.items()
        if (loss - minimum).mean() <= (loss - minimum).std(ddof=1) / np.sqrt(len(loss))
    ]

    def cost(key):
        encoder, _, signal, *_ = key
        return 0 if not signal else (1 if encoder == "mpnet" else 2)

    encoder, reg, signal, mix, penalty, shape, retention = min(
        eligible, key=lambda key: (cost(key), risks[key].mean())
    )
    return dict(
        encoder=encoder,
        ridge=reg,
        kernel=shape,
        signal=signal,
        peer_mix=mix,
        peer_penalty=penalty,
        source_retention=retention,
        domain_readout="panel-template",
        probe_count=len(x),
        fit_selection="Four-fold contender-pair risk; prefer cheaper encoders within one paired standard error.",
    )
