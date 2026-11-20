"""Decode model capability embeddings directly from a new question's words and format on CPU."""

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

from coldstart.character_tfidf import character_tfidf
from coldstart.conditional_text import lexical
from panel_capabilities import fitted_templates, template_membership, template_probabilities
from panel_rasch import subtype_matrix
from panel_tabular_skills import ability_factors


def surface(texts):
    return [t if len(t) <= 2200 else t[:420] + " " + t[-1780:] for t in texts]


def fit_kernel(texts, domains):
    words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=16000, dtype=np.float32)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=12000, dtype=np.float32)
    samples = surface(texts)
    numeric = lexical(texts, [])
    scales = numeric.std(0).clip(0.25)
    return dict(
        words=words,
        chars=chars,
        word_support=words.fit_transform(samples).T,
        char_support=chars.fit_transform(samples).T,
        scale=scales,
        numeric=np.column_stack((numeric / scales, subtype_matrix(texts))),
        domains=np.asarray(domains),
        texts=np.asarray(texts),
    )


def kernel(pack, texts, member):
    samples = surface(texts)
    word = (pack["words"].transform(samples) @ pack["word_support"]).toarray()
    char = (character_tfidf(pack["chars"], samples) @ pack["char_support"]).toarray()
    features = np.column_stack((lexical(texts, []) / pack["scale"], subtype_matrix(texts)))
    format_similarity = np.exp(-cdist(features, pack["numeric"], "sqeuclidean") / features.shape[1])
    same_source = member[:, pack["domains"]]
    return (0.5 * word + 0.25 * char + 0.25 * format_similarity) * (
        0.3 + 0.7 * same_source
    ) + 0.5 * same_source


def fit_text_kernel(dom, grades, texts, meta, size, seed, return_mix=False):
    regularizers = np.asarray([0.15, 0.75, 3.0])
    signals = np.asarray([0.0, 0.4, 1.0, 1.8])
    risks = np.empty((len(regularizers), len(signals), len(dom)))
    rng = np.random.default_rng(seed + 723)
    fold = np.empty(len(dom), int)
    sources = np.eye(int(dom.max()) + 1)
    for source in np.unique(dom):
        rows = rng.permutation(np.flatnonzero(dom == source))
        fold[rows] = np.arange(len(rows)) % 3
    for held in range(3):
        a, b = fold != held, fold == held
        z, axes, target = ability_factors(dom[a], grades[:, a], meta, size)
        support = fit_kernel(texts[a], dom[a])
        gram = kernel(support, texts[a], sources[dom[a]])
        member = template_membership(texts[a], texts[b], dom[a])
        cross = kernel(support, texts[b], member)
        mean = z @ member.T
        contenders = np.argsort(-mean, axis=0)[:15]
        for j, penalty in enumerate(regularizers):
            weight = np.linalg.solve(gram + penalty * np.eye(len(gram)), target)
            shift = axes @ (cross @ weight).T
            for k, scale in enumerate(signals):
                error = np.take_along_axis(mean + scale * shift - grades[:, b], contenders, axis=0)
                risks[j, k, b] = ((error - error.mean(0)) ** 2).mean(0)
    loss = risks.mean(2)
    delta = risks - risks[np.unravel_index(loss.argmin(), loss.shape)]
    probability = softmax(-delta.mean(2) / np.maximum(0.00015, delta.std(2) / np.sqrt(len(dom))))
    z, axes, target = ability_factors(dom, grades, meta, size)
    support = fit_kernel(texts, dom)
    gram = kernel(support, texts, sources[dom])
    # Combine the within-panel settings into one readout for ordinary CPU serving.
    weights = sum(
        (probability[j] @ signals) * np.linalg.solve(gram + p * np.eye(len(gram)), target)
        for j, p in enumerate(regularizers)
    )
    detail = dict(within_panel_risk=float(loss.min()))
    if return_mix:
        detail.update(regularizers=regularizers, coefficients=probability @ signals)
    return dict(text_kernel=support, z=z, axes=axes, weights=weights), detail


def text_kernel_scores(pack, texts):
    support = pack["text_kernel"]
    if "source_readout" not in support:
        support["source_readout"] = fitted_templates(tuple(support["texts"]), tuple(support["domains"]))
    member = template_probabilities(support["source_readout"], texts)
    requirements = kernel(support, texts, member) @ pack["weights"]
    return pack["z"] @ member.T + pack["axes"] @ requirements.T


def score_margin(scores):
    if len(scores) == 1:
        return np.full(scores.shape[1], np.inf)
    top = np.partition(scores, -2, axis=0)[-2:]
    return top[1] - top[0]
