"""A CPU question readout for relative model strengths within a benchmark."""

import re
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.extmath import randomized_svd


SUBJECT = re.compile(r"multiple choice questions \(with answers\) about ([^.\n]+)", re.I)
CLUSTERS = (5, 18, 45)


def normalized(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def subjects(texts):
    return [match.group(1).lower() if (match := SUBJECT.search(text)) else "general" for text in texts]


def fit_topics(x, texts, domains):
    x = normalized(x)
    centers = []
    for j in np.unique(domains):
        data = x[domains == j]
        center = data.mean(0)
        for n in CLUSTERS:
            km = MiniBatchKMeans(n_clusters=n, batch_size=512, max_iter=35, n_init=2, random_state=471)
            km.fit(data - center)
            centers.append(km.cluster_centers_ + center)
    mean = x.mean(0)
    _, _, axes = randomized_svd(x - mean, 80, n_iter=3, random_state=471)
    return dict(
        centers=np.vstack(centers),
        widths=np.tile(CLUSTERS, len(np.unique(domains))),
        mean=mean,
        axes=axes,
        subjects=np.unique(subjects(texts)),
    )


def features(pack, x, texts):
    x = normalized(x)
    centers = pack["centers"]
    # Squared distance up to an additive query constant, cancelled by each softmax.
    logits = 16 * (2 * x @ centers.T - np.sum(centers**2, axis=1))
    groups = np.split(logits, np.cumsum(pack["widths"])[:-1], axis=1)
    vocab = {name: i for i, name in enumerate(pack["subjects"])}
    topics = [vocab.get(sub, vocab["general"]) for sub in subjects(texts)]
    lengths = [
        [len(q), len(q.split()), q.count("\n"), sum(c.isdigit() for c in q), q.count("?"), q.count(",")]
        for q in texts
    ]
    return np.column_stack(
        (
            (x - pack["mean"]) @ pack["axes"].T,
            np.eye(len(vocab))[topics],
            *(softmax(g, axis=1) for g in groups),
            np.log1p(lengths),
        )
    )


class TopicReadout:
    def __init__(self, folder: Path, stage: str):
        from catboost import CatBoostRegressor

        with np.load(folder / "topic_fit.npz") as f:
            self.pack = dict(f)
        self.boost = CatBoostRegressor()
        self.boost.load_model(str(folder / f"{stage}.cbm"))

    def predict(self, x, texts, membership):
        p = self.pack
        relative = self.boost.predict(features(p, x, texts), thread_count=2) * p["scale"]
        return membership @ p["domain_means"] + relative @ p["model_axes"].T, membership
