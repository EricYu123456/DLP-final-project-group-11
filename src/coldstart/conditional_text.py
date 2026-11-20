"""Predict historical model strengths from prompt content and optional independent answer evidence."""

from itertools import combinations
import re

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import randomized_svd

from .capability import KernelReadout
from .topic_readout import subjects


ASCII_DIGIT = re.compile(r"[0-9]")
CUES = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:not|except|false|least)\b",
        r"\b(?:if|unless|assume|implies)\b",
        r"\b(?:patient|woman|man|child|years old)\b",
        r"\b(?:following|which|what)\b",
        r"\b(?:how many|how much|percent|probability|ratio)\b",
        r"\b(?:most likely|best|primary)\b",
    )
)


def lexical(texts, vocab):
    index = {v: i for i, v in enumerate(vocab)}
    one = np.zeros((len(texts), len(vocab)), np.float32)
    if index:
        for i, name in enumerate(subjects(texts)):
            if name in index:
                one[i, index[name]] = 1
    details = []
    for t in texts:
        low = t.lower()
        details.append(
            [
                len(t),
                len(t.split()),
                t.count("\n"),
                t.count("?"),
                len(ASCII_DIGIT.findall(t))
                if t.isascii()
                else sum(c.isdigit() for c in t),
                *(len(pattern.findall(low)) for pattern in CUES),
            ]
        )
    return np.column_stack((one, np.log1p(details))).astype(np.float32)


def consensus_features(proxies):
    feats, words = [], []
    for probabilities, answers in proxies:
        p = probabilities[:, :, :4]
        avg = p.mean(1)
        probs = np.sort(avg, axis=1)
        feats.extend(
            (
                probabilities.reshape(len(p), -1),
                avg,
                probs[:, -1:] - probs[:, -2:-1],
                np.abs(p[:, 0] - p[:, 1]).sum(1, keepdims=True),
            )
        )
        words.append(
            [set(re.findall(r"\d+(?:\.\d+)?|[a-z]+", s.lower())) for s in answers]
        )
    for a, b in combinations(range(len(proxies)), 2):
        p = proxies[a][0][:, :, :4].mean(1)
        q = proxies[b][0][:, :, :4].mean(1)
        agreement = [
            (
                len(left & right) / max(1, len(left | right)),
                int(bool(left & right) and left == right),
                int(bool({v for v in left & right if any(c.isdigit() for c in v)})),
            )
            for left, right in zip(words[a], words[b])
        ]
        feats.extend(
            (
                (p * q).sum(1, keepdims=True),
                (p.argmax(1) == q.argmax(1))[:, None],
                np.asarray(agreement),
            )
        )
    return np.column_stack(feats).astype(np.float32)


def text_pack(x, texts):
    mean = x.mean(0)
    _, _, axes = randomized_svd(x - mean, 48, n_iter=3, random_state=1349)
    return dict(mean=mean, axes=axes, vocabulary=np.unique(subjects(texts)))


def text_features(pack, vectors, texts, evidence=None):
    out = [
        (vectors - pack["mean"]) @ pack["axes"].T,
        lexical(texts, pack["vocabulary"]),
    ]
    if evidence:
        out.append(consensus_features(evidence))
    return np.column_stack(out).astype(np.float32)


class ConditionalReadout:
    """A shared kernel and tree readout. Regional training uses cross-fitted predictions."""

    @classmethod
    def fit(cls, vectors, features, history, domains):
        model = cls()
        model.kernel = KernelReadout.fit(vectors, history, domains)
        n_domains = int(domains.max()) + 1
        model.means = np.stack(
            [history[:, domains == i].mean(1) for i in range(n_domains)]
        )
        rel = history.T - model.means[domains]
        difficulty = rel.mean(1, keepdims=True)
        residual = rel - difficulty
        _, _, model.axes = randomized_svd(
            residual, min(24, len(history) - 1), n_iter=4, random_state=1349
        )
        targets = np.column_stack((difficulty, residual @ model.axes.T))
        model.scale = targets.std(0).clip(0.06)
        model.forest = ExtraTreesRegressor(
            n_estimators=88,
            max_features=0.65,
            min_samples_leaf=14,
            max_depth=15,
            random_state=1349,
            n_jobs=5,
        )
        wt = np.sqrt(len(domains) / (n_domains * np.bincount(domains)[domains]))
        model.forest.fit(features, targets / model.scale, sample_weight=wt)
        return model

    def predict(self, vectors, features):
        scores, mem = self.kernel.predict(vectors)
        coords = self.forest.predict(features) * self.scale
        forest = mem @ self.means + coords[:, :1] + coords[:, 1:] @ self.axes
        return (0.48 * scores + 0.52 * forest).clip(0, 1), mem
