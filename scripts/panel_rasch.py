"""Estimate conditional model abilities and shared item difficulties from one graded panel."""

import re

import numpy as np
from scipy.special import expit, logit

from panel_capabilities import normalized, question_kernel, template_membership


SUBJECT = re.compile(r"multiple choice questions \(with answers\) about ([^.\n]+)", re.I)
STEM = re.compile(
    r"algebra|astronomy|biolog|chemistr|computer|engineering|learning|mathematic|physics|statistics"
)
HUMANITIES = re.compile(r"history|philosophy|logic|law|jurisprudence|moral|religion|fallac")
SOCIAL = re.compile(
    r"economic|geograph|government|politic|psycholog|sexuality|sociolog|relations|security studies|foreign policy"
)


def prompt_subtype(text):
    """MMLU's four public subject groups and GPQA's visible elicitation formats."""
    subject = SUBJECT.match(text)
    if subject:
        name = subject.group(1).lower()
        for i, pattern in enumerate((STEM, HUMANITIES, SOCIAL)):
            if pattern.search(name):
                return i
        return 3
    examples = text.startswith("Here are some example questions from experts")
    if examples or text.startswith("What is the correct answer to this question:"):
        cot = "think step by step" in text[-55:].lower()
        return 4 + 2 * int(examples) + int(cot)
    return 8


def subtype_matrix(texts):
    return np.eye(9)[[prompt_subtype(t) for t in texts], :8]


def item_offsets(design, ability, item, y):
    p = expit(ability @ design.T + item)
    change = (y - p).sum(0) - 0.15 * item
    return item + np.clip(change / ((p * (1 - p)).sum(0) + 0.15), -1.2, 1.2)


def pack_rasch(x, domains, y, texts, ids, names, cfg):
    groups = np.eye(int(domains.max()) + 1)[domains]
    count = groups.sum(0)
    subtype = subtype_matrix(texts)
    centers = (groups.T @ subtype) / count[:, None]
    design = np.column_stack((groups, subtype - groups @ centers))
    n, d = len(y), len(count)
    # First estimate each source's marginal variance on the log-odds scale.
    item = logit((y.sum(0) + 0.5) / (n + 1))
    marginal = np.zeros((n, d))
    for _ in range(6):
        p = expit(marginal @ groups.T + item)
        info = (p * (1 - p)) @ groups + 0.35
        marginal += np.clip(((y - p) @ groups - 0.35 * marginal) / info, -1.2, 1.2)
        item = item_offsets(groups, marginal, item, y)
    cov = np.cov(marginal.T) - np.diag((1 / info).mean(0))
    eig, vec = np.linalg.eigh(0.93 * cov + 0.07 * np.diag(np.diag(cov)))
    precision = np.eye(design.shape[1]) * cfg["subtype_penalty"]
    precision[:d, :d] = (vec / eig.clip(0.045)) @ vec.T
    ability = np.column_stack((marginal, np.zeros((n, subtype.shape[1]))))
    for _ in range(7):
        p = expit(ability @ design.T + item)
        curvature = design.T[None, :, :] @ ((p * (1 - p))[:, :, None] * design)
        gradient = (y - p) @ design - ability @ precision
        delta = np.linalg.solve(curvature + precision, gradient[:, :, None])[:, :, 0]
        ability += np.clip(delta, -1.2, 1.2)
        item = item_offsets(design, ability, item, y)
    p = expit(ability @ design.T + item)
    residual = (y - p) / (p * (1 - p)).clip(0.14)
    residual -= residual.mean(0)
    axes = np.linalg.svd(residual, full_matrices=False)[0][:, :12]
    support = normalized(x)
    kernel = question_kernel(support, support, "rbf")
    coef = np.linalg.solve(kernel + 1.5 * np.eye(len(x)), (axes.T @ residual).T)
    return dict(
        method="rasch",
        model_names=names,
        vectors=np.column_stack((ability, axes)),
        support=support,
        coef=coef,
        signal=cfg["signal"],
        kernel="rbf",
        encoder="mpnet",
        texts=texts,
        domains=domains,
        prompt_ids=ids,
        subtype_centers=centers,
        domain_item=(groups.T @ item) / count,
    )


def rasch_design(router, x, texts, membership=None):
    if membership is None:
        membership = template_membership(router["texts"], texts, router["domains"])
    subtype = subtype_matrix(texts) - membership @ router["subtype_centers"]
    signal = float(router["signal"])
    semantic = np.zeros((len(texts), router["coef"].shape[1]))
    if signal:
        semantic = signal * (question_kernel(x, router["support"], "rbf") @ router["coef"])
    return np.column_stack((membership, subtype, semantic))


def select_measurement(x, domains, y, texts, ids, names, seed, base_cfg):
    """Compare bounded alternatives using only pairwise outcomes inside the same 200 prompts."""
    from panel_capabilities import pack_panel, panel_design

    rng = np.random.default_rng(seed)
    fold = np.empty(len(domains), dtype=int)
    for j in np.unique(domains):
        arr = rng.permutation(np.flatnonzero(domains == j))
        fold[arr] = np.arange(len(arr)) % 4
    settings = [(penalty, signal) for penalty in (1.5, 6.0) for signal in (0.0, 0.5, 1.0)]
    losses = {key: [] for key in ["empirical", *settings]}
    for i in range(4):
        a, b = fold != i, fold == i
        train = x[a], domains[a], y[:, a], texts[a], ids[a], names
        pm = template_membership(texts[a], texts[b], domains[a])
        standard = pack_panel(*train, base_cfg)
        score = standard["vectors"] @ panel_design(standard, x[b], texts[b]).T
        contenders = np.argsort(-score, axis=0)[:12]
        left, right = np.triu_indices(len(contenders), k=1)
        obs = np.take_along_axis(y[:, b], contenders, axis=0)
        diff = obs[left] - obs[right]

        def record(key, prediction):
            p = np.take_along_axis(prediction, contenders, axis=0)
            losses[key].extend(((p[left] - p[right] - diff) ** 2).mean(0))

        record("empirical", score)
        for penalty in (1.5, 6.0):
            pack = pack_rasch(*train, dict(subtype_penalty=penalty, signal=1.0))
            for signal in (0.0, 0.5, 1.0):
                pack["signal"] = signal
                raw = pack["vectors"] @ rasch_design(pack, x[b], texts[b], pm).T
                record((penalty, signal), expit(raw + pm @ pack["domain_item"]))
    risks = {key: np.asarray(value) for key, value in losses.items()}
    best = min(risks, key=lambda key: risks[key].mean())
    diff = risks["empirical"] - risks[best]
    if best == "empirical" or diff.mean() <= diff.std(ddof=1) / np.sqrt(len(diff)):
        return dict(base_cfg, method="empirical", paired_risk=float(risks["empirical"].mean()))
    return dict(
        method="rasch",
        probe_count=len(x),
        encoder="mpnet",
        subtype_penalty=best[0],
        signal=best[1],
        paired_risk=float(risks[best].mean()),
        incumbent_risk=float(risks["empirical"].mean()),
    )
