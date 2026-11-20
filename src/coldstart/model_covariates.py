"""Descriptive LLM covariates from model names and documented parent architectures."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def model_descriptions(names, table):
    rows = pd.read_csv(table).fillna("")
    lookup = {str(row.model_name).lower().replace("__", "/"): row for row in rows.itertuples()}
    features, sizes = [], []
    for name in names:
        key = str(name).lower().replace("__", "/")
        row = lookup.get(key)
        parents = str(row.parent_models).lower() if row is not None else ""
        family = str(row.family).lower() if row is not None else ""
        features.append(" ".join((key.split("/")[-1], parents, family, family)))
        pattern = r"(?<![a-z0-9])([0-9]+(?:[._][0-9]+)?)[bB](?![a-z])"
        amount = re.findall(pattern, key.replace("phi-1_5", "phi-1.5 1.3b"))
        if not amount:
            amount = re.findall(pattern, parents)
        sizes.append(max([float(value.replace("_", ".")) for value in amount] or [float("nan")]))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)
    desc = vec.fit_transform(features)
    size = np.log(np.asarray(sizes))
    known = np.isfinite(size)
    size = np.where(known, size, 0)
    match = np.exp(-((size[:, None] - size) ** 2) / 0.65) * (known[:, None] & known)
    return (desc @ desc.T).toarray(), match


def architecture_kernel(names, table):
    """Pair instruction format, specialization and measurable transformer geometry without leaderboard scores."""
    data = pd.read_csv(table).fillna("").set_index("name")
    values = []
    patterns = (
        r"instruct|chat|hermes|dpo|sft|vicuna|tulu|wizard|monarch|zephyr",
        r"base|pretrain",
        r"math|arith|physics",
        r"bio|medic|clinical|health",
        r"codellama|coding|coder|code-|starcoder",
        r"merge|mix|moe|dare",
        r"llama-?3|llama[_ ]3",
        r"llama-?2|llama[_ ]2",
        r"yi-",
        r"mistral|mixtral",
        r"qwen",
        r"gemma",
        r"phi-",
        r"bloom|falcon",
    )
    for name in names:
        key = str(name).replace("__", "/")
        row = data.loc[key] if key in data.index else {}
        text = " ".join(
            [key, *(str(row.get(s, "")) for s in ("parents", "family", "tags", "architecture"))]
        ).lower()
        tags = [float(bool(re.search(p, text))) for p in patterns]
        numeric = [
            float(row.get(c, 0) or 0)
            for c in (
                "parameters",
                "layers",
                "width",
                "heads",
                "kv_heads",
                "experts",
                "intermediate",
                "vocab",
            )
        ]
        values.append([*tags, *numeric])
    arr = np.array(values)
    t = arr[:, : len(patterns)]
    norm = np.linalg.norm(t, axis=1).clip(1)
    discrete = (t @ t.T) / (norm[:, None] * norm)
    numeric = arr[:, len(patterns) :]
    known = numeric > 0
    logs = np.log(numeric.clip(1))
    shared = known[:, None] & known
    sq = np.minimum((logs[:, None] - logs) ** 2, 4) * shared
    distance = sq.sum(-1) / shared.sum(-1).clip(1)
    geometry = np.exp(-distance / 0.42) * (shared.sum(-1) / numeric.shape[1])
    return 0.53 * discrete + 0.47 * geometry
