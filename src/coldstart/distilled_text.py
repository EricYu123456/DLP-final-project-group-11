"""Distill auxiliary solvers and historical model disagreements into a text-only routing encoder."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .switch_calibration import peer_vectors


BACKBONE = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def tokenize(texts, limit):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION)
    encoded = tok(list(texts), add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    budget = limit - 2

    def shorten(row):
        if len(row) > budget:
            start = int(0.40 * budget)
            row = row[:start] + row[-(budget - start) :]
        return [tok.cls_token_id, *row, tok.sep_token_id]

    return [shorten(r) for r in encoded], tok.pad_token_id


def batch_ids(tokens, idx, pad, device):
    seq = [tokens[i] for i in idx]
    longest = (max(map(len, seq)) + 7) // 8 * 8
    a = np.full((len(seq), longest), pad, dtype=np.int64)
    for i, row in enumerate(seq):
        a[i, : len(row)] = row
    ids = torch.as_tensor(a, device=device)
    return dict(input_ids=ids, attention_mask=ids.ne(pad).long())


class DistilledText(nn.Module):
    def __init__(self, historical, domains, auxiliary):
        super().__init__()
        from transformers import AutoModel

        self.text = AutoModel.from_pretrained(BACKBONE, revision=REVISION)
        width = self.text.config.hidden_size
        self.read = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(0.13), nn.Linear(width, 160), nn.GELU(), nn.Dropout(0.12)
        )
        self.skill = nn.Linear(160, 32)
        self.historical = nn.Linear(32, historical, bias=False)
        self.difficulty = nn.Linear(160, 1)
        self.source = nn.Linear(160, domains)
        self.aux = nn.Linear(160, auxiliary)

    def forward(self, data):
        encoded = self.text(**data).last_hidden_state
        mask = data["attention_mask"][:, :, None].to(encoded.dtype)
        h = self.read((encoded * mask).sum(1) / mask.sum(1).clamp_min(1))
        phi = self.skill(h).tanh()
        return phi, self.historical(phi), self.difficulty(h), self.source(h), self.aux(h)


def training_loss(net, data, y, dom, aux, rates, pair_a, pair_b):
    _, contrast, difficulty, source, proxy = net(data)
    contrast = contrast.float()
    y, base = y.float(), rates[dom]
    centered = contrast - contrast.mean(1, keepdim=True)
    absolute = base + (difficulty.float() + centered)
    delta = y[:, pair_a] - y[:, pair_b]
    # Estimate deviations from the historical default rather than relearning model ordering per query.
    pair_logit = base[:, pair_a] - base[:, pair_b] + 1.25 * (centered[:, pair_a] - centered[:, pair_b])
    paired = F.binary_cross_entropy_with_logits(pair_logit, (delta > 0).float(), reduction="none")
    pair_weight = delta.abs() * (0.55 + torch.sigmoid(torch.maximum(base[:, pair_a], base[:, pair_b])))
    bin_loss = F.binary_cross_entropy_with_logits(absolute, y)
    pair_loss = (paired * pair_weight).sum() / pair_weight.sum().clamp_min(1)
    return (
        bin_loss
        + 0.95 * pair_loss
        + 0.38 * F.smooth_l1_loss(proxy.float(), aux)
        + 0.16 * F.cross_entropy(source.float(), dom)
    )


@torch.inference_mode()
def encode_tokens(net, seq, pad, device, batch=56):
    was_training = net.training
    net.eval()
    outputs = np.empty((len(seq), 32), np.float32)
    order = np.argsort([len(v) for v in seq])
    for start in range(0, len(seq), batch):
        ix = order[start : start + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            phi = net(batch_ids(seq, ix, pad, device))[0]
        outputs[ix] = phi.float().cpu().numpy()
    net.train(was_training)
    return outputs


def bank_features(x, dt):
    center = np.stack([x[dt == j].mean(0) for j in np.unique(dt)])
    sd = (x - center[dt]).std(0).clip(0.045)
    return dict(center=center, sd=sd)


def query_features(pack, x, dt):
    base = ((x - pack["center"][dt]) / pack["sd"]).clip(-4, 4)
    local = (np.eye(len(pack["center"]))[dt, :, None] * base[:, None, :5]).reshape(len(x), -1)
    return np.column_stack((base, local))


def historical_embeddings(x, dt, h):
    pack = bank_features(x, dt)
    z = query_features(pack, x, dt)
    means = np.stack([h[:, dt == j].mean(1) for j in np.unique(dt)])
    residual = h.T - means[dt]
    # All candidates compete on the same question, so common item difficulty cancels.
    residual -= residual.mean(1, keepdims=True)
    wt = np.sqrt(0.55 + 0.45 * len(dt) / (len(means) * np.bincount(dt)[dt]))
    empirical = np.linalg.solve(
        (z * wt[:, None]).T @ (z * wt[:, None]) + 80 * np.eye(z.shape[1]),
        (z * wt[:, None]).T @ (residual * wt[:, None]),
    )
    return dict(**pack, means=means, history=empirical.T)


def fit_distilled(pack, x, dt, h, y, idx, nearest, reg):
    w, off = peer_vectors(h, y, idx, dt, nearest)
    coeff = w @ pack["history"]
    raw = query_features(pack, x[idx], dt[idx])
    residual = y[:, idx] - w @ h[:, idx] - off[:, dt[idx]]
    for i, v in enumerate(residual):
        take = np.isfinite(v)
        z = raw[take]
        coeff[i] += np.linalg.solve(z.T @ z + reg * np.eye(z.shape[1]), z.T @ v[take])
    return dict(skill=coeff, source=w @ pack["means"].T + off)


def distilled_scores(pf, x, dom, pack, signal):
    coordinates = query_features(pack, x, dom)
    return pf["source"][:, dom] + signal * pf["skill"] @ coordinates.T
