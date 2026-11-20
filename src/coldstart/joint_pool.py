"""Jointly learn historical and newcomer LLM embeddings from reference labels and 200 measured answers."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class JointPool(nn.Module):
    def __init__(self, width, history_rates, measured_rates, transfer, dropout):
        super().__init__()
        h, d = history_rates.shape
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(width, 224),
            nn.LayerNorm(224),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(224, 112),
            nn.LayerNorm(112),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(112, 41),
        )
        dim = 32 + d * 8 + d
        self.history = nn.Parameter(torch.randn(h, dim) * 0.055)
        self.delta = nn.Parameter(torch.zeros(len(transfer), dim))
        self.register_buffer("transfer", torch.as_tensor(transfer, dtype=torch.float32))
        prior = torch.logit(torch.as_tensor(history_rates, dtype=torch.float32).clamp(0.025, 0.975))
        target = torch.logit(torch.as_tensor(measured_rates, dtype=torch.float32).clamp(0.025, 0.975))
        with torch.no_grad():
            self.history[:, -d:] = prior
            self.delta[:, -d:] = target - self.transfer @ prior
        self.register_buffer("origin", self.delta.detach().clone())

    def coordinates(self, x, membership):
        raw = self.encoder(x)
        return torch.cat(
            (raw[:, :32], (membership[:, :, None] * raw[:, None, 32:40]).flatten(1), membership), dim=1
        ), raw[:, 40:]

    def vectors(self):
        return self.transfer @ self.history + self.delta

    def forward(self, x, membership):
        phi, difficulty = self.coordinates(x, membership)
        return difficulty + phi @ self.history.T, difficulty + phi @ self.vectors().T


def pair_loss(logits, truth, anchors, soft=False):
    if len(anchors) == 0:
        return logits.sum() * 0
    delta = logits[:, :, None] - logits[:, None, anchors]
    if soft:
        a = truth[:, :, None] * (1 - truth[:, None, anchors])
        b = (1 - truth[:, :, None]) * truth[:, None, anchors]
    else:
        y = truth[:, :, None] - truth[:, None, anchors]
        a, b = (y > 0).float(), (y < 0).float()
    return (F.softplus(-delta) * a + F.softplus(delta) * b).sum() / (a + b).sum().clamp_min(1)


def fit_joint(x, membership, h, y, teacher, idx, weights, seed, stages, callback, real_weight=0.22):
    device = "cuda"
    torch.manual_seed(seed + 2401)
    rng = np.random.default_rng(seed + 2401)
    dom = membership.argmax(1)
    d = membership.shape[1]
    hist_rates = np.stack([h[:, dom == j].mean(1) for j in range(d)], axis=1)
    new_rates = np.stack([teacher[:, dom == j].mean(1) for j in range(d)], axis=1)
    model = JointPool(x.shape[1], hist_rates, new_rates, weights, 0.20).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.00055, weight_decay=0.025)
    vals = [
        torch.as_tensor(v, dtype=torch.float32, device=device)
        for v in (x, membership, h.T, teacher.T, y[:, idx].T)
    ]
    features, members, history, soft, observed = vals
    support = torch.as_tensor(idx, device=device)
    hist_top = np.argsort(-hist_rates.mean(1))[:8]
    new_top = np.argsort(-new_rates.mean(1))[: min(4, len(y))]
    mass = 0.45 / len(x) + 0.55 / (d * np.bincount(dom)[dom])
    for i in range(1, max(stages) + 1):
        rows = torch.as_tensor(rng.choice(len(x), 196, p=mass), device=device)
        shots = torch.as_tensor(rng.choice(len(idx), 64), device=device)
        his, nov = model(features[rows], members[rows])
        _, actual = model(features[support[shots]], members[support[shots]])
        ha = rng.choice(hist_top, 3, replace=False).tolist()
        na = rng.choice(new_top, min(2, len(new_top)), replace=False).tolist()
        valid = torch.isfinite(observed[shots])
        lbl = observed[shots].nan_to_num(0)
        hl = F.binary_cross_entropy_with_logits(his, history[rows]) + 0.38 * pair_loss(his, history[rows], ha)
        nl = F.binary_cross_entropy_with_logits(nov, soft[rows]) + 0.55 * pair_loss(nov, soft[rows], na, True)
        al = (
            F.binary_cross_entropy_with_logits(actual, lbl, reduction="none") * valid
        ).sum() / valid.sum().clamp_min(1)
        delta = model.delta - model.origin
        loss = (
            hl
            + 0.48 * nl
            + real_weight * (al + 0.55 * pair_loss(actual, observed[shots], na))
            + 0.025 * delta.square().mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        opt.step()
        if i in stages:
            model.eval()
            with torch.inference_mode():
                callback(i, model)
            model.train()
    return model.eval()
