"""Learn question axes by differentiating through a 200-answer capability fit."""

import torch
from torch import nn
from torch.nn import functional as F


class EpisodicRouter(nn.Module):
    def __init__(self, width, n_models, n_domains, rank=12, learned=8):
        super().__init__()
        self.rank, self.learned = rank, learned
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("axes", torch.zeros(n_models, rank))
        self.register_buffer("domain_axes", torch.zeros(n_domains, rank))
        self.register_buffer("axis_scale", torch.ones(rank))
        self.encoder = nn.Sequential(
            nn.Linear(width, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, rank + learned + n_domains),
        )

    def text(self, vectors):
        raw = self.encoder((vectors - self.center) / self.scale)
        residual, skill, logits = raw.split([self.rank, self.learned, len(self.domain_axes)], dim=-1)
        memberships = logits.softmax(-1)
        predicted = memberships @ self.domain_axes + self.axis_scale * residual
        return predicted, 0.12 * skill.tanh(), logits

    def observed_axes(self, history, reference):
        rows = history[:, reference]
        difficulty = rows.mean(-1, keepdim=True)
        measured = (rows - difficulty) @ self.axes[reference] * (len(self.axes) / len(reference))
        return measured, difficulty

    def design(self, measured, skill, memberships):
        return torch.cat([torch.full_like(skill[:, :1], 0.25), 0.15 * memberships, measured, skill], dim=1)

    def fit(self, design, target, difficulty):
        identity = torch.eye(design.shape[1], device=design.device)
        y = target - difficulty
        if torch.isfinite(y).all():
            return torch.linalg.solve(design.T @ design + 0.30 * identity, design.T @ y).T
        fitted = []
        for row in y.T:
            observed = torch.isfinite(row)
            x = design[observed]
            fitted.append(torch.linalg.solve(x.T @ x + 0.30 * identity, x.T @ row[observed]))
        return torch.stack(fitted)

    def support_design(self, vectors, domains, history, reference):
        pred, skill, _ = self.text(vectors)
        measured, difficulty = self.observed_axes(history, reference)
        membership = F.one_hot(domains, len(self.domain_axes)).float()
        return self.design(0.75 * measured + 0.25 * pred, skill, membership), difficulty

    @torch.inference_mode()
    def onboard(self, vectors, domains, history, responses):
        data = [torch.as_tensor(x) for x in (vectors, domains, history)]
        ref = torch.arange(len(self.axes))
        design, difficulty = self.support_design(*data, ref)
        return self.fit(design, torch.as_tensor(responses.T.copy(), dtype=torch.float32), difficulty).numpy()

    @torch.inference_mode()
    def query(self, vectors, *, historical_scores=None, mix=0.0):
        pred, skill, logits = self.text(torch.as_tensor(vectors))
        if mix:
            s = torch.as_tensor(historical_scores, dtype=torch.float32)
            predicted_pc = (s - s.mean(-1, keepdim=True)) @ self.axes
            pred = (1 - mix) * pred + mix * predicted_pc
        return self.design(pred, skill, logits.softmax(-1)).numpy()

    @classmethod
    def restore(cls, file):
        pack = torch.load(file, weights_only=True, map_location="cpu")
        model = cls(*pack["shape"])
        model.load_state_dict(pack["state"])
        return model.eval()
