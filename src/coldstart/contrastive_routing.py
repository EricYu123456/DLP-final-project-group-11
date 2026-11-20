"""Learn the likelihood ratio of a challenger-only success versus an incumbent-only success."""

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.nn import functional as F

from .switch_calibration import peer_vectors


class ContrastiveDecoder(nn.Module):
    def __init__(self, width, n_models, n_sources):
        super().__init__()
        self.d = n_sources
        self.encoder = nn.Sequential(
            nn.Linear(width, 224),
            nn.LayerNorm(224),
            nn.GELU(),
            nn.Dropout(0.23),
            nn.Linear(224, 112),
            nn.LayerNorm(112),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(112, 48),
        )
        self.absolute = nn.Parameter(torch.randn(n_models, 24 + 8 * n_sources) * 0.095)
        self.left = nn.Parameter(torch.randn(n_models, 16) * 0.17)
        self.right = nn.Parameter(torch.randn(n_models, 16) * 0.17)

    def coordinates(self, x, source):
        phi = self.encoder(x)
        local = (F.one_hot(source, self.d)[:, :, None] * phi[:, None, 24:32]).flatten(1)
        return torch.cat((phi[:, :24], local, phi[:, 32:]), 1)

    def pair(self, first, second):
        return torch.cat(
            (
                self.absolute[first] - self.absolute[second],
                self.left[first] * self.right[second] - self.left[second] * self.right[first],
            ),
            1,
        )


def historical_event_rates(h, domains):
    out = []
    for j in np.unique(domains):
        yes = h[:, domains == j].astype(np.float64)
        out.append((yes @ (1 - yes).T + 1) / (yes.shape[1] + 2))
    return np.stack(out)


def train_contrasts(x, dom, h, steps, callback, device):
    torch.manual_seed(7419)
    rng = np.random.default_rng(7419)
    d = int(dom.max()) + 1
    net = ContrastiveDecoder(x.shape[1], len(h), d).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.00065, weight_decay=0.045)
    xx = torch.as_tensor(x, dtype=torch.float32, device=device)
    dd = torch.as_tensor(dom, dtype=torch.long, device=device)
    truth = torch.as_tensor(h.T.copy(), dtype=torch.float32, device=device)
    rates = torch.as_tensor(historical_event_rates(h, dom), dtype=torch.float32, device=device)
    pa, pb = np.triu_indices(len(h), 1)
    strengths = np.stack([h[:, dom == j].mean(1) for j in range(d)]).mean(0)
    importance = 0.4 + np.maximum(strengths[pa], strengths[pb])
    importance /= importance.sum()
    mass = 0.52 / len(dom) + 0.48 / (d * np.bincount(dom)[dom])
    for step in range(1, max(steps) + 1):
        ix = rng.choice(len(dom), size=146, p=mass)
        pairs = rng.choice(len(pa), size=min(240, len(pa)), replace=False, p=importance)
        a = torch.as_tensor(pa[pairs], device=device)
        b = torch.as_tensor(pb[pairs], device=device)
        src = dd[ix]
        logits = net.coordinates(xx[ix], src) @ net.pair(a, b).T
        difference = truth[ix][:, a] - truth[ix][:, b]
        # Remove each pair's base odds during training. A measured base rate is reintroduced at routing time.
        good = (difference > 0) / rates[src[:, None], a, b].clamp(0.025)
        bad = (difference < 0) / rates[src[:, None], b, a].clamp(0.025)
        loss = (good * F.softplus(-logits) + bad * F.softplus(logits)).sum() / (good + bad).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in steps:
            net.eval()
            callback(step, net)
            net.train()
    return net.eval()


@torch.inference_mode()
def features(net, x, dom):
    device = next(net.parameters()).device
    return (
        net.coordinates(
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(dom, dtype=torch.long, device=device),
        )
        .cpu()
        .numpy()
    )


def base_events(h, y, idx, dt, w, source_offset):
    """Estimate paired base odds using historical joint outcomes as control variates."""
    conditional = (w @ h + source_offset[:, dt]).clip(0, 1)
    size = len(y)
    advantage, discord = [], []
    for j in np.unique(dt):
        group = dt == j
        seen = idx[dt[idx] == j]
        teacher = conditional[:, group]
        expected = teacher @ (1 - teacher).T / teacher.shape[1]
        base = conditional[:, seen]
        observed = y[:, seen]
        valid = np.isfinite(observed).astype(float)
        clean = np.nan_to_num(observed)
        count = valid @ valid.T
        residual = clean @ (valid - clean).T - (valid * base) @ (valid * (1 - base)).T
        events = (expected + residual / (count + 8)).clip(0.002, 0.995)
        np.fill_diagonal(events, 0.5 / size)
        advantage.append(np.log(events / events.T))
        discord.append((events + events.T).clip(0.01, 1))
    return np.stack(advantage), np.stack(discord)


def embed_challengers(net, x, h, y, idx, dom, prior, nearest):
    w, off = peer_vectors(h, y, idx, dom, nearest)
    historical = [v.detach().cpu().numpy() for v in (net.absolute, net.left, net.right)]
    u, v, k = [(w @ m).astype(np.float32) for m in historical]
    odds, discord = base_events(h, y, idx, dom, w, off)
    means = np.stack([h[:, dom == j].mean(1) for j in range(net.d)])
    weights = 0.45 / net.d + 0.55 * np.bincount(dom) / len(dom)
    strength = (w @ means.T + off) @ weights
    # Compare each candidate against the same measured historical models. The effective sample
    # size is the number of probes, so normalize the likelihood by its number of reference comparators.
    if prior > 0:
        obs = torch.as_tensor(y[:, idx].T, dtype=torch.float32)
        ref = torch.as_tensor(h[:, idx].T, dtype=torch.float32)
        diff = obs[:, :, None] - ref[:, None, :]
        pos, neg = (diff > 0).float(), (diff < 0).float()
        dl = torch.as_tensor(dom[idx], dtype=torch.long)
        coords = torch.as_tensor(x[idx], dtype=torch.float32)
        u0 = torch.from_numpy(u)
        var = nn.Parameter(torch.zeros_like(u0))
        hu, hv, hk = [torch.from_numpy(m) for m in historical]
        cross = torch.from_numpy(v)[:, None] * hk - hv[None] * torch.from_numpy(k)[:, None]
        joint_y = np.vstack((y, h))
        ext_w = np.vstack((w, np.eye(len(h))))
        ext_off = np.vstack((off, np.zeros((len(h), net.d))))
        ref_odds, _ = base_events(h, joint_y, idx, dom, ext_w, ext_off)
        intercept = torch.as_tensor(ref_odds[:, : len(y), len(y) :], dtype=torch.float32)[dl]
        initial = torch.cat((u0[:, None] - hu[None], cross), dim=2)
        logits0 = torch.einsum("qf,mhf->qmh", coords, initial) + intercept
        opt = torch.optim.Adam([var], lr=0.023)
        for _ in range(75):
            logits = logits0 + (coords[:, : u.shape[1]] @ var.T)[:, :, None]
            loss = (pos * F.softplus(-logits) + neg * F.softplus(logits)).sum() / len(
                h
            ) + prior * var.square().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        u = u + var.detach().numpy()
    return dict(u=u, v=v, k=k, odds=odds, discord=discord, strength=strength)


def contrast_scores(pf, x, dom, cfg):
    anchors = np.argsort(-pf["strength"])[: min(int(cfg["anchors"]), len(pf["u"]))]
    result = np.zeros((len(pf["u"]), len(x)))
    for a in anchors:
        emb = np.column_stack((pf["u"] - pf["u"][a], pf["v"] * pf["k"][a] - pf["v"][a] * pf["k"]))
        logits = cfg["signal"] * (emb @ x.T) + pf["odds"][dom, :, a].T
        result += (2 * expit(logits) - 1) * pf["discord"][dom, :, a].T
    return result / len(anchors)
