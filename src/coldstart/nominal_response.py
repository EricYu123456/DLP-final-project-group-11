"""Predict a model's answer distribution from a short profile and independently estimate the correct answer."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .latent_answers import tensors
from .switch_calibration import peer_vectors


class NominalResponse(nn.Module):
    def __init__(self, experts, domains):
        super().__init__()
        self.domains = domains
        self.global_rank, self.local_rank = 16, 4
        width = 3 * experts + 4 * experts + domains + 6
        self.shared = nn.Sequential(
            nn.Linear(width, 72), nn.LayerNorm(72), nn.GELU(), nn.Linear(72, 48), nn.GELU()
        )
        self.judge = nn.Linear(48, 1)
        self.prototype = nn.Linear(48, 1)
        self.features = nn.Linear(48, self.global_rank + self.local_rank)
        self.difficulty = nn.Linear(48, 1)
        self.width = domains + self.global_rank + domains * self.local_rank

    def query(self, raw, dom):
        votes = raw["votes"].transpose(1, 2)
        batch, k, e = votes.shape
        mask = raw["mask"]
        mc = raw["mc"].float()
        top = (votes == votes.amax(1, keepdim=True)) & (votes > 1 / mask.sum(1)[:, None, None] + 0.001)
        detail = torch.cat((votes, votes.clamp_min(0.007).log() / 4, top.float()), dim=-1)
        opts = torch.zeros((k, 4), device=dom.device)
        opts[:4] = torch.eye(4, device=dom.device)
        by_option = opts[None] * mc[:, None, None]
        unknown = (torch.arange(k, device=dom.device)[None] == mask.sum(1)[:, None] - 1) & (
            ~raw["mc"][:, None]
        )
        context = torch.cat((raw["quality"], F.one_hot(dom, self.domains).float(), mc[:, None]), dim=1)
        x = self.shared(
            torch.cat(
                (detail, by_option, unknown[:, :, None].float(), context[:, None].expand(-1, k, -1)), dim=2
            )
        )
        base = votes.mean(2).clamp_min(0.015).log()
        truth = (0.8 * base + 2 * self.judge(x).squeeze(-1)).masked_fill(~mask, -1000).softmax(1)
        common = (0.5 * base + self.prototype(x).squeeze(-1)).masked_fill(~mask, -1000)
        feat = self.features(x).tanh()
        level = self.difficulty((truth[:, :, None] * x).sum(1)).squeeze(-1)
        return truth, common, feat, level

    def correctness(self, z, q, domain, with_choices=False):
        truth, common, feat, level = q
        n = self.domains
        general = z[:, n : n + self.global_rank]
        local = z[:, n + self.global_rank :].reshape(len(z), n, self.local_rank)
        pref = common[:, :, None] + torch.einsum("bkd,md->bkm", feat[:, :, : self.global_rank], general)
        pref += torch.einsum("bkd,mbd->bkm", feat[:, :, self.global_rank :], local[:, domain])
        pick = pref.softmax(1)
        ability = z[:, :n][:, domain].T + level[:, None]
        conditional = (torch.logit(pick.clamp(0.0002, 0.9998)) + ability[:, None]).sigmoid()
        estimate = (truth[:, :, None] * conditional).sum(1)
        return (estimate, conditional, pick) if with_choices else (estimate, conditional)


def known_prob(p, gold, alternative):
    i = torch.arange(len(gold), device=gold.device)
    observed = p[i, gold.clamp_min(0)]
    return torch.where((gold >= 0)[:, None], observed, alternative)


def train_reference(raw, dom, keys, h, steps, *, seed=4451, device="cpu", checkpoint=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = int(dom.max()) + 1
    net = NominalResponse(raw["votes"].shape[1], d)
    rate = np.stack([h[:, dom == j].mean(1) for j in range(d)], axis=1)
    init = torch.zeros(len(h), net.width)
    init[:, :d] = torch.logit(torch.tensor(rate, dtype=torch.float32).clamp(0.025, 0.975)) + 0.6
    init[:, d:] = torch.randn(len(h), net.width - d) * 0.09
    net = net.to(device)
    z = nn.Parameter(init.to(device).clone())
    init = init.to(device)
    opt = torch.optim.AdamW(
        [dict(params=net.parameters(), lr=0.0014), dict(params=[z], lr=0.013)], weight_decay=0.015
    )
    inputs = tensors(raw, device)
    ids = torch.as_tensor(dom, dtype=torch.long, device=device)
    observed = torch.tensor(h.T.copy(), dtype=torch.float32, device=device)
    gold = torch.tensor(keys, dtype=torch.long, device=device)
    weights = 0.58 / len(dom) + 0.42 / (d * np.bincount(dom)[dom])
    for step in range(steps):
        ix = torch.as_tensor(rng.choice(len(dom), 112, replace=True, p=weights), device=device)
        ds = ids[ix]
        target = observed[ix]
        q = net.query({k: v[ix] for k, v in inputs.items()}, ds)
        pred, conditional = net.correctness(z, q, ds)
        cond = known_prob(conditional, gold[ix], pred)
        present = gold[ix] >= 0
        correct = -torch.log(q[0][torch.arange(len(ix), device=device), gold[ix].clamp_min(0)].clamp_min(1e-7))
        loss = (
            0.70 * F.binary_cross_entropy(cond, target)
            + 0.55 * F.binary_cross_entropy(pred, target)
            + 0.55 * (correct * present).sum() / present.sum().clamp_min(1)
        )
        diff = pred - target
        loss += (
            0.3 * (diff - diff.mean(1, keepdim=True)).square().mean()
            + 0.003 * (z[:, d:] - init[:, d:]).square().mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.5)
        opt.step()
        if checkpoint is not None and (step + 1) % 850 == 0 and checkpoint(step + 1, net, z) is False:
            break
    return net.eval().cpu(), z.detach().cpu().numpy()


@torch.no_grad()
def query(net, raw, dom):
    points = []
    for a in range(0, len(dom), 360):
        s = slice(a, a + 360)
        points.append(
            net.query(tensors({k: v[s] for k, v in raw.items()}), torch.tensor(dom[s], dtype=torch.long))
        )
    return tuple(torch.cat(v) for v in zip(*points))


def outputs(net, z, q, dom):
    if not torch.is_tensor(z):
        z = torch.as_tensor(z, dtype=torch.float32)
    with torch.no_grad():
        return np.concatenate(
            [
                net.correctness(
                    z, tuple(v[i : i + 360] for v in q), torch.as_tensor(dom[i : i + 360], dtype=torch.long)
                )[0].numpy()
                for i in range(0, len(dom), 360)
            ]
        ).T


def choice_features(raw, domain, n_domains):
    votes = raw["votes"].transpose(0, 2, 1)
    n, k, experts = votes.shape
    top = (votes == votes.max(1, keepdims=True)) & (votes > 1 / raw["mask"].sum(1)[:, None, None] + 0.01)
    letters = np.broadcast_to(np.eye(k, 4), (n, k, 4)) * raw["mc"][:, None, None]
    local = votes[:, :, None, :] * np.eye(n_domains)[domain, None, :, None]
    return np.concatenate((votes, top, letters, 0.55 * local.reshape(n, k, n_domains * experts)), axis=2)


@torch.no_grad()
def fit_choice_tilt(net, profile, q, domain, raw, targets, weight):
    """One regularized multinomial Newton step from the historical decoder to actual panel choices."""
    _, _, picks = net.correctness(
        torch.as_tensor(profile["z"]), q, torch.as_tensor(domain, dtype=torch.long), with_choices=True
    )
    phi = choice_features(raw, domain, net.domains)
    penalty = 5.0 * np.eye(phi.shape[2])
    delta = np.zeros((len(targets), phi.shape[2]))
    for m, prob in enumerate(picks.numpy().transpose(2, 0, 1)):
        if not weight[m].any():
            continue
        centered = phi - np.einsum("nk,nkf->nf", prob, phi)[:, None]
        variance = (weight[m, :, None] * prob).reshape(-1, 1)
        flat = centered.reshape(-1, phi.shape[2])
        info = flat.T @ (variance * flat)
        change = np.einsum("n,nk,nkf->f", weight[m], targets[m] - prob, centered, optimize=True)
        delta[m] = np.linalg.solve(info + penalty, change)
    return delta.astype(np.float32)


@torch.no_grad()
def choice_effect(net, profile, q, domain, raw):
    z = torch.as_tensor(profile["z"])
    effects = []
    for start in range(0, len(domain), 360):
        take = slice(start, start + 360)
        local = tuple(v[take] for v in q)
        dom = torch.as_tensor(domain[take], dtype=torch.long)
        base, _, pick = net.correctness(z, local, dom, with_choices=True)
        features = choice_features({k: v[take] for k, v in raw.items()}, domain[take], net.domains)
        shift = torch.from_numpy(np.asarray(features @ profile["choice_tilt"].T, dtype=np.float32))
        pick = (pick.clamp_min(1e-12).log() + shift).softmax(1)
        ability = z[:, : net.domains][:, dom].T + local[3][:, None]
        conditional = (torch.logit(pick.clamp(0.0002, 0.9998)) + ability[:, None]).sigmoid()
        effects.append(((local[0][:, :, None] * conditional).sum(1) - base).numpy())
    return np.concatenate(effects).T


def add_choice_signal(matrix, net, profile, q, domain, raw):
    if "choice_tilt" not in profile:
        return matrix
    change = choice_effect(net, profile, q, domain, raw) - profile["choice_center"][:, domain]
    return matrix + profile["choice_signal"] * change


def profile_objective(net, z, center, q, dom, target, observed, key, prior, distribution=None, weight=None):
    p, cond, pick = net.correctness(z, q, dom, with_choices=True)
    c = known_prob(cond, key, p)
    err = 0.60 * F.binary_cross_entropy(c, target, reduction="none") + 0.40 * F.binary_cross_entropy(
        p, target, reduction="none"
    )
    if distribution is not None:
        err -= weight * (distribution * pick.clamp_min(1e-7).log()).sum(1)
    delta = z - center
    return (
        (err * observed).sum()
        + prior * (0.75 * delta[:, :net.domains].square().sum() + delta[:, net.domains:].square().sum())
    ) / observed.sum()


def calibrated_profile(net, z, question, domain, peers, source):
    sample = np.concatenate(
        [np.flatnonzero(domain == j)[:: max(1, np.sum(domain == j) // 115)] for j in range(net.domains)]
    )
    cal = outputs(net, z, tuple(v[sample] for v in question), domain[sample])
    expected = np.stack([cal[:, domain[sample] == j].mean(1) for j in range(net.domains)], axis=1)
    return dict(z=z, w=peers, source=source, expected=expected)


def fit_new(
    net,
    history,
    means,
    question,
    domain,
    keys,
    h,
    y,
    idx,
    near,
    prior,
    response_distribution=None,
    response_weight=None,
):
    peers, off = peer_vectors(h, y, idx, domain, near)
    center = torch.as_tensor(peers @ history, dtype=torch.float32)
    center[:, : net.domains] += torch.tensor(off * 3.3, dtype=torch.float32)
    z = nn.Parameter(center.clone())
    opt = torch.optim.Adam([z], lr=0.035)
    q = tuple(v[idx] for v in question)
    dom = torch.tensor(domain[idx], dtype=torch.long)
    target = torch.tensor(y[:, idx].T.copy(), dtype=torch.float32)
    ok = target.isfinite()
    target = target.nan_to_num()
    key = torch.tensor(keys[idx], dtype=torch.long)
    for _ in range(105):
        loss = profile_objective(net, z, center, q, dom, target, ok, key, prior, response_distribution, response_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return calibrated_profile(net, z.detach().numpy(), question, domain, peers, peers @ means.T + off)


def scores(pf, nominal, peers, dom, means, cfg):
    local = nominal - pf["expected"][:, dom]
    control = pf["w"] @ (peers - means[dom]).T
    return pf["source"][:, dom] + cfg["signal"] * ((1 - cfg["peer"]) * local + cfg["peer"] * control)
