"""Infer a latent correct answer from independent solvers and learn each LLM's response tendencies."""

import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .switch_calibration import peer_vectors


def answer_signature(text):
    s = str(text).strip().lower()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", s.replace(",", ""))
    stop = {
        "the",
        "answer",
        "is",
        "a",
        "an",
        "therefore",
        "it",
        "of",
        "to",
        "in",
        "and",
        "are",
        "approximately",
    }
    terms = set(re.findall(r"[a-z]+", s)) - stop
    return numbers[-1] if numbers else "", terms


def answer_arrays(evidence):
    """Use four displayed options, or align independent short answers plus an unobserved answer."""
    n, e = len(evidence[0][0]), len(evidence)
    k = max(5, e + 1)
    votes = np.zeros((n, e, k), np.float32)
    allowed = np.zeros((n, k), bool)
    mc = evidence[0][0][:, 0, :4].sum(1) > 0.5
    allowed[mc, :4] = True
    quality = []
    for j, (p, _) in enumerate(evidence):
        avg = p[:, :, :4].mean(1)
        votes[mc, j, :4] = avg[mc]
        stability = np.abs(p[:, 0, :4] - p[:, 1, :4]).sum(1)
        quality.extend((avg.max(1), stability, np.tanh(p[:, 0, 4] / 3), np.tanh(p[:, 0, 5] / 3)))
    for i in np.flatnonzero(~mc):
        groups = []
        pos = []
        for p, answers in evidence:
            if p.shape[-1] > 6 and not p[i, 0, 6]:
                pos.append(None)
                continue
            numeric, tokens = answer_signature(answers[i])
            match = next(
                (
                    l
                    for l, (v, other) in enumerate(groups)
                    if (numeric and numeric == v)
                    or (tokens and len(tokens & other) / max(1, len(tokens | other)) >= 0.68)
                ),
                None,
            )
            if match is None:
                match = len(groups)
                groups.append((numeric, tokens))
            pos.append(match)
        allowed[i, : len(groups) + 1] = True
        for j, t in enumerate(pos):
            if t is None:
                votes[i, j, : len(groups) + 1] = 1 / (len(groups) + 1)
                continue
            certainty = 0.62 + 0.32 * np.exp(min(0, evidence[j][0][i, 0, 5]))
            votes[i, j, : len(groups) + 1] = (1 - certainty) / len(groups)
            votes[i, j, t] = certainty
    votes = np.where(allowed[:, None, :], votes.clip(1e-6), 0)
    votes /= votes.sum(2, keepdims=True)
    return dict(votes=votes, mask=allowed, mc=mc, quality=np.column_stack(quality).astype(np.float32))


def tensors(data, device="cpu"):
    return {k: torch.as_tensor(v, device=device) for k, v in data.items()}


class AnswerDecoder(nn.Module):
    """A permutation-equivariant gate estimates answer probabilities from the committee's votes."""

    def __init__(self, experts, domains):
        super().__init__()
        self.e, self.d = experts, domains
        width = 5 * experts + domains + 1
        self.gate = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 1))
        self.level = nn.Sequential(nn.Linear(4 * experts + domains + 1, 32), nn.GELU(), nn.Linear(32, 1))
        self.reliability = nn.Parameter(torch.zeros(domains, experts))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def query(self, raw, dom):
        v, mask = raw["votes"], raw["mask"]
        global_input = torch.cat(
            (raw["quality"], F.one_hot(dom, self.d).float(), raw["mc"][:, None].float()), -1
        )
        features = torch.cat((v.transpose(1, 2), global_input[:, None].expand(-1, v.shape[2], -1)), -1)
        mean = (v * self.reliability[dom].softmax(-1)[:, :, None]).sum(1)
        logits = mean.clamp_min(1e-6).log() + self.gate(features).squeeze(-1)
        return logits.masked_fill(~mask, -1e5).log_softmax(-1), self.level(global_input).squeeze(-1), mean

    @property
    def width(self):
        return self.d + self.e + self.d * self.e + 1 + 4

    def correctness(self, z, raw, dom, item):
        d, e = self.d, self.e
        base = z[:, :d][:, dom].T
        general = z[:, d : d + e]
        conditional = z[:, d + e : d + e + d * e].reshape(len(z), d, e)
        combine = (general[None] + conditional[:, dom].permute(1, 0, 2)).softmax(-1)
        picks = torch.einsum("bme,bek->bmk", combine, raw["votes"])
        offset = F.pad(z[:, -4:], (0, picks.shape[2] - 4))
        dist = (
            (picks.clamp_min(1e-6).log() + raw["mc"][:, None, None] * offset[None])
            .masked_fill(~raw["mask"][:, None, :], -1e5)
            .softmax(-1)
        )
        chance = raw["mask"].sum(1).sub(1).clamp_min(1).float().log()
        signal = torch.logit(dist.clamp(0.002, 0.998)) + chance[:, None, None]
        sensitivity = F.softplus(z[:, -5]).clamp(0.05, 3)
        return base[:, :, None] + item[:, None, None] + sensitivity[None, :, None] * signal


def correct_posterior(log_probs, logits, outcomes, temperature=0.24):
    truth = outcomes[:, :, None]
    evidence = (truth * F.logsigmoid(logits) + (1 - truth) * F.logsigmoid(-logits)).sum(1)
    return (log_probs + temperature * evidence).log_softmax(1)


def measured_posterior(posterior, batch):
    if "option_key" not in batch:
        return posterior
    keys = batch["option_key"].long()
    known = (keys >= 0) & batch["mc"]
    actual = F.one_hot(keys.clamp_min(0), posterior.shape[1]).float()
    return torch.where(known[:, None], actual, posterior)


def fit_history(raw, dom, h, steps=1850):
    torch.manual_seed(2593)
    rng = np.random.default_rng(2593)
    d, e = int(dom.max()) + 1, raw["votes"].shape[1]
    net = AnswerDecoder(e, d)
    rates = np.stack([h[:, dom == j].mean(1) for j in range(d)], axis=1)
    init = torch.zeros(len(h), net.width)
    init[:, :d] = torch.logit(torch.tensor(rates).float().clamp(0.015, 0.985))
    init[:, -5] = -0.75
    z = nn.Parameter(init.clone())
    opt = torch.optim.AdamW([*net.parameters(), z], lr=0.0025, weight_decay=0.013)
    x = tensors(raw)
    labels = torch.as_tensor(h.T.copy(), dtype=torch.float32)
    ds = torch.as_tensor(dom, dtype=torch.long)
    mass = 0.55 / len(dom) + 0.45 / (d * np.bincount(dom)[dom])
    for _ in range(steps):
        ix = rng.choice(len(dom), size=128, p=mass)
        batch = {k: v[ix] for k, v in x.items()}
        lp, difficulty, prior = net.query(batch, ds[ix])
        pred = net.correctness(z, batch, ds[ix], difficulty)
        posterior = measured_posterior(correct_posterior(lp, pred, labels[ix]).exp().detach(), batch)
        pos = pred.sigmoid()
        conditional = F.binary_cross_entropy_with_logits(
            pred, labels[ix, :, None].expand_as(pred), reduction="none"
        )
        marginal = (lp.exp()[:, None] * pos).sum(2)
        loss = (
            0.6 * F.binary_cross_entropy(marginal, labels[ix])
            + 0.7 * (conditional * posterior[:, None]).sum(2).mean()
        )
        loss -= 0.45 * (posterior * lp).sum(1).mean()
        loss += 0.045 * (lp.exp() * (lp - prior.clamp_min(1e-6).log())).sum(1).mean()
        loss += (
            0.003 * (z[:, d:] - init[:, d:]).square().mean()
            + 0.018 * (z[:, :d] - init[:, :d]).square().mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net.eval(), z.detach()


def fit_newcomers(net, hist_vectors, raw, dom, h, y, idx, prior_steps, nearest):
    d = net.d
    peers, offset = peer_vectors(h, y, idx, dom, nearest)
    prior = torch.as_tensor(peers, dtype=torch.float32) @ hist_vectors
    prior[:, :d] += (4 * torch.as_tensor(offset, dtype=torch.float32)).clip(-1.4, 1.4)
    z = nn.Parameter(prior.clone())
    opt = torch.optim.Adam([z], lr=0.032)
    data = tensors({k: v[idx] for k, v in raw.items()})
    ds = torch.as_tensor(dom[idx], dtype=torch.long)
    hpanel = torch.from_numpy(h[:, idx].T.copy()).float()
    answers = torch.from_numpy(y[:, idx].T.copy()).float()
    good = torch.isfinite(answers)
    values = answers.nan_to_num()
    with torch.no_grad():
        lp, level, _ = net.query(data, ds)
        old = net.correctness(hist_vectors, data, ds, level)
        gold = measured_posterior(correct_posterior(lp, old, hpanel).exp(), data)
    for _ in range(145):
        pred = net.correctness(z, data, ds, level)
        observed = F.binary_cross_entropy_with_logits(
            pred, values[:, :, None].expand_as(pred), reduction="none"
        )
        conditional = (observed * gold[:, None]).sum(2)
        actual = F.binary_cross_entropy((lp.exp()[:, None] * pred.sigmoid()).sum(2), values, reduction="none")
        delta = z - prior
        penalty = prior_steps * (0.60 * delta[:, :d].square().sum() + delta[:, d:].square().sum())
        loss = (((0.78 * conditional + 0.22 * actual) * good).sum() + penalty) / good.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return z.detach().numpy()


@torch.inference_mode()
def predict_answers(net, z, raw, domain):
    z = torch.as_tensor(z, dtype=torch.float32)
    values = []
    for i in range(0, len(domain), 320):
        data = tensors({k: v[i : i + 320] for k, v in raw.items()})
        dom = torch.as_tensor(domain[i : i + 320], dtype=torch.long)
        lp, level, _ = net.query(data, dom)
        scores = (lp.exp()[:, None] * net.correctness(z, data, dom, level).sigmoid()).sum(2)
        values.append(scores.numpy())
    return np.concatenate(values).T
