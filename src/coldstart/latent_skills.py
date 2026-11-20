"""Learn latent item types from historical response patterns and predict their distribution from text."""

import numpy as np
import torch
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from torch import nn

from .conditional_capability import paired_profile


class ItemTypeGate(nn.Module):
    def __init__(self, width, domains, classes):
        super().__init__()
        self.domains = domains
        self.classes = classes
        self.network = nn.Sequential(
            nn.Linear(width, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.23),
            nn.Linear(160, domains * classes),
        )

    def forward(self, x, domains):
        logits = self.network(x).reshape(-1, self.domains, self.classes)
        return logits[torch.arange(len(x), device=x.device), domains]


def item_posteriors(h, domain, n_classes):
    """Tempered Bernoulli-mixture EM avoids treating related historical LLMs as independent votes."""
    responsibilities = np.zeros((h.shape[1], n_classes), dtype=np.float32)
    means = []
    for source in np.unique(domain):
        use = domain == source
        outcomes = np.asarray(h[:, use].T, dtype=np.float32)
        center = outcomes.mean(0)
        relative = outcomes - center
        relative -= relative.mean(1, keepdims=True) * 0.78
        _, _, vt = randomized_svd(relative, min(19, len(h) - 1), n_iter=4, random_state=3207)
        projected = relative @ vt.T
        labels = KMeans(n_clusters=n_classes, n_init=5, random_state=3207).fit_predict(projected)
        posterior = np.eye(n_classes, dtype=np.float32)[labels]
        for _ in range(26):
            counts = posterior.sum(0)
            rates = (posterior.T @ outcomes + 9 * center) / (counts[:, None] + 9)
            rates = rates.clip(0.012, 0.988)
            loglik = outcomes @ np.log(rates / (1 - rates)).T + np.log(1 - rates).sum(1)
            posterior = softmax(0.40 * loglik + np.log((counts + 1) / (len(posterior) + n_classes)), axis=1)
        means.append((posterior.T @ outcomes + 9 * center) / (posterior.sum(0)[:, None] + 9))
        responsibilities[use] = posterior
    return np.stack(means), responsibilities


def train_gate(x, domains, soft_labels, steps, callback, device="cpu"):
    """Use soft cluster memberships as supervision, with an equal-source component in each batch."""
    torch.manual_seed(3207)
    rng = np.random.default_rng(3207)
    n_sources = int(domains.max()) + 1
    gate = ItemTypeGate(x.shape[1], n_sources, soft_labels.shape[1]).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=0.0008, weight_decay=0.06)
    inputs = torch.as_tensor(x, dtype=torch.float32, device=device)
    sources = torch.as_tensor(domains, dtype=torch.long, device=device)
    targets = torch.as_tensor(soft_labels, device=device)
    weights = 0.5 / len(x) + 0.5 / (n_sources * np.bincount(domains)[domains])
    for step in range(1, max(steps) + 1):
        ix = rng.choice(len(x), size=256, p=weights)
        logits = gate(inputs[ix], sources[ix])
        loss = -(targets[ix] * logits.log_softmax(1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in steps:
            gate.eval()
            callback(step, gate)
            gate.train()
    return gate.eval()


@torch.inference_mode()
def gate_scores(gate, x, domain):
    dev = next(gate.parameters()).device
    logits = gate(
        torch.as_tensor(x, dtype=torch.float32, device=dev),
        torch.as_tensor(domain, dtype=torch.long, device=dev),
    )
    return logits.softmax(1).cpu().numpy()


def measure_types(h, y, ix, domain, rates, posterior, shrink, neighbor):
    """Encode each candidate's paired residual strengths on the same observed probes."""
    population = np.bincount(domain)
    historical_means = np.stack([h[:, domain == j].mean(1) for j in range(len(population))])
    embeddings = []
    for row in y:
        seen = ix[np.isfinite(row[ix])]
        dt = domain[seen]
        weights, mass = paired_profile(
            h[:, seen], row[None, seen], dt, population, np.arange(len(population))
        )
        weights = weights[0]
        distance = np.abs(h[:, seen] - row[seen]) @ mass
        nearest = softmax(-distance / 0.033)
        weights = (1 - neighbor) * weights + neighbor * nearest
        marginal = historical_means @ weights
        prior = rates @ weights
        residual = row[seen] - weights @ h[:, seen]
        for j in range(len(population)):
            mask = dt == j
            within = residual[mask]
            marginal[j] += within.sum() / (len(within) + 10)
            deviations = within - (marginal[j] - historical_means[j] @ weights)
            resp = posterior[seen[mask]]
            prior[j] += marginal[j] - historical_means[j] @ weights
            prior[j] += (resp.T @ deviations) / (resp.sum(0) + shrink)
        embeddings.append(np.r_[weights, marginal, prior.reshape(-1)])
    return np.stack(embeddings)


def route_types(z, h_pred, q_probs, q_source, h_means, cfg):
    nh, nd = h_pred.shape[1], len(h_means)
    weights = z[:, :nh]
    marginal = z[:, nh : nh + nd]
    skills = z[:, nh + nd :].reshape(len(z), nd, -1)
    # Only contrasts between candidates affect routing; subtract within-source baselines once.
    per_type = np.einsum("mqk,qk->mq", skills[:, q_source], q_probs) - marginal[:, q_source]
    peer = weights @ (h_pred - h_means[q_source]).T
    mix = cfg["type_mix"]
    return marginal[:, q_source] + cfg["signal"] * (mix * per_type + (1 - mix) * peer)
