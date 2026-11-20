"""Meta-train on historical tournaments with disjoint 200-question supports and query sets."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ProbeBank, ROOT, input_context, metrics
from coldstart.episodic_router import EpisodicRouter
from run_experiment import atomic_json


def train(args):
    start = perf_counter()
    d, s, c, *_ = input_context(args.run)
    bank = ProbeBank(d, s, c["seeds"])
    tr = d.subset(s.train_models, s.train_items)
    x, y, domains = [torch.as_tensor(v, device=args.device) for v in (tr.embeddings, tr.labels.T, tr.domains)]
    shape = (x.shape[1], y.shape[1], len(tr.benchmark_names), 12, 8)
    torch.manual_seed(409)
    rng = np.random.default_rng(409)
    net = EpisodicRouter(*shape).to(args.device)
    net.center.copy_(x.mean(0))
    net.scale.copy_(x.std(0).clamp_min(1e-4))
    net.axes.copy_(torch.as_tensor(bank.basis.axes[:, : net.rank]))
    hist_pc = (y - y.mean(1, keepdim=True)) @ net.axes
    net.axis_scale.copy_(hist_pc.std(0))
    for i in range(len(tr.benchmark_names)):
        net.domain_axes[i] = hist_pc[domains == i].mean(0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0007, weight_decay=0.003)
    pools = [np.flatnonzero(tr.domains == k) for k in range(len(tr.benchmark_names))]
    all_models = np.arange(tr.n_models)
    # Some tournaments contain mostly competitive peers, as the top-two decision drives routing.
    strong = np.argsort(bank.domain_means.mean(0))[-(tr.n_models // 2) :]
    base = np.ones(tr.n_models)
    base[strong] += 1.5
    base /= base.sum()
    stages = [150, 500, 1200, 2500, 5000]
    output = args.output / "episodic"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(1, stages[-1] + 1):
        draws = np.stack([rng.choice(p, 80, replace=False) for p in pools])
        support = torch.as_tensor(draws[:, :40].ravel(), device=args.device)
        queries = torch.as_tensor(draws[:, 40:].ravel(), device=args.device)
        models = rng.choice(all_models, 13, replace=False, p=base if step % 2 else None)
        refs = torch.as_tensor(np.setdiff1d(all_models, models), device=args.device)
        models = torch.as_tensor(models, device=args.device)
        design, difficulty = net.support_design(x[support], domains[support], y[support], refs)
        profiles = net.fit(design, y[support][:, models], difficulty)
        pred, skill, domain_logits = net.text(x[queries])
        query = net.design(pred, skill, domain_logits.softmax(-1))
        scores = profiles @ query.T
        truth = y[queries][:, models].T
        difference = scores[:, None, :] - scores[None, :, :]
        won = truth[:, None, :] * (1 - truth[None, :, :])
        top = torch.softmax(scores.detach().mean(-1) * 10, dim=0)
        importance = (1 + 13 * (top[:, None] + top[None, :]))[:, :, None]
        pair_weights = won * importance
        ranking = (F.softplus(-5 * difference) * pair_weights).sum() / pair_weights.sum().clamp_min(1)
        measured, _ = net.observed_axes(y[queries], refs)
        loss = (
            ranking
            + 0.10 * ((pred - measured) / net.axis_scale).square().mean()
            + 0.06 * F.cross_entropy(domain_logits, domains[queries])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        optimizer.step()
        if step not in stages:
            continue
        net.cpu().eval()
        torch.save({"shape": shape, "state": net.state_dict()}, output / f"step_{step}.pt")
        query = net.query(d.embeddings[s.val_items])
        val_truth = d.labels[np.ix_(s.val_models, s.val_items)]
        for policy in ("balanced", "informative", "representative"):
            choices = []
            for seed in bank.seeds:
                sel = bank.orders[policy, seed]
                items = bank.items[sel]
                z = net.onboard(
                    d.embeddings[items],
                    d.domains[items],
                    bank.history[:, sel].T,
                    d.labels[np.ix_(s.val_models, items)],
                )
                choices.append((z @ query.T).argmax(0))
            rows.append(
                dict(
                    step=step,
                    policy=policy,
                    **metrics(np.stack(choices), val_truth, d.domains[s.val_items]),
                    seconds=perf_counter() - start,
                )
            )
        atomic_json(output / "validation.json", rows)
        print(max(rows[-3:], key=lambda row: row["objective"]), flush=True)
        net.to(args.device).train()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--device", default="cpu")
    opts = p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        train(opts)
