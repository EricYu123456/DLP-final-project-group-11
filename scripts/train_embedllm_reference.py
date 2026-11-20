"""Train the original TextMF architecture on the project's matched question split."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from evaluate_metadata_excluded import load_oracle
from pilot_routing_learning import ROOT, input_context


class TextMF(nn.Module):
    """EmbedLLM's model lookup, noisy linear text projection and two-class head."""

    def __init__(self, models, text_dim):
        super().__init__()
        self.profile = nn.Embedding(models, 232)
        self.projection = nn.Linear(text_dim, 232)
        self.classifier = nn.Linear(232, 2)

    def forward(self, models, vectors):
        noisy_text = vectors + 0.05 * torch.randn_like(vectors)
        return self.classifier(self.profile(models) * self.projection(noisy_text))


def fit(vectors, labels, model_ids, prompt_ids, output, mode):
    # The public CSV is model-major and prompt-ordered. Its DataLoader has shuffle=False.
    order = np.argsort(prompt_ids)
    labels = labels[:, order]
    rows, cols = np.where(np.isfinite(labels))
    rows_gpu = torch.from_numpy(rows).cuda()
    cols_gpu = torch.from_numpy(cols).cuda()
    truth = torch.from_numpy(labels[rows, cols].astype(np.int64)).cuda()
    questions = torch.from_numpy(vectors[order]).cuda()
    torch.manual_seed(42)
    net = TextMF(len(model_ids), questions.shape[1]).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    torch.cuda.synchronize()
    start = perf_counter()
    for epoch in range(1, 51):
        for first in range(0, len(rows), 2048):
            batch = slice(first, first + 2048)
            logits = net(rows_gpu[batch], questions[cols_gpu[batch]])
            loss = nn.functional.cross_entropy(logits, truth[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            torch.cuda.synchronize()
            print(f"{mode} epoch {epoch}/50, {perf_counter() - start:.1f}s", flush=True)
    with torch.no_grad():
        weight = torch.diff(net.classifier.weight, dim=0).squeeze(0)
        projection = weight[:, None] * net.projection.weight
        bias = weight * net.projection.bias
        np.savez(
            output / f"embedllm_{mode}.npz",
            profiles=net.profile.weight.cpu().numpy(),
            projection=projection.cpu().numpy(),
            bias=bias.cpu().numpy(),
            offset=float(torch.diff(net.classifier.bias).item()),
            model_ids=model_ids,
            n_fit_per_model=np.isfinite(labels).sum(axis=1),
            seconds=perf_counter() - start,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--mode", choices=["historical", "known_ids"], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    d, s, _, _, prepared, family = input_context(args.run)
    bank = np.asarray(s.train_items)
    if args.mode == "historical":
        ids = d.model_ids[s.train_models]
        labels = d.labels[np.ix_(s.train_models, bank)]
    else:
        extra = load_oracle(d, prepared / "embedllm.npz", family)
        ids = np.r_[d.model_ids, extra.model_ids]
        labels = np.vstack([d.labels[:, bank], extra.labels[:, bank]])
        order = ids.argsort()
        ids, labels = ids[order], labels[order]
    fit(d.embeddings[bank], labels, ids, d.prompt_ids[bank], args.output, args.mode)


if __name__ == "__main__":
    main()
