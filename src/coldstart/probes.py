"""Static probe selection using question metadata and text embeddings only."""

from __future__ import annotations

from typing import Callable
import numpy as np


STRATEGIES = ("random", "balanced_random", "diversity", "balanced_diversity")


def _diverse_order(
    embeddings: np.ndarray,
    indices: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    *,
    device: str = "cpu",
    check_deadline: Callable[[], None] | None = None,
) -> np.ndarray:
    """Greedy cosine k-center, with a seeded first point and deterministic ties."""
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    first = int(rng.integers(len(indices)))
    selected = []
    nearest_similarity = np.full(len(indices), -np.inf, dtype=np.float32)
    chosen = np.zeros(len(indices), dtype=bool)
    tensor = None
    if device != "cpu":
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            tensor = torch.as_tensor(vectors, device=device)
    for step in range(budget):
        if check_deadline is not None:
            check_deadline()
        current = first if step == 0 else int(np.argmin(nearest_similarity))
        selected.append(current)
        chosen[current] = True
        if tensor is not None:
            similarity = (tensor @ tensor[current]).cpu().numpy()
        else:
            similarity = vectors @ vectors[current]
        nearest_similarity = np.maximum(nearest_similarity, similarity)
        nearest_similarity[chosen] = np.inf
    return indices[np.asarray(selected)]


def select_probe_order(
    embeddings: np.ndarray,
    benchmark_ids: np.ndarray,
    candidate_indices: np.ndarray,
    budget: int,
    strategy: str,
    seed: int,
    *,
    device: str = "cpu",
    check_deadline: Callable[[], None] | None = None,
) -> np.ndarray:
    """Select a nested probe order; each prefix is a valid smaller budget.

    Balanced policies cycle through sorted benchmark IDs, skipping exhausted
    benchmarks. Selection never reads any response labels.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown static probe strategy: {strategy}")
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    benchmark_ids = np.asarray(benchmark_ids, dtype=np.int64)
    if candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_indices must be a vector of unique indices")
    if budget < 0 or budget > len(candidates):
        raise ValueError("budget must be between zero and the candidate pool size")
    if np.any(candidates < 0) or np.any(candidates >= len(embeddings)):
        raise ValueError("candidate indices are out of range")
    if len(benchmark_ids) != len(embeddings):
        raise ValueError("benchmark IDs must match embeddings")
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return rng.permutation(candidates)[:budget]
    if strategy == "diversity":
        return _diverse_order(embeddings, candidates, budget, rng, device=device,
                              check_deadline=check_deadline)
    groups = {int(domain): candidates[benchmark_ids[candidates] == domain]
              for domain in np.unique(benchmark_ids[candidates])}
    # Derive counts from the same cycling rule that creates the final order.
    counts = {domain: 0 for domain in groups}
    domain_order = []
    while len(domain_order) < budget:
        for domain, indices in groups.items():
            if counts[domain] < len(indices) and len(domain_order) < budget:
                domain_order.append(domain)
                counts[domain] += 1
    orders = {}
    for domain, indices in groups.items():
        domain_rng = np.random.default_rng(np.random.SeedSequence([seed, domain, 7411]))
        if strategy == "balanced_random":
            orders[domain] = domain_rng.permutation(indices)[:counts[domain]]
        else:
            orders[domain] = _diverse_order(
                embeddings, indices, counts[domain], domain_rng, device=device,
                check_deadline=check_deadline,
            )
    positions = {domain: 0 for domain in groups}
    selected = []
    for domain in domain_order:
        selected.append(orders[domain][positions[domain]])
        positions[domain] += 1
    return np.asarray(selected, dtype=np.int64)
