"""Predeclared checks on information contained in revealed probe responses."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def permute_support_within_benchmark(
    responses_prefix: Sequence[float],
    benchmark_ids_prefix: Sequence[Any],
    seed: int,
) -> np.ndarray:
    """Shuffle only revealed answers within their benchmark, preserving totals.

    Both arguments must already be sliced to the experiment's budget. Passing
    only the revealed prefix keeps responses outside that budget unavailable to
    this diagnostic. The acquisition order and selected prompt IDs stay fixed.
    Missing positions remain missing, and inputs are never mutated.
    """
    responses = np.asarray(responses_prefix)
    benchmarks = np.asarray(benchmark_ids_prefix)
    if responses.ndim != 1 or benchmarks.ndim != 1 or responses.shape != benchmarks.shape:
        raise ValueError("responses and benchmark IDs must be equal-length one-dimensional prefixes")
    if responses.dtype.kind not in "fiu":
        raise ValueError("responses must be numeric binary values or NaN")
    if np.isinf(responses).any():
        raise ValueError("responses may contain NaN, but not infinity")
    finite = np.isfinite(responses)
    if not np.isin(responses[finite], [0.0, 1.0]).all():
        raise ValueError("revealed responses must be binary or NaN")
    if benchmarks.dtype.kind in "fc" and not np.isfinite(benchmarks).all():
        raise ValueError("benchmark IDs must be finite")
    shuffled = responses.copy()
    rng = np.random.default_rng(seed)
    for benchmark in np.unique(benchmarks):
        positions = np.flatnonzero(finite & (benchmarks == benchmark))
        shuffled[positions] = rng.permutation(responses[positions])
    return shuffled
