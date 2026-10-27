"""Choose probes to reduce approximate benchmark-score uncertainty.

The caller supplies a reference bank from the allowed support pool, never final
target items. This is a local Laplace/Fisher approximation, not a calibrated
confidence interval or a guarantee of smaller benchmark prediction error.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from .adaptation import adapt_map, sigmoid


@dataclass
class AdaptiveProbeResult:
    indices: np.ndarray
    response_count: int
    seconds: float
    trace: list[dict]


def benchmark_score_gradients(
    reference_design: np.ndarray,
    reference_offsets: np.ndarray,
    reference_benchmark_ids: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    """One gradient of the mean predicted score per reference benchmark."""
    design = np.asarray(reference_design, dtype=np.float64)
    offsets = np.asarray(reference_offsets, dtype=np.float64)
    domains = np.asarray(reference_benchmark_ids)
    latent = np.asarray(latent, dtype=np.float64)
    if design.ndim != 2 or len(design) == 0:
        raise ValueError("reference_design must be a nonempty matrix")
    if offsets.shape != (len(design),) or domains.shape != (len(design),):
        raise ValueError("Reference offsets and benchmark IDs must match design rows")
    if latent.shape != (design.shape[1],):
        raise ValueError("Latent dimension does not match reference design")
    if not (np.isfinite(design).all() and np.isfinite(offsets).all()
            and np.isfinite(latent).all()):
        raise ValueError("Reference features and latent must be finite")
    probabilities = sigmoid(offsets + design @ latent)
    weighted = (probabilities * (1 - probabilities))[:, None] * design
    return np.stack([weighted[domains == domain].mean(axis=0)
                     for domain in np.unique(domains)])


def score_variance_reduction(
    candidate_design: np.ndarray,
    candidate_offsets: np.ndarray,
    latent: np.ndarray,
    covariance: np.ndarray,
    score_gradients: np.ndarray,
) -> np.ndarray:
    """Expected local reduction of equally weighted benchmark-score variance.

    For candidate feature c, Bernoulli information w, covariance S, and each
    benchmark gradient g, the reduction is w*(g.T*S*c)^2/(1+w*c.T*S*c).
    This exact rank-one identity holds for the fixed local gradients/Hessian.
    """
    design = np.asarray(candidate_design, dtype=np.float64)
    offsets = np.asarray(candidate_offsets, dtype=np.float64)
    latent = np.asarray(latent, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    gradients = np.asarray(score_gradients, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError("candidate_design must be a matrix")
    n, dimension = design.shape
    if (offsets.shape != (n,) or latent.shape != (dimension,)
            or covariance.shape != (dimension, dimension)
            or gradients.ndim != 2 or gradients.shape[1] != dimension
            or len(gradients) == 0):
        raise ValueError("Candidate, covariance, or gradient dimensions differ")
    if not all(np.isfinite(value).all() for value in
               (design, offsets, latent, covariance, gradients)):
        raise ValueError("Acquisition inputs must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-9, rtol=1e-9):
        raise ValueError("Covariance must be symmetric")
    probabilities = sigmoid(offsets + design @ latent)
    weights = probabilities * (1 - probabilities)
    projected = design @ covariance
    latent_variance = np.einsum("ij,ij->i", projected, design)
    if np.any(latent_variance < -1e-8):
        raise ValueError("Covariance produces a negative candidate variance")
    score_projection = projected @ gradients.T
    return (weights * np.mean(score_projection ** 2, axis=1)
            / (1 + weights * np.maximum(latent_variance, 0)))


def select_adaptive_probe_order(
    candidate_design: np.ndarray,
    candidate_offsets: np.ndarray,
    candidate_benchmark_ids: np.ndarray,
    responses: np.ndarray | Callable[[int], float],
    reference_design: np.ndarray,
    reference_offsets: np.ndarray,
    reference_benchmark_ids: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    budget: int,
    strategy: str,
    seed: int,
    check_deadline: Callable[[], None] | None = None,
    *,
    warmup: int = 10,
) -> AdaptiveProbeResult:
    """Return candidate-local indices, revealing exactly one response per step.

    Both policies start with seeded balanced random probes. ``fisher`` then
    selects the greatest expected score-variance reduction globally;
    ``balanced_fisher`` cycles through benchmarks with available candidates.
    The reference bank must cover the candidate benchmarks. No unselected
    response is read, including for missing-value checks or tie breaking.
    """
    start = time.perf_counter()
    design = np.asarray(candidate_design, dtype=np.float64)
    offsets = np.asarray(candidate_offsets, dtype=np.float64)
    domains = np.asarray(candidate_benchmark_ids)
    mean = np.asarray(prior_mean, dtype=np.float64)
    precision = np.asarray(prior_precision, dtype=np.float64)
    if precision.ndim == 1:
        precision = np.diag(precision)
    if strategy not in {"fisher", "balanced_fisher"}:
        raise ValueError("strategy must be fisher or balanced_fisher")
    if design.ndim != 2 or offsets.shape != (len(design),) or domains.shape != (len(design),):
        raise ValueError("Candidate arrays have inconsistent shapes")
    if not 0 <= budget <= len(design) or warmup < 0:
        raise ValueError("Invalid budget or warmup")
    if not set(np.unique(domains)).issubset(set(np.unique(reference_benchmark_ids))):
        raise ValueError("Reference bank must cover every candidate benchmark")
    if not callable(responses) and np.shape(responses) != (len(design),):
        raise ValueError("Responses must match candidates or be an index callback")
    if check_deadline:
        check_deadline()
    # Empty-support MAP validates the prior and obtains its covariance.
    fit = adapt_map(design[:0], offsets[:0], np.empty(0), mean, precision,
                    check_deadline=check_deadline)
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(design))
    available = np.ones(len(design), dtype=bool)
    domain_cycle = list(np.unique(domains))
    warmup_orders = {
        domain: np.random.default_rng(np.random.SeedSequence([seed, int(domain), 7411]))
        .permutation(np.flatnonzero(domains == domain))
        for domain in domain_cycle
    }
    next_domain = 0
    selected: list[int] = []
    observed: list[float] = []
    trace: list[dict] = []
    for step in range(budget):
        if check_deadline:
            check_deadline()
        eligible = available.copy()
        balanced_step = step < warmup or strategy == "balanced_fisher"
        if balanced_step:
            # Preserve quotas for nested prefixes and skip exhausted benchmarks.
            for _ in domain_cycle:
                domain = domain_cycle[next_domain % len(domain_cycle)]
                next_domain += 1
                if np.any(available & (domains == domain)):
                    eligible &= domains == domain
                    break
        eligible_order = tie_order[eligible[tie_order]]
        if len(eligible_order) == 0:
            raise RuntimeError("No eligible candidate before budget was reached")
        score = None
        if step < warmup:
            # Match the balanced_random control's prefix for this same seed.
            random_order = warmup_orders[domain]
            index = int(random_order[available[random_order]][0])
        else:
            gradients = benchmark_score_gradients(reference_design, reference_offsets,
                                                  reference_benchmark_ids, fit.latent)
            values = score_variance_reduction(design[eligible_order], offsets[eligible_order],
                                              fit.latent, fit.covariance, gradients)
            local = int(np.argmax(values))
            index = int(eligible_order[local])
            score = float(values[local])
        response = float(responses(index) if callable(responses) else responses[index])
        # Failed/missing observations still consume a probe, but do not fit a label.
        if np.isinf(response) or (np.isfinite(response) and response not in (0.0, 1.0)):
            raise ValueError("An observed response must be binary or NaN")
        selected.append(index)
        observed.append(response)
        available[index] = False
        fit = adapt_map(design[selected], offsets[selected], np.asarray(observed), mean,
                        precision, initial=fit.latent, check_deadline=check_deadline)
        trace.append({"step": step + 1, "candidate_index": index,
                      "benchmark_id": int(domains[index]), "acquisition_score": score,
                      "observed": bool(np.isfinite(response)),
                      "converged": fit.converged})
    return AdaptiveProbeResult(np.asarray(selected, dtype=np.int64), len(selected),
                               time.perf_counter() - start, trace)

