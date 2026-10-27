"""Convex inference of a new model's latent vector from binary responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class AdaptationResult:
    latent: np.ndarray
    covariance: np.ndarray
    converged: bool
    iterations: int
    objective: float
    observed_responses: int
    gradient_norm: float

    def metadata(self) -> dict:
        return {
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "objective": float(self.objective),
            "observed_responses": int(self.observed_responses),
            "gradient_norm": float(self.gradient_norm),
        }


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate a sigmoid without overflow for extreme logits."""
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def predict_from_latent(
    latent: np.ndarray, offsets: np.ndarray, design: np.ndarray
) -> np.ndarray:
    """Return probabilities for one latent vector or an array of vectors."""
    latent = np.asarray(latent, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or offsets.shape != (design.shape[0],):
        raise ValueError("design must be [questions, latent_dim] and offsets [questions]")
    if latent.shape[-1] != design.shape[1]:
        raise ValueError("latent and design dimensions differ")
    return sigmoid(latent @ design.T + offsets).astype(np.float32)


def adapt_map(
    design: np.ndarray,
    offsets: np.ndarray,
    responses: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    max_iterations: int = 25,
    tolerance: float = 1e-7,
    check_deadline: Callable[[], None] | None = None,
) -> AdaptationResult:
    """Minimize SUM Bernoulli NLL plus a Gaussian prior using Newton steps.

    Missing responses are omitted. A positive-definite prior makes the optimum
    unique, including when all responses agree or the design is rank deficient.
    The returned covariance is the inverse posterior Hessian at the MAP point.
    """
    design = np.asarray(design, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    mean = np.asarray(prior_mean, dtype=np.float64)
    precision = np.asarray(prior_precision, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError("design must have shape [questions, latent_dim]")
    n_questions, latent_dim = design.shape
    if offsets.shape != (n_questions,) or responses.shape != (n_questions,):
        raise ValueError("offsets and responses must match design rows")
    if mean.shape != (latent_dim,):
        raise ValueError("prior_mean must match the latent dimension")
    if precision.ndim == 1:
        precision = np.diag(precision)
    if precision.shape != (latent_dim, latent_dim):
        raise ValueError("prior_precision must be a square matrix or diagonal")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(offsets)):
        raise ValueError("question features must be finite")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(precision)):
        raise ValueError("the prior must be finite")
    if not np.allclose(precision, precision.T, rtol=1e-9, atol=1e-10):
        raise ValueError("prior_precision must be symmetric")
    # Validate strictly positive precision before attempting the optimization.
    np.linalg.cholesky(precision)
    observed = np.isfinite(responses)
    if np.any((responses[observed] < 0) | (responses[observed] > 1)):
        raise ValueError("observed responses must lie in [0, 1]")
    design, offsets, responses = design[observed], offsets[observed], responses[observed]
    latent = mean.copy() if initial is None else np.asarray(initial, dtype=np.float64).copy()
    if latent.shape != mean.shape or not np.all(np.isfinite(latent)):
        raise ValueError("initial latent must be finite and match prior_mean")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")

    def objective(value: np.ndarray) -> float:
        logits = offsets + design @ value
        delta = value - mean
        return float(np.sum(np.logaddexp(0.0, logits) - responses * logits)
                     + 0.5 * delta @ precision @ delta)

    converged = False
    iterations = 0
    for iteration in range(max_iterations):
        if check_deadline is not None:
            check_deadline()
        logits = offsets + design @ latent
        probabilities = sigmoid(logits)
        gradient = design.T @ (probabilities - responses) + precision @ (latent - mean)
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (weights[:, None] * design) + precision
        gradient_norm = float(np.max(np.abs(gradient))) if latent_dim else 0.0
        iterations = iteration + 1
        if gradient_norm <= tolerance:
            converged = True
            break
        step = np.linalg.solve(hessian, gradient)
        decrement = float(gradient @ step)
        before = objective(latent)
        step_size = 1.0
        accepted = False
        for _ in range(30):
            candidate = latent - step_size * step
            if objective(candidate) <= before - 1e-4 * step_size * decrement:
                latent = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            # At machine precision the line search can no longer lower the NLL.
            converged = gradient_norm <= max(tolerance, 1e-6)
            break
        if float(np.max(np.abs(step_size * step))) <= tolerance:
            converged = True
            break

    probabilities = sigmoid(offsets + design @ latent)
    weights = probabilities * (1.0 - probabilities)
    hessian = design.T @ (weights[:, None] * design) + precision
    gradient = design.T @ (probabilities - responses) + precision @ (latent - mean)
    gradient_norm = float(np.max(np.abs(gradient))) if latent_dim else 0.0
    return AdaptationResult(
        latent=latent,
        covariance=np.linalg.solve(hessian, np.eye(latent_dim)),
        converged=bool(converged or gradient_norm <= tolerance),
        iterations=iterations,
        objective=objective(latent),
        observed_responses=int(observed.sum()),
        gradient_norm=gradient_norm,
    )


def adapt_many(
    design: np.ndarray,
    offsets: np.ndarray,
    responses: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    **kwargs,
) -> list[AdaptationResult]:
    """Fit one independent latent per row of a response matrix."""
    responses = np.asarray(responses)
    if responses.ndim != 2:
        raise ValueError("responses must have shape [models, questions]")
    return [adapt_map(design, offsets, row, prior_mean, prior_precision, **kwargs)
            for row in responses]
