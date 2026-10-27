"""Score unseen-model predictions without treating responses as independent models."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _matrix(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError(f"{name} must have shape [models, items] and be nonempty")
    return result


def _row_mean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.where(finite, values, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def _observations(
    y_true: Any, y_prob: Any, mask: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = _matrix(y_true, "y_true")
    predicted = _matrix(y_prob, "y_prob")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if np.isinf(actual).any():
        raise ValueError("y_true may contain NaN for missing labels, but not infinity")
    observed = np.isfinite(actual)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.ndim == 1 and actual.shape[0] == 1:
            supplied = supplied[None, :]
        if supplied.shape != actual.shape:
            raise ValueError("mask must have the same shape as y_true")
        observed &= supplied
    if ((actual[observed] < 0.0) | (actual[observed] > 1.0)).any():
        raise ValueError("observed labels must be in [0, 1]")
    observed_predictions = predicted[observed]
    if not np.isfinite(observed_predictions).all():
        raise ValueError("predictions must be finite for every observed label")
    if ((observed_predictions < 0.0) | (observed_predictions > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return actual, predicted, observed


def benchmark_metrics(
    y_true: Any,
    y_prob: Any,
    domain_ids: Sequence[Any],
    mask: Any | None = None,
    *,
    require_all_domains: bool = True,
    epsilon: float = 1e-7,
) -> dict[str, Any]:
    """Return benchmark score errors and response-level proper scores.

    Rows are held-out models; columns are the same held-out items for every
    predictor. A benchmark's true and predicted scores use identical observed
    cells. By default every model must have an observed item in every benchmark;
    this prevents an incomplete model from silently receiving an easier macro
    average. ``require_all_domains=False`` permits coverage diagnostics, with NaN
    for missing benchmark scores. Scores use proportions, not percentage points.
    """
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    actual, predicted, observed = _observations(y_true, y_prob, mask)
    domains = np.asarray(domain_ids)
    if domains.ndim != 1 or len(domains) != actual.shape[1]:
        raise ValueError("domain_ids must have one entry per item")
    if domains.dtype.kind in "fc" and not np.isfinite(domains).all():
        raise ValueError("domain_ids must be finite")
    unique_domains = np.unique(domains)
    shape = (actual.shape[0], len(unique_domains))
    counts = np.zeros(shape, dtype=np.int64)
    true_scores = np.full(shape, np.nan)
    pred_scores = np.full(shape, np.nan)
    domain_brier = np.full(shape, np.nan)
    domain_nll = np.full(shape, np.nan)
    safe_probabilities = np.clip(predicted, epsilon, 1.0 - epsilon)
    squared_error = (predicted - actual) ** 2
    log_loss = -(
        actual * np.log(safe_probabilities)
        + (1.0 - actual) * np.log1p(-safe_probabilities)
    )
    for index, domain in enumerate(unique_domains):
        cells = observed & (domains[None, :] == domain)
        count = cells.sum(axis=1)
        counts[:, index] = count
        for source, destination in (
            (actual, true_scores),
            (predicted, pred_scores),
            (squared_error, domain_brier),
            (log_loss, domain_nll),
        ):
            total = np.where(cells, source, 0.0).sum(axis=1)
            destination[:, index] = np.divide(
                total, count, out=np.full(actual.shape[0], np.nan), where=count > 0
            )
    if require_all_domains and (counts == 0).any():
        raise ValueError("every model must have observed labels in every domain")
    domain_count = (counts > 0).sum(axis=1)
    item_count = counts.sum(axis=1)
    signed_error = pred_scores - true_scores
    absolute_error = np.abs(signed_error)
    absolute_for_max = np.where(np.isfinite(absolute_error), absolute_error, -np.inf)
    positive_for_max = np.where(np.isfinite(signed_error), np.maximum(signed_error, 0), -np.inf)
    worst_domain = absolute_for_max.max(axis=1)
    worst_overestimate = positive_for_max.max(axis=1)
    worst_domain[domain_count == 0] = np.nan
    worst_overestimate[domain_count == 0] = np.nan

    def item_average(values: np.ndarray) -> np.ndarray:
        total = np.where(observed, values, 0.0).sum(axis=1)
        return np.divide(
            total, item_count, out=np.full(actual.shape[0], np.nan), where=item_count > 0
        )

    weighted_error = np.where(counts > 0, absolute_error, 0.0) * counts
    per_model = {
        "macro_mae": _row_mean(absolute_error),
        "worst_domain_mae": worst_domain,
        "worst_overestimate": worst_overestimate,
        "micro_score_mae": np.abs(item_average(predicted) - item_average(actual)),
        "weighted_domain_mae": np.divide(
            weighted_error.sum(axis=1),
            item_count,
            out=np.full(actual.shape[0], np.nan),
            where=item_count > 0,
        ),
        "brier": item_average(squared_error),
        "macro_brier": _row_mean(domain_brier),
        "nll": item_average(log_loss),
        "macro_nll": _row_mean(domain_nll),
        "domain_count": domain_count,
        "item_count": item_count,
    }
    summary: dict[str, Any] = {
        name: float(values[np.isfinite(values)].mean())
        if np.isfinite(values).any()
        else float("nan")
        for name, values in per_model.items()
        if name not in {"domain_count", "item_count"}
    }
    summary.update(
        models=int(actual.shape[0]),
        domains=int(len(unique_domains)),
        complete_models=int((domain_count == len(unique_domains)).sum()),
        observed_responses=int(observed.sum()),
    )
    return {
        "domain_ids": unique_domains,
        "counts": counts,
        "true_scores": true_scores,
        "pred_scores": pred_scores,
        "absolute_error": absolute_error,
        "signed_error": signed_error,
        "domain_brier": domain_brier,
        "domain_nll": domain_nll,
        "per_model": per_model,
        "summary": summary,
    }


def paired_bootstrap(
    values_a: Any,
    values_b: Any | None = None,
    *,
    groups: Sequence[Any] | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Bootstrap a model mean or a paired difference (A minus B).

    Inputs have shape [models] or [seeds, models]. Seeds are averaged within
    each model before resampling. With lineage ``groups``, whole groups are
    resampled, preserving the model-weighted estimand even for unequal group
    sizes. At least two independent units are needed for an interval. This is
    a percentile bootstrap interval, not a hypothesis-test p-value.
    """
    left = _matrix(values_a, "values_a")
    if values_b is None:
        differences = left
    else:
        right = _matrix(values_b, "values_b")
        if right.shape != left.shape:
            raise ValueError("paired values must have the same [seeds, models] shape")
        differences = left - right
    if not np.isfinite(differences).all():
        raise ValueError("paired values must be finite with a complete set of seeds")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    model_values = differences.mean(axis=0)
    model_count = len(model_values)
    if groups is None:
        group_labels = np.arange(model_count)
    else:
        group_labels = np.asarray(groups)
        if group_labels.ndim != 1 or len(group_labels) != model_count:
            raise ValueError("groups must have one entry per model")
    unique_groups, inverse = np.unique(group_labels, return_inverse=True)
    group_count = len(unique_groups)
    if group_count < 2:
        raise ValueError("a bootstrap interval needs at least two independent groups")
    totals = np.bincount(inverse, weights=model_values, minlength=group_count)
    sizes = np.bincount(inverse, minlength=group_count)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, 1024):
        stop = min(start + 1024, n_bootstrap)
        draws = rng.integers(group_count, size=(stop - start, group_count))
        samples[start:stop] = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "estimate": float(model_values.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "confidence": float(confidence),
        "n_models": int(model_count),
        "n_seeds": int(left.shape[0]),
        "n_groups": int(group_count),
        "n_bootstrap": int(n_bootstrap),
        "resampling_unit": "model" if groups is None else "group",
        "difference": "a_minus_b" if values_b is not None else "mean",
    }


def routing_metrics(
    y_true: Any,
    y_prob: Any,
    mask: Any | None = None,
    *,
    fixed_model: int | None = None,
) -> dict[str, Any]:
    """Describe quality-only routing on items observed for every candidate.

    Candidate rows are chosen without target labels. ``fixed_model`` must have
    been selected using training or validation data. Hindsight references are
    explicitly named; neither is a deployable policy. Incomplete item columns
    are excluded jointly so routers and fixed references use identical items.
    """
    actual, predicted, observed = _observations(y_true, y_prob, mask)
    common = observed.all(axis=0)
    if not common.any():
        raise ValueError("routing needs an item observed for every candidate model")
    truth = actual[:, common]
    probability = predicted[:, common]
    choices = probability.argmax(axis=0)
    quality = float(truth[choices, np.arange(truth.shape[1])].mean())
    oracle_quality = float(truth.max(axis=0).mean())
    result = {
        "quality": quality,
        "hindsight_oracle_quality": oracle_quality,
        "hindsight_oracle_regret": oracle_quality - quality,
        "hindsight_best_fixed_quality": float(truth.mean(axis=1).max()),
        "n_candidates": int(truth.shape[0]),
        "n_items": int(truth.shape[1]),
        "chosen_model_positions": choices,
        "evaluated_item_mask": common,
    }
    if fixed_model is not None:
        if not isinstance(fixed_model, (int, np.integer)) or not 0 <= fixed_model < truth.shape[0]:
            raise ValueError("fixed_model must index a candidate row")
        fixed_quality = float(truth[fixed_model].mean())
        result["validation_fixed_quality"] = fixed_quality
        result["gain_over_validation_fixed"] = quality - fixed_quality
    return result
