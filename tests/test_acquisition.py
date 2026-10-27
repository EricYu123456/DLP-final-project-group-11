import numpy as np
import pytest

from coldstart.acquisition import (benchmark_score_gradients,
                                   score_variance_reduction,
                                   select_adaptive_probe_order)
from coldstart.adaptation import sigmoid
from coldstart.probes import select_probe_order


def test_fisher_reduction_matches_direct_rank_one_inverse():
    rng = np.random.default_rng(531)
    raw = rng.normal(size=(5, 5))
    precision = raw.T @ raw + np.eye(5)
    covariance = np.linalg.inv(precision)
    design = rng.normal(size=(11, 5))
    offsets = rng.normal(size=11)
    latent = rng.normal(size=5)
    gradients = rng.normal(size=(3, 5))
    actual = score_variance_reduction(design, offsets, latent, covariance, gradients)
    weights = sigmoid(offsets + design @ latent)
    weights *= 1 - weights
    expected = []
    before = np.mean(np.einsum('ij,jk,ik->i', gradients, covariance, gradients))
    for row, weight in zip(design, weights):
        after_covariance = np.linalg.inv(precision + weight * np.outer(row, row))
        after = np.mean(np.einsum('ij,jk,ik->i', gradients, after_covariance, gradients))
        expected.append(before - after)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-11)
    assert np.all(actual >= 0)


def test_score_gradient_matches_finite_difference():
    rng = np.random.default_rng(44)
    design = rng.normal(size=(12, 4))
    offsets = rng.normal(size=12)
    domains = np.arange(12) % 3
    latent = rng.normal(size=4)
    actual = benchmark_score_gradients(design, offsets, domains, latent)
    for coordinate in range(4):
        shift = np.zeros(4)
        shift[coordinate] = 1e-6
        difference = (sigmoid(offsets + design @ (latent + shift))
                      - sigmoid(offsets + design @ (latent - shift))) / 2e-6
        expected = [difference[domains == domain].mean() for domain in range(3)]
        np.testing.assert_allclose(actual[:, coordinate], expected, atol=1e-9)


@pytest.mark.parametrize('strategy', ['fisher', 'balanced_fisher'])
def test_adaptive_selection_queries_only_selected_unique_responses(strategy):
    rng = np.random.default_rng(4)
    design = rng.normal(size=(30, 4))
    domains = np.arange(30) % 3
    responses = rng.integers(0, 2, size=30)
    reads = []
    def oracle(index):
        assert index not in reads
        reads.append(index)
        return responses[index]
    kwargs = dict(candidate_design=design, candidate_offsets=np.zeros(30),
                  candidate_benchmark_ids=domains, reference_design=design[:12],
                  reference_offsets=np.zeros(12), reference_benchmark_ids=domains[:12],
                  prior_mean=np.zeros(4), prior_precision=np.eye(4),
                  budget=12, strategy=strategy, seed=7, warmup=3)
    result = select_adaptive_probe_order(responses=oracle, **kwargs)
    np.testing.assert_array_equal(result.indices, reads)
    assert result.response_count == 12
    unselected = np.setdiff1d(np.arange(30), result.indices)
    changed = responses.copy()
    changed[unselected] = 1 - changed[unselected]
    repeat = select_adaptive_probe_order(responses=changed, **kwargs)
    np.testing.assert_array_equal(result.indices, repeat.indices)
    if strategy == 'balanced_fisher':
        np.testing.assert_array_equal(np.bincount(domains[result.indices]), [4, 4, 4])


def test_adaptive_deadline_is_checked_before_reading():
    class Stop(Exception):
        pass
    def stop():
        raise Stop
    with pytest.raises(Stop):
        select_adaptive_probe_order(np.ones((3, 1)), np.zeros(3), np.zeros(3),
                                    np.zeros(3), np.ones((3, 1)), np.zeros(3), np.zeros(3),
                                    np.zeros(1), np.ones((1, 1)), 1, 'fisher', 1, stop)


def test_nested_budgets_share_balanced_random_warmup_and_missing_costs_budget():
    rng = np.random.default_rng(19)
    design = rng.normal(size=(30, 3))
    domains = np.arange(30) % 3
    labels = np.full(30, np.nan)
    args = (design, np.zeros(30), domains, labels, design[:12], np.zeros(12),
            domains[:12], np.zeros(3), np.eye(3))
    short = select_adaptive_probe_order(*args, 12, 'balanced_fisher', 32, warmup=6)
    long = select_adaptive_probe_order(*args, 20, 'balanced_fisher', 32, warmup=6)
    control = select_probe_order(design, domains, np.arange(30), 6, 'balanced_random', 32)
    np.testing.assert_array_equal(short.indices, long.indices[:12])
    np.testing.assert_array_equal(short.indices[:6], control)
    assert short.response_count == 12
    assert all(not item['observed'] for item in short.trace)
