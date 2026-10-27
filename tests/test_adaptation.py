import numpy as np
from scipy.optimize import minimize

from coldstart.adaptation import adapt_map, predict_from_latent, sigmoid


def test_map_matches_independent_optimizer_and_posterior_hessian():
    rng = np.random.default_rng(71)
    design = rng.normal(size=(67, 5))
    offsets = rng.normal(scale=0.3, size=67)
    mean = rng.normal(scale=0.2, size=5)
    precision = np.diag([0.7, 1.0, 2.0, 3.0, 4.0])
    y = rng.binomial(1, sigmoid(offsets + design @ rng.normal(size=5)))
    fit = adapt_map(design, offsets, y, mean, precision)

    def objective(latent):
        logits = offsets + design @ latent
        delta = latent - mean
        return np.logaddexp(0, logits).sum() - y @ logits + 0.5 * delta @ precision @ delta

    reference = minimize(objective, mean, method="BFGS", options={"gtol": 1e-7})
    np.testing.assert_allclose(fit.latent, reference.x, atol=2e-6)
    assert fit.converged
    assert fit.gradient_norm < 1e-6
    probabilities = sigmoid(offsets + design @ fit.latent)
    hessian = design.T @ ((probabilities * (1 - probabilities))[:, None] * design) + precision
    np.testing.assert_allclose(fit.covariance @ hessian, np.eye(5), atol=1e-10)


def test_empty_observations_recover_prior_and_separation_is_finite():
    design = np.ones((20, 2))
    precision = np.diag([0.3, 2.0])
    mean = np.array([0.7, -0.2])
    empty = adapt_map(design, np.zeros(20), np.full(20, np.nan), mean, precision)
    np.testing.assert_allclose(empty.latent, mean)
    np.testing.assert_allclose(empty.covariance, np.linalg.inv(precision))
    separated = adapt_map(design, np.zeros(20), np.ones(20), mean, precision)
    assert separated.converged and np.all(np.isfinite(separated.latent))
    assert np.all(np.linalg.eigvalsh(separated.covariance) > 0)


def test_likelihood_is_sum_so_more_observations_reduce_prior_shrinkage():
    single = adapt_map(np.ones((1, 1)), np.zeros(1), np.ones(1), np.zeros(1), np.ones(1))
    repeated = adapt_map(np.ones((20, 1)), np.zeros(20), np.ones(20), np.zeros(1), np.ones(1))
    assert repeated.latent[0] > single.latent[0]
    assert repeated.covariance[0, 0] < single.covariance[0, 0]


def test_extreme_logits_and_batched_prediction_are_stable():
    np.testing.assert_allclose(sigmoid(np.array([-10000.0, 0.0, 10000.0])), [0, 0.5, 1])
    predictions = predict_from_latent(np.array([[0.0], [1.0]]), np.zeros(3), np.ones((3, 1)))
    assert predictions.shape == (2, 3)
    np.testing.assert_allclose(predictions[0], 0.5)

