import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("hybrid_pilot", Path(__file__).resolve().parents[1] / "scripts/analyze_hybrid_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_evidence_matches_exact_gaussian_marginal_likelihood():
    # Laplace integration is exact for this independent Gaussian reference.
    rng = np.random.default_rng(44)
    design = rng.normal(size=(7, 3))
    truth = rng.normal(size=7)
    precision = np.diag([.4, 1.2, 2.3])
    variance = .7
    hessian = precision + design.T @ design / variance
    posterior_covariance = np.linalg.inv(hessian)
    latent = posterior_covariance @ design.T @ truth / variance
    residual = truth - design @ latent
    objective = .5 * (residual @ residual / variance + len(truth) * np.log(2 * np.pi * variance)
                       + latent @ precision @ latent)
    actual = pilot.laplace_log_evidence([objective], precision, posterior_covariance[None])[0]
    marginal_covariance = variance * np.eye(len(truth)) + design @ np.linalg.inv(precision) @ design.T
    expected = -.5 * (len(truth) * np.log(2 * np.pi) + np.linalg.slogdet(marginal_covariance)[1]
                      + truth @ np.linalg.solve(marginal_covariance, truth))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_zero_support_and_affine_coordinate_invariance():
    precision = np.diag([.5, 1., 2.])
    covariance = np.diag([.3, .4, .2])
    zero = pilot.laplace_log_evidence([0], precision, np.linalg.inv(precision)[None])
    np.testing.assert_allclose(zero, [0], atol=1e-12)
    transform = np.array([[2., .2, 0], [0, .7, .1], [.1, 0, 1.2]])
    inverse = np.linalg.inv(transform)
    actual = pilot.laplace_log_evidence([9.5], precision, covariance[None])
    changed = pilot.laplace_log_evidence([9.5], inverse.T @ precision @ inverse,
                                        (transform @ covariance @ transform.T)[None])
    np.testing.assert_allclose(actual, changed, atol=1e-12)
