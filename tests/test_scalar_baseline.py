import numpy as np
import pytest

from coldstart.adaptation import sigmoid
from coldstart.models import ScalarLogitShift


def test_scalar_training_rates_are_calibrated_and_offsets_satisfy_sufficient_statistics():
    labels = np.array([[0, 0, 1, np.nan], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=float)
    domains = np.array([0, 0, 1, 1])
    model = ScalarLogitShift(2).fit(labels, domains)
    np.testing.assert_allclose(sigmoid(model.benchmark_offsets), [0.5, 0.8])
    np.testing.assert_array_equal(model.training_counts, [[2, 1], [2, 2], [2, 2]])
    for ability, counts, successes in zip(model.training_offsets, model.training_counts, model.training_successes):
        gradient = counts @ sigmoid(model.benchmark_offsets + ability) - successes.sum() + 1e-6 * ability
        assert abs(gradient) < 1e-9
    assert model.training_offsets[0] < model.training_offsets[1] < model.training_offsets[2]


def test_scalar_prior_floor_empty_probes_and_additional_evidence_are_coherent():
    labels = np.tile([0, 1, 1, 0], (4, 1))
    model = ScalarLogitShift(2).fit(labels, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(model.prior_precision, [[25.0]])
    target = np.array([0, 1])
    empty = model.predict(np.empty(0), np.empty(0, dtype=int), target)
    np.testing.assert_allclose(empty[0], sigmoid(model.benchmark_offsets + model.prior_mean[0]))
    one = model.predict(np.ones(1), np.zeros(1, dtype=int), target)
    many = model.predict(np.ones(20), np.zeros(20, dtype=int), target)
    assert np.all(many > one) and np.all(one > empty)


def test_scalar_shift_uses_only_revealed_domain_totals_and_rejects_empty_training_models():
    model = ScalarLogitShift(2).fit(np.array([[0, 1, 0, 1], [0, 0, 1, 1]]), np.array([0, 0, 1, 1]))
    probes = np.array([0, 1, 0, 1, 0, 1])
    a = model.predict(np.array([0, 1, 1, 1, 1, 0]), probes, np.array([0, 1]))
    b = model.predict(np.array([1, 0, 0, 1, 1, 1]), probes, np.array([0, 1]))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="Every training model"):
        ScalarLogitShift(2).fit(np.array([[0, 1], [np.nan, np.nan]]), np.array([0, 1]))
