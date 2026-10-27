"""Scientific metric checks: weighting, missingness, pairing, and resampling."""

import numpy as np
import pytest

from coldstart.metrics import benchmark_metrics, paired_bootstrap, routing_metrics


def test_macro_weights_benchmarks_instead_of_response_count():
    actual = [[1, 1, 1, 0]]
    predicted = [[1, 1, 1, 1]]
    result = benchmark_metrics(actual, predicted, [0, 0, 0, 1])
    assert result["summary"]["macro_mae"] == pytest.approx(0.5)
    assert result["summary"]["micro_score_mae"] == pytest.approx(0.25)
    assert result["summary"]["weighted_domain_mae"] == pytest.approx(0.25)
    assert result["summary"]["brier"] == pytest.approx(0.25)
    assert result["summary"]["macro_brier"] == pytest.approx(0.5)
    assert result["summary"]["worst_domain_mae"] == pytest.approx(1.0)
    assert result["summary"]["worst_overestimate"] == pytest.approx(1.0)


def test_signed_domain_errors_do_not_cancel_in_macro_mae():
    result = benchmark_metrics([[0, 1]], [[0.5, 0.5]], [0, 1])
    assert result["summary"]["micro_score_mae"] == pytest.approx(0.0)
    assert result["summary"]["macro_mae"] == pytest.approx(0.5)
    np.testing.assert_allclose(result["signed_error"], [[0.5, -0.5]])


def test_missing_labels_are_not_zero_and_share_prediction_denominator():
    result = benchmark_metrics(
        [[1, np.nan, 0, 1]], [[0.75, np.nan, 0.25, 0.5]], [0, 0, 1, 1]
    )
    np.testing.assert_array_equal(result["counts"], [[1, 2]])
    np.testing.assert_allclose(result["true_scores"], [[1.0, 0.5]])
    np.testing.assert_allclose(result["pred_scores"], [[0.75, 0.375]])
    assert result["summary"]["macro_mae"] == pytest.approx(0.1875)


def test_incomplete_domain_cannot_silently_change_primary_metric():
    with pytest.raises(ValueError, match="every domain"):
        benchmark_metrics([[1, np.nan]], [[0.5, np.nan]], [0, 1])
    diagnostic = benchmark_metrics(
        [[1, np.nan]], [[0.5, np.nan]], [0, 1], require_all_domains=False
    )
    assert diagnostic["summary"]["complete_models"] == 0
    assert diagnostic["per_model"]["domain_count"].tolist() == [1]


def test_observed_probability_validation_and_finite_boundary_log_loss():
    with pytest.raises(ValueError, match="finite"):
        benchmark_metrics([[1]], [[np.nan]], [0])
    with pytest.raises(ValueError, match="probabilities"):
        benchmark_metrics([[1]], [[1.1]], [0])
    result = benchmark_metrics([[0, 1]], [[1, 0]], [0, 0])
    assert np.isfinite(result["summary"]["nll"])
    assert result["summary"]["brier"] == 1.0


def test_repeating_seeds_does_not_narrow_model_bootstrap_interval():
    a = np.array([0.1, 0.4, 0.2, 0.6])
    b = np.array([0.2, 0.3, 0.4, 0.5])
    once = paired_bootstrap(a, b, n_bootstrap=2000, seed=7)
    repeated = paired_bootstrap(
        np.tile(a, (3, 1)), np.tile(b, (3, 1)), n_bootstrap=2000, seed=7
    )
    assert repeated["n_seeds"] == 3
    assert repeated["n_models"] == 4
    for key in ("estimate", "ci_low", "ci_high"):
        assert once[key] == pytest.approx(repeated[key])


def test_group_bootstrap_resamples_related_models_together():
    result = paired_bootstrap(
        [0.0, 0.0, 1.0], groups=["family_a", "family_a", "family_b"],
        n_bootstrap=4000, seed=19,
    )
    assert result["estimate"] == pytest.approx(1 / 3)
    assert result["n_groups"] == 2
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 1.0
    with pytest.raises(ValueError, match="two independent groups"):
        paired_bootstrap([0.1, 0.2], groups=["a", "a"])


def test_bootstrap_is_paired_before_resampling():
    result = paired_bootstrap([0.2, 0.6, 0.9], [0.1, 0.5, 0.8], seed=31)
    assert result["estimate"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)
    assert result["difference"] == "a_minus_b"


def test_routing_uses_joint_support_and_labels_hindsight_references():
    actual = [[1, 0, np.nan], [0, 1, 1]]
    probability = [[0.9, 0.8, np.nan], [0.1, 0.7, 0.9]]
    result = routing_metrics(actual, probability, fixed_model=1)
    assert result["n_items"] == 2
    assert result["quality"] == pytest.approx(0.5)
    assert result["hindsight_oracle_quality"] == pytest.approx(1.0)
    assert result["hindsight_oracle_regret"] == pytest.approx(0.5)
    assert result["validation_fixed_quality"] == pytest.approx(0.5)
    np.testing.assert_array_equal(result["chosen_model_positions"], [0, 0])
