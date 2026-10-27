import numpy as np
import pytest

from coldstart.probes import STRATEGIES, select_probe_order


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_probe_selection_is_nested_unique_eligible_and_reproducible(strategy):
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(100, 8)).astype(np.float32)
    domains = np.arange(100) % 4
    candidates = np.arange(10, 90)
    long = select_probe_order(embeddings, domains, candidates, 32, strategy, 17)
    short = select_probe_order(embeddings, domains, candidates, 16, strategy, 17)
    np.testing.assert_array_equal(long[:16], short)
    assert len(np.unique(long)) == 32
    assert set(long) <= set(candidates)
    if strategy.startswith("balanced"):
        np.testing.assert_array_equal(np.bincount(domains[long]), [8, 8, 8, 8])


def test_balanced_selection_redistributes_exhausted_domains():
    embeddings = np.eye(10, dtype=np.float32)
    domains = np.array([0] + [1] * 9)
    selected = select_probe_order(embeddings, domains, np.arange(10), 7, "balanced_random", 9)
    np.testing.assert_array_equal(np.bincount(domains[selected]), [1, 6])
