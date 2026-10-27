import numpy as np
import torch

from coldstart.models import (
    ColdStartDecoder, DecoderConfig, MeanBaselines, TrainConfig, fit_decoder,
    create_decoder, load_decoder, question_knn, save_decoder,
)


def test_rank_zero_has_only_global_and_benchmark_effects_and_checkpoint_roundtrip(tmp_path):
    decoder = ColdStartDecoder(DecoderConfig(6, 3, 4, rank=0, dropout=0.0))
    decoder.eval()
    x = np.random.default_rng(1).normal(size=(12, 6)).astype(np.float32)
    domains = np.arange(12) % 3
    offset, design = decoder.export_features(x, domains)
    assert design.shape == (12, 4)
    np.testing.assert_array_equal(design[:, 0], 1)
    np.testing.assert_array_equal(design[:, 1:].sum(axis=1), 1)
    path = tmp_path / "decoder.pt"
    save_decoder(path, decoder, {"training_models": [1, 3, 5, 7]})
    recovered, metadata = load_decoder(path)
    recovered_offset, recovered_design = recovered.export_features(x, domains)
    np.testing.assert_array_equal(offset, recovered_offset)
    np.testing.assert_array_equal(design, recovered_design)
    assert metadata["training_models"] == [1, 3, 5, 7]


def test_tiny_training_fits_only_provided_training_models_and_questions():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(48, 5)).astype(np.float32)
    domains = np.arange(48) % 2
    labels = rng.binomial(1, 0.6, size=(3, 48)).astype(np.float32)
    decoder = ColdStartDecoder(DecoderConfig(5, 2, 3, rank=2, hidden_dim=8, dropout=0.0))
    train = np.arange(32)
    config = TrainConfig(epochs=2, batch_questions=16, patience=2, warmup_epochs=0, device="cpu")
    result = fit_decoder(decoder, x, domains, labels, train, np.arange(32, 40), config)
    np.testing.assert_allclose(decoder.input_mean.numpy(), x[train].mean(axis=0), atol=1e-6)
    assert decoder.model_latents.weight.shape[0] == 3
    assert len(result["history"]) == 2
    mean, precision = decoder.latent_prior()
    assert mean.shape == (5,)
    assert np.all(np.linalg.eigvalsh(precision) > 0)


def test_means_and_knn_respect_missing_probes_and_empty_domains():
    baseline = MeanBaselines(2, smoothing=2).fit(np.array([[0, 1, 1, 1]]), np.array([0, 0, 1, 1]))
    predicted = baseline.probe_mean(np.array([[1, np.nan]]), np.array([0, 0]), np.array([0, 1]))
    np.testing.assert_allclose(predicted, [[2 / 3, 1]])
    predictions = question_knn(
        np.eye(2), np.array([[1, np.nan], [np.nan, np.nan]]), np.eye(2),
        prior_probabilities=np.array([0.2, 0.8]), n_neighbors=1,
    )
    np.testing.assert_allclose(predictions, [[0.6, 0.8], [0.2, 0.8]])


def test_seed_controls_both_initialization_and_complete_training():
    rng = np.random.default_rng(55)
    x = rng.normal(size=(24, 4)).astype(np.float32)
    labels = rng.binomial(1, 0.6, size=(3, 24)).astype(np.float32)
    domains = np.arange(24) % 2
    decoder_config = DecoderConfig(4, 2, 3, rank=2, hidden_dim=8, dropout=0.1)
    states = []
    for _ in range(2):
        decoder = create_decoder(decoder_config, seed=13)
        fit_decoder(decoder, x, domains, labels, np.arange(16), np.arange(16, 20),
                    TrainConfig(epochs=2, batch_questions=8, warmup_epochs=0, seed=13, device="cpu"))
        states.append({name: value.detach().clone() for name, value in decoder.state_dict().items()})
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name]), name


def test_coldstart_callback_selects_checkpoints_only_after_interaction_warmup():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(24, 4)).astype(np.float32)
    labels = rng.binomial(1, 0.6, size=(3, 24)).astype(np.float32)
    decoder = create_decoder(DecoderConfig(4, 2, 3, rank=2, hidden_dim=8), seed=4)
    called = []

    def selection(model, epoch):
        called.append(epoch)
        return {"metric": "coldstart_macro_mae", "value": abs(epoch - 4) / 100, "prior_strength": 1}

    result = fit_decoder(decoder, x, np.arange(24) % 2, labels, np.arange(16), np.arange(16, 20),
        TrainConfig(epochs=6, batch_questions=8, warmup_epochs=2, selection_interval=2, patience=1, seed=4, device="cpu"),
        selection_callback=selection)
    assert called == [4, 6]
    assert result["best_epoch"] == 4
    assert result["best_selection_metric"] == "coldstart_macro_mae"
    assert result["best_selection_value"] == 0
    assert all(not row["checkpoint_eligible"] for row in result["history"][:2])


def test_knn_unanimous_neighbors_stay_inside_probability_bounds():
    rng = np.random.default_rng(51)
    probes = rng.normal(size=(50, 16)).astype(np.float32)
    target = rng.normal(size=(200, 16)).astype(np.float32)
    predictions = question_knn(probes, np.stack([np.ones(50), np.zeros(50)]), target,
                               n_neighbors=20, prior_weight=0)
    np.testing.assert_array_equal(predictions[0], 1)
    np.testing.assert_array_equal(predictions[1], 0)

