"""Neural item decoder and few-response prediction baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adaptation import adapt_map, sigmoid
from .runtime import atomic_replace


@dataclass
class DecoderConfig:
    input_dim: int
    num_benchmarks: int
    num_models: int
    rank: int = 8
    hidden_dim: int = 128
    mode: str = "structured"
    dropout: float = 0.1
    num_item_biases: int = 0

    @property
    def latent_dim(self) -> int:
        return self.rank + (1 + self.num_benchmarks if self.mode == "structured" else 0)


@dataclass
class TrainConfig:
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    latent_l2: float = 0.002
    epochs: int = 100
    batch_questions: int = 1024
    patience: int = 10
    min_delta: float = 0.0001
    warmup_epochs: int = 5
    selection_metric: str = "validation_bce"
    selection_interval: int = 2
    selection_budget: int = 100
    selection_prior_strengths: tuple[float, ...] = (1.0,)
    seed: int = 0
    device: str = "auto"


class ColdStartDecoder(nn.Module):
    """A text-conditioned logistic factor model linear in each model's latent.

    Structured mode uses a global ability, benchmark residuals, and learned
    text interactions. The rank-zero version removes the text interactions.
    Lowrank mode retains only the learned interactions, as an EmbedLLM-style
    factorization control. Item bias is optional and only valid in a fixed bank.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        if config.mode not in {"structured", "lowrank"}:
            raise ValueError("mode must be structured or lowrank")
        if config.rank < 0 or config.latent_dim < 1:
            raise ValueError("the decoder needs at least one latent dimension")
        self.config = config
        self.register_buffer("input_mean", torch.zeros(config.input_dim))
        self.register_buffer("input_scale", torch.ones(config.input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.difficulty_head = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.difficulty_head.weight)
        nn.init.zeros_(self.difficulty_head.bias)
        self.benchmark_bias = nn.Parameter(torch.zeros(config.num_benchmarks))
        self.interaction_head = (nn.Linear(config.hidden_dim, config.rank)
                                 if config.rank else None)
        self.model_latents = nn.Embedding(config.num_models, config.latent_dim)
        nn.init.normal_(self.model_latents.weight, std=0.1)
        if config.mode == "structured":
            with torch.no_grad():
                self.model_latents.weight[:, :1 + config.num_benchmarks].zero_()
        self.item_bias = (nn.Embedding(config.num_item_biases, 1)
                          if config.num_item_biases else None)
        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)

    @torch.no_grad()
    def fit_input_normalizer(self, training_embeddings: torch.Tensor) -> None:
        if training_embeddings.shape[1] != self.config.input_dim:
            raise ValueError("embedding dimension differs from decoder config")
        mean = training_embeddings.mean(dim=0)
        scale = training_embeddings.std(dim=0, unbiased=False)
        floor = max(float(scale.square().mean().sqrt()) * 0.1, 1e-6)
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale.clamp_min(floor))

    @torch.no_grad()
    def initialize_abilities(self, labels: torch.Tensor, benchmark_ids: torch.Tensor) -> None:
        """Initialize intercepts using training-model/training-item labels only."""
        finite = torch.isfinite(labels)
        values = torch.nan_to_num(labels)
        counts = finite.sum(dim=1).clamp_min(1)
        overall = (values.sum() + 0.5) / (finite.sum() + 1.0)
        model_probability = (values.sum(dim=1) + 0.5) / (counts + 1.0)
        global_ability = torch.logit(model_probability.clamp(1e-4, 1 - 1e-4)) - torch.logit(overall)
        for benchmark in range(self.config.num_benchmarks):
            selected = benchmark_ids == benchmark
            group_finite = finite[:, selected]
            group_values = values[:, selected]
            probability = (group_values.sum() + 0.5) / (group_finite.sum() + 1.0)
            self.benchmark_bias[benchmark] = torch.logit(probability.clamp(1e-4, 1 - 1e-4))
            if self.config.mode == "structured":
                per_model = (group_values.sum(dim=1) + 0.5) / (group_finite.sum(dim=1) + 1.0)
                self.model_latents.weight[:, 1 + benchmark] = (
                    torch.logit(per_model.clamp(1e-4, 1 - 1e-4))
                    - self.benchmark_bias[benchmark] - global_ability
                )
        if self.config.mode == "structured":
            self.model_latents.weight[:, 0] = global_ability

    def item_features(
        self,
        embeddings: torch.Tensor,
        benchmark_ids: torch.Tensor,
        item_ids: torch.Tensor | None = None,
        *,
        include_interactions: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (embeddings - self.input_mean) / self.input_scale
        hidden = self.encoder(normalized)
        offsets = self.difficulty_head(hidden).squeeze(-1) + self.benchmark_bias[benchmark_ids]
        parts = []
        if self.config.mode == "structured":
            parts.extend([
                torch.ones((len(embeddings), 1), dtype=hidden.dtype, device=hidden.device),
                F.one_hot(benchmark_ids, num_classes=self.config.num_benchmarks).to(hidden.dtype),
            ])
        if self.interaction_head is not None:
            interaction = self.interaction_head(hidden)
            # Bounding the features prevents arbitrary encoder/latent rescaling.
            interaction = torch.tanh(interaction) / math.sqrt(self.config.rank)
            if not include_interactions:
                interaction = interaction * 0.0
            parts.append(interaction)
        design = torch.cat(parts, dim=1)
        if self.item_bias is not None:
            if item_ids is None:
                raise ValueError("fixed-bank item bias requires item_ids")
            offsets = offsets + self.item_bias(item_ids).squeeze(-1)
        return offsets, design

    def forward(
        self,
        embeddings: torch.Tensor,
        benchmark_ids: torch.Tensor,
        model_ids: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        *,
        include_interactions: bool = True,
    ) -> torch.Tensor:
        offsets, design = self.item_features(
            embeddings, benchmark_ids, item_ids, include_interactions=include_interactions
        )
        latents = (self.model_latents.weight if model_ids is None else self.model_latents(model_ids))
        return offsets.unsqueeze(0) + latents @ design.T

    @torch.no_grad()
    def export_features(
        self,
        embeddings: np.ndarray,
        benchmark_ids: np.ndarray,
        item_ids: np.ndarray | None = None,
        *,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        device = next(self.parameters()).device
        offsets, design = [], []
        for start in range(0, len(embeddings), batch_size):
            stop = start + batch_size
            batch_offsets, batch_design = self.item_features(
                torch.as_tensor(embeddings[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(benchmark_ids[start:stop], dtype=torch.long, device=device),
                (torch.as_tensor(item_ids[start:stop], dtype=torch.long, device=device)
                 if item_ids is not None else None),
            )
            offsets.append(batch_offsets.cpu().numpy())
            design.append(batch_design.cpu().numpy())
        return np.concatenate(offsets), np.concatenate(design)

    @torch.no_grad()
    def latent_prior(
        self,
        *,
        variance_floor: float = 0.2,
        variance_shrinkage: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate a diagonal training-model prior with a positive variance floor."""
        if variance_floor <= 0 or not 0 <= variance_shrinkage <= 1:
            raise ValueError("invalid prior variance floor or shrinkage")
        latents = self.model_latents.weight.detach().cpu().numpy().astype(np.float64)
        mean = latents.mean(axis=0)
        variances = latents.var(axis=0, ddof=1 if len(latents) > 1 else 0)
        blocks = ([slice(0, 1), slice(1, 1 + self.config.num_benchmarks),
                   slice(1 + self.config.num_benchmarks, None)]
                  if self.config.mode == "structured" else [slice(None)])
        for block in blocks:
            if len(variances[block]):
                variances[block] = ((1 - variance_shrinkage) * variances[block]
                                    + variance_shrinkage * variances[block].mean())
        variances = np.maximum(variances, variance_floor ** 2)
        return mean, np.diag(1.0 / variances)


def select_device(requested: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def create_decoder(config: DecoderConfig, seed: int) -> ColdStartDecoder:
    """Seed before parameter construction, as required for reproducible runs."""
    seed_everything(seed)
    return ColdStartDecoder(config)


def fit_decoder(
    decoder: ColdStartDecoder,
    embeddings: np.ndarray,
    benchmark_ids: np.ndarray,
    labels: np.ndarray,
    train_items: np.ndarray,
    val_items: np.ndarray,
    config: TrainConfig,
    *,
    check_deadline: Callable[[], None] | None = None,
    checkpoint_callback: Callable[[ColdStartDecoder, dict], None] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    selection_callback: Callable[[ColdStartDecoder, int], dict] | None = None,
) -> dict:
    """Fit training models and select a checkpoint by BCE or a supplied score."""
    seed_everything(config.seed)
    if config.epochs < 1 or config.batch_questions < 1 or config.patience < 1 or config.selection_interval < 1:
        raise ValueError("epochs, batch_questions, patience, and selection_interval must be positive")
    device = select_device(config.device)
    decoder.to(device)
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    domains = torch.as_tensor(benchmark_ids, dtype=torch.long, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    train_ids = np.asarray(train_items, dtype=np.int64)
    val_ids = np.asarray(val_items, dtype=np.int64)
    if y.shape[0] != decoder.config.num_models or y.shape[1] != len(x):
        raise ValueError("labels must have one row per training model and one column per item")
    needs_interaction_warmup = decoder.config.mode == "structured" and decoder.config.rank > 0
    if needs_interaction_warmup and config.epochs <= config.warmup_epochs:
        raise ValueError("epochs must exceed warmup_epochs so interaction parameters are trained")
    if len(train_ids) == 0 or len(val_ids) == 0 or np.intersect1d(train_ids, val_ids).size:
        raise ValueError("training and validation items must be nonempty and disjoint")
    train_tensor = torch.as_tensor(train_ids, device=device)
    decoder.fit_input_normalizer(x[train_tensor])
    decoder.initialize_abilities(y[:, train_tensor], domains[train_tensor])
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    generator = np.random.default_rng(config.seed)
    history = []
    best_state = None
    best_loss = float("inf")
    best_validation_bce = float("inf")
    best_selection_metric = "validation_bce"
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    def evaluate(item_indices: np.ndarray) -> float:
        decoder.eval()
        total_loss, total_count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(item_indices), config.batch_questions):
                indices = torch.as_tensor(item_indices[start:start + config.batch_questions], device=device)
                target = y[:, indices]
                observed = torch.isfinite(target)
                logits = decoder(x[indices], domains[indices], item_ids=indices)
                losses = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target), reduction="none")
                total_loss += float(losses[observed].sum())
                total_count += int(observed.sum())
        return total_loss / max(total_count, 1)

    try:
        for epoch in range(config.epochs):
            if check_deadline is not None:
                check_deadline()
            decoder.train()
            total_loss, total_count = 0.0, 0
            order = generator.permutation(train_ids)
            include_interactions = epoch >= config.warmup_epochs or decoder.config.mode == "lowrank"
            for start in range(0, len(order), config.batch_questions):
                if check_deadline is not None:
                    check_deadline()
                indices = torch.as_tensor(order[start:start + config.batch_questions], device=device)
                target = y[:, indices]
                observed = torch.isfinite(target)
                if not bool(observed.any()):
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = decoder(x[indices], domains[indices], item_ids=indices,
                                 include_interactions=include_interactions)
                losses = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target), reduction="none")
                likelihood = losses[observed].mean()
                # Equal per-coordinate shrinkage across rank-zero/rank-eight fits.
                penalized_latents = decoder.model_latents.weight
                if needs_interaction_warmup and not include_interactions:
                    penalized_latents = penalized_latents[:, :1 + decoder.config.num_benchmarks]
                penalty = config.latent_l2 * penalized_latents.square().sum(dim=1).mean()
                if decoder.item_bias is not None:
                    penalty = penalty + config.latent_l2 * decoder.item_bias.weight.square().mean()
                loss = likelihood + penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
                optimizer.step()
                total_loss += float(losses[observed].detach().sum())
                total_count += int(observed.sum())
            validation_bce = evaluate(val_ids)
            row = {"epoch": epoch + 1, "train_bce": total_loss / max(total_count, 1),
                   "validation_bce": validation_bce, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            eligible = include_interactions or not needs_interaction_warmup
            row["checkpoint_eligible"] = eligible
            selection = None
            if eligible:
                if selection_callback is None:
                    selection = {"metric": "validation_bce", "value": validation_bce}
                elif epoch == 0 or (epoch + 1) % config.selection_interval == 0:
                    selection = selection_callback(decoder, epoch + 1)
                    if not np.isfinite(float(selection["value"])):
                        raise ValueError("Checkpoint selection callback returned a non-finite score")
            if selection is not None:
                row["selection_metric"] = selection["metric"]
                row["selection_value"] = float(selection["value"])
                if "prior_strength" in selection:
                    row["selection_prior_strength"] = float(selection["prior_strength"])
            if selection is not None and selection["value"] < best_loss - config.min_delta:
                best_loss, best_epoch, stale_epochs = float(selection["value"]), epoch + 1, 0
                best_selection_metric = selection["metric"]
                best_validation_bce = validation_bce
                best_state = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
                if checkpoint_callback is not None:
                    checkpoint_callback(decoder, {"history": history, "best_epoch": best_epoch,
                        "best_validation_bce": best_validation_bce,
                        "best_selection_metric": best_selection_metric,
                        "best_selection_value": best_loss})
            elif selection is not None:
                stale_epochs += 1
            if progress_callback is not None:
                progress_callback(row)
            if stale_epochs >= config.patience and epoch >= config.warmup_epochs:
                break
    finally:
        if best_state is not None:
            decoder.load_state_dict(best_state)
        decoder.eval()
    return {"history": history, "best_epoch": best_epoch, "best_validation_bce": best_validation_bce,
            "best_selection_metric": best_selection_metric, "best_selection_value": best_loss,
            "minimum_validation_bce": min(row["validation_bce"] for row in history),
            "elapsed_seconds": time.perf_counter() - started, "device": str(device),
            "train_config": asdict(config), "decoder_config": asdict(decoder.config)}


def save_decoder(path: str | Path, decoder: ColdStartDecoder, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    state = {key: value.detach().cpu() for key, value in decoder.state_dict().items()}
    torch.save({"decoder_config": asdict(decoder.config), "state_dict": state,
                "metadata": metadata or {}}, temporary)
    atomic_replace(temporary, path)


def load_decoder(path: str | Path, device: str = "cpu") -> tuple[ColdStartDecoder, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    decoder = ColdStartDecoder(DecoderConfig(**checkpoint["decoder_config"]))
    decoder.load_state_dict(checkpoint["state_dict"])
    decoder.to(device).eval()
    return decoder, checkpoint.get("metadata", {})


class MeanBaselines:
    """Training priors and smoothed response means fitted without query labels."""

    def __init__(self, num_benchmarks: int, smoothing: float = 5.0):
        self.num_benchmarks = num_benchmarks
        self.smoothing = smoothing
        self.global_prior = 0.5
        self.benchmark_prior = np.full(num_benchmarks, 0.5, dtype=np.float64)

    def fit(self, labels: np.ndarray, benchmark_ids: np.ndarray) -> "MeanBaselines":
        labels = np.asarray(labels, dtype=np.float64)
        self.global_prior = float(np.nanmean(labels))
        for benchmark in range(self.num_benchmarks):
            group = labels[:, np.asarray(benchmark_ids) == benchmark]
            if np.any(np.isfinite(group)):
                self.benchmark_prior[benchmark] = float(np.nanmean(group))
            else:
                self.benchmark_prior[benchmark] = self.global_prior
        return self

    def training_mean(self, target_benchmarks: np.ndarray, num_models: int = 1) -> np.ndarray:
        return np.broadcast_to(self.benchmark_prior[target_benchmarks],
                               (num_models, len(target_benchmarks))).copy().astype(np.float32)

    def probe_mean(
        self, responses: np.ndarray, probe_benchmarks: np.ndarray, target_benchmarks: np.ndarray,
        *, by_benchmark: bool = True,
    ) -> np.ndarray:
        responses = np.atleast_2d(np.asarray(responses, dtype=np.float64))
        predictions = np.empty((len(responses), len(target_benchmarks)), dtype=np.float64)
        if not by_benchmark:
            counts = np.isfinite(responses).sum(axis=1)
            means = (np.nansum(responses, axis=1) + self.smoothing * self.global_prior) / (counts + self.smoothing)
            predictions[:] = means[:, None]
        else:
            for benchmark in range(self.num_benchmarks):
                selected = np.asarray(probe_benchmarks) == benchmark
                values = responses[:, selected]
                counts = np.isfinite(values).sum(axis=1)
                means = (np.nansum(values, axis=1) + self.smoothing * self.benchmark_prior[benchmark]) / (counts + self.smoothing)
                predictions[:, np.asarray(target_benchmarks) == benchmark] = means[:, None]
        return predictions.astype(np.float32)


class ScalarLogitShift:
    """A single Bayesian ability shift around training benchmark base rates."""

    def __init__(self, num_benchmarks: int, variance_floor: float = 0.2,
                 training_ridge: float = 1e-6):
        if num_benchmarks < 1 or variance_floor <= 0 or training_ridge <= 0:
            raise ValueError("Scalar baseline dimensions, scale floor, and ridge must be positive")
        self.num_benchmarks = num_benchmarks
        self.variance_floor = variance_floor
        self.training_ridge = training_ridge
        self.benchmark_offsets = None
        self.prior_mean = None
        self.prior_precision = None
        self.training_offsets = None
        self.training_counts = None
        self.training_successes = None

    def fit(self, labels: np.ndarray, benchmark_ids: np.ndarray) -> "ScalarLogitShift":
        """Fit offsets and a scalar population prior using supplied training data."""
        from scipy.optimize import brentq

        labels = np.asarray(labels, dtype=np.float64)
        domains = np.asarray(benchmark_ids, dtype=np.int64)
        if labels.ndim != 2 or domains.shape != (labels.shape[1],) or not len(labels):
            raise ValueError("Training labels must be [models, items] with one benchmark per item")
        if np.any((domains < 0) | (domains >= self.num_benchmarks)):
            raise ValueError("Unknown training benchmark ID")
        observed = np.isfinite(labels)
        if np.isinf(labels).any() or not np.isin(labels[observed], [0.0, 1.0]).all():
            raise ValueError("Training labels must be binary or NaN")
        counts = np.stack([observed[:, domains == domain].sum(axis=1)
                           for domain in range(self.num_benchmarks)], axis=1)
        successes = np.stack([np.nansum(labels[:, domains == domain], axis=1)
                              for domain in range(self.num_benchmarks)], axis=1)
        if np.any(counts.sum(axis=0) == 0) or np.any(counts.sum(axis=1) == 0):
            raise ValueError("Every training model and benchmark needs at least one observation")
        rates = np.clip(successes.sum(axis=0) / counts.sum(axis=0), 1e-6, 1 - 1e-6)
        offsets = np.log(rates) - np.log1p(-rates)
        abilities = []
        for model_counts, model_successes in zip(counts, successes):
            def gradient(ability):
                return float(model_counts @ sigmoid(offsets + ability)
                             - model_successes.sum() + self.training_ridge * ability)
            abilities.append(brentq(gradient, -64.0, 64.0, xtol=1e-12))
        abilities = np.asarray(abilities, dtype=np.float64)
        variance = max(float(abilities.var(ddof=1 if len(abilities) > 1 else 0)),
                       self.variance_floor ** 2)
        self.benchmark_offsets = offsets
        self.prior_mean = np.array([abilities.mean()])
        self.prior_precision = np.array([[1.0 / variance]])
        self.training_offsets = abilities
        self.training_counts = counts
        self.training_successes = successes
        return self

    def predict(self, responses: np.ndarray, probe_benchmarks: np.ndarray,
                target_benchmarks: np.ndarray, strength: float = 1.0,
                *, check_deadline: Callable[[], None] | None = None) -> np.ndarray:
        if self.benchmark_offsets is None:
            raise ValueError("Fit the scalar baseline before prediction")
        if strength <= 0 or not np.isfinite(strength):
            raise ValueError("Prior strength must be finite and positive")
        responses = np.atleast_2d(np.asarray(responses, dtype=np.float64))
        probe_benchmarks = np.asarray(probe_benchmarks, dtype=np.int64)
        target_benchmarks = np.asarray(target_benchmarks, dtype=np.int64)
        if responses.ndim != 2 or probe_benchmarks.shape != (responses.shape[1],):
            raise ValueError("Responses must match the supplied probe benchmark IDs")
        if target_benchmarks.ndim != 1 or np.any((target_benchmarks < 0) | (target_benchmarks >= self.num_benchmarks)) or np.any((probe_benchmarks < 0) | (probe_benchmarks >= self.num_benchmarks)):
            raise ValueError("Unknown benchmark ID")
        design = np.ones((responses.shape[1], 1), dtype=np.float64)
        offsets = self.benchmark_offsets[probe_benchmarks]
        predictions = []
        for response in responses:
            fit = adapt_map(design, offsets, response, self.prior_mean,
                            strength * self.prior_precision, check_deadline=check_deadline)
            predictions.append(sigmoid(self.benchmark_offsets[target_benchmarks] + fit.latent[0]))
        return np.asarray(predictions, dtype=np.float32)

    def metadata(self) -> dict:
        if self.benchmark_offsets is None:
            raise ValueError("Fit the scalar baseline before requesting metadata")
        return {"benchmark_offsets": self.benchmark_offsets.tolist(),
                "prior_mean": self.prior_mean.tolist(), "prior_precision": self.prior_precision.tolist(),
                "training_offsets": self.training_offsets.tolist(),
                "training_counts": self.training_counts.tolist(),
                "training_successes": self.training_successes.tolist(),
                "variance_floor_standard_deviation": self.variance_floor,
                "training_ridge": self.training_ridge}


def question_knn(
    probe_embeddings: np.ndarray,
    responses: np.ndarray,
    target_embeddings: np.ndarray,
    *,
    prior_probabilities: np.ndarray | float = 0.5,
    n_neighbors: int = 5,
    prior_weight: float = 1.0,
    temperature: float = 0.1,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict from cosine-nearest observed probes with a smoothed local mean."""
    probes = np.asarray(probe_embeddings, dtype=np.float32)
    targets = np.asarray(target_embeddings, dtype=np.float32)
    responses = np.atleast_2d(np.asarray(responses, dtype=np.float64))
    if probes.shape[0] != responses.shape[1]:
        raise ValueError("responses must have one column per probe")
    if n_neighbors < 1 or prior_weight < 0 or temperature <= 0:
        raise ValueError("invalid KNN parameters")
    priors = np.broadcast_to(np.asarray(prior_probabilities, dtype=np.float64), (len(targets),))
    if not np.all(np.isfinite(priors)) or np.any((priors < 0) | (priors > 1)):
        raise ValueError("prior_probabilities must be finite values in [0, 1]")
    predictions = np.broadcast_to(priors, (len(responses), len(targets))).copy()
    if not len(probes):
        return predictions.astype(np.float32)
    probes = probes / np.maximum(np.linalg.norm(probes, axis=1, keepdims=True), 1e-12)
    targets = targets / np.maximum(np.linalg.norm(targets, axis=1, keepdims=True), 1e-12)
    neighbors = min(n_neighbors, len(probes))
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        similarity = targets[start:stop] @ probes.T
        indices = np.argpartition(-similarity, neighbors - 1, axis=1)[:, :neighbors]
        scores = np.take_along_axis(similarity, indices, axis=1)
        weights = np.exp((scores - scores.max(axis=1, keepdims=True)) / temperature).astype(np.float64)
        for model, row in enumerate(responses):
            values = row[indices]
            observed = np.isfinite(values)
            effective_weights = weights * observed
            numerator = (effective_weights * np.nan_to_num(values)).sum(axis=1) + prior_weight * priors[start:stop]
            denominator = effective_weights.sum(axis=1) + prior_weight
            predictions[model, start:stop] = np.divide(
                numerator, denominator, out=priors[start:stop].copy(), where=denominator > 0
            )
    return np.clip(predictions, 0.0, 1.0).astype(np.float32)
