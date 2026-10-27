# Validation-only diagnosis of the first model pilot

Date: 2026-10-28, Asia/Taipei. No final-test performance was inspected for this diagnosis.

The first CUDA runtime pilot completed ten shared-training epochs in approximately 1.08 seconds, but selected a structured rank-eight checkpoint before its interaction term had received a likelihood gradient. It cannot support a claim that the residual embedding was learned from model behavior. This is an implementation/selection issue to resolve before locking the main experiment.

## Evidence

The pilot used 62 training models, five benchmarks, 15,733 training questions, 3,372 validation questions, hidden width 128, rank eight, learning rate `0.002`, latent penalty `0.002`, and five warm-up epochs. The relevant record is `artifacts/pilot_model_runtime/training/structured8_seed_11.json`.

| Epoch | Training BCE | Validation-item BCE | Residual likelihood training |
| --- | --- | --- | --- |
| 1 | 0.62564 | 0.61383 | Disabled |
| 3, selected | 0.60896 | 0.61198 | Disabled |
| 5 | 0.60042 | 0.61272 | Disabled |
| 6 | 0.59392 | 0.61358 | Enabled |
| 10 | 0.55532 | 0.62562 | Enabled |

The saved checkpoint has global-latent standard deviation `0.549` and benchmark-latent standard deviations approximately `0.155–0.320`. Its residual-coordinate standard deviations are only `0.033–0.060`. Residual RMS fell from the seeded initialization's `0.1043` to `0.0476`; all eight fitted residual prior standard deviations hit the configured `0.2` floor. The interaction head's weight change from seeded initialization has norm `1.5e-5`, consistent with weight decay rather than response-likelihood training. Details are saved in `tmp/review/pilot_initial_latent_diagnostics.json`.

The code disabled interaction contributions for the first five training epochs while still regularizing those latent coordinates. It allowed any epoch to become the retained checkpoint. Validation also evaluated the full interaction expression during warm-up, while the training likelihood omitted it. The shared text encoder did learn difficulty features, so its outputs changed; that does not make the untrained interaction head and shrink-only residual latents a learned behavioral factorization.

The falling training BCE and rising validation-item BCE also indicate item-side overfitting in this pilot. That trend alone does not establish how cold-start benchmark macro-MAE behaves. Epoch selection should be checked against the actual transfer task using validation models.

## Bounded pre-test comparison

The root agent approved a validation-only grid over the same three model variants: structured rank eight, structured rank zero, and low-rank-only rank eight. It changes no final test membership or evaluation labels.

| Setting | Values |
| --- | --- |
| Warm-up | Zero in the main grid |
| Hidden width | 32, 128 |
| Latent penalty | `0.0001`, `0.002` |
| Learning rate | `0.001`, `0.002` |
| Main selection task | New validation models, fixed balanced-random probes |
| Checkpoint support budget | 100 responses |
| Prior multipliers | `0.3`, `1`, `3`, selected globally across validation models |
| Final validation budgets | 50, 100, 200 responses |

Use the same eight training configurations and checkpoint criterion for each variant. Fix probe IDs, validation model IDs, and initialization seeds across the development comparison. Retain rank zero even if it wins; the purpose is to establish whether residual embeddings add useful information.

The root agent approved one shared hidden-width/learning-rate/latent-penalty configuration for all three variants. For each candidate shared configuration, first tune each variant's prior separately, then average validation macro-MAE equally across the three variants and the three response budgets. Select the shared configuration with the lowest average. This keeps the rank ablation's training capacity and regularization choices matched and prevents a configuration search from favoring only the full model. Save the full grid and the selection rule before final-test scoring.

The recommended callback evaluates held-out-model macro-MAE every two epochs and at epoch one. It can minimize over the declared three prior multipliers for the whole validation-model group, avoiding an arbitrary prior scale deciding epoch selection. Keep train/validation BCE in the history as separate monitoring fields. Re-score retained configurations across all three budgets and the same prior grid before the final protocol is locked. The callback must receive only permitted training/probe features and validation-model responses; final test labels remain unused.

Regardless of checkpoint criterion, a structured residual checkpoint must have completed at least one epoch with active interactions. If a warm-up is retained for a diagnostic run, do not shrink inactive residual coordinates during that warm-up and begin eligible checkpoint selection after it. Record the active-interaction epoch count with the checkpoint.

## Required diagnosis after this comparison

- Report selected epochs, validation cold-start macro-MAE, and held-out-item BCE separately.
- Inspect residual training variance, prior-floor usage, and within-benchmark item-feature variation.
- Compare rank eight and rank zero on common balanced-random probes with separately tuned priors.
- Apply the preregistered within-benchmark permutation control to distinguish useful question-response alignment from benchmark totals.
- Freeze the retained configuration and source/data hashes before final test evaluation.

This document records the initial failure and the pre-test development rationale. Subsequent selected configurations and validation results should be added with their own artifact paths rather than replacing the first pilot's evidence.

## First interaction-trained development configuration

An independent CPU inspection of `artifacts/development_v2/h32_lr0.001_l20.0001` found a genuine learned residual term. This is one configuration and one seed in the ongoing validation search, not the selected final model.

| Check | Structured rank eight |
| --- | --- |
| Selected epoch | 40, with interactions active from epoch one |
| Residual latent RMS | `0.3592`, versus seeded initialization `0.0999` |
| Residual coordinate standard deviations | `0.224–0.265` |
| Interaction-head weight change from initialization | Norm `3.5360` |
| Equal-benchmark within-feature variance | `0.3082` |
| Between-benchmark feature variance | `0.0153` |
| Within fraction | `0.9526` |

The interaction features therefore vary within benchmarks and their head received response-driven updates. These facts establish that the implementation trains a residual representation; they do not establish that it improves new-model prediction.

At 100 fixed balanced-random validation-model responses and the separately selected prior multiplier `1`, residual posterior-to-prior variance ratios were `0.962–0.972`. Their mean squared prior-whitened adaptation displacement was `0.0368`. The global coordinate's variance ratio was `0.139`, and benchmark coordinates' ratios were approximately `0.727–0.885`. In this coordinate system, most response-driven identification concerns global and benchmark effects; the learned residual coordinates remain close to their priors at this budget. These are conditional Laplace diagnostics, not calibrated confidence statements.

For this single configuration, validation macro-MAE averaged over the three response budgets after prior selection was `0.05545` for the structured model, `0.05683` for rank zero, and `0.05200` for the low-rank-only control. The full grid and retained shared configuration determine what comparison is ultimately reported. No conclusion should be chosen from this first completed cell.

The feature and adaptation evidence is saved in `tmp/review/development_first_config_diagnostics.json`. It uses training-pool item features and revealed validation-model probe responses only; final-test model responses and final-test item features do not enter this diagnostic.

## Retained shared development configuration

The eight-configuration search finished without using final-test performance. The shared selection rule retained hidden width `128`, learning rate `0.001`, and latent penalty `0.002`, with zero warm-up and cold-start checkpoint selection. The development run is `artifacts/development_v2/h128_lr0.001_l20.002`; the full configuration comparison and rule are saved under `reports/development/`.

| Variant | Selected epoch | Selected prior multiplier | Validation macro-MAE, averaged over budgets |
| --- | --- | --- | --- |
| Structured rank eight | 36 | 1 | 0.05339 |
| Global + benchmark, rank zero | 58 | 1 | 0.05151 |
| Low-rank-only rank eight | 54 | 0.3 | 0.04919 |

These are selection-stage results from seed 11 and thirteen validation models. They are not a final test estimate. The structured residual term does not improve over rank zero in this retained development cell. Nonmonotonic errors across the three support budgets also prevent claiming that every increase in response count improves accuracy for every finite probe path.

The selected structured model has residual latent RMS `0.3244`, residual standard deviations `0.246–0.280`, and interaction-head weight change norm `3.9509`. Its equal-benchmark within/between feature variances are `0.4006` and `0.1728`, giving within fraction `0.6987`. Thus the revised training is no longer selecting an untrained residual head.

At 100 balanced-random responses, the structured model's residual posterior variance ratios average `0.9352`, and its mean squared prior-whitened displacement is `0.0643`. The low-rank-only control has corresponding values `0.6934` and `0.2192`. Residual coordinates in the structured model remain more prior-driven in this development setting, while the low-rank-only representation changes more with the observed responses. Coordinate-level contraction is conditional on each model's fitted prior and parameterization; it supplements, rather than replaces, prediction comparisons.

The selected-checkpoint evidence is saved in `tmp/review/development_selected_config_diagnostics.json`. The reproducible CPU checker `tmp/review/inspect_checkpoint.py` accepts a run directory, seed, and response budget, and reads training-bank features plus revealed validation-model support responses. Three-seed main training and the locked final-test comparisons remain separate subsequent stages.
