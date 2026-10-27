# Cold-start model embeddings: architecture and review contract

Design date: 2026-10-28, Asia/Taipei. This document specifies the approved model direction and the checks needed before interpreting results. Hyperparameters below are initial development settings, not measured winners.

The main task is to infer a compact representation of an unseen LLM from exactly 50, 100, or 200 binary responses, then predict its scores on separately held-out benchmark questions. The shared decoder is learned from permitted training models and training questions. At cold start, only the new model's representation changes.

Current retained development setting: hidden width 128, learning rate `0.001`, per-model latent penalty `0.002`, no interaction warm-up, and held-out-model macro-MAE checkpoint selection every two epochs (also epoch one). A validation-only eight-configuration search selected one shared training configuration for structured rank eight, structured rank zero, and low-rank-only rank eight; priors are tuned separately. The initial BCE-stopping pilot selected an untrained residual checkpoint and was superseded before final-test scoring. See `docs/reviews/pilot_model_diagnosis.md` and `reports/development/` for the recorded evidence and selection rule.

## Approved scope

Use a small neural text-conditioned logistic decoder with global ability, benchmark residuals, and a low-rank interaction. Freeze it before estimating a new model. The final logit is linear in the model representation, making cold-start fitting a small convex optimization problem. Evaluate the representation independently of how probes are selected.

The primary protocol holds out both models and item groups: final evaluation-item labels are unavailable for every model during training and tuning. Probe items come from the training-item bank. A calibrated fixed-item-bank protocol is a separate secondary experiment; its results and stronger item-parameter baselines must have a separate table.

This is a controlled course-project extension. Frozen-model profile estimation and fitted multivariate Gaussian profile priors already appear in [SkillEval, Section 4.2 and Appendix G](https://arxiv.org/html/2610.28767v1#S4.SS2). The proposed convex decoder, implementation checks, and acquisition comparisons should be presented as engineering and empirical choices rather than a new general principle. [EmbedLLM's official implementation](https://github.com/richardzhuang0412/EmbedLLM) remains the starting reference.

## Decoder and representation

For model `m`, question `q`, text embedding `x_q`, and benchmark index `d(q)`, use

```text
h_q       = GELU(W_hidden preprocess(x_q) + b_hidden)
b_q       = w_difficulty^T h_q + beta[d(q)] + beta_0
v_q       = interaction_head(h_q)
logit_mq  = b_q + a_m + u_m[d(q)] + z_m^T v_q / sqrt(r)
p_mq      = sigmoid(logit_mq)
```

The retained shared item network has hidden width 128. The experiment compares residual rank `r=8` with `r=0`; rank 16 is outside the current locked search. A rank of zero removes only the residual interaction; the shared text difficulty and model global/benchmark effects remain.

The representation is `eta_m = [a_m, u_m, z_m]`, with `a_m` a global logit shift, `u_m` benchmark effects, and `z_m` residual behavior coordinates. The benchmark effects are a statistical summary of this dataset, not named human skills. The residual dimensions have no inherent semantic labels.

An orthonormal contrast matrix `C` of shape `D x (D-1)`, with `C^T 1=0`, can represent the benchmark effects without redundant coordinates: `u_m=C t_m`. Then

```text
c_q   = concat(1, C[d(q), :], v_q / sqrt(r))
eta_m = concat(a_m, t_m, z_m)
logit = b_q + c_q^T eta_m
```

For ten benchmarks and rank eight, the contrast representation has 18 coordinates. Using all ten one-hot benchmark coordinates plus a global coordinate is also computationally valid with a positive definite Gaussian prior, but its global and benchmark coefficients are redundant. In that version, interpret predictions and combined effects rather than individual redundant coordinates.

Input preprocessing must be saved with the checkpoint. Use training-question statistics, or a fixed scale such as `sqrt(768)` for verified unit-L2 vectors. Do not fit a scaler/PCA with held-out correctness labels. Target texts may be transformed by the already-fitted preprocessing pipeline.

The interaction head and model embedding have a scale ambiguity: multiplying one by a constant and dividing the other preserves logits. Control it through bounded/normalized item features plus a model-latent penalty, or through explicitly recorded frozen-feature standardization. Save the actual feature scales; prior precision and posterior covariance depend on them. A stable solver cannot repair an unrecorded coordinate change.

## Shared training

Train only permitted training models on the permitted training-question bank. Never load the public all-model `optimal_mf_model_embeddings.pth` for a family-held-out experiment.

Initialize global/benchmark coordinates from permitted training-model response rates and residual coordinates at `Normal(0, 0.1)`. The retained setting uses no warm-up, so all checkpoints have interaction likelihood training. Use AdamW with the selected learning rate `0.001`, hidden width 128, weight decay `1e-4`, and a per-model sum latent penalty `0.002`. Record whether one optimization step processes all model-question pairs or one minibatch; an epoch count alone is ambiguous for this small matrix.

A reasonable training loss is average binary cross-entropy plus separately recorded neural and model-latent penalties. A macro-benchmark weighted training loss is an optional ablation, not a silent default: it changes the training target relative to the original item-weighted loss. Keep weighting identical between rank-zero and rank-eight comparisons.

Use validation models to simulate actual cold start at 50, 100, and 200 responses. Select prior strength and compare retained model configurations using validation benchmark macro-MAE. Tune prior strength separately for rank-zero, full structured, and low-rank-only decoders with the same development budget. Reusing a poorly suited rank-eight prior for rank zero would make the ablation unfair. The initial preregistered pilot used training-model validation-item BCE for shared-epoch stopping; its selected checkpoint preceded active interaction training. The documented pre-test correction uses held-out-model macro-MAE at 100 fixed balanced-random responses, minimized over the same prior grid, to select epochs; BCE remains a monitor. The shared training configuration is then chosen by mean validation macro-MAE across variants and all three budgets. Keep the final model family and final item labels sealed while making these decisions.

Retained development grid and setting:

| Choice | Retained setting | Validation comparison |
| --- | --- | --- |
| Residual rank | 8, with rank-zero control | 0, 8 |
| Hidden width | 128 | 32, 128 |
| Shared learning rate | 0.001 | 0.001, 0.002 |
| Model-latent penalty | 0.002 per-model sum | 0.0001, 0.002 |
| Prior multiplier | 1 | 0.3, 1, 3 |
| Prior covariance | Estimated diagonal with block variance shrinkage | Correlated prior outside current search |
| Training seeds | 11, 23, 37 for main runs | Seed 11 development search |

The actual number of independent model examples is the number of training LLMs, not the number of repeated support episodes. Prefer this small decoder and analytic adaptation over a large support-set encoder until the controlled baselines are complete.

## Prior and cold-start objective

For support labels `y_i`, frozen offsets `b_i`, and frozen design rows `c_i`, estimate

```text
L(eta) = sum_i [softplus(b_i + c_i^T eta) - y_i*(b_i + c_i^T eta)]
         + 0.5*(eta-mu)^T Lambda*(eta-mu)
```

`Lambda` must be symmetric positive definite. Start at `mu`, or warm-start sequential acquisition from the previous solution. With zero responses, return the prior mean and prior covariance.

The sum likelihood gives a fixed Gaussian prior whose relative influence decreases as more responses arrive. If an implementation uses mean BCE, divide the prior penalty by the support count to preserve this objective. A fixed penalty added to mean BCE describes fixed regularization instead; it must not be interpreted as the same fixed-prior posterior across different budgets.

The first prior can be fixed block-isotropic with explicit scales for global, benchmark, and residual coordinates. A data-estimated prior may use

```text
mu        = mean_training_models(eta_m)
S         = sample_covariance_training_models(eta_m)
Sigma0    = (1-gamma)*S + gamma*diag(S) + epsilon*I
Lambda    = prior_multiplier * inverse(Sigma0)
```

Estimate it after the decoder and coordinate scales are frozen. Use training models only; tune `gamma`/the multiplier on validation models. Diagonal shrinkage and a variance/eigenvalue floor are necessary when model count is small or some coordinates are weakly used. Describe this as a fitted or empirical prior only when those quantities are actually estimated; a hard-coded isotropic penalty does not qualify. The fitted profiles are noisy, regularized estimates, so the plug-in covariance is an approximation to the population prior.

## Newton fitting and returned uncertainty

For `s=b+C eta`, `p=sigmoid(s)`, and `w=p*(1-p)`,

```text
gradient = C^T*(p-y) + Lambda*(eta-mu)
Hessian  = C^T*diag(w)*C + Lambda
direction = solve(Hessian, gradient)
eta_next = eta - step_size*direction
```

Use float64 for the small CPU solve, stable softplus/sigmoid, and backtracking until the full objective decreases. A budget of 25 iterations is ample as an initial cap for about twenty coordinates; return convergence status rather than silently treating the cap as success. Sequential warm starts should converge to the same MAP solution as a fresh prior-mean initialization within numerical tolerance.

Return the covariance `Hessian_at_final_eta^-1`, plus the exact prior precision, convergence information, iteration count, and support IDs. Symmetrize tiny floating-point asymmetries. Numerical damping used only to solve a linear system must be recorded and must not silently redefine the statistical prior.

This covariance is a Laplace approximation conditional on the frozen shared decoder and label model. It omits uncertainty in neural item parameters, the estimated prior, unresolved evaluation conditions, and new-family distribution shift. It can guide probe selection; it is not automatically a calibrated benchmark-score confidence interval.

## Interface contract

The implementation may use different names, but these objects must be available without accessing held-out labels:

```python
item_features(text_features, benchmark_ids) -> ItemDesign
# ItemDesign.offset: [Q]
# ItemDesign.features: [Q, P]
# ItemDesign.item_ids: [Q], plus benchmark_ids and coordinate metadata

fit_shared(training_data, model_ids, question_ids, config) -> FrozenDecoder
estimate_prior(training_model_latents, config) -> GaussianPrior
# GaussianPrior.mean: [P], precision: [P, P]

adapt_map(support_design, support_labels, prior, initial=None) -> AdaptedModel
# AdaptedModel.latent: [P], covariance: [P, P]
# objective, gradient_norm, converged, iterations, support_ids

predict(adapted_model, target_design) -> probabilities[Q]
benchmark_scores(probabilities, target_benchmark_ids) -> scores[D]
```

Checkpoints should contain the frozen network, preprocessing, coordinate order, rank, benchmark mapping, training-model IDs, training-question IDs, prior parameters, random seed, data revision/hash, and selected development configuration. A new model's saved representation should include its support IDs and responses so inference can be reproduced.

## Score-targeted acquisition

The research-methods component supplies the detailed derivation. Given current covariance `Sigma`, candidate feature `c_q`, and `w_q=p_q*(1-p_q)`, a local Fisher update is

```text
Sigma_after = Sigma - w_q*Sigma*c_q*c_q^T*Sigma
                        / (1 + w_q*c_q^T*Sigma*c_q)
```

Before experiments, define an unlabeled reference bank `R_d` from the permitted training/support pool for each benchmark. For benchmark `d`, define the reference-score prediction and its gradient

```text
score_d = mean_{j in R_d} sigmoid(b_j + c_j^T eta)
g_d     = mean_{j in R_d} [p_j*(1-p_j)*c_j]
```

Select the unused candidate maximizing the approximate equally weighted reduction in benchmark-score variance:

```text
gain(q) = mean_d w_q*(g_d^T*Sigma*c_q)^2
                    / (1 + w_q*c_q^T*Sigma*c_q)
```

This is a local Bayesian experimental-design approximation. It uses training/support-pool reference texts and benchmark IDs only. Final target text features enter prediction after the support path and adaptation have finished; they do not enter selection. Reference-score uncertainty is a proxy for final held-out benchmark-score uncertainty and depends on the reference and target distributions matching. Candidate probabilities and gradients are recomputed after observing selected responses. Maximal predictive ambiguity alone is not the same objective.

Compare against uniform random, balanced random, semantic diversity, and balanced semantic diversity. Start adaptive acquisition with a predeclared balanced-random warm-up. A balanced score-targeted version can isolate benchmark allocation from within-benchmark selection. Cache one nested path up to 200 responses per policy/model/seed, then evaluate prefixes at all budgets. The final fitting method, priors, and stopping rules must stay identical across policies for a given predictor.

## Showing that the embedding is useful

First compare predictors on exactly the same stored probe sets. This identifies representation value before acquisition value:

| Predictor | Model-specific information |
| --- | --- |
| Zero-probe training-domain mean | No new-model answers |
| Smoothed observed benchmark mean | Benchmark response counts only |
| Question KNN | New-model responses on semantically nearby probes |
| Global + benchmark latent, rank zero | Global ability and benchmark residuals |
| Full decoder, rank eight | Same hierarchy plus residual interaction |
| Low-rank-only decoder, rank eight | Learned response coordinates without explicit model global/benchmark effects |

Train rank-zero and rank-eight models separately on the same allowed data and tune each one's prior on validation models. The primary test of residual embedding value is their paired difference in macro benchmark MAE on common probes, with item-level Brier/NLL as secondary measures. Better plots or a distinct latent geometry do not establish useful prediction. If rank zero matches or beats rank eight, retain it as the simpler supported conclusion.

An informative negative control permutes support labels among questions within each benchmark. It preserves every benchmark's support count and score while destroying question-response alignment. If the full decoder's advantage persists under this control, interpret it as evidence of shrinkage or benchmark-score structure, rather than item-specific information. This control uses a fixed stored balanced-random probe set and a frozen decoder; it must not trigger new data-dependent selection. For each budget, slice to the first `k` responses before permuting. Permuting all 200 responses before taking a 50-response prefix could import responses outside the declared budget. Preserve missing-response positions and permute finite observed labels only.

Also report item-feature variance within versus between benchmarks. If `v_q` is effectively constant within each benchmark, it supplies another parameterization of benchmark effects rather than transferable within-benchmark distinctions. Compare within-model, within-benchmark Brier/NLL to check whether any macro-score gain also reflects improved question discrimination. Prior-whitened support Fisher eigenvalues and posterior contraction by coordinate block can reveal whether 50 responses identify the residual coordinates at all.

Use training-reference features for the variance diagnostic. Let `mu_d=mean_{q in R_d} v_q` and `mu_bar=mean_d mu_d`. Report

```text
within  = mean_d mean_{q in R_d} ||v_q - mu_d||^2
between = mean_d ||mu_d - mu_bar||^2
within_fraction = within / (within + between)
```

These are equally weighted benchmark summaries, so large MMLU item counts do not dominate. Also retain each benchmark's within variance and the eigenvalues of the pooled within-benchmark covariance. A high within fraction alone does not prove useful signal: the item variation could be noise. For identifiability, the eigenvalues of the prior-whitened support Fisher matrix `Sigma0^(1/2) C_support^T W C_support Sigma0^(1/2)` quantify directions constrained by responses relative to their priors. Report these as diagnostics, not success targets for final-test tuning.

Then hold the predictor fixed and compare probe policies. Do not compare a new decoder with adaptive probes against an old decoder with random probes and attribute all gains to the embedding.

For robustness, show per-benchmark and per-family errors, worst-benchmark error, and model-level paired uncertainty. Raw Euclidean latent distance is not invariant to coordinate transformations. Use a fixed reference-question set's predicted response profile, or prior-whitened coordinates with a saved transformation, when comparing model representations geometrically.

## Review checklist before a result enters the main table

- The shared decoder and prior were fitted using permitted training models and questions only.
- Model families, target item groups, label semantics, and exclusions are recorded.
- Every predictor sees the exact same support IDs for a paired policy/budget comparison.
- Every adaptation uses the declared sum-likelihood objective and the Hessian matches it.
- The MAP gradient is small, the objective decreases, and prior-mean and warm-start fitting agree.
- Rank-zero and rank-eight use matched information, preprocessing, training weighting, and validation budgets, with separately selected priors.
- Acquisition reference banks use training/support-pool text only; final item features enter prediction after adaptation.
- Hyperparameters and policies were fixed before final test labels were scored.
- Main metrics aggregate by model and benchmark; response cells are not treated as independent replicates.
- Labels from unresolved duplicate records are described as public aggregate targets rather than known single-run correctness.
- No claim of a reliable confidence interval follows only from a Laplace covariance or Brier score.

## Implementation review log

The initial implementation is in `src/coldstart/models.py` and `src/coldstart/adaptation.py`. `ColdStartDecoder` uses a trained input normalizer, a shared 128-unit GELU/dropout layer, a text difficulty head, benchmark intercepts, and a bounded `tanh` interaction head divided by `sqrt(rank)`. Its structured representation uses all benchmark one-hot effects plus a global effect; the prior resolves the redundant gauge. The initial fitted prior is diagonal, with within-block variance shrinkage and a standard-deviation floor of `0.2`.

Actual interoperability methods are `decoder.item_features(...)`, `decoder.export_features(...)`, `decoder.latent_prior(...)`, `adapt_map(...)`, and `predict_from_latent(...)`. `adapt_map` returns latent, covariance, objective, gradient norm, iterations, convergence, and observed-response count. The runner must save the exact prior used for each fit along with support IDs.

Independent numerical checks on 2026-10-28 compared MAP fitting with SciPy BFGS on ordinary, rank-deficient, extreme-offset, 200-response, and zero-response cases. The largest coordinate difference against the reference solution was below `4e-9`; inverse-Hessian identity error was below `9e-16`. All-zero and all-one support cases remained finite and converged. Reconstructed logits from exported item features matched the neural forward pass exactly in a small CPU check. Results are stored in `tmp/review/architecture_numerical_checks.json`.

Review identified two corrections, acknowledged by the implementation agent: seed RNGs before constructing the neural module, and use a per-model sum of squared latent coordinates so the per-coordinate training penalty does not weaken as rank increases. These numerical checks establish implementation agreement for small controlled inputs; they provide no evidence yet that the learned representation improves benchmark prediction. Retained hyperparameters and real-data findings remain to be added after validation.

The implementation now provides `create_decoder(config, seed)` to seed construction and uses a per-model sum for the latent penalty. After the first runtime pilot exposed checkpoint selection before interaction training, a documented validation-only amendment adopted zero warm-up and the same held-out-model checkpoint criterion for all three variants. The eight-configuration grid retained the shared setting recorded at the top of this document. In its seed-11 development result, structured rank eight had macro-MAE `0.05339`, rank zero `0.05151`, and low-rank-only `0.04919`, averaged over budgets after prior selection. These development numbers show no structured-residual benefit and must not be presented as final-test performance.
