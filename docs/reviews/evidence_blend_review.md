# Rejected evidence-weighted blend: validation-only review

The bounded validation prototype did not beat its fixed-weight control. The root agent rejected the candidate without temperature, threshold, or model-prior tuning. It will not be added to the 24-model cohort evaluation. The original frozen predictors, main results, and family-study results remain unchanged.

The proposal was motivated after inspection of existing main and family test results. It is therefore a post hoc hypothesis, not part of the original preregistered experiment. The prototype used only stored main-validation fits and predictions; the 24-model cohort's prediction performance remained unseen during this decision. Scope is recorded in `reports/exploratory_hybrid/validation_only/scope.json`.

## Mathematical review

For a frozen decoder with sum Bernoulli negative log-likelihood and Gaussian-prior quadratic, the conditional Laplace log evidence is

```text
E = -MAP_objective + 0.5*(logdet(prior_precision) + logdet(posterior_covariance))
```

The Gaussian normalization and Laplace integration constants cancel. Comparing dimensions eight and six is valid when each calculation uses the same observed support labels, the actual normalized prior precision, and the final MAP Hessian. A redundant prior-only coordinate cancels from the determinant ratio.

The candidate used equal model prior odds, `w=sigmoid(E8-E0)`, and prediction `w*p8+(1-w)*p0`, where eight denotes the frozen low-rank-only decoder and zero denotes the structured rank-zero decoder. It reused the same response budget for both fits. Because component probabilities were existing MAP predictions rather than latent-marginalized predictive integrals, the accurate description is a Laplace-evidence-weighted MAP blend.

## Validation result

Mean benchmark macro-MAE across the three response budgets:

| Predictor/control | Macro-MAE |
| --- | --- |
| Structured rank zero | 0.051402 |
| Low-rank-only eight-dimensional profile | 0.049973 |
| Evidence-weighted blend | 0.049846 |
| Uniform 50/50 blend | 0.048889 |
| Best global constant from `0, 0.25, 0.5, 0.75, 1` | 0.048889, weight 0.5 |

The evidence blend's average weight on the eight-dimensional predictor was approximately `0.421`, with individual weights spanning nearly zero to one. Its error was about `0.000957` higher than the uniform blend, or `0.096` percentage points. It did not provide evidence that support-derived adaptive weighting added value beyond a fixed combination.

Per-budget means and model-level rows are retained in `reports/exploratory_hybrid/validation_only/summary.csv` and `per_model.csv`. The best constant above is one global validation-selected weight; it was not separately selected for each budget or model.

## Interpretation and decision

Support marginal evidence measures fit and latent complexity for observed responses. It does not directly optimize benchmark-score MAE or guarantee extrapolation under a new model family. The validation comparison confirms the practical concern: a mathematically coherent gate need not improve the downstream objective. This observation does not identify a unique cause of the gate's failure or invalidate the Laplace formula.

The proposed 15-number state would have contained two profiles plus a mixing weight, tied to two frozen decoders. It would not have established a newly learned 15-dimensional skill representation or a globally novel method.

The candidate is closed as a negative exploratory result. No gate, fitted constant blend, or further blend tuning enters the 24-model evaluation; that evaluation resumes with its original seven frozen predictors and declared reference. The rejected probe remains separate from the core implementation and original result tables.
