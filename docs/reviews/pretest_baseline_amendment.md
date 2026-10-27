# Main-run baseline amendment before final-test access

Date: 2026-10-28, Asia/Taipei. Status: root-approved, main experiment only. Final-test performance has not been used to make this amendment.

Three-seed validation placed the smoothed benchmark-mean baseline at its largest tested smoothing value, `20`, and question KNN at its largest tested prior weight, `3`. Their errors were still improving toward those boundaries. The existing global response mean also assigned the same probability to every benchmark, omitting benchmark difficulty already available from training data. These observations motivated a bounded correction to baseline coverage before final-test locking.

## Approved changes

- Extend main benchmark/global response-mean smoothing by adding `50` and `100`.
- Extend main question-KNN prior weights by adding `10` and `30`.
- Use main adaptation prior multipliers `0.1, 0.3, 1, 3, 10` for the three neural variants and the scalar baseline below.
- Add one non-neural scalar logit-shift baseline calibrated to training benchmark response rates.
- Retain the original training-domain mean as the exact no-response reference. It also reveals whether very strong smoothing effectively discards new-model responses.

The selected neural architecture, training configuration, model/item splits, response budgets, and acquisition policies remain fixed. Neural checkpoint selection retains its original prior candidates `0.3, 1, 3`; expanding the final adaptation grid does not retroactively retrain/select epochs under a different criterion.

The family study retains its original instrument and grids, with the scalar baseline disabled. Main-run development results must not be used to tune family targets. These restrictions were explicitly approved by the root agent.

## Scalar benchmark-calibrated ability baseline

For training model `m` and benchmark `d`, compute observed response count `n_md` and success count `s_md`, using training models and training questions only. Let

```text
p_d = clip(sum_m s_md / sum_m n_md, 1e-6, 1-1e-6)
b_d = logit(p_d)
```

Every retained benchmark must have observed training labels. Fit one scalar offset per training model by minimizing

```text
sum_d [n_md*softplus(b_d+a_m) - s_md*(b_d+a_m)] + 0.5*1e-6*a_m^2
```

The tiny fixed ridge gives a finite optimum even for separated labels. Its derivative is strictly increasing:

```text
g(a) = sum_d [n_md*sigmoid(b_d+a) - s_md] + 1e-6*a
g'(a) = sum_d n_md*p_md*(1-p_md) + 1e-6
```

A bracketed one-dimensional root finder or backtracked Newton solve therefore obtains the unique optimum. With the clipped benchmark offsets and positive ridge, the bracket `[-64, 64]` safely covers the practical training solution. A model with no training observations should not contribute an artificial zero offset to the empirical prior.

Estimate prior mean `mu=mean_m a_m` and variance `sigma2=max(sample_variance_m a_m, 0.2^2)`. For a new model's revealed support labels, estimate

```text
a_new = argmin_a sum_i [softplus(b_d(i)+a) - y_i*(b_d(i)+a)]
                  + 0.5*strength*(a-mu)^2/sigma2
p_target(d) = sigmoid(b_d+a_new)
```

This is exactly the existing convex `adapt_map` problem with a one-column all-ones design. Tune one prior multiplier over the same main validation models, seeds, and three budgets as the other predictors. The baseline needs benchmark IDs but no text encoder, model names, or final-test labels.

Its predictions differ across benchmarks because `b_d` captures their training difficulty, while model-specific adaptation has only one degree of freedom. It is a practical comparison for whether the neural representation outperforms a simple ability correction. It is not a fully matched neural-rank ablation: its item offsets are benchmark averages rather than learned text-conditioned difficulties. The structured rank-zero comparison remains necessary for the latter question.

## Review invariants

The implementation review should verify:

- Training sufficient-statistic optimization agrees with an independent scalar likelihood fit.
- New-model adaptation agrees with the declared sum-likelihood Gaussian-prior optimum.
- Zero/entirely missing support returns the prior mean, and extreme prior strength approaches it.
- More consistent positive evidence shifts ability upward and more negative evidence shifts it downward.
- Within-benchmark support-label permutation leaves the result unchanged.
- Missing positions consume the declared attempted-response budget but do not contribute labels.
- Predictions depend only on revealed support labels and permitted training priors.
- The original training-domain mean remains separate: the scalar prior prediction is `sigmoid(b_d+mu)`, which need not equal `p_d` when `mu` is nonzero.

This is the final approved method scope for the main-run baseline amendment. Boundary behavior after the expanded validation grid should be reported honestly; it does not authorize an open-ended search or additional predictors.

## Independent implementation review

`ScalarLogitShift` in `src/coldstart/models.py` matches the approved sufficient-statistic fit and one-dimensional Gaussian-prior adaptation. Its training root uses `brentq`; its cold-start prediction calls the existing `adapt_map` with an all-ones design. The two runner fit sites both use only `split.train_models x split.train_items`. The family configuration explicitly disables this baseline.

Independent synthetic checks, saved in `tmp/review/scalar_logit_shift_numerical_checks.json`, found:

- Training offsets agreed with direct full-likelihood scalar minimization within `3.8e-8`, with derivative residual below `1.6e-13`.
- New-model predictions agreed with an independent scalar root solution within `2.6e-8` for prior multipliers `0.1`, `1`, and `10`.
- Empty and entirely missing support returned the prior prediction; prior strength `1e12` approached the same limit.
- Positive/negative evidence moved the ability in the correct direction, and benchmark probabilities remained distinct.
- Within-benchmark label permutation left predictions unchanged.
- The variance floor and training models with separated all-zero/all-one outcomes remained finite.

No blocking mathematical or information-flow defect was found in this review. The checks use synthetic data only and provide no estimate of the baseline's real prediction quality. Final empirical comparisons must use the frozen main-run configuration and saved probe paths.
