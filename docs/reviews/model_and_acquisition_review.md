# Model and adaptive acquisition review

Reviewer: architecture/model agent. Initial review: 2026-10-28, Asia/Taipei.

The reviewed implementation correctly separates a frozen neural item decoder from convex cold-start fitting. The acquisition score correctly computes a local reduction in equally weighted benchmark-score variance. The numerical checks below establish consistency of the implementation; real-data predictive value remains an experimental question.

## Reviewed components

- `src/coldstart/models.py`: text preprocessing, shared item network, global/benchmark/residual design, priors, means and question KNN.
- `src/coldstart/adaptation.py`: stable logistic likelihood, Gaussian prior, Newton updates, line search, and Hessian covariance.
- `src/coldstart/acquisition.py`: reference-score gradients, rank-one information reduction, sequential oracle reads, and balanced acquisition.
- `tests/test_models.py`, `tests/test_adaptation.py`, and `tests/test_acquisition.py`.

## Scientific assessment

The structured decoder is a hierarchical multidimensional logistic response model. Its frozen logit is affine in a new model's latent representation, so sum Bernoulli NLL plus an SPD Gaussian prior is strictly convex. This gives a unique MAP representation and allows direct Newton fitting. The full benchmark one-hot effects plus global effect have redundant coordinates; their positive definite prior resolves the optimization, while individual redundant coefficients should not receive an independent semantic interpretation.

The current fitted diagonal prior has a training-model mean and block-shrunk training-model variances. This is a fitted empirical prior, with diagonal covariance. It is not the full correlated Gaussian prior used by some related methods. The standard-deviation floor prevents unused coordinates from receiving nearly infinite precision, but its value and the overall precision multiplier must be checked on held-out-model validation data.

The primary rank-zero comparison is necessary. Its representation contains global and benchmark effects, and the rank-eight model adds item-dependent interactions. Tune their priors separately with equal validation effort, then compare on common stored probes. Training-item likelihood alone does not establish new-model representation quality.

A within-benchmark label permutation is an especially clean preregistered diagnostic. For rank zero, the support design depends only on benchmark identity, so its gradient depends on labels only through each benchmark's response total. Preserving those totals leaves its MAP and predictions unchanged, even if text difficulty offsets vary. The scalar objective can change by a label-offset constant, so objective equality is not an appropriate invariant. The full rank-eight model can change because its design varies among questions within a benchmark.

For every diagnostic budget, slice the revealed responses to that budget before permuting. Do not permute a 200-response vector and then take its first 50 entries. Use fixed balanced-random support IDs; the diagnostic must not rerun an adaptive selector or influence final-test hyperparameters.

## Acquisition derivation and interpretation

For current covariance `S`, a candidate design row `c`, response information `w=p(1-p)`, and benchmark-score gradient `g`, the code uses

```text
delta_variance = w*(g^T S c)^2 / (1 + w*c^T S c)
```

This equals the direct difference obtained from a rank-one precision update while keeping the local score gradients and response weights fixed. Averaging this quantity over benchmark gradients implements equal benchmark weighting. It is a local experimental-design approximation, not a guarantee that the next answer will reduce realized score error.

The reference bank must be drawn from the training/support-pool texts. Final target features are passed to prediction only after selection and adaptation are fixed. The acquisition module accepts generic feature arrays and therefore cannot enforce this provenance internally; the runner must record reference-bank IDs and assert their membership in the allowed pool.

The callback API reads exactly one selected scalar response per step. The implementation does not inspect unselected responses to resolve missingness or ties. A missing response still consumes one attempted probe and is omitted from the likelihood; its treatment must be consistent with the declared response budget.

Returned covariance conditions on the neural decoder, fitted prior, and response-label model. Its use for acquisition does not establish calibrated benchmark-score intervals under unseen model families or unresolved response-label aggregation.

## Numerical evidence

The command

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/test_acquisition.py tests/test_adaptation.py tests/test_models.py -q
```

passed all 12 tests at the initial review. These include finite-difference score gradients, direct rank-one inverse agreement, unique selected oracle reads, invariance to changing unselected labels, deadline checks, MAP agreement with SciPy, empty support, separated labels, fixed sum-likelihood behavior, stable extreme logits, checkpoint round trips, and training-only preprocessing.

An independent CPU check in `tmp/review/architecture_numerical_checks.json` additionally exercised non-diagonal SPD priors, duplicate design columns, 200 responses, extreme offsets, random starting points, and all-zero/all-one labels. MAP coordinates differed from SciPy BFGS by less than `4e-9`; inverse-Hessian identity error was below `9e-16`. Exported features reconstructed neural forward logits exactly for the tested input.

## Corrections and remaining checks

Two training implementation corrections were acknowledged by the implementation agent:

1. Seed RNGs before constructing the neural module. Seeding only inside the training function leaves initialization uncontrolled.
2. Penalize a per-model sum of squared latent coordinates, or explicitly tune dimension-dependent penalties. A mean over all coordinates weakens per-coordinate regularization when rank increases.

The initial adaptive warm-up used a global seeded permutation, whereas the static balanced-random selector used per-benchmark RNG streams. Both are balanced random distributions, but matching their exact warm-up IDs would isolate post-warm-up acquisition more cleanly. The root agent was asked to share the same prefix or explicitly record the difference.

Recommended focused acquisition checks are nested-prefix equality for separately requested budgets and explicit confirmation that a NaN response consumes a probe. A separate permutation-helper review should verify preserved per-benchmark counts/totals, unchanged NaN masks, no input mutation, reproducibility, and the rank-zero MAP invariant.

The prefix-only helper in `src/coldstart/diagnostics.py` was reviewed independently. It validates binary finite labels, preserves missing positions, copies its input, and permutes only finite revealed labels within benchmark. Tests in `tests/test_evaluation_leakage.py` cover response totals, missingness, reproducibility, input immutability, and independence from labels after the budget prefix. The rank-zero MAP invariant was requested as an additional integration check.

The root agent retained training-model validation-item BCE for shared-epoch stopping and held-out-model macro-MAE for prior selection in the first preregistered pilot. This is a deliberate separation of objectives. If the pilot reveals a mismatch, any change must be based on validation evidence and recorded before final-test scoring.

The following checks remain for real-data runs: train/reference/target ID provenance, model-family exclusions, validation-only prior selection, MAP convergence summaries, matched predictor probe sets, and within-/between-benchmark interaction variation. These must be completed before a result enters the main table.

## Reporting integration review

The initial `scripts/report_results.py` correctly averages seeds within models before averaging models. Its lineage-cluster bootstrap resamples group totals and sizes, retaining the model-weighted estimand when family sizes differ. Paired comparisons use shared `(seed, model_id)` cells, rather than response cells.

The following integration issues were sent to the root agent before real-data reporting:

- Predictor names emitted by the runner (`coldstart_structured8`, `coldstart_rank0`, and `coldstart_neural_lowrank`) differ from the report's initial default (`structured_rank8`). The default would omit policy plots and paired contrasts.
- Validation output filenames have a `validation_` prefix, while generic `per_model_metrics.csv` and `per_domain_metrics.csv` are written only for the test phase. The initial report reader did not select phase-specific filenames.
- A saved partial run can contain an incomplete model/seed grid. Pandas averaging and inner joins would silently use available cells, so final reports should require phase completion or explicitly validate common comparison grids and report paired counts.
- The zero-probe reference has strategy `zero_probe` and is omitted by balanced-random filtering. It needs its own table row or a labeled horizontal reference in the budget plot.
- Acquisition initially read an obsolete `fisher_warmup` config key and used the whole support pool as its score reference, while the protocol declared nested `probes.fisher.warmup` and 64 reference items per benchmark. The runner should implement the declared train-only reference sampling and save its IDs, or update the protocol before locking.

These are review observations on the initial integration, not claims that they remain unresolved in the current files. Each correction should be verified against the actual pilot and frozen configuration.
