# Implementation log

## 2026-10-28, Asia/Taipei

The implementation session began at 03:39 and has a hard stop at 12:00.
Training must finish by 11:45; a process watchdog stops registered project jobs
by 11:55. The initial six-agent team comprised two research agents, two architecture
agents, and two implementation agents. The lead agent owns integration,
scientific review, reporting, and Git publication.

### Decisions before experiments

- Primary question: how accurately can a new LLM embedding, inferred from
  50, 100, or 200 responses, predict its benchmark scores on held-out items?
- Primary protocol: held-out models and grouped held-out items. Final item
  labels are unavailable to every training model. A fixed-bank experiment,
  if added, must have a separate name and table.
- Audit the full source CSV before choosing a structural single-record subset.
  Published tensor labels remain a proxy target, not established inference truth.
- Use a frozen text-conditioned logistic decoder and fit the new embedding
  with regularized convex optimization. Compare a global/benchmark/residual
  representation with rank-zero and low-rank-only ablations.
- Test model embeddings against smoothed benchmark means and question KNN on
  identical probe IDs. Separate the value of the representation from selection.
- Existing work already covers cold-start profiles, Gaussian priors, adaptive
  testing, and family holdout. Claims will focus on measured contributions and
  reproducible controlled experiments.

Results, scope changes, validation decisions, and final commands will be added
as they occur. No performance result was available at session start.

### Data freeze and validation development

- The complete audit examined 4,586,288 raw records. All 3,995,376 model/prompt
  cells are present; 77,581 have conflicting raw labels. All released tensor
  entries equal the maximum of their corresponding raw rows.
- Structural singleton eligibility in the five primary sources is identical
  across all 112 models. Restricting the cohort to 88 reviewed models and
  retaining one representative per detected question group leaves 22,475 items.
- A ten-epoch runtime pilot found that item-BCE early stopping could select a
  checkpoint before residual interactions were trained. We removed warm-up for
  the study and added a checkpoint eligibility check. Checkpoint selection now
  uses held-out-model validation macro MAE, with a fixed 100-probe policy.
- An eight-configuration, validation-only search selected one shared setting
  for all three neural variants: hidden width 128, learning rate 0.001, latent
  regularization 0.002. See `reports/development/selection.json` for every
  candidate, selection rule, and the preserved failed first attempt.
- The eight-dimensional neural variant had the lowest development score. It
  was selected as the practical primary candidate and Fisher selector before
  final test evaluation. The structured and rank-zero variants remain controls.
- Family experiments retain the original hidden width 128, learning rate
  0.002, and latent regularization 0.002. Main development models overlap some
  later family targets, so the main architecture search is not transferred to
  that separate study. Family adaptation settings use the disjoint Qwen
  validation group, pooled across the two matched training conditions.

### Pre-test reliability and baseline amendments

- The first three-seed main validation process stopped on a Windows file
  replacement error. No test evaluation ran. Atomic writes now retry bounded
  transient permission errors; persistent errors still fail. A fresh main run
  preserves the previous artifacts and records the corrected source hashes.
- Validation selected the largest smoothing values in the original mean/KNN
  grids. The main study therefore expands those finite grids and adds a scalar
  benchmark-calibrated logit-shift baseline. Neural adaptation multipliers are
  also evaluated on the fixed `[0.1, 0.3, 1, 3, 10]` grid. Shared neural training
  settings and the checkpoint criterion remain unchanged. Family experiments
  keep their separate, original controls. These decisions precede test scoring.

### Team adjustment requested by the user

At approximately 04:11, the user changed the research roles to two
`gpt-6-sol` agents at `high` effort and reduced implementation to one
`gpt-6.1-sol` agent at `max` effort. The former research agents and second
implementation agent were stopped. The remaining implementation agent received
the data and family-study handoff; both architecture roles remained unchanged.
