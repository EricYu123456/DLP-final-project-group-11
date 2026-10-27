# Reproduction and saved artifacts

The repository contains source, frozen configuration, evidence tables, small
result files, and exported figures. Raw questions, downloaded tensors, prepared
arrays, and local checkpoints are excluded from Git. The primary experiment is
an offline simulation using released binary responses; it does not call an LLM
API or measure current API prices.

## Environment

The recorded local run uses Python 3.10.6, PyTorch
`2.10.0.dev20251209+cu128`, and an NVIDIA RTX 5070 Ti with 16 GB memory.
`requirements-tested.txt` records numerical dependency versions. The project
also runs on CPU. The test workflow checks Linux/Python 3.10 and Windows/Python
3.12 with CPU PyTorch; numerical trajectories can differ across hardware and
library versions, so reproduction is not a promise of byte-identical weights.

Install a compatible PyTorch build and run:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/download_data.py
python scripts/prepare_data.py --no-split
```

The download is about 3.4 GB. The pinned revision and expected SHA-256 hashes
are in `src/coldstart/data.py` and `data/metadata/dataset_source.json`.
Preparation scans every source CSV, distinguishes missing/duplicate/conflicting
records, and saves arrays and audit metadata in `data/processed/`.

The default raw-label policy is strict structural singleton, with one
deterministic representative per detected question group. Model filtering uses
the reviewed include decisions in `data/metadata/model_families.csv` before
freezing the common question set. The final primary dataset fingerprint is
`d37f4c6eeb850831f0e6ceb439377ad4b434d943b3df51174af0cfa515fad9ba`.

## Primary experiment

Use an empty output directory:

```bash
python scripts/run_experiment.py --config configs/protocol.json --phase train_validate --output artifacts/reproduction --cpu-threads 1 --evaluate-validation --save-predictions
python scripts/run_experiment.py --config configs/protocol.json --phase test --output artifacts/reproduction --cpu-threads 1 --save-predictions
python scripts/report_results.py --run artifacts/reproduction --output reports/reproduction
```

`--device auto` selects CUDA when available; `--device cpu` forces CPU.
`--deadline` accepts an optional timezone-aware ISO timestamp. All experiment
jobs from the implementation session used an explicit cutoff before noon.

Training touches training-model labels on support questions. Checkpoint
selection uses held-out validation models and validation items, with a fixed
100-probe policy. Final prior strengths and simple-baseline settings are
selected over validation models, budgets, and seeds. `test_lock.json` binds
those decisions to the actual prepared-file bytes, split, normalized source
hashes, and checkpoint hashes. The test command rejects changed locks.

`--resume` can reuse compatible completed checkpoints. An output directory that
already contains final-test results cannot be reused for new development.
The completed local run is `artifacts/main_v2`; `artifacts/main` preserves an
earlier validation attempt that failed on a Windows atomic replacement error.

Key files in each completed run are:

| Artifact | Contents |
| --- | --- |
| `effective_config.json`, `split.json` | Actual settings and original model/prompt IDs |
| `training/*.json`, `checkpoints/` | Curves, retained epochs, weights, and training provenance |
| `validation_grid.csv`, `selected_hyperparameters.json` | All adaptation/baseline candidates and decisions |
| `test_lock.json` | Pre-test configuration, source, data, split, and checkpoint checksums |
| `test_completed.json`, `per_model_metrics.csv`, `per_domain_metrics.csv` | Complete test results |
| `test_probes.json`, `test_acquisition.jsonl` | Revealed probe order and adaptive acquisition trace |
| `test_embeddings/*.npz` | New-model vectors, covariance, probe IDs, and decoder digest |
| `test_predictions/*.npz` | Item probabilities and their model/prompt IDs when enabled |

Source hashes normalize UTF-8 line endings to LF. Data and checkpoint hashes
refer to actual binary bytes. A vector belongs to the decoder whose checksum
is stored with it; vectors from independently trained decoders need not share
an identifiable coordinate system.

## Historical development search

`reports/development/` preserves the eight-configuration validation search and
its exact selection rule. It selected one shared width/learning-rate/latent
penalty across the three neural variants. Its adaptation grid was `[0.3,1,3]`.
The later main-only baseline amendment expands adaptation to `[0.1,0.3,1,3,10]`,
adds a scalar calibrated ability, and enlarges finite mean/KNN smoothing grids.
The shared neural architecture and checkpoint criterion are unchanged.

`scripts/run_development_grid.py` honors the declared historical prior grid.
The provided `configs/protocol.json` already contains the selected final
architecture, so rerunning development is unnecessary to reproduce the main
experiment. No final target correctness selected the architecture or policy.

## Original matched family study

The family protocol uses its original fixed neural settings, not the main
validation-selected configuration. It tests structured 14D and global/benchmark
6D representations. The six prepared splits hold target models, Qwen validation
models, and item groups fixed within each pair; both training pools have 40
models. The family study shares adaptation settings selected from the pooled
Qwen validation grids across the two conditions.

PowerShell example with a user-chosen two-hour runtime bound:

```powershell
$familyDeadline = (Get-Date).AddHours(2).ToString("o")
python scripts/run_family_study.py --config configs/family_protocol.json --output artifacts/family_reproduction --phase prepare --device auto --cpu-threads 1 --deadline $familyDeadline
python scripts/run_family_study.py --config configs/family_protocol.json --output artifacts/family_reproduction --phase validate --cpu-threads 1 --deadline $familyDeadline
python scripts/run_family_study.py --config configs/family_protocol.json --output artifacts/family_reproduction --phase test --cpu-threads 1 --deadline $familyDeadline --save-predictions
python scripts/report_family_results.py --run artifacts/family_reproduction --output reports/family_reproduction
```

Validation never automatically starts family tests. Both arms must be relocked
to the same pooled validation selection first. The completed original local
study is `artifacts/family_study_v4`. Earlier prepared versions preserve the
development trail; they were not used for the reported final results.

A historical timing flag in the immutable family configuration is corrected
by `reports/reviews/family_config_timing_correction.json`: the dedicated file
was written 14 seconds after the grid process finished, while its fixed numeric
defaults predated the search. The addendum preserves locked hashes and changes
no hyperparameter or result.

`scripts/run_secondary_lowrank_family.py` reproduces the separate 8D coverage
follow-up using `--control` and `--main` completed runs. It has the same explicit
prepare/validate/test stages and deadline requirement. The completed local
version is `artifacts/family_lowrank_secondary_v2`. Its scope is explicitly
post-main-inspection, and its exports contain only the new neural predictor;
copied baseline results are not pooled as additional evidence.

`scripts/analyze_hybrid_pilot.py --run artifacts/main_v2` reproduces the rejected
validation-only evidence-blend pilot. It reads saved validation profiles and
scores, with no new decoder training or coverage-cohort scoring.

## External-model embedding and replay

`scripts/embed_new_model.py plan` exports local prompts and empty binary-response
cells. After supplying correctness, `infer` writes `embedding.npz` and
`prediction.json`. It performs no live LLM call or free-form grading. Original
scoring compatibility is unresolved by the source release.

The local `artifacts/demo_validation_replay/` replays 100 support answers from
an existing validation model through the CSV interface. It recovers the stored
eight-dimensional validation vector within `1.81e-7` and covariance within
`2.68e-8`. This checks the reusable software interface; it is not an evaluation
of a newly queried LLM.

## Statistical interpretation

Main tables average seeds inside each model, then models. Macro score error
weights the five benchmarks equally. Confidence intervals resample five
reviewed lineage groups and remain descriptive; there is no multiplicity
correction. Family results are paired within fixed target cohorts, with
per-family lineage intervals suppressed. Different family and main presets
must not be pooled as a single generalization estimate.

See `reports/FINAL_REPORT.md` and the independent review files for supported
claims, failures, and the distinction between original and secondary analyses.

## Coverage of metadata-excluded IDs

`scripts/evaluate_metadata_excluded.py` has a structural `--phase prepare`
stage and a separate `--phase score` stage. It reuses a completed main run
through `--main`, and stores the exact 24-ID list, parent/checkpoint/data/loader
digests, old scalar-prior snapshots, probe orders, and availability counts before
scoring. The score stage requires an explicit deadline and an independent
`--gate` JSON whose `status` starts with `PASS` and whose `manifest_sha256`
matches that preparation. A new reproduction needs its own reviewed gate;
an audit from another set of checkpoint bytes cannot be reused.

The original local run is `artifacts/metadata_excluded_cohort_v2`. After scoring,
generate its descriptive report with:

```bash
python scripts/report_results.py --run artifacts/metadata_excluded_cohort_v2 --output reports/metadata_excluded --no-intervals
```

No lineage intervals are estimated for these models, and their results are
kept separate from the primary and matched family studies. The candidate
evidence-blend prototype was rejected before this final scoring pass and is
absent from its predictor list.
