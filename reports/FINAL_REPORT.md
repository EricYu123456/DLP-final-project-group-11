# Efficient LLM Routing with Compact Model Embeddings

NYCU Deep Learning Practice, Group 11. Routing study date 2026-11-15 (Taipei). Secondary benchmark-score study date 2026-10-28. Fixed reference [EmbedLLM: Learning Compact Representations of Large Language Models, ICLR 2025](https://openreview.net/forum?id=Fs9EabmQrJ).

## Executive summary

An 8D model embedding inferred from 50, 100, or 200 balanced-random binary probe answers enables single-candidate query routing with a frozen shared question decoder. Across 3,370 unseen test queries from five benchmarks in a simultaneous pool of 13 held-out models, compact 8D embeddings reach 46.41%, 52.70%, and 52.72% macro accuracy, beating domain probe means and question KNN. Yet they trail the same-budget fixed candidate (50.30%, 54.62%, and 54.62%) by 1.90 percentage points at 200 probes (conditional 95% paired query-bootstrap interval [-2.96, -0.88] points). Even full 15,733-probe embeddings reach 52.53%, leaving a -2.09 percentage-point deficit behind the fixed candidate.

In add-one trials where each of the 13 held-out newcomers challenges a validation-selected incumbent (`Meta-Llama-3-70B-Instruct`, 58.60% macro accuracy), raw argmax drops macro accuracy to 54.73%, 55.16%, and 56.51% because newcomer switches trigger far more harms than rescues. Validation-selected probability margins (0.075, 0.1, and 0.15) bring macro accuracy to 58.51%, 58.56%, and 58.62% by routing almost every query to the incumbent (0.66%, 0.14%, and 0.07% newcomer share), so at 200 probes the gated estimate is 0.02 percentage points above the incumbent, with the paired interval crossing zero. Both protocols fail to establish an accuracy gain over choosing the fixed probe-best model or keeping the incumbent.

In an aligned 200-probe checkpoint-selection comparison across seeds 11, 23, and 37, choosing the 8D decoder checkpoint by validation routing accuracy (`B`) raised test macro accuracy by +3.53 and +3.48 percentage points over validation score-MAE selection (`A`) on the 13-model (53.83% vs. 50.29%) and expanded 37-model (57.09% vs. 53.61%) pools. Adding softmax expected routing regret (`C`) shifted 200-probe test macro accuracy by +0.06 and +0.12 percentage points relative to `B`, with conditional 95% intervals crossing zero. Both `B` (-0.80 on 13 models and +0.61 on 37 models) and `C` (-0.73 and +0.73 points) fell short of the planned +1.0-point margin over the fixed comparators with a positive paired interval, and the expanded 37-model pool remains a retrospective cohort with no lineage-blind evidence for its 24 excluded IDs.

Our secondary study from 2026-10-28 remains valid within its original boundaries. Cold-start 8D embeddings lower macro benchmark-score MAE from 6.26 to 4.27 percentage points across 50 to 200 probes and rely on item-response alignment beyond score totals, while 8D-minus-1D lineage intervals cross zero. We independently implement few-answer onboarding and test incumbent masking directly without claiming originality for cold-start routing. Original EmbedLLM already performs per-query routing for learned model-ID vectors on unseen questions, and [UniRoute](https://arxiv.org/abs/2502.08773) and [ZeroRouter](https://arxiv.org/abs/2601.06220) study few-answer onboarding of unseen model IDs. Our variant targets onboarding unseen model IDs through a frozen decoder.

## Routing protocol and information boundaries

Routing reuses the frozen `main_v2` `neural_lowrank` 8D decoder (`sigmoid(offset(question) + feature(question)^T model_embedding)`), trained on 62 historical models and 15,733 support questions. Legacy hyperparameter selection used 13 validation models and 3,372 validation queries to minimize cold-start score MAE. Onboarding never refits the shared decoder. For each held-out model, Newton's method solves the convex MAP problem to fit only that candidate's 8D vector from nested balanced-random sets of 50, 100, or 200 binary probe answers across seeds 11, 23, and 37. Every query independently scores all pool embeddings and selects one candidate before reading target correctness, using released binary labels strictly for offline replay scoring.

We evaluate two protocols on the same 3,370 unseen test question groups across `LogiQA`, `MMLU`, `MathQA`, `MedMCQA`, and `TruthfulQA` in [the routing runner](../scripts/run_routing.py) and [the validation record](routing/validation_selection.json).

1. **Simultaneous all-unseen pool** places all 13 held-out candidates together (650, 1,300, or 2,600 total probe responses across the pool, plus a 15,733-probe full-support comparator totaling 204,529 responses). The router picks the `argmax` predicted correctness per query and compares against same-budget `fixed_candidate` (highest probe macro accuracy, lowest-ID tiebreak), `domain_probe_mean`, `question_knn`, `uniform_zero_probe_expected`, `lowest_id_zero_probe`, and a hindsight per-query `oracle`.
2. **Incumbent masking with add-one newcomer** pairs the validation-best training model (`meta-llama__Meta-Llama-3-70B-Instruct`, ID 50, 58.36% validation macro accuracy and 15,733 support responses) against each of the 13 held-out newcomers independently (50, 100, or 200 probe responses per episode, totaling 650, 1,300, or 2,600 across the 13 trials). Validation queries select one probability margin per method and budget from `{0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0}`, replacing the incumbent only when `p_newcomer - p_incumbent > margin`. The incumbent retains its jointly trained vector while newcomers fit independent MAP vectors, and this leaves estimator asymmetry as a possible influence on score comparison. An **opportunity** occurs when the newcomer answers right and the incumbent answers wrong, a **rescue** occurs when the router switches on an opportunity, and a **harm** occurs when the router switches onto a wrong newcomer against a right incumbent.

Efficiency in this study denotes probe label budget, 8D embedding storage, and one intended downstream model call per query. We perform no live LLM calls and make no deployment cost or latency claims. Because this routing evaluation reuses test models and questions already scored in the benchmark study, it is an exploratory downstream reanalysis even though validation selects all margins and test labels never enter fitting or selection. The 13 test models span only five reviewed lineage groups including nine Llama models, and the paired bootstrap (2,000 draws, seed 8675309) resamples paired queries within benchmark after averaging seeds and add-one trials. Those intervals condition on the fixed candidate roster and support probe draws and leave lineage-robust population bounds unestablished.

## Primary study on cold-start routing

### Simultaneous all-unseen pool

Macro accuracy averages per-benchmark correctness equally across all five benchmarks, and micro accuracy averages across all 3,370 queries. Both hindsight oracle (81.40% macro, 89.61% micro) and uniform expected selection (36.23% macro, 42.66% micro) mark upper and zero-information baselines in [the routing summary](routing/test_summary.csv).

| Router | 50 probes macro / micro (%) | 100 probes macro / micro (%) | 200 probes macro / micro (%) | 15,733 full macro / micro (%) |
| --- | ---: | ---: | ---: | ---: |
| Compact embedding, 8D | 46.41 / 57.62 | 52.70 / 64.07 | 52.72 / 64.11 | 52.53 / 64.10 |
| Fixed probe-best candidate | 50.30 / 60.52 | 54.62 / 65.37 | 54.62 / 65.37 | 54.62 / 65.37 |
| Domain probe mean | 45.23 / 53.81 | 45.66 / 56.96 | 51.27 / 64.54 | — |
| Question KNN | 41.73 / 51.36 | 43.15 / 52.12 | 46.43 / 56.44 | — |
| Lowest-ID zero-probe | 54.62 / 65.37 | 54.62 / 65.37 | 54.62 / 65.37 | 54.62 / 65.37 |

Compact 8D exceeds uniform zero-probe selection (+10.18, +16.46, and +16.49 percentage points) and question KNN (+4.68, +9.55, and +6.29 points), and at 200 probes it leads domain probe mean by +1.45 points (conditional 95% interval [+0.06, +2.97] points). Within-benchmark shuffling that preserves each candidate's per-benchmark selection share yields 45.79%, 51.85%, and 51.94% expected macro accuracy at 50, 100, and 200 probes, putting compact 8D ahead by +0.62, +0.84, and +0.78 percentage points as a descriptive query-matching signal without an interval.

Every few-shot and full-support 8D router still trails `fixed_candidate` by -3.89 points ([-5.17, -2.70]), -1.92 points ([-3.10, -0.76]), -1.90 points ([-2.96, -0.88]), and -2.09 points ([-3.20, -1.08]). Meanwhile, `lowest_id_zero_probe` hits 54.62% only because `ConvexAI__Luminex-34B-v0.1` carries source ID 1 and also achieves the highest test macro accuracy in the 13-model roster. That score reflects source-ID coincidence inside this fixed roster. Candidate selection shares show that at 15,733 probes compact 8D sends 92.19% of queries to `Luminex-34B-v0.1` and 6.25% to `google__gemma-2b-it` (versus 92.58% and 5.72% at 200 probes). Adding 15,533 extra support responses shifts 200-probe macro accuracy by -0.19 points ([-0.78, +0.37]), showing little observed change under this decoder with ordinary unweighted full-support MAP, as [the paired intervals](routing/test_paired.csv) and [the pool selection shares](routing/test_pool_selections.csv) document. The full-support mix places 9,689 of 15,733 responses on MMLU, while balanced 200-probe MAP uses 40 per benchmark, so the comparison also changes likelihood weights by benchmark.

### Incumbent masking and add-one newcomer rescue versus harm

Across the 13 newcomer trials, `Meta-Llama-3-70B-Instruct` achieves 58.60% macro accuracy (71.57% micro), while pairwise hindsight oracle reaches 65.75% macro (77.56% micro). Individual newcomers create between 176 and 255 opportunities per trial, averaging 201.69 queries out of 3,370.

| Policy | Budget | Margin | Macro / micro (%) | Share (%) | Rescues / trial | Harms / trial | Capture (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent only | 0 | — | 58.60 / 71.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| Compact embedding, gated | 50 | 0.075 | 58.51 / 71.42 | 0.66 | 0.79 | 5.87 | 0.39 |
| Compact embedding, gated | 100 | 0.100 | 58.56 / 71.56 | 0.14 | 0.38 | 0.69 | 0.19 |
| Compact embedding, gated | 200 | 0.150 | 58.62 / 71.57 | 0.07 | 0.38 | 0.38 | 0.19 |
| Compact embedding, gated | 15,733 | 0.200 | 58.59 / 71.50 | 1.51 | 3.95 | 6.28 | 1.96 |
| Compact embedding, raw argmax | 50 | 0.000 | 54.73 / 67.15 | 18.55 | 37.31 | 186.31 | 18.50 |
| Compact embedding, raw argmax | 100 | 0.000 | 55.16 / 67.84 | 16.23 | 32.90 | 158.85 | 16.31 |
| Compact embedding, raw argmax | 200 | 0.000 | 56.51 / 69.24 | 13.04 | 26.03 | 104.54 | 12.90 |
| Compact embedding, raw argmax | 15,733 | 0.000 | 57.09 / 69.92 | 12.19 | 26.23 | 81.95 | 13.01 |
| Domain probe mean | 50 | 0.050 | 58.60 / 71.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| Domain probe mean | 100 | 0.025 | 58.55 / 71.56 | 0.09 | 0.13 | 0.44 | 0.06 |
| Domain probe mean | 200 | 0.050 | 58.64 / 71.58 | 0.19 | 0.72 | 0.46 | 0.36 |
| Fixed candidate | 50 | 0.000 | 58.50 / 71.41 | 2.56 | 6.54 | 11.90 | 3.24 |
| Fixed candidate | 100 | 0.000 | 58.60 / 71.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| Fixed candidate | 200 | 0.000 | 58.60 / 71.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| Question KNN | 50 | 0.150 | 58.52 / 71.54 | 0.25 | 0.56 | 1.59 | 0.28 |
| Question KNN | 100 | 1.000 | 58.60 / 71.57 | 0.00 | 0.00 | 0.00 | 0.00 |
| Question KNN | 200 | 0.200 | 58.60 / 71.57 | 0.12 | 0.38 | 0.59 | 0.19 |

Unfiltered `argmax` switching hurts across few-shot budgets, losing 3.87, 3.44, and 2.08 macro percentage points against the incumbent because harmful switches outnumber rescues by four to five times. Validation gating recovers +3.78, +3.40, and +2.11 macro points over raw `argmax` (all conditional 95% intervals exclude zero), including raising the threshold to 1.0 (`incumbent_only`) for 100-probe KNN. Under validation gating, 50 probes shows a small decline of -0.09 points ([-0.11, -0.06]), 100 probes (-0.04 points, [-0.11, +0.03]) and full support (-0.01 points, [-0.08, +0.07]) have intervals crossing zero, and at 200 probes the gated estimate is 0.02 percentage points above the incumbent, with the paired interval crossing zero ([-0.02, +0.06]).

Per-candidate logs show how sparse and asymmetric those switches are. At 50 probes, gated compact 8D loses 183 net correct answers in seed 37 on `TheBloke__tulu-30B-fp16` (26 rescues against 209 harms across 802 selections), driving most of the budget-50 deficit. At 200 probes, only four of 39 seed-by-candidate episodes switch at all (`NousResearch__Nous-Hermes-13b` switches once with zero net change in seed 23), while `Luminex-34B-v0.1` gains +5 net in seed 23 (9 rescues vs. 4 harms), `BioMistral-7B` gains +1 in seed 23 (2 vs. 1), and `google__gemma-2b-it` loses -6 in seed 11 (4 vs. 10). Under `fixed_candidate` at 50 probes, seed 23 switches all 3,370 queries to `Luminex-34B-v0.1` for 255 rescues and 464 harms (-209 net). Full episode counts live in [the add-one candidate log](routing/test_add_one_by_candidate.csv).

![Two-panel routing accuracy plot comparing 13-model pool routing on the left and validation-gated add-one gain over the incumbent on the right across 50, 100, and 200 probes.](routing/routing_accuracy.png)

*Macro routing accuracy across 50, 100, and 200 balanced-random probes per candidate. The left panel compares four pool routers across all 13 held-out models alongside the full 15,733-probe 8D reference line. The right panel plots add-one macro gain over the Meta-Llama-3-70B-Instruct incumbent under validation margins with 95% paired, conditional query-bootstrap intervals.*

### Single-prompt inference path

For one future prompt, the single-prompt CLI loads a frozen `neural_lowrank` decoder, encodes the prompt with `all-mpnet-base-v2` (or reads a saved `.npy` vector), and computes `latents @ query_design` for every trained ID or fitted `.npz` embedding. Pure low-rank ranking needs no `--benchmark` because the shared question difficulty offset cancels across candidates, while passing `--benchmark` adds predicted correctness probabilities and enables `--minimum-gain` two-candidate gating. The tool prints JSON naming one selected candidate so a caller can dispatch one downstream model call without an intermediate server. On a held-out validation prompt and a 50-probe fitted newcomer, local raw-prompt encoding matched the released feature vector within `5.25e-7` maximum score error and selected the identical model in [the query router](../scripts/route_query.py).

The `fit` subcommand fits a newcomer embedding from score-labeled probes and reuses the same frozen checkpoint across fitting and routing for the routing-selected BCE objective. Passing `--stdin` consumes one nonempty prompt per line and emits one independent JSON choice per line while reusing the decoder and text encoder, as shown in [the README](../README.md).

### Checkpoint selection and routing-aware training

Original EmbedLLM learned model-ID embeddings for unseen questions. Our variant freezes a shared 8D `neural_lowrank` decoder to onboard unseen model IDs from few responses, linking [few-observation LLM profiles](https://arxiv.org/abs/2609.37362) with a [softmax expected routing regret](https://arxiv.org/abs/2505.16037) training surrogate. In [the routing-learning study](../scripts/pilot_routing_learning.py), all three arms train 8D embeddings on the 62 historical training models and 15,733 support items across seeds `11`, `23`, and `37`, applying identical inverse-frequency benchmark weights (mean weight 1) and the same balanced 200-probe validation procedure across 13 validation models and 3,372 validation queries (`prior_strength = 0.3`). Each seed runs two fixed 100-epoch trajectories evaluated at the same 51 candidate epochs (epoch `1` and every `2` epochs through `100`). Arm `A` (`A_bce_score_selection`) and arm `B` (`B_bce_routing_selection`) select from one benchmark-macro BCE trajectory by minimum validation macro score MAE versus maximum validation macro routed accuracy. Arm `C` (`C_bce_regret_routing_selection`) adds softmax expected routing regret over 13 sampled training models per minibatch (`lambda = tau = 1.0`) and selects its checkpoint by validation macro routed accuracy.

Test evaluation pools candidate IDs as either the 13 audited test models (`audited_13`) or 37 models (`expanded_37`) combining those 13 with all 24 lineage-uncertain metadata-excluded IDs across the same 3,370 common test question groups. A budget of 200 probes per candidate corresponds to 2,600 probe attempts on `audited_13` and 7,400 on `expanded_37`, and 100 probes acts as a secondary budget (1,300 and 3,700 attempts). Both test pools come from retrospective same-release cohorts with historical partition reuse and provide no lineage-blind evidence for the 24 excluded models. Because the two rosters share queries and 13 candidates, the expansion provides coverage, with no independent replication claim. Same-budget fixed comparators track the single highest-scoring probe model (`support_best_fixed`) and the top predicted model over a disjoint reference bank of 64 support-only prompts per benchmark (`embedding_mean_fixed`, 320 prompts total). Within each seed, both fixed selectors choose the same candidate and coincide on test at 100 and 200 probes (`54.62%` on 13 models and `56.49%` on 37 models) and on validation at 200 probes (`52.92%`), while their chosen candidate can differ across seeds.

| Cohort | Probes | `A` (BCE + MAE) | `B` (BCE + routing) | `C` (BCE+regret + routing) | `support_best_fixed` | `embedding_mean_fixed` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (13 models) | 100 | 51.54% | 53.94% | 53.94% | 49.37% | 52.92% |
| Validation (13 models) | 200 | 51.60% | 55.16% | 54.61% | 52.92% | 52.92% |
| Audited test (13 models) | 100 | 48.15% | 53.19% | 53.34% | 54.62% | 54.62% |
| Audited test (13 models) | 200 | 50.29% | 53.83% | 53.89% | 54.62% | 54.62% |
| Expanded test (37 models) | 100 | 53.76% | 56.58% | 56.61% | 56.49% | 56.49% |
| Expanded test (37 models) | 200 | 53.61% | 57.09% | 57.21% | 56.49% | 56.49% |

Paired 200-probe contrasts use a within-benchmark paired query bootstrap (2,000 draws after averaging seeds `11`, `23`, and `37`), conditional on the fixed candidate rosters, balanced probe draws, and selected validation checkpoints. Validation routing-accuracy selection (`B - A`) raised test macro accuracy by +3.53 ([+2.38, +4.73]) percentage points (+2.28 micro) on `audited_13` and +3.48 ([+1.72, +5.08]) percentage points (+3.31 micro) on `expanded_37`, with all three seeds positive in both pools. Adding softmax expected regret (`C - B`) shifted macro accuracy by +0.06 ([-0.70, +0.83]) points (-0.04 micro) on `audited_13` and +0.12 ([-0.70, +1.00]) points (-0.94 micro) on `expanded_37`.

Relative to both coinciding fixed comparators at 200 probes, `B - fixed` measured -0.80 ([-1.50, -0.19]) macro points on `audited_13` and +0.61 ([-0.75, +1.95]) on `expanded_37`. Meanwhile `C - fixed` measured -0.73 ([-1.75, +0.16]) on `audited_13` and +0.73 ([-0.74, +2.14]) macro points (-0.20 micro) on `expanded_37`. Both `B` and `C` fell short of the planned +1 percentage-point target over both fixed comparators with a positive paired interval.

At 200 probes in `expanded_37`, [the per-seed test log](routing/learning/test_by_seed.csv) shows that both fixed selectors choose ID 76 in seeds 11 and 23 and ID 49 in seed 37. Across those three seeds, `B - fixed` macro gaps are -1.72, -1.05, and +4.59 percentage points, and `C - fixed` gaps are -0.05, -0.84, and +3.07 points, so the positive mean relies on the third seed. In `audited_13`, `B` routes 97.3–98.0% of queries to the fixed candidate versus 81.5–90.3% for `A`, and that difference primarily reflects better routing choices in this pool, with no demonstrated increase in same-usage within-benchmark matching.

In full 15,733-item support (containing 9,689 `MMLU` prompts), switching arm `B` on `expanded_37` from ordinary unweighted MAP fitting (55.62% macro, 72.65% micro) to equal-weight per-benchmark support (56.74% macro, 68.91% micro) shifted test accuracy by +1.12 ([-0.39, +2.78]) macro points and -3.74 micro points. Full summary tables and paired bootstrap intervals live in [the validation learning summary](routing/learning/validation_summary.csv), [the test learning summary](routing/learning/test_summary.csv), and [the paired test comparisons](routing/learning/test_paired.csv).

![Two-panel macro accuracy difference plot comparing 13 audited models on the left and 37 held-out model IDs on the right across routing selection versus score selection, added regret loss versus BCE, and routing-selected BCE versus the fixed comparator.](routing/learning/checkpoint_comparison.png)

*Test macro accuracy differences at 200 probes per candidate on the 13 audited and 37 held-out model pools, with paired 95% query-bootstrap intervals conditional on the candidate roster, seeds, and selected checkpoints.*

Optional two-step reproduction defaults to `--run artifacts/main_v2`, `--checkpoints artifacts/routing_learning`, and `--output reports/routing/learning`. Users on a fresh clone can pass `--run` pointing to a freshly trained run such as `artifacts/reproduction` alongside separate `--checkpoints` and `--output` paths to keep checked-in reports unchanged.

```bash
python scripts/pilot_routing_learning.py --phase train_validate
python scripts/pilot_routing_learning.py --phase test
```

### External transfer evaluation on LLMRouterBench

External transfer evaluates the official LLMRouterBench JSON-results archive (`NPULH/LLMRouterBench`, revision `b95870f9e60c589e82f7972c91ffa8de9ca89f85`) across all 20 performance-roster model IDs with complete original binary grades. These model IDs are absent from EmbedLLM by normalized literal alias, while weight-family overlap remains possible. Matching Unicode NFKC casefolded word and math-operator tokens against raw EmbedLLM prompts and extracted stems removes 588 of 1,001 MMLU-Pro stems and all 198 GPQA stems, so the evaluation uses predesignated KORBench alongside MMLU-Pro and BBH while undetected paraphrases may remain. Deduplicating 21 BBH and 65 KORBench stems leaves 2,657 question groups split into 1,327 support groups and 1,330 query groups (`207` MMLU-Pro, `530` BBH, and `593` KORBench) in [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench), [Hugging Face](https://huggingface.co/datasets/NPULH/LLMRouterBench), [the external protocol](routing/external_transfer/protocol.json), and [the external routing script](../scripts/run_external_routing.py).

Frozen `A` and `B` checkpoints across seeds `11`, `23`, and `37` transfer with zero external training or selection under `100` and `200` nested model-shared balanced-random probes per candidate (`2,000` and `4,000` responses per roster). Unknown benchmark offsets add the unweighted mean of all five historical benchmark biases to the frozen learned text difficulty head under the same MAP prior (`prior_strength = 0.3`), and each query scores all 20 fitted embeddings to pick one model before historical labels are scored against same-budget `support_best_fixed`, two arm `embedding_mean_fixed` comparators from a disjoint support reference (`60` prompts per source), `smoothed_domain`, `question_knn`, and hindsight `oracle` (`92.32%` macro). Historical generation prompts differ from encoder features on BBH (final question with choices) and KORBench (query first and task rule, with worked examples stripped). Under the `384`-token `all-mpnet-base-v2` cap, `755` of `1,185` extracted KORBench inputs exceed the cap (`5` long question stems alone), along with `45` MMLU-Pro and `9` BBH full feature inputs, so context loss may impact transfer while making no causal claim in [the prompt audit](routing/external_transfer/prompt_audit.json).

| Policy | 100 probes macro / micro (%) | 200 probes macro / micro (%) |
| --- | ---: | ---: |
| `A` (`A_bce_score_selection`) | 69.78 / 67.29 | 70.60 / 68.77 |
| `B` (`B_bce_routing_selection`) | 72.21 / 70.10 | 72.60 / 71.25 |
| `A` `embedding_mean_fixed` | 72.94 / 71.55 | 73.47 / 72.33 |
| `B` `embedding_mean_fixed` | 73.65 / 72.06 | 73.47 / 72.33 |
| `support_best_fixed` | 74.35 / 72.56 | 74.18 / 72.83 |
| Domain probe mean | 73.95 / 72.16 | 74.44 / 73.13 |
| Question KNN | 73.62 / 72.61 | 74.08 / 72.86 |

At 200 probes, `B` achieves `72.60%` macro (`71.25%` micro) accuracy versus `70.60%` macro (`68.77%` micro) for `A`, yielding a `B - A` gain of `+2.00` macro percentage points (conditional 95% paired interval `[+1.08, +2.93]`, `+2.48` micro points, `2` of `3` seeds positive), while at 100 probes `B - A` reaches `+2.42` macro points (`[+1.31, +3.52]`). At 200 probes, comparators reach `74.18%` macro for `support_best_fixed`, `73.47%` for `B` `embedding_mean_fixed`, `74.44%` for domain probe mean, `74.08%` for question KNN, and `92.32%` for the hindsight oracle, leaving `B - support_best_fixed` at `-1.58` macro points (`[-2.44, -0.74]`) and `B - B embedding_mean_fixed` at `-0.87` macro points (`[-1.56, -0.21]`). This external evidence supports the checkpoint-selection effect here and indicates that the router still trails simple fixed baselines, with zero current real API cost or latency measurements in [the external summary](routing/external_transfer/summary.csv), [the paired external intervals](routing/external_transfer/paired.csv), [the per-benchmark breakdown](routing/external_transfer/by_benchmark.csv), and [the per-seed breakdown](routing/external_transfer/by_seed.csv).

Safe optional reproduction uses the pinned archive placed at `data/raw/llmrouterbench/bench-release.tar.gz` (`1,283,503,080` bytes) with the optional `sentence-transformers` packages already documented in [the README](../README.md). Omitting `--run`, `--checkpoints`, `--learning-selection`, `--audit`, and `--archive` reuses the same checkpoints and protocol (`artifacts/main_v2`, `artifacts/routing_learning/checkpoints`, `reports/routing/learning/selection.json`, and `reports/routing/external_transfer/protocol.json`), while `--output artifacts/llmrouterbench_reproduction` and `--features artifacts/llmrouterbench_reproduction/features.npz` keep checked-in files untouched and `--device cuda` (or `--device cpu` on CPU hosts, as the script passes `--device` directly to `SentenceTransformer`) encodes prompts in [the external routing script](../scripts/run_external_routing.py).

```bash
python scripts/run_external_routing.py --phase prepare --device cuda --features artifacts/llmrouterbench_reproduction/features.npz --output artifacts/llmrouterbench_reproduction
python scripts/run_external_routing.py --phase score --features artifacts/llmrouterbench_reproduction/features.npz --output artifacts/llmrouterbench_reproduction
```

### Sequential question selection and adaptive stopping

We next evaluate choosing both which onboarding questions to ask and when to stop requesting answers across all three routing-selected BCE (`B`) checkpoints (seeds 11, 23, and 37) on the add-one newcomer-versus-incumbent problem. Fixed-budget evaluations in the core studies remain intact as fair controls, and this sequential arm re-fits incumbent ID 50 (`Meta-Llama-3-70B-Instruct`) from all 15,733 historical support items via the same unweighted 8D MAP solver (`prior_strength = 0.3`) used for newcomers, so the incumbent vector differs from the original jointly fitted anchor. True online test replay stops probe requests and MAP refits once the stopping criterion fires, while `sequential_probes_fixed_200` runs separately afterward as a retrospective 200-probe comparator verified to match the online prefix. Matched balanced random (`balanced_random_same_attempts`) fits each arrival at its exact realized probe count using the same 10 shared warm-up probes per seed and the same 500-prompt label-blind support bank. Both question selection and stopping count operate strictly on support data without target texts or target labels during onboarding, exposing only query features at routing time before target labels score outcomes in [the sequential routing runner](../scripts/run_sequential_routing.py), [the sequential validation record](routing/sequential/validation_selection.json), and [the learning selection record](routing/learning/selection.json).

Every newcomer starts from 10 shared balanced warm-up probes per seed (two per benchmark) and advances under deterministic round-robin benchmark quotas up to a cap of 200 probes from 500 candidate support prompts (100 per benchmark), tracked against a disjoint 160-prompt equal-domain support reference bank (32 per benchmark). At each step, the selector pre-screens at most 48 eligible same-domain candidates using a Laplace variance approximation and evaluates up to 8 top-ranked candidates on both hypothetical Bernoulli outcomes (`0` and `1`) using actual warm-started regularized 8D MAP refits, as available same-domain candidates can fall below either bound. The two post-probe reference prediction shifts are centered to zero expectation under the current probe probability, picking the candidate that maximizes centered expected post-probe positive advantage over the incumbent beyond the admission margin while revealing only that single queried answer with zero live ensemble generation. A separate stopping score computes mean approximate expected value of perfect information (EVPI) near the admission margin across the 160 reference questions from the current MAP inverse-Hessian covariance. Posterior EVPI serves as a heuristic conditional on the fitted decoder and Gaussian Laplace approximation, offering no formal accuracy guarantee. Across 13 validation newcomers and 3,372 validation queries, one threshold grid first selects admission margin `0.025` on 200 balanced-random validation probes. Under that fixed margin, the sequential selector is separately continued to 200 probes (`tolerance = 0`), and a second threshold grid then picks EVPI tolerance `0.0003` (62.18 mean probes, 26 median) on those continued trajectories to minimize mean responses subject to at most 0.10 percentage points macro loss relative to running the selector to 200, with zero tuning on test. This design adapts [expected-error-reduction active learning](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) and [knowledge-gradient acquisition](https://doi.org/10.1287/ijoc.1080.0314), alongside [precision-based adaptive stopping](https://arxiv.org/abs/2608.14425) as a close alternative, without claiming novelty over prior active-learning or stopping literature in [the validation summary](routing/sequential/validation_summary.csv), [the validation margin grid](routing/sequential/validation_margin_grid.csv), [the validation tolerance grid](routing/sequential/validation_tolerance_grid.csv), and [the validation paired intervals](routing/sequential/validation_paired.csv).

| Policy | Mean / median probes | Total probes | Macro / micro (%) | Newcomer share (%) | Rescues / harms | Capture (%) | Local CPU (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential probes + EVPI stop | 45.44 / 14 | 1,772 | 58.5352 / 71.5651 | 0.4664 | 1.95 / 2.21 | 0.97 | 0.128 |
| Sequential probes, fixed 200 | 200.00 / 200 | 7,800 | 58.5030 / 71.5598 | 0.4976 | 2.05 / 2.49 | 1.02 | 0.700 |
| Matched balanced random | 45.44 / 14 | 1,772 | 58.5502 / 71.5643 | 0.5326 | 2.18 / 2.46 | 1.08 | 0.0002 |
| Balanced random 50 | 50.00 / 50 | 1,950 | 58.5286 / 71.5620 | 0.4390 | 1.77 / 2.13 | 0.88 | 0.0002 |
| Balanced random 100 | 100.00 / 100 | 3,900 | 58.5329 / 71.5636 | 0.5159 | 2.03 / 2.33 | 1.00 | 0.0002 |
| Balanced random 200 | 200.00 / 200 | 7,800 | 58.5472 / 71.5651 | 0.5577 | 2.31 / 2.56 | 1.14 | 0.0003 |
| Incumbent only | 0.00 / 0 | 0 | 58.5998 / 71.5727 | 0.0000 | 0.00 / 0.00 | 0.00 | 0.0000 |

Across 13 held-out test IDs and 3 seeds (39 arrivals on 3,370 queries), adaptive onboarding requests a mean of 45.44 probes (median 14, range 10–200), totaling 1,772 requests versus 7,800 at the cap (77.3% fewer). Three arrivals reach the 200-probe cap, 36 stop early at the EVPI threshold while showing at least one reference edge above the admission margin (a heuristic trigger without confidence certification), and zero stop without a positive reference edge (`weak_on_reference = 0`). Conditional 95% paired macro accuracy differences place adaptive +0.0322 pp ([+0.0065, +0.0649]) above the same selector fixed at 200, -0.0150 pp ([-0.0515, +0.0261]) against matched balanced random, and -0.0646 pp ([-0.3845, +0.2066]) against incumbent-only. With 0.4664% newcomer selection share and 1.95 mean rescues versus 2.21 harms across 201.69 opportunities (0.97% capture), these savings occur on an incumbent-dominated setup yielding few additional beneficial queries and zero advantage over balanced random probing at matched depth. Local observed CPU wall-time for the selection and MAP refit algorithm averages 0.128s versus 0.700s at the cap, while variable actual LLM generation time remains unmeasured in [the sequential test summary](routing/sequential/test_summary.csv), [the sequential paired intervals](routing/sequential/test_paired.csv), [the per-arrival log](routing/sequential/test_by_arrival.csv), [the per-benchmark breakdown](routing/sequential/test_by_benchmark.csv), and [the adaptive onboarding figure](routing/sequential/adaptive_onboarding.png).

![Two-panel sequential onboarding plot showing empirical probe-count stops across 39 arrivals on the left and adaptive test macro accuracy differences with conditional 95% intervals against fixed-200 selection, matched balanced random, and incumbent-only baselines on the right.](routing/sequential/adaptive_onboarding.png)

*Empirical probe-count stops across 39 test arrivals on the left (mean 45.4 probes) and adaptive test macro accuracy differences with conditional 95% paired query-bootstrap intervals on the right, using [the sequential test summary](routing/sequential/test_summary.csv) and [the sequential paired intervals](routing/sequential/test_paired.csv).*

Safe reproduction omits `--run`, `--checkpoints`, `--learning-selection`, and `--incumbent-selection` to reuse the local defaults (`artifacts/main_v2`, `B` checkpoints in `artifacts/routing_learning/checkpoints`, `reports/routing/learning/selection.json`, and `reports/routing/validation_selection.json`), while passing `--output artifacts/sequential_reproduction` keeps checked-in reports unchanged in [the sequential routing script](../scripts/run_sequential_routing.py).

```bash
python scripts/run_sequential_routing.py --phase validation --output artifacts/sequential_reproduction
python scripts/run_sequential_routing.py --phase test --output artifacts/sequential_reproduction
```

For practical onboarding of a new model, [the onboarding CLI](../scripts/onboard_model.py) serves one `next_probe` prompt per step until stopping and writes paired newcomer and incumbent embeddings for gated per-prompt routing with [the query router](../scripts/route_query.py), as documented in [the README](../README.md).

### Validation-only routing-aware probe pilot

We also ran a validation-only active acquisition pilot using the frozen seed-11 `neural_lowrank` decoder, 13 validation newcomers, and 3,372 validation queries. Disjoint support-only banks allocate 32 reference prompts and 100 probe candidates per benchmark (160 reference and 500 candidate prompts total). Every newcomer starts from a common set of 10 balanced warm-up probes and advances with identical round-robin benchmark quotas to budgets 50 and 100 across four arms (`balanced_random`, `score_fisher`, `d_optimal`, and `routing_expected_improvement`) in [the pilot script](../scripts/pilot_routing_acquisition.py).

To score each candidate probe for routing decisions against the incumbent under the fixed 0.10 margin, define the current decision gap on a reference prompt as `d = p_new - p_incumbent - 0.10` and let `gamma` denote the approximate standard deviation of a one-probe posterior-mean update to `p_new` under the Laplace covariance. Following [knowledge-gradient acquisition](https://doi.org/10.1287/ijoc.1080.0314), the expected upside on that reference prompt is `gamma * phi(|d| / gamma) - |d| * Phi(-|d| / gamma)`, where `phi` and `Phi` are the standard normal density and cumulative distribution function, and each candidate probe is scored by averaging this upside over the 160 reference prompts.

| Arm | Budget 50 fixed 0.10 (%) | Budget 100 fixed 0.10 (%) | Budget 50 argmax (%) | Budget 100 argmax (%) |
| --- | ---: | ---: | ---: | ---: |
| Balanced random | 58.6104 | 58.6735 | 58.1923 | 57.9167 |
| Score-Fisher | 58.2523 | 58.6448 | 57.8888 | 58.2040 |
| D-optimal | 58.2630 | 58.5507 | 58.0572 | 57.9988 |
| Routing-aware expected improvement | 58.4819 | 58.6320 | 57.9145 | 58.3272 |

Under the fixed 0.10 admission rule, routing-aware expected improvement reaches 58.4819% and 58.6320% macro accuracy at budgets 50 and 100 versus 58.6104% and 58.6735% for balanced random, showing -0.13 and -0.04 percentage points with zero improvement over random probing. Under the unfiltered `argmax` diagnostic at budget 100, routing-aware acquisition reaches 58.3272% and leads balanced random (57.9167%) by +0.41 percentage points, yet it remains below incumbent-only validation accuracy (58.3574%). This descriptive pilot reused the validation split, evaluated zero test queries, and makes no positive empirical claim in [the pilot summary](routing/acquisition_pilot/summary.csv) and [the paired pilot log](routing/acquisition_pilot/paired.csv).

```bash
python scripts/pilot_routing_acquisition.py --run artifacts/main_v2 --validation-selection reports/routing/validation_selection.json --output reports/routing/acquisition_pilot
```

### Exploratory support-selected fallback router

We also ran an exploratory fallback routing check across the 13 validation candidates and 3,372 validation queries using saved `neural_lowrank` predictions in [the anchored pilot script](../scripts/pilot_anchored_routing.py), [the anchored pilot summary](routing/anchored_pilot/summary.csv), [the fold breakdown](routing/anchored_pilot/folds.csv), and [the per-seed breakdown](routing/anchored_pilot/by_seed.csv). Per seed (11, 23, and 37) and budget (50, 100, and 200), the policy selects one default candidate from the same balanced-random probe answers by highest probe macro accuracy and lowest source ID on ties, then switches independently for each prompt only when the highest embedding correctness score exceeds the default by a margin. Two benchmark-stratified validation query folds (1,687 and 1,685 queries) learn each budget's macro-gain-maximizing margin on the opposite fold. Decoder, checkpoint, and model tuning had prior full validation access, so only margin selection is cross-fit. This result is exploratory validation, with no final-test score.

| Budget | Default macro / micro (%) | Cross-fit macro / micro (%) | Macro gain (pp) | Conditional 95% CI (pp) | Micro gain (pp) | Switch (%) | Rescues / harms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 51.51 / 62.21 | 51.95 / 62.24 | +0.44 | [-0.10, +1.01] | +0.03 | 1.89 | 11.67 / 10.67 |
| 100 | 49.37 / 62.59 | 51.88 / 62.83 | +2.51 | [+1.65, +3.39] | +0.24 | 5.95 | 42.00 / 34.00 |
| 200 | 52.92 / 65.69 | 53.69 / 65.32 | +0.77 | [-0.17, +1.77] | -0.37 | 6.41 | 32.67 / 45.00 |

Macro gains against the same-budget default are +0.44, +2.51, and +0.77 percentage points at 50, 100, and 200 probes, with conditional 95% intervals of [-0.10, +1.01], [+1.65, +3.39], and [-0.17, +1.77] points and micro gains of +0.03, +0.24, and -0.37 points. Each interval conditions on the two selected fold thresholds and the candidate roster. At 100 probes, 5.95% of queries switch, averaging 42 rescues and 34 harms per seed, and both folds gain macro accuracy (2.97 and 2.04 percentage points under margins 0.05 and 0.075). That budget-100 gain concentrates in decoder and probe seed 11 (+6.50 macro points, 101 rescues against 81 harms), where support chose the weaker default ID 64 while seeds 23 and 37 chose ID 5 (+1.03 and 0.00 macro points). Across budgets, results show no general advantage, and micro accuracy falls at 200 probes. Reproducing this analysis on a fresh run requires `--evaluate-validation --save-predictions` at `train_validate`.

```bash
python scripts/pilot_anchored_routing.py --run artifacts/main_v2 --output reports/routing/anchored_pilot
```

## Secondary study on historical benchmark-score estimates (2026-10-28)

Our original October study audited the pinned `RZ412/EmbedLLM` release (`266ae3e180ccb533c86afc9cfb2f5db99d171aea`) across all 4,586,288 CSV rows of the 112-model by 35,673-prompt matrix, finding 553,616 duplicate model/prompt cells, 77,581 conflicting cells, and released tensor labels equal to raw cell maxima. Structural-singleton filtering over reviewed-lineage models yielded 88 models (62 train, 13 validation, 13 test) and 22,475 representative question groups (15,733 support, 3,372 validation, 3,370 test items, 975,446 training response cells), while MedMCQA matched its upstream validation split and conservative stem grouping carried minor over-grouping or paraphrase risk in [the data audit](data_audit/summary.json) and [the source review](../docs/research/data_sources.md).

With balanced random probes, macro benchmark-score MAE (percentage points, lower is better) fell across all learned predictors, yet 8D-minus-1D scalar intervals over the five reviewed lineage groups crossed zero at 100 (-0.512, [-1.257, +0.653]) and 200 (-0.506, [-1.573, +0.419]) responses.

| Predictor | 50 responses | 100 responses | 200 responses |
| --- | ---: | ---: | ---: |
| Neural low-rank embedding, 8D | 6.26 | 5.10 | 4.27 |
| Global + benchmark embedding, 6D | 6.77 | 5.77 | 4.77 |
| Structured embedding, 14D | 6.83 | 5.65 | 4.57 |
| Scalar calibrated ability, 1D | 6.20 | 5.61 | 4.78 |
| Smoothed benchmark response mean | 6.77 | 6.60 | 6.02 |
| Question KNN | 6.57 | 6.46 | 5.41 |
| Training benchmark mean, zero responses | 6.69 | 6.69 | 6.69 |

![Embedding comparison](main/embedding_budget_curve.png)

Within-benchmark response permutation raised 8D MAE by +0.188, +0.760, and +0.912 points with lineage intervals excluding zero, confirming item-response alignment beyond score totals, whereas 14D permuted point errors slightly improved over unpermuted runs. Among probe policies on the 8D decoder (`Random` 5.42 / 5.23 / 4.94, `Balanced random` 6.26 / 5.10 / 4.27, `Semantic diversity` 5.69 / 5.40 / 5.44, `Balanced diversity` 5.96 / 5.57 / 5.48, `Score-aware Fisher` 5.46 / 4.82 / 4.46, `Balanced score-aware Fisher` 5.48 / 5.09 / 4.45), global Fisher beat global random by 0.410 and 0.481 points at 100 and 200 responses while balanced diversity degraded accuracy by 1.201 points at 200. TruthfulQA showed the largest 8D benchmark error at 200 balanced probes (6.28 MAE, -5.06 signed bias), alongside retained single-model misses on OpenBuddy CodeLlama (45.90% vs. 34.36% predicted) and Gemma-2b-it MMLU (28.23% vs. 37.37% predicted), which [the main results](main/RESULTS.md) document.

![Response alignment control](main/response_alignment.png)

![Probe policies](main/probe_policy_curve.png)

Matched 40-model training-composition swaps under the fixed family preset gave mixed 200-probe excluded-minus-allowed MAE gaps for `14D / 6D` on Llama (+0.611 / +0.008, 8 targets), Mistral (-0.041 / -0.146, 6 targets), and Yi (+0.334 / +0.489, 3 targets). An exploratory 8D follow-up yielded -0.438 on Llama (4.500 vs. 4.938), +0.453 on Mistral (4.299 vs. 3.846), and a +2.225-point transfer gap on Yi (6.944 vs. 4.719). A validation-only Laplace-evidence blend of 8D and 6D averaged 4.985 MAE versus 4.889 for a fixed 50/50 blend and 4.997 for 8D alone, so we rejected it before testing. Finally, evaluating frozen main checkpoints on the 24 lineage-uncertain model IDs without intervals gave 6.593, 5.574, and 4.735 MAE for 8D versus 6.967, 6.392, and 6.320 for 1D scalar and 8.977 for the zero-probe prior, which [the family protocol](../configs/family_protocol.json), [the matched family report](family/RESULTS.md), [the exploratory 8D report](family_lowrank_secondary/RESULTS.md), [the blend review](../docs/reviews/evidence_blend_review.md), [the hybrid script](../scripts/analyze_hybrid_pilot.py), and [the 24-model coverage note](metadata_excluded/README.md) detail.

![Matched family comparison](family/matched_family_curves.png)

![Exploratory 8D family coverage](family_lowrank_secondary/matched_family_curves.png)

## Related work and limitations

Few-response profiling, empirical priors, adaptive probes, softmax regret training, and cold-start routing appear across [EmbedLLM: Learning Compact Representations of Large Language Models, ICLR 2025](https://openreview.net/forum?id=Fs9EabmQrJ), [UniRoute](https://arxiv.org/abs/2502.08773), [ZeroRouter](https://arxiv.org/abs/2601.06220), [softmax regret routing](https://arxiv.org/abs/2505.16037), [few-observation LLM profiles](https://arxiv.org/abs/2609.37362), [tinyBenchmarks](https://proceedings.mlr.press/v235/maia-polo24a.html), [AEA](https://proceedings.mlr.press/v267/li25bp.html), [ATLAS](https://proceedings.mlr.press/v306/li26fx.html), and [SkillEval](https://arxiv.org/abs/2610.28767). Our contribution is an audited empirical study spanning convex onboarding, unseen-pool routing, add-one newcomer rescue versus harm, checkpoint selection alongside softmax regret training, and negative controls. Unresolved upstream evaluator revisions, potential benchmark paraphrases or pretraining contamination, historical test-label reuse, and the lack of live generation cost or latency measurements remain open limits alongside [the project overview](../README.md).
