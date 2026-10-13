# Results log

Every eval run and finding, so we can compare everything in step 4. Newest findings at the bottom of each section.
Weave project: https://wandb.ai/cz2630-columbia-university/mathAgent/weave

## Dataset v1 (retired): `money_math_eval` 55 rows
| Source | Rows | How selected |
|---|---|---|
| GSM8K **test** split (1,319 problems) | 40 | Uniform random sample, `random.Random(seed=7)`: reproducible and unbiased. 40 is a cost/time budget. |
| Hand-written finance | 15 | Mortgages, loans, savings, payoff time, CAGR. Labels computed by reference code in `evals/finance_questions.py`. |

**Known caveats**
- GSM8K is public and old (2021), so modern models have very likely seen it in training (contamination). Near-perfect no-tool scores are expected and don't prove the model can do arithmetic on new numbers.
- `fin-08` ("how many months to pay off") is ambiguous: 92.11 months vs 93 payments. Both models answered 92.11, so this is a labeling issue, not a math error. To be reworded.

## Eval runs

### Run 1: v1 vs v2, Qwen/Qwen3.6-35B-A3B, full 55 rows, parallel=20 (Oct 11)
| Version | Accuracy | Median latency/question | p90 | Max | Wall clock (run) | Sandbox execs/question |
|---|---|---|---|---|---|---|
| v1_no_tool | 54/55 (98.2%) | 11.9s | 18.8s | 33.3s | 59s | 0 |
| v2_sandbox | 54/55 (98.2%) | 31.2s | 36.8s | **613.6s** | 614s | 1.05 |

- The only miss in both versions is `fin-08` (the label ambiguity above).
- v2 time split across all spans: ~75% in LLM calls, ~25% in sandbox tool calls.
- Sandbox cold start: median 5.6s, max 16.2s (python:3.11-slim, 1 CPU / 1Gi).
- **Outlier:** `gsm8k-1097` took 613.6s. One inference request hung for the OpenAI client's 600s default timeout before a retry succeeded, and the whole eval waited on it. **Fixed:** client `timeout=90, max_retries=2`.
- **Takeaway:** with a strong reasoning model, the sandbox did not improve accuracy on this dataset and cost ~2.6x latency. Its value here is auditability: every number comes from code that's visible in the trace.

## Dataset v2 (current): `money_math_eval` 50 rows
**Composition:** 30 FinanceReasoning-Hard (ACL 2025) + 20 GSM-Symbolic-P2 (Apple 2024). Random sample with seed 7. Built by `evals/build_dataset.py`.

**Why we switched:** GSM8K is likely memorized (98% without tools), and the hand-written finance labels were ambiguous.

**Scorer:** `numeric_match` counts an answer as correct within 0.5% relative error OR within the reference answer's rounding unit. Trade-off: for large integers, 0.5% is lenient (1153 vs 1152 passes).

### Run 2: version × model matrix, dataset v2, parallel=20 (Oct 11)
| Model | Version | Accuracy | p50 latency | p90 | Wall clock | $ / question | Sandbox execs/q |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | v1_no_tool | 45/50 (90%) | 21.2s | 32.8s | 68s | $0.0036 | 0 |
| Qwen3.6-35B-A3B | v2_sandbox | 45/50 (90%) | 37.5s | 50.0s | 343s | $0.0038 | 1.06 |
| Qwen3.6-35B-A3B | v3_sandbox_check | 46/50 (92%) | 48.3s | 119.4s | 315s | $0.0062 | 2.22 |
| gpt-oss-120b | v1_no_tool | 41/50 (82%) | 11.0s | 19.1s | 50s | $0.00013 | 0 |
| gpt-oss-120b | v2_sandbox | 43/50 (86%) | 32.3s | 35.8s | 86s | $0.00013 | 1.14 |
| gpt-oss-120b | v3_sandbox_check | 44/50 (88%) | 49.5s | 77.7s | 149s | $0.00046 | 2.12 |

Costs are computed from Weave token counts × Weave's per-token prices. Full per-question failures: run `python scripts/eval_report.py --hours N --wrong` (a local copy is in `result_backup/matrix_report.txt`, which is not committed).

> **Statistical caveat (added Oct 13, from the ARIA report review):** each cell is ONE run on the same 50 questions. Paired exact McNemar tests on Qwen vs gpt-oss within each version: v1 p=0.219, v2 p=0.727, v3 p=0.625, v4 p=1.000. None is significant, and the 2–6 point version gaps are similar in size. Read the accuracy findings below as **trends/hypotheses**. The cost differences (~28×) and latency differences are large and robust. Confirming accuracy effects needs more questions (a few hundred) and repeated trials.

**Findings**
1. **The dataset now discriminates.** The no-tool baseline fell from 98% (GSM8K) to 90% for Qwen and 82% for gpt-oss.
2. **Trend (not significant):** tools and verification appear to help the weaker model more: gpt-oss 82 → 86 → 88% vs Qwen 90 → 90 → 92%.
3. **Cost:** gpt-oss-120b is ~28× cheaper per question than Qwen3.6, because Qwen spends its budget on long reasoning output. gpt-oss v3 (88%) costs about 1/8 of Qwen v1 (90%).
4. **The v3 checker** costs 1.7–3.5× more and adds tail latency (Qwen p90 119s) for +1–2 correct answers. Marginal. In production you'd invoke it selectively (high-stakes or low-confidence answers) and use a different model as the checker for independence.
5. **Infra:** one Qwen v2 miss (`gsmp2-7-2`) was an `APITimeoutError` from the new 90s client timeout. Long reasoning outputs need more headroom, so this is a latency-vs-hang trade-off.
6. **Failure review:**
   - **Shared by all 6 runs, likely label or convention issues:**
     - `finreason-test-2017` (everyone answers 3000, label 4200)
     - `finreason-test-2144` (74.5 rounded to 75 vs label 74)
     - `finreason-test-2109` (sign convention −84.211 vs 84.211)
     - `gsmp2-26-26` (ambiguous "additional 25%")
   - **Genuine reasoning errors:**
     - gpt-oss truck-trip questions (`gsmp2-7-*`): v1 had 3 wrong, v2's sandbox fixed 2
     - `finreason-test-2062` and `-2012` (wrong formula)
     - `finreason-test-2164` (wrong discounting)

### v4_clarify: design + pre-eval findings (Oct 11)
**What v4 adds over v2:**
- `numpy-financial` in the sandbox (vetted `npf.pmt`, `npf.nper`, ... instead of hand-written formulas).
- An ambiguity policy:
  - **Chat:** ask ONE clarifying question when readings differ materially.
  - **Eval (one-shot):** a mandatory `ASSUMPTIONS:` line (or "none") plus `ALTERNATIVE: <reading> -> <number>`.

**New scorer `flags_ambiguity`:** an LLM judge (gpt-oss-120b) decides (a) whether the question is materially ambiguous and (b) whether the reply flagged it.
- `appropriate = flagged == ambiguous`, so always hedging is penalized.
- Manual labels (`KNOWN_AMBIGUOUS`: 4 questions from the run-2 failure review) override the judge's ambiguity call.

**Sandbox setup cost:** the numpy-financial install adds ~5.2s per sandbox on top of the ~4.7s cold start. In production you'd bake a custom image or keep a warm pool.

**Findings while building it:**
1. **Chat mode:** on the apartment question, Qwen asked a precise clarifying question ("original $600k or discounted $480k?"). gpt-oss answered without asking.
2. **Conditional instructions get ignored:** the first eval-mode prompt ("*if* ambiguous, state assumptions") flagged 0 of 2 known-ambiguous questions, because models are overconfident in their own reading. A **mandatory** `ASSUMPTIONS:` step fixed it: both models flagged the apartment question with the correct alternative, and both wrote "none" on a clear loan question (no false alarms).
3. **gpt-oss is inconsistent on ambiguous wording:** the same apartment question gave 108,000 in one run and 92,400 in another. On the pizza question it produced a third reading (2130). The inconsistency is itself a signal of ambiguity.

### Run 3: v4_clarify on both models, dataset v2, parallel=20 (Oct 11)
| Model | Version | Accuracy | Excl. timeouts | Final **or** stated alternative correct | p50 / p90 | $ / question |
|---|---|---|---|---|---|---|
| Qwen3.6 | v1_no_tool | 90% | 90% | 90% | 21s / 33s | $0.0036 |
| Qwen3.6 | v2_sandbox | 90% | 92% (1 timeout) | 90% | 38s / 50s | $0.0038 |
| Qwen3.6 | v3_sandbox_check | **92%** | 92% | 92% | 48s / 119s | $0.0062 |
| Qwen3.6 | v4_clarify | 84% | 89% (3 timeouts) | 88% | 35s / 69s | $0.0034 |
| gpt-oss-120b | v1_no_tool | 82% | 82% | 82% | 11s / 19s | $0.00013 |
| gpt-oss-120b | v2_sandbox | 86% | 86% | 86% | 32s / 36s | $0.00013 |
| gpt-oss-120b | v3_sandbox_check | **88%** | 88% | 88% | 50s / 78s | $0.00046 |
| gpt-oss-120b | v4_clarify | 84% | 84% | 86% | 38s / 46s | $0.00021 |

The "excl. timeouts" and "final or alternative" columns come from `scripts/alt_metric.py`, computed post hoc from Weave data with no rerun.

**Findings**
1. **v4 did not improve strict accuracy.** v3 (checker) has the highest observed score on both models, but all version gaps are within noise at n=50 (see the caveat above).
2. **The ambiguity policy costs latency on reasoning models.** Qwen p90 went from 50s to 69s, with 3 timeouts even at 180s. Excluding timeouts, v4 ≈ v2.
3. **"Right reading, wrong pick":** in several misses, the correct answer appears as the stated ALTERNATIVE (Qwen: finreason-2017, gsmp2-32-17; gpt-oss: finreason-2132). That's valuable to a customer (who can pick their reading) but invisible to strict benchmark scoring.
4. **numpy-financial: no measurable gain.** The remaining errors are interpretation and labels, not formulas. It added ~5s of sandbox setup.
5. **Judge bias (important):** the judge called 24/50 questions ambiguous when grading Qwen's replies but 34/50 when grading gpt-oss's, even though the questions are the same. Its ambiguity verdict is swayed by the reply under review (replies that list assumptions "convince" it). Human review confirms only 4. **Fix:** judge question ambiguity once, from the question alone (cached per row), and judge only "flagged?" from the reply. Measure judge-vs-human agreement before trusting it (→ Agent Lens judge alignment).
   - Raw judge output: Qwen flagged 14/24 judge-ambiguous, 3/26 false alarms. gpt-oss flagged 30/34, 7/16 false alarms.

### Re-score: precision-aware `numeric_match` v1 (Oct 13; no agent rerun)
**Why:** the ARIA report showed that v0's 0.5% relative tolerance passed 851.313 for an expected 854.466 on a "three decimal places" question.

**v1 rule:**
- `correct` = within half a unit of the precision the question asks for (or of the label's own precision, if the label is coarser).
- With no stated precision: half a unit of the label's last digit, or 0.01% relative.
- Raw `abs_error` / `rel_error` are kept.

**Logged to Weave:**
- `scripts/rescore.py --apply` ran `call.apply_scorer()` on all 400 saved predict calls.
- Each answer now carries both `numeric_match:ErXl…` (v0, the original eval score) and `numeric_match:u6Ry…` (v1) as feedback.
- The original eval summaries are unchanged.

| Version | Model | v0 (0.5%) | **v1 strict** | rel 0.1% | rel 1% |
|---|---|---|---|---|---|
| v1_no_tool | Qwen3.6 | 45 | **43** | 45 | 45 |
| v1_no_tool | gpt-oss-120b | 41 | **37** | 40 | 41 |
| v2_sandbox | Qwen3.6 | 45 | **45** | 45 | 45 |
| v2_sandbox | gpt-oss-120b | 43 | **43** | 43 | 43 |
| v3_sandbox_check | Qwen3.6 | 46 | **46** | 46 | 46 |
| v3_sandbox_check | gpt-oss-120b | 44 | **43** | 43 | 44 |
| v4_clarify | Qwen3.6 | 42 | **42** | 42 | 42 |
| v4_clarify | gpt-oss-120b | 42 | **41** | 41 | 42 |

All counts are out of 50.

**Finding:** a strict tolerance hurts **only the no-tool versions** (6 of 8 flips are v1). That fits the sandbox's purpose: mental math drifts in the last digits (97.732 vs 97.731, 1665.75 vs 1665.31, 103615 vs 103628), while code doesn't.
- Under v1, gpt-oss goes **74% (no tool) → 86% (sandbox)**. That's a 12-point gap, larger than any gap under v0 (still one run, n=50).
- Sandbox versions lose only `finreason-2164`, which is a genuinely wrong discounting (851.313 vs 854.466).

**Side effects to be aware of:**
- The first dry run called the scorer as a traced op and logged ~400 standalone `numeric_match` calls.
- The first `--apply` attempt failed (it passed the question as an ObjectRef), and logged ~400 failed scorer calls.
- Both batches are root-level calls, not attached to any eval. They clutter the Traces list and can be filtered out or deleted in the UI.

### Human ambiguity labels + judge fix (Oct 13; no agent rerun)
**Human labels:** a blind review of all 50 questions in `docs/ambiguity_review.xlsx` gave **1 ambiguous** (`gsmp2-26-26`, pizza) and 49 clear (18 marked N, 31 left blank and treated as clear, per the reviewer).
- The reviewer disagreed with 3 of my 4 failure-review picks (apartment fees, futures sign, 74.5 rounding), so even humans differ on "ambiguous".
- Stored as dataset columns `human_ambiguous`, `human_reviewed`, `human_alt_reading` and `human_notes`. This is a new version of `money_math_eval` (`3mWqXXZX…`); rows and labels are unchanged.

**Judge fix:**
- The single contaminated judge was split into `judge_question_ambiguity` (sees the question only) and `flags_ambiguity` v2 (judges only whether the reply surfaces an ambiguity).
- `appropriate` is now scored against the human label.

**Judge alignment** (`evals/judge_alignment.py`, a Weave Evaluation `judge_alignment_question_ambiguity`):

| | Human: ambiguous | Human: clear |
|---|---|---|
| Judge: ambiguous | 1 | **27** |
| Judge: clear | 0 | 22 |

Agreement is 46% and **Cohen's κ = 0.03** (chance level). Even without seeing replies, the LLM judge calls 28/50 questions ambiguous where the human found 1. **It is not usable as a substitute for human labels** without recalibration (stricter rubric, few-shot examples from the human labels).

**Flagging vs human labels** (`scripts/rejudge_flags.py`, attached to the saved answers as feedback).

The first flag-judge version (`zmnWP1…`) counted "the reply states the interpretation it used" as flagging, which was too loose. The judge was tightened so "flagged" requires naming an *alternative* reading or asking a clarifying question. It was re-judged as `flags_ambiguity:eGbfPG…` (v1 → v2 shown):

| Version | Model | Flagged | Caught (of 1) | False alarms (of 49 clear) | Appropriate |
|---|---|---|---|---|---|
| v4_clarify | Qwen3.6 | 16/50 (was 17) | 1 | 15 (was 16) | 35/50 |
| v4_clarify | gpt-oss-120b | 38/50 (was 44) | 1 | **37** (was 43) | **13/50** |

**Finding:** v4's mandatory ASSUMPTIONS/ALTERNATIVE step catches the real ambiguity but mostly produces **hedging noise**: gpt-oss names an alternative on 37/49 clear questions, Qwen on 15/49.
- **Lesson (judge design):** the first flag judge's definition was too loose. Reading the judge's reasons on a few cases exposed it. Same lesson as the ambiguity judge: inspect and validate judges before trusting them.
- With only 1 ambiguous question, "caught" can't be measured meaningfully. That needs a dataset designed with ambiguous questions.
- **Caveat:** the "flagged" judge is not human-validated (its task is concrete, but unverified).

### Speed summary (for step 4)
- **No-tool p50:** 11s (gpt-oss) / 21s (Qwen).
- **Sandbox adds ~15–20s p50:** a second LLM round trip plus the sandbox (cold start median 5.6s).
- **The checker adds another ~10–15s p50,** and much more at p90.
- **Wall clock depends on parallelism and the slowest tail question.** Hence the per-request timeout.

## Model checks (2-turn mortgage conversation, v2_sandbox; not a full eval)
Expected: turn 1 = 2528.27, turn 2 = 205557.14 interest saved.
| Model | Turn 1 | Turn 2 | Notes |
|---|---|---|---|
| Qwen/Qwen3.6-35B-A3B | ✅ | ✅ (205555.90, rounding) | Current default |
| deepseek-ai/DeepSeek-V3.1 | ✅ | ✅ exact | |
| openai/gpt-oss-120b | ✅ | ✅ (rounding) | Works after the tool_choice fix below |
| openai/gpt-oss-20b | ✅ | ✅ exact | |
| meta-llama/Llama-3.3-70B-Instruct | ❌ 2543.29 | ❌ | Buggy code, then invented numbers |

## Bugs found via Weave traces (demo story material)
1. **0 tool calls:** a chatty persona made the model *describe* the computation and stop. Fix: put the tool rule first in the prompt, plus `tool_choice="required"` on the first step.
2. **Hallucinated number:** the model printed `f"${{M:.2f}}"`, so stdout was the literal `${M:.2f}` and the model invented $2,543.29. Fix: the tool warns when stdout has no digits, and the prompt says "only report numbers from tool output".
3. **gpt-oss empty replies:** the endpoint returns an empty message for gpt-oss-120b when `tool_choice="required"` (fine with `"auto"`). Fix 1 had caused this. Fix: tool_choice is now model-aware.
4. **Eval stall:** one hung request plus the 600s default client timeout stalled the run. Fix: 90s timeout and 2 retries.
5. **Missing eval in Weave:** the Evaluation and the Dataset had the same name, so the Evaluation publish failed and its calls were dropped (404s). Fix: renamed it `money_math_benchmark`.

## Pending (step 3 continued)
- [x] Replace dataset (v2: FinanceReasoning-Hard + GSM-Symbolic-P2)
- [x] Version × model matrix (Qwen3.6, gpt-oss-120b × v1/v2/v3)
- [x] v4: numpy-financial + clarify/assumptions policy + `flags_ambiguity` scorer
- [x] Fix judge bias: question-only judge + human labels; alignment measured (κ = 0.03)
- [x] Precision-aware `numeric_match` v1, re-scored and logged to Weave
- [~] ~~Sandbox serial vs parallel benchmark~~: dropped. Sandbox depth comes from code execution + cold-start/setup data.
- [ ] Agent Lens: judge-vs-human alignment, failure clustering (if setup is quick)
- [ ] ON HOLD: clean final pass (would add duplicate eval records; the existing 8 runs are already comparable)
- [ ] Parallel vs serial sandbox benchmark (PDF extra-credit item)
