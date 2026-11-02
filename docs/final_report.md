# Consolidation Drift: Final Report

**Question.** When an LLM agent repeatedly rewrites its memory into a shorter form, does it only lose information (H0), or does each rewrite also push the content in a consistent direction, for example from "slightly prefers mornings" toward "requires mornings" (H1)?

**Short answer.** Yes, for one kind of statement. In a pre-registered confirmation run on 48 new items (Stage A), the hierarchical pipeline made obligation and preference statements stronger over repeated rewrites, for example "I should water the plants" turning into "definitely responsible for watering". The shift is +0.33 levels on the 1-to-5 scale (95% CI +0.10 to +0.59, p = 0.003). Certainty and number statements did not shift (-0.03). The proposed fix, AVR, cut statement loss at 20 filler turns from 21.1% to 1.1% and raised exact-level accuracy from 78.0% to 87.4%, while the strengthening itself stayed at +0.32. These numbers come from an LLM grader; the planned human check of the grader is still to be done.

---

## 1. What was built

**Dataset (DriftBench v1).** 200 short statements, each with a known strength level from 1 to 5 on its own scale (preference, certainty, obligation, number) or a true/false polarity. 20 were written by hand and 180 were generated. Each statement is also embedded in 0, 8 or 20 unrelated conversation turns taken from LongMemEval-S, so memory has to compete with other content. That gives 600 test rows. The filler never changes between rounds, so the only thing that changes is the memory.

**Pipelines (how memory is rewritten each round).**
- *Sliding window:* the last 4 turns stay as written and older turns are folded into a running summary. Turns arrive a few at a time over the rounds.
- *Hierarchical:* turn notes, then chunk summaries, then one global memory capped at 120 words. The next round starts from that global memory.

**Grader.** A separate, larger model (`gpt-oss-120b`, Groq) reads the memory and returns the strength level from 1 to 5, or `NOT_PRESENT` if the attribute is no longer there. Drift is the graded level minus the true level.

**Metrics.** Mean signed drift (direction), present-but-wrong rate, attrition (statement gone), and the Attractor Regression Coefficient (ARC). ARC is the slope of drift against distance from the middle level. A negative slope means memories are pulled toward level 3. ARC confidence intervals resample whole items, because rounds of the same item are not independent.

**Fix: AVR (Anchor, Verify, Restore).** Before any rewrite, the writer model flags statements whose exact strength matters (anchors); it is not told which statement is the target. After each round, every anchor is checked against the new memory, and any anchor that is changed or missing is put back word for word. The check uses the writer model, not the grader that measures the results.

**Tests.** 31 offline unit tests pass.

## 2. How the experiment went

| Step | What happened |
|---|---|
| First pilot (Sep 17) | 12 items, 72 chains, 720 grades. It showed a large drop for hierarchical at 20 turns. |
| Diagnosis | The drop was a measuring artifact: the grader had to guess a level even when the statement was gone, and the hierarchical step split text on ". ", which broke it. |
| Fixes (Oct 11 to 15) | Added the `NOT_PRESENT` option, line-based splitting, progressive turn feeding for sliding, item-clustered confidence intervals, a stricter polarity parser. |
| Re-run on gpt-oss-20b | Sliding was usable. Hierarchical was not: 111 of 137 "not present" grades came from a bug. The model sometimes spent its whole 2048-token output limit thinking and returned an empty reply, and the next round then answered the empty prompt with "please share your notes". |
| Fix | The output limit is now 8192 tokens, and empty replies are retried instead of passed on. |
| Final run on Qwen | The Groq free tier would have taken about two more days, so all 72 chains were re-run on a self-hosted `qwen3.8-27b` through an SSH tunnel. Sliding and hierarchical used the same model. The grader stayed on Groq `gpt-oss-120b`, so the writer and the grader are different models. |
| Pre-registration (Nov 1) | The hypothesis, the single primary test and the cut-offs were written down and committed before Stage A ran (`docs/preregistration.md`). |
| Stage A (Nov 1 to 2) | 48 new items picked at random (12 per graded category, none from the pilot), hierarchical pipeline with and without AVR, 0, 8 and 20 filler turns, 20 rounds: 288 chains and 5,760 grades. Run on the self-hosted Qwen server at its 48-request limit, graded on Groq in 7 parallel shards. No chain had an empty memory, so none was excluded. |

## 3. Results

### Stage A: pre-registered confirmation run (main result)

`qwen3.8-27b` writes, `gpt-oss-120b` grades. 48 items, 20 rounds, 288 chains, 5,760 grades (`data/experiments/stage_a`, analysis in `validation/stage_a_analysis.py`). Drift is the graded level minus the true level; each item is averaged first and items are resampled (5,000 times) for the confidence intervals.

**Primary test: hierarchical pipeline, obligation and preference items (24), all filler levels and rounds.**

| | Mean signed drift | 95% CI | p |
|---|---|---|---|
| Primary test | **+0.33** | +0.10 to +0.59 | 0.003 |

**Secondary tests (Holm-corrected).**

| Group | Mean signed drift | 95% CI | Holm p |
|---|---|---|---|
| 0 filler turns | +0.18 | +0.03 to +0.40 | 0.002 |
| 8 filler turns | +0.48 | +0.16 to +0.84 | 0.002 |
| 20 filler turns | +0.33 | +0.01 to +0.70 | 0.081 |
| Controls: certainty and number items | -0.03 | -0.08 to +0.00 | 0.082 |

**Plain hierarchical compared with hierarchical + AVR.**

| | Hierarchical | Hierarchical + AVR |
|---|---|---|
| Statement lost ("not present"), 0 filler turns | 0.0% | 0.0% |
| Statement lost, 8 filler turns | 4.2% | 0.0% |
| Statement lost, 20 filler turns | 21.1% | 1.1% |
| Graded at exactly the true level | 78.0% | 87.4% |
| Mean signed drift, obligation and preference | +0.33 | +0.32 |
| Mean signed drift, certainty and number | -0.03 | -0.01 |

Per item, the AVR arm has a lower mean absolute error than the plain arm by 0.05 levels (95% CI +0.00 to +0.10, p = 0.047). The difference in signed drift between the arms is +0.01 (95% CI -0.10 to +0.11). AVR anchored the target statement in all 144 of its chains, plus 2.1 filler statements per chain on average, and restored an anchor in 32.6% of its 9,000 checks.

### Earlier pilot runs (12 items, 10 rounds)

**gpt-oss-20b re-run (`data/experiments/phase2_v2`), 12 items, 10 rounds, fixed grader and pipelines.** Only the sliding rows are usable. The hierarchical rows are damaged by the empty-reply bug described above: of 137 "not present" grades, 111 were non-answers ("please share your notes") or empty memories, so those rows say nothing about drift.

| pipeline | filler turns | records | not present | mean signed drift (all rounds) | mean signed drift (round 10) | present but wrong (round 10) | ARC slope [95% CI] | ARC p (clustered) |
|---|---|---|---|---|---|---|---|---|
| hierarchical | 0 | 120 | 3 | +0.15 (n=117) | +0.27 (n=11) | 3/11 | -0.02 [-0.27, +0.24] | 0.64 |
| hierarchical | 20 | 120 | 118 | -1.00 (n=2) | n/a | n/a | -2.00 | n/a |
| hierarchical | 8 | 120 | 16 | -0.15 (n=104) | -0.09 (n=11) | 4/11 | -0.41 [-0.92, +0.04] | 0.07 |
| sliding | 0 | 120 | 0 | -0.01 (n=120) | +0.00 (n=12) | 0/12 | +0.00 [-0.01, +0.01] | 1.00 |
| sliding | 20 | 120 | 0 | -0.06 (n=120) | -0.08 (n=12) | 1/12 | -0.09 [-0.28, +0.00] | 0.64 |
| sliding | 8 | 120 | 0 | -0.06 (n=120) | -0.17 (n=12) | 2/12 | -0.09 [-0.22, +0.00] | 0.22 |

How to read it: "not present" is the number of grades where the statement was missing from the memory. ARC slope below 0 means pulled toward the middle level. The clustered p-value accounts for the 12 items being re-used across rounds.

**Qwen run (`data/experiments/phase2_qwen`), the main result.** `qwen3.8-27b` writes, `gpt-oss-120b` grades. 12 items, 10 rounds, 72 chains, 720 grades. This run has no empty memories and no "please share your notes" non-answers. Only 8 grades were "not present", all in hierarchical chains with 20 filler turns, and the memory text had content in each.

| pipeline | filler turns | records | not present | mean signed drift (all rounds) | mean signed drift (round 10) | present but wrong (round 10) | ARC slope [95% CI] | ARC p (clustered) |
|---|---|---|---|---|---|---|---|---|
| hierarchical | 0 | 120 | 0 | +0.31 (n=120) | +0.33 (n=12) | 3/12 | -0.03 [-0.31, +0.29] | 0.91 |
| hierarchical | 20 | 120 | 8 | +0.17 (n=112) | +0.18 (n=11) | 2/11 | -0.07 [-0.26, +0.23] | 0.55 |
| hierarchical | 8 | 120 | 0 | -0.02 (n=120) | +0.00 (n=12) | 4/12 | +0.06 [-0.18, +0.34] | 0.61 |
| sliding | 0 | 120 | 0 | -0.03 (n=120) | +0.00 (n=12) | 0/12 | -0.04 [-0.12, +0.00] | 0.74 |
| sliding | 20 | 120 | 0 | +0.03 (n=120) | -0.08 (n=12) | 1/12 | -0.05 [-0.20, +0.09] | 0.53 |
| sliding | 8 | 120 | 0 | -0.04 (n=120) | -0.08 (n=12) | 3/12 | -0.12 [-0.28, +0.05] | 0.21 |

Because rounds of one item are not independent, mean signed drift was also tested by averaging each item first and resampling items (`validation/mean_drift_bootstrap.py`):

| pipeline | filler turns | mean signed drift (item level) | 95% CI | p | items drifting up / flat / down |
|---|---|---|---|---|---|
| sliding | 0 | -0.02 | [-0.07, +0.00] | 0.712 | 0 / 11 / 1 |
| sliding | 8 | -0.04 | [-0.17, +0.08] | 0.772 | 1 / 9 / 2 |
| sliding | 20 | +0.03 | [-0.10, +0.20] | 0.902 | 1 / 10 / 1 |
| hierarchical | 0 | +0.31 | [+0.06, +0.61] | 0.017 | 4 / 8 / 0 |
| hierarchical | 8 | -0.02 | [-0.23, +0.18] | 0.906 | 2 / 7 / 3 |
| hierarchical | 20 | +0.23 | [+0.03, +0.47] | 0.029 | 4 / 7 / 1 |
| all cells pooled | | +0.068 | [-0.030, +0.196] | 0.238 | |

Descriptive counts: 712 records were graded and 8 more were "not present". Of the 712, 599 (84%) match the true level exactly and 113 miss. Of the misses, 79 are too strong (73 by one step, 6 by two) and 34 are too weak (33 by one step, 1 by two). Rounds of the same item repeat, so treat this as a description, not a test.

Which items drive the hierarchical shift at 0 filler turns (mean drift per item): obl_001 +0.7, obl_002 +1.4, obl_003 +0.5, pref_002 +1.1. The other eight items are exactly 0.

## 4. Interpretation

- **Directional drift exists for obligation and preference statements.** Under repeated hierarchical rewriting they get stronger, and the shift passed the single pre-registered test on items the pilot never used.
- **It is specific.** Certainty and number statements, run in the same chains, stayed at about 0.
- **It is not a pull toward the middle.** The drift goes up the scale, toward stronger wording, so it is a strengthening bias rather than the regression to level 3 that ARC was designed to find.
- **AVR mainly prevents loss.** It almost removed lost statements at 20 filler turns (21.1% to 1.1%) and raised exact-level accuracy by 9.4 points. Strengthening stayed at the same level in both arms.
- **The pilot's earlier hint was the right direction.** The 12-item pilot gave +0.31 and +0.23 for hierarchical, which did not pass correction on its own; Stage A confirmed the direction on new items.
- **Caveat.** All levels come from one LLM grader. The pre-registration makes a human check of the grader part of the confirmation, and it has not been done yet.

## 5. Limitations

- **Sample.** Stage A used 48 of the 200 items (24 in the primary test). The full set has not been run.
- **One writer model.** Stage A used Qwen only. A general claim needs two or three writer models.
- **Grader not checked against humans.** A 100-sample blind sheet (`validation/label_sheet.txt`) was labelled by another model (Gemini), not by a person. It agreed with the grader on the exact level in 44 of 100 and within one level in 98 of 100 (kappa 0.25). That is a second-model comparison, not the human check the pre-registration requires.
- **No real memory-size limit.** "Budget" here means the amount of filler, not a fixed word limit.
- **Rolling-summary pipeline, negation items and the other proposed metrics were not run.**
- **Synthetic items.** 180 of the 200 statements were generated.

## 6. Next steps

1. Have a person label the grader-check samples, starting with the ones the grader called "too strong", and report agreement.
2. Run a revised AVR aimed at strengthening, and compare it on the same 48 items.
3. Repeat Stage A with a second and third writer model.
4. Run all 200 items, add the rolling pipeline and the negation items, and add a real memory-size limit.

## Appendix: reproduce

```
python3 -m pytest tests/ -q
python3 -m src.phase2.run --pilot --pipelines sliding,hierarchical --rounds 10 \
    --backend selfhosted --model qwen3.8-27b --delay 0 --out data/experiments/phase2_qwen
python3 -m src.phase2.grade --dir data/experiments/phase2_qwen --model openai/gpt-oss-120b --delay 2.0
python3 validation/results_tables.py data/experiments/phase2_qwen
python3 validation/stage_a_analysis.py data/experiments/stage_a
```
