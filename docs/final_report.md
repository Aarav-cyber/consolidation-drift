# Consolidation Drift: Final Report

**Question.** When an LLM agent repeatedly rewrites its memory into a shorter form, does it only lose information (H0), or does each rewrite also push the content in a consistent direction, for example from "slightly prefers mornings" toward "requires mornings" (H1)?

**Short answer.** A weak hint, not a finding. On the clean Qwen run, no pipeline pulls memories toward the middle level (ARC slopes are small and every confidence interval includes 0). The hierarchical pipeline does show a small upward shift in strength, for example "should" becoming "must": +0.31 at 0 filler turns (p = 0.017) and +0.23 at 20 turns (p = 0.029). It comes from 4 of the 12 items (three obligation items and one preference item), it does not survive correction for testing six cells, and pooled over all cells it is +0.07 (p = 0.24). The sliding pipeline shows none. Twelve items is too few to settle it.

---

## 1. What was built

**Dataset (DriftBench v1).** 200 short statements, each with a known strength level from 1 to 5 on its own scale (preference, certainty, obligation, number) or a true/false polarity. 20 were written by hand and 180 were generated. Each statement is also embedded in 0, 8 or 20 unrelated conversation turns taken from LongMemEval-S, so memory has to compete with other content. That gives 600 test rows. The filler never changes between rounds, so the only thing that changes is the memory.

**Pipelines (how memory is rewritten each round).**
- *Sliding window:* the last 4 turns stay as written and older turns are folded into a running summary. Turns arrive a few at a time over the rounds.
- *Hierarchical:* turn notes, then chunk summaries, then one global memory capped at 120 words. The next round starts from that global memory.

**Grader.** A separate, larger model (`gpt-oss-120b`, Groq) reads the memory and returns the strength level from 1 to 5, or `NOT_PRESENT` if the attribute is no longer there. Drift is the graded level minus the true level.

**Metrics.** Mean signed drift (direction), present-but-wrong rate, attrition (statement gone), and the Attractor Regression Coefficient (ARC). ARC is the slope of drift against distance from the middle level. A negative slope means memories are pulled toward level 3. ARC confidence intervals resample whole items, because rounds of the same item are not independent.

**Tests.** 27 offline unit tests pass.

## 2. How the experiment went

| Step | What happened |
|---|---|
| First pilot (Sep 17) | 12 items, 72 chains, 720 grades. It showed a large drop for hierarchical at 20 turns. |
| Diagnosis | The drop was a measuring artifact: the grader had to guess a level even when the statement was gone, and the hierarchical step split text on ". ", which broke it. |
| Fixes (Oct 11 to 15) | Added the `NOT_PRESENT` option, line-based splitting, progressive turn feeding for sliding, item-clustered confidence intervals, a stricter polarity parser. |
| Re-run on gpt-oss-20b | Sliding was usable. Hierarchical was not: 111 of 137 "not present" grades came from a bug. The model sometimes spent its whole 2048-token output limit thinking and returned an empty reply, and the next round then answered the empty prompt with "please share your notes". |
| Fix | The output limit is now 8192 tokens, and empty replies are retried instead of passed on. |
| Final run on Qwen | The Groq free tier would have taken about two more days, so all 72 chains were re-run on a self-hosted `qwen3.8-27b` through an SSH tunnel. Sliding and hierarchical used the same model. The grader stayed on Groq `gpt-oss-120b`, so the writer and the grader are different models. |

## 3. Results

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

- **No pull toward the middle.** ARC, the main metric in the proposal, is null on both pipelines and all filler levels. Memories are not collapsing toward level 3.
- **A small push toward stronger wording, in the hierarchical pipeline only.** Items that start at "should" or "slightly prefers" end up as "must" or "prefers". This is the direction the H1 example describes, but it is a different shape from the one ARC tests, so ARC cannot see it.
- **It is not established.** Two of six cells have p below 0.05 (0.017 and 0.029). Correcting for six comparisons needs about p < 0.008, and neither passes. Pooled over all cells the shift is +0.07 with a 95% interval of -0.03 to +0.20.
- **Sliding window is clean.** Its mean drift is within 0.04 of 0 in every cell, and the 0-filler case reads 0 as an inert control should.
- **The first pilot's big result was an artifact.** The -1.01 slope for hierarchical at 20 turns came from forcing the grader to guess a level, plus the text-splitting and empty-reply bugs.
- **Status: "no confirmed drift on 12 items; a hint of strengthening in hierarchical rewrites worth testing on all 200 items", not "drift exists" and not "drift does not exist".**

## 5. Limitations

- **Small sample.** Only 12 of the 200 items were run, at 10 rounds instead of 20. Each pipeline-and-filler cell has 120 grades from 12 items.
- **One usable writer model for hierarchical.** The gpt-oss-20b hierarchical results were damaged by the empty-reply bug, so the hierarchical result rests on Qwen alone. A general claim needs two or three writer models.
- **Multiple comparisons.** Six cells were tested, so single p-values near 0.02 are not reliable on their own.
- **Grader not checked against humans.** A 100-row blind labelling sheet, built from the Qwen outputs, is prepared in `validation/` (`label_sheet.txt`, scored with `score.py`) but has not been filled in. In the gpt-oss re-run, 26 "not present" grades had the target in the text, so the grader does make mistakes.
- **No real memory-size limit.** "Budget" here means the amount of filler, not a fixed word limit.
- **Rolling-summary pipeline, negation items and the other proposed metrics were not run.**
- **Synthetic items.** 180 of the 200 statements were generated.

## 6. Next steps

1. Fill in the labelling sheet and report grader agreement.
2. Run all 200 items for 20 rounds on two or three writer models, including the rolling pipeline.
3. Add a real memory-size limit so loss and bias can be separated.
4. If drift is confirmed, build and test the proposed fix (AVR: Anchor, Verify, Reconstruct). It pins the statements most likely to drift, accepts a new summary only if it still agrees with the previous one on those statements, and restores the pinned statement when the check fails. It is designed but not built.

## Appendix: reproduce

```
python3 -m pytest tests/ -q
python3 -m src.phase2.run --pilot --pipelines sliding,hierarchical --rounds 10 \
    --backend selfhosted --model qwen3.8-27b --delay 0 --out data/experiments/phase2_qwen
python3 -m src.phase2.grade --dir data/experiments/phase2_qwen --model openai/gpt-oss-120b --delay 2.0
python3 validation/results_tables.py data/experiments/phase2_qwen
```
