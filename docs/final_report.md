# Consolidation Drift: Final Report

**Question.** When an LLM agent repeatedly rewrites its memory into a shorter form, does it only lose information (H0), or does each rewrite also push the content in a consistent direction, for example from "slightly prefers mornings" toward "requires mornings" (H1)?

**Short answer.** Not established yet. After the fixes, the sliding pipeline shows no statistically significant drift on the 12-item pilot. The hierarchical results from the gpt-oss re-run are unusable because of a pipeline bug (fixed since). The clean Qwen run is finished but its grading is still in progress, so this report will be updated with those numbers.

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

**Qwen run (`data/experiments/phase2_qwen`).** All 72 chains finished with the fixed pipeline. Grading is still running, so no numbers yet. Generate the table with `python3 validation/results_tables.py data/experiments/phase2_qwen` once `evaluations.json` exists.

## 4. Interpretation

- Sliding window: slopes are small (about -0.09) and every confidence interval touches 0. No significant drift was found. Sliding with 0 filler turns is an inert control and reads exactly 0, as it should.
- The large effect in the first pilot (-1.01 for hierarchical at 20 turns) did not survive the fixes. It was an artifact of forcing the grader to pick a level.
- No result in the clean data supports H1 (directional bias) yet. The honest status is "no drift found on 12 items", not "no drift exists".
- Final interpretation waits for the Qwen grades.

## 5. Limitations

- **Small sample.** Only 12 of the 200 items were run, at 10 rounds instead of 20. Each pipeline-and-filler cell has 120 grades from 12 items.
- **One writer model in the final run.** The gpt-oss-20b run is partial and the Qwen run is complete. A general claim needs two or three writer models.
- **Grader not checked against humans.** A 100-row blind labelling sheet is prepared in `validation/` (`label_sheet.csv`, scored with `score.py`) but has not been filled in. In the gpt-oss re-run, 26 "not present" grades had the target in the text, so the grader does make mistakes.
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
