# Consolidation Drift

## Measuring Semantic Changes in Repeated LLM Memory Consolidation

**Project:** Consolidation Drift Research
**Status:** Phase 2 framework complete. 12-item pilot run and graded across 2 of 3 pipelines and all 3 context levels. Full 200-item dataset built and validated but not yet run end-to-end.

---

## 1. Overview

Large Language Models (LLMs) can summarize and consolidate information into shorter memories. However, repeatedly rewriting the same memory may gradually change its original meaning.

This research investigates whether repeated memory consolidation causes **semantic drift** — a situation where a memory is not completely forgotten, but its meaning, certainty, intensity, numerical value, or polarity changes over multiple consolidation rounds.

For example:

**Original memory:**

> I slightly prefer morning meetings, but afternoon meetings are fine too.

**Possible drift:**

> I strongly prefer morning meetings.

The second statement retains the general topic but changes the strength of the original preference.

The objective of this project is to measure whether such changes occur systematically and whether they depend more on the number of consolidation rounds or the available memory budget.

---

## 2. Research Question

Does repeatedly consolidating an LLM-generated memory produce measurable and directional changes in its original meaning?

The research also investigates:

* Does repeated consolidation cause information loss?
* Does the meaning of a memory become stronger or weaker over time?
* Are some types of information more vulnerable to drift than others?
* Does semantic error depend on consolidation depth (round count) or memory budget (context load), or on the consolidation architecture itself?

---

## 3. Research Hypotheses

### H0 — Loss or Diffusion

Repeated consolidation causes information loss or accuracy degradation, primarily related to compression and memory budget. Error appears early (as soon as budget pressure exists) and does not grow further with additional rounds.

### H1 — Directional Semantic Drift

Repeated consolidation causes a memory's meaning to move systematically in a particular direction, and that movement compounds as more rounds are applied.

```text
Slight preference
        ↓
Preference
        ↓
Strong preference
        ↓
Requirement
```

We test this with a growth exponent `b` fit to `error(round) ≈ a · round^b` per (pipeline, context level, category): `b ≈ 0` reads as H0-like (flat, budget-driven), `b > 0` reads as H1-like (compounding with depth). See §6.3.

**Important:** These are research hypotheses, not conclusions. The experiments are designed to test them, and the pilot data below shows evidence pointing in *both* directions depending on the pipeline — see §7.

---

## 4. Project History

### 4.1 Phase 0/1 — Proof of concept (superseded)

The project began with a single hand-written item and one pipeline:

* A 20-item starter dataset (`data/atomic/atomic_v0.jsonl`), 4 items each across 5 categories (preference intensity, epistemic certainty, obligation strength, numeric magnitude, polarity/negation), each with a fixed 1–5 ground-truth level.
* An LLM connection was tested first via the Gemini API (Google GenAI SDK), which hit a free-tier quota error under repeated requests, then switched to Groq (`openai/gpt-oss-20b`).
* A single `rolling` consolidation loop (`src/consolidate.py`): the model rewrites its own previous output for 10 rounds, output saved round-by-round to `data/experiments/rolling_summary/pref_001.json`.

This proved the basic mechanics (iterative rewriting, preservation-instructed prompting, delayed pacing against rate limits) but tested nothing systematically — one item, one pipeline, no embedded context, no automated grading.

### 4.2 Phase 2 — Full experimental harness (current)

The dataset, pipeline set, and evaluation harness were rebuilt from scratch to support a real experiment:

* The dataset grew from 20 to **200 graded items** (`data/atomic/atomic_v1.jsonl`), built against real conversational filler from the **LongMemEval-S** benchmark rather than synthetic-only context (LoCoMo was evaluated and kept as a documented fallback — see `consolidation-drift/docs/dataset_sources.md`).
* Each item is now embedded at **three context/budget levels** and pushed through **three consolidation pipeline architectures** (not just `rolling`), each graded automatically against its ground truth by a separate LLM call.
* A 12-item pilot has been run and graded across two of the three pipelines at all three context levels, 10 rounds each (720 graded records) — see §7 for results.

---

## 5. Repository Structure

```text
consolidation-drift/
├── README.md
├── consolidation-drift/docs/
│   └── dataset_sources.md        # provenance for filler + atomic items, build log
├── data/
│   ├── atomic/
│   │   ├── atomic_v0.jsonl       # 20-item phase 0/1 dataset (kept verbatim inside v1)
│   │   └── atomic_v1.jsonl       # 200-item graded dataset (current)
│   ├── processed/
│   │   └── filler_pool.jsonl     # 77,431 neutral turns filtered from LongMemEval-S
│   ├── embedded/
│   │   ├── embedded_v1.jsonl     # 600 rows = 200 items x {c0, c8, c20}
│   │   └── validation_report.json
│   └── experiments/
│       ├── rolling_summary/      # phase 0/1 single-item output (superseded format)
│       └── phase2/
│           ├── sliding/          # {item_id}.{level}.json chains
│           ├── hierarchical/     # {item_id}.{level}.json chains
│           ├── evaluations.json  # 720 graded records from the pilot
│           └── drift_metrics.json
├── src/
│   ├── consolidate.py            # phase 0/1 rolling script (superseded by phase2/pipelines.py)
│   ├── analyze_results.py        # phase 0/1 analysis (superseded, expects old data shape)
│   ├── calculate_drift.py        # phase 0/1 drift calc (superseded by phase2/metrics.py)
│   ├── dataset/
│   │   ├── build_filler_pool.py  # LongMemEval-S -> filler_pool.jsonl
│   │   ├── expand_atomic.py      # atomic_v0 (20) -> atomic_v1 (200)
│   │   ├── build_embedded.py     # atomic_v1 + filler -> embedded_v1.jsonl
│   │   └── validate_drift_ready.py
│   └── phase2/
│       ├── llm.py                # shared Groq client, retry/key-rotation, PRESERVE prompt
│       ├── pipelines.py          # rolling_step / sliding_step / hierarchical_step
│       ├── run.py                # pushes embedded rows through a pipeline, resume-aware
│       ├── evaluate.py           # grade() for 1-5 scales, polarity() for negation
│       ├── grade.py              # grades saved chains, writes evaluations + drift_metrics
│       └── metrics.py            # mean signed/abs drift, growth exponent, survival-vs-truth
├── tests/
│   ├── dataset/                  # 6 tests: filler pool, atomic_v1, embedded_v1, drift-ready
│   └── phase2/                   # pipeline + metrics tests
└── logs/
    ├── pilot_run.log / _run2.log / _run3.log   # resumed pilot generation runs
    └── pilot_grade.log
```

> Note: `src/metrics.py` (top-level, empty) is leftover from an earlier layout and unrelated to `src/phase2/metrics.py`. `consolidation-drift/consolidation-drift/docs/` also has a duplicated nesting level from a re-zip — worth a cleanup commit.

---

## 6. Methodology

### 6.1 Dataset

**Atomic items** (`atomic_v1.jsonl`, 200 items, 40 per category): each item is a short statement with a `category`, an `attribute`, a fixed `true_level` (1–5), and the scale's own wording for each level (e.g. level 2 = "slightly prefers morning"), so grading later uses the item's own anchors rather than a generic rubric. Polarity/negation items instead carry a boolean `true_value` and are graded as PRESERVED / FLIPPED / LOST rather than on a 1–5 scale.

**Filler pool** (`filler_pool.jsonl`, 77,431 turns): real conversational turns from LongMemEval-S, filtered to 10–280 characters and stripped of evidence turns, answer sessions, and anything containing graded keywords, so filler can't leak the answer.

**Embedded items** (`embedded_v1.jsonl`, 600 rows = 200 items × 3 levels):

* `c0` — the bare target statement, no surrounding context.
* `c8` — the target embedded with 8 filler turns.
* `c20` — the target embedded with 20 filler turns.

Filler is frozen per item (seeded by `md5(item_id)`) and `c8` is a strict prefix of `c20`, so the three levels form a clean nested budget ladder rather than three independently-sampled contexts — isolating the effect of context load from sampling noise.

A `validate_drift_ready.py` check and `tests/dataset/` (6 tests, passing) confirm prompts render correctly, scales are well-formed, and there's no filler/target leakage before anything is sent to a model.

### 6.2 Consolidation pipelines

All three pipelines share identical preservation instructions (`llm.PRESERVE`: preserve meaning, qualifiers, uncertainty, negation, numeric values, and intensity; add nothing) so the pipeline *architecture* — not prompt wording — is the experimental variable:

* **`rolling`** — `m_{k+1} = rewrite(m_k)`. The naive baseline: the model only ever sees its own last output.
* **`sliding`** — keeps the last 4 turns verbatim in a window; whenever a turn falls out of the window it is folded into a running summary via one LLM call. Simulates a fixed-size context buffer over an ongoing conversation.
* **`hierarchical`** — bottom-up re-derivation every round: turns → grouped notes (4 turns/call) → chunk summaries (4 notes/call) → one budget-capped (120-word) global memory, which is split back into statements to seed the next round. Nothing survives verbatim between rounds; everything is re-derived.

### 6.3 Evaluation and metrics

A separate LLM call — never shown the ground truth — regrades every round's output:

* Graded categories: mapped back onto the item's own 1–5 scale, giving `drift = predicted_level − true_level`.
* Negation items: a binary `PRESERVED` / `FLIPPED` / `LOST` verdict against the statement's original truth value.

`src/phase2/metrics.py` aggregates graded records (`compute_metrics`) into, per (pipeline, context level, category):

* mean signed drift and mean absolute drift, broken out by round,
* a **growth exponent** `b` fit from `error(round) ≈ a · round^b` (log-log least squares, round ≥ 1) — the H0-vs-H1 dial: `b ≈ 0` is flat/H0-like, `b > 0` grows with depth/H1-like,
* `survival_vs_truth`: the fraction of present-but-wrong graded rows per (pipeline, level, round).

Note: negation records currently carry a `verdict`, not a `level`, and `compute_metrics` only aggregates records that have a `level` — so negation results are not yet summarized anywhere (see §8).

---

## 7. Pilot Results (current)

Run: 12 items (`PILOT_IDS` in `run.py`, 3 per graded category) × 3 context levels × **2 pipelines** (`sliding`, `hierarchical`; `rolling` not yet run under phase2) × 10 rounds, model `openai/gpt-oss-120b` on Groq. 720 graded records in `data/experiments/phase2/evaluations.json`; aggregates in `drift_metrics.json`. Sample size per cell is small (n=3 items/round), so treat these as pilot-scale signal, not final results.

**Growth exponents (`b`) by pipeline / level / category:**

| Pipeline | Level | epistemic_certainty | numeric_magnitude | obligation_strength | preference_intensity |
|---|---|---:|---:|---:|---:|
| sliding | c0 | +0.000 | +0.000 | +0.000 | +0.000 |
| sliding | c8 | +0.000 | +0.000 | −0.037 | +0.000 |
| sliding | c20 | +0.000 | +0.000 | +0.000 | +0.000 |
| hierarchical | c0 | +0.000 | +0.000 | +0.347 | +0.014 |
| hierarchical | c8 | +0.000 | +0.195 | +0.234 | +0.000 |
| hierarchical | c20 | +0.163 | +0.004 | −0.038 | −0.208 |

**The clearest single signal in the pilot:** `sliding|c20|obligation_strength` has a mean signed drift of exactly **−1.00 at every round from round 1 through round 10** — a constant, immediate error that never grows or shrinks. That's about as clean an H0 (budget-driven, not depth-driven) result as pilot data gets. By contrast, `hierarchical|c0|obligation_strength` (b=+0.347) and `hierarchical|c8|obligation_strength` (b=+0.234) show drift that grows with round count — H1-like — for the same category under the pipeline that re-derives everything from scratch each round.

Taken together, this suggests the two architectures may sit on opposite sides of the H0/H1 line for the same category, which is the open thread most worth writing up next.

---

## 8. Current Limitations / Open Questions

* `rolling` has not been run through the phase2 harness at all — only the incompatible phase 0/1 format exists for it, so there's no 3-way pipeline comparison yet.
* Negation/polarity items are excluded from the pilot (`PILOT_IDS` has none) and, even if graded, `compute_metrics` does not currently aggregate verdict-based records — negation has zero summarized results.
* 188 of the 200 dataset items are untouched by any pipeline run.
* No plotting/visualization exists for `drift_metrics.json`; `analyze_results.py` / `calculate_drift.py` are phase 0/1 scripts written for the old single-item data shape and don't apply to phase2 output.
* Everything so far uses a single model (`openai/gpt-oss-120b` via Groq) — no cross-model replication yet.
* Sample size per (pipeline, level, category) cell in the pilot is 3 items — enough to see a signal, not enough to trust a growth exponent on its own.

---

## 9. Suggested Next Steps

1. Run the `sliding|c20|obligation_strength` flat-vs-growing result — and the hierarchical growth exponents — through a proper statistical test (round-1 vs round-10 drift) before treating either as a finding.
2. Scale the pilot to the full 200-item dataset across all three pipelines and levels.
3. Run `rolling` through phase2 for the first true 3-way pipeline comparison.
4. Add negation items to a pilot run and extend `compute_metrics` to aggregate verdict rates per (pipeline, level, round).
5. Build the actual drift plot(s) (mean drift vs round, faceted by pipeline/level) — still missing entirely.
6. Replicate the pilot on a second, architecturally distinct model (e.g. Gemini) to check the sliding/hierarchical split isn't a `gpt-oss` artifact.

---

## 10. Running the Pilot

```bash
# Generate consolidation chains (resume-aware: skips files already on disk)
python -m src.phase2.run --pilot --pipelines sliding,hierarchical --levels c0,c8,c20 --rounds 10

# Grade every chain against ground truth and compute drift metrics
python -m src.phase2.grade --dir data/experiments/phase2
```

Requires `GROQ_API_KEY` in a gitignored `.env` file (see `src/phase2/llm.py`).