# Engineering & Implementation Log: Metric Validation, ARC, and Protocol Fixes

**Project:** Consolidation Drift  
**Author:** AI Research Assistant  
**Date:** 2026-10-11  

---

## 1. Context & Objectives

The goal of this session was to address the findings in the *Consolidation Drift Metric Validation & New Metric Proposal* dossier:
1. Implement the **Attractor Regression Coefficient (ARC)** in `src/phase2/metrics.py` to overcome the cancellation blind spot of mean signed drift.
2. Implement the **`ATTRIBUTE_NOT_PRESENT` gate** in `src/phase2/evaluate.py` to eliminate forced-choice hallucination on degraded memories.
3. Fix the **sentence degeneration confound** in `src/phase2/pipelines.py:hierarchical_step` by replacing naive `.split(". ")` with bullet/line statement extraction.
4. Update `src/phase2/grade.py` to handle `NOT_PRESENT` records, compute attrition rates, and report ARC pooled metrics.
5. Add unit tests for all new components in `tests/phase2/test_phase2.py`.
6. Create an automated visualization suite in `src/phase2/visualize.py` producing the Budget $\times$ Depth headline heatmap and ARC trajectories.
7. Document the entire process, including rationale, thinking processes, and any failures encountered.

---

## 2. Chronological Log & Thinking Process

### Step 1: Initial Codebase Inspection & Test Suite Baseline
* **Action:** Ran `pytest` on existing repository.
* **Thinking Process:** Before touching any files, verify that all existing tests pass cleanly to establish a baseline.
* **Outcome:** 12 passed in 6.17s.

---

### Step 2: Protocol Fix in `src/phase2/evaluate.py`
* **Thinking Process:** In forced-choice classification, LLMs faced with a passage that omits the target attribute will randomly guess or pick the middle category (level 3). This mimics an attractor effect when it is actually an attrition effect. To distinguish attrition (loss) from drift (bias), the evaluator must have an explicit escape token: `NOT_PRESENT`.
* **Changes Made:**
  * Updated `grade()` prompt to instruct: *"If the consolidated memory does not mention, retain, or reflect this attribute at all, return ONLY 'NOT_PRESENT'."*
  * Added logic in `grade()` to check for `"NOT_PRESENT"` in the response and return `None`.
  * Preserved regex matching `\b[1-5]\b` for valid graded responses.

---

### Step 3: Hierarchical Pipeline Splitting Fix in `src/phase2/pipelines.py`
* **Thinking Process:** 
  The string split `glob.replace("\n", " ").split(". ")` was identified in the pilot review as causing severe sentence fragmentation (e.g., breaking `10 a.m.` into `10 a` and `m`, or breaking numbered lists at the decimal point). 
  We needed an extractor that preserves whole statements, strips list formatting, and filters out noise fragments.
* **Failure / Edge Case Identified:**
  Initially considered filtering lines with `len(line) >= 15`. However, checking `tests/phase2/test_phase2.py:test_hierarchical_fuses_chunks()` revealed that the mock client returns `"GLOBAL MEMORY."` (length 14). A threshold of 15 would have discarded valid mock outputs and broken unit tests.
* **Resolution:**
  Calibrated the minimum length threshold to $\ge 10$ characters, ensuring both synthetic test outputs and concise natural statements are preserved while sub-word artifacts (like `"m"`, `"1."`, `"3"`) are safely rejected.

---

### Step 4: ARC Metric Implementation in `src/phase2/metrics.py`
* **Thinking Process:**
  ARC fits $y_i = a + b \cdot x_i$ where $x_i = \text{true\_level}_i - 3$ and $y_i = \text{level}_i - \text{true\_level}_i$.
  To make ARC reliable for academic publication:
  1. It should compute not just the point slope $b$, but also $R^2$, Standard Error $SE(b)$, $t$-statistic, and a two-tailed $p$-value.
  2. It must be computed at multiple granularities: pooled per `(pipeline, context_level)`, by round $k \in \{1 \dots 10\}$, and by category.
  3. Records with `level is None` (`NOT_PRESENT`) must feed into an **Attrition Rate** table rather than skewing the numerical OLS fit.
* **Changes Made:**
  * Added `compute_arc()` function with full OLS regression and Student's $t$ inference (using `scipy.stats` when available, with pure Python fallback).
  * Updated `compute_metrics()` to include `arc_pooled`, `arc_by_round`, and `attrition` tracking.

---

### Step 5: Updating `src/phase2/grade.py`
* **Thinking Process:**
  When `grade()` returns `None`, `grade.py` should record `present: False`, `level: None`, and omit the `drift` key.
  When printing the evaluation summary, it should print the pooled ARC table alongside mean signed drift and growth exponents.
* **Changes Made:**
  * Added `present` boolean and `level: None` handling in `grade.py`.
  * Added formatted ARC summary printout displaying slope, $R^2$, $p$-value, and sample size $n$.

---

### Step 6: Unit Test Expansion in `tests/phase2/test_phase2.py`
* **Thinking Process:**
  We must verify all new logic offline without requiring live LLM calls:
  1. `test_grade_not_present`: Verifies `NOT_PRESENT` returns `None`.
  2. `test_compute_arc_attractor`: Verifies that a theoretical collapse to level 3 produces an exact slope of $-1.000$ and $R^2 = 1.000$.
  3. `test_compute_arc_zero_drift`: Verifies that zero drift yields slope $0.000$.
  4. `test_metrics_handles_attrition`: Verifies records with `level: None` feed attrition metrics and are excluded from drift averages.
* **Execution & Result:**
  Ran `pytest`. **All 16 tests passed** in 2.60s.

---

### Step 7: Verification and Execution Failures / Diagnoses
* **Encountered Failure 1 (PowerShell CLI Quote Syntax Error):**
  * *Command:* Attempted to run a python one-liner updating `drift_metrics.json` directly from the terminal.
  * *Error:* `SyntaxError: f-string expression part cannot include a backslash` due to nested quotation escaping (`f'p={v[\"p_value\"]}'`) inside the PowerShell string.
  * *Diagnosis & Fix:* Replaced f-string with classic printf string formatting (`'p=%.6f' % pv`), executing cleanly on the second attempt.
* **Recomputation Outcome:**
  `data/experiments/phase2/drift_metrics.json` was successfully recomputed from `evaluations.json` with the new ARC metrics.
  * Sanity check verified: `sliding|c0` yielded slope $+0.0000$ ($p = 1.000$).
  * Key attractor confirmed: `hierarchical|c0` yielded slope $-0.2000$ ($p = 0.0089$).
  * Target washout confirmed: `hierarchical|c20` yielded slope $-1.0125$ ($p < 10^{-12}$).

---

### Step 8: Building the Visualization Suite `src/phase2/visualize.py`
* **Thinking Process:**
  The research proposal identified a 2D Budget $\times$ Depth grid as the primary headline figure for the paper.
  We built `src/phase2/visualize.py` using `matplotlib` to render:
  1. `results/figures/budget_depth_heatmap.png`: A $2 \times 2$ panel comparing Mean Signed Drift and ARC across budget levels (`c0`, `c8`, `c20`) and rounds ($1 \dots 10$) for both pipelines.
  2. `results/figures/arc_trajectory.png`: Line plots tracking how the ARC slope evolves over consolidation rounds.
  3. `results/figures/arc_scatter_ols.png`: Jittered scatter plots of $(x_i, y_i)$ with fitted OLS regression lines for all experimental configurations.
* **Execution & Result:**
  Executed `python -m src.phase2.visualize`. All 3 publication-ready figures were generated in `results/figures/`.

---

## 3. Summary of Deliverables Created / Updated

| File | Status | Description |
| :--- | :--- | :--- |
| `src/phase2/evaluate.py` | Updated | Added `ATTRIBUTE_NOT_PRESENT` gate. |
| `src/phase2/pipelines.py` | Updated | Fixed hierarchical statement extraction and fragment filtering. |
| `src/phase2/metrics.py` | Updated | Added `compute_arc()` OLS regression, $R^2$, SE, $p$-values, and attrition tracking. |
| `src/phase2/grade.py` | Updated | Handled missing attributes and integrated ARC table reporting. |
| `tests/phase2/test_phase2.py` | Updated | Added 4 new unit tests (16/16 tests passing). |
| `src/phase2/visualize.py` | Created | Automated generation of headline heatmaps and ARC trajectory plots. |
| `docs/arc_metric_specification.md` | Created | Formal mathematical and theoretical specification for ARC. |
| `docs/protocol_confounds_and_fixes.md` | Created | Analysis of the 3 pilot confounds and their protocol solutions. |
| `docs/implementation_log.md` | Created | This engineering log documenting the thinking process and execution history. |
| `results/figures/*.png` | Generated | 3 headline figures generated and saved. |

---

## 4. Phase A Remediation: Pipeline Mechanics, Polarity Hardening & Clustered Statistics (2026-10-15)

### Objectives
Address the four critical issues raised in peer code review:
1. `sliding` pipeline stalling after round 1 due to empty `tail_add`.
2. Pooled ARC pseudo-replication inflating statistical significance across repeated rounds.
3. Polarity evaluator parsing vulnerability (`"NOT PRESERVED"` matching `"PRESERVED"`).
4. Missing dedicated negation error rates in `compute_metrics()`.

### Thinking Process & Implementations
* **Sliding Pipeline Redesign:**
  * Context turns are partitioned across $K$ rounds (`src/phase2/run.py`).
  * `sliding_step` ingests `new_turns` progressively. When the window overflows $W=4$, older turns evict and merge into the summary.
  * When turns are exhausted, an autonomous re-compaction pass maintains summary conciseness.
  * For $c0$ (0 turns), the window retains the target verbatim as an inert control.
* **Item-Clustered Block Bootstrap for ARC:**
  * Evaluated the pseudo-replication confound: 120 observations across 12 items are autocorrelated.
  * Implemented an item-level block bootstrap ($B=500$ iterations) resampling whole item trajectories with replacement.
  * Generates cluster-robust standard errors, empirical 95% CIs, and cluster-adjusted $p$-values.
* **Strict Polarity Regex Parser:**
  * Replaced substring containment with isolated word boundaries: `\b(PRESERVED|FLIPPED|LOST)\b`.
  * Explicitly detects negation phrases (`NOT PRESERVED`, `UNPRESERVED`) and prevents false positive assignment to `PRESERVED`.
  * Enforces a strict single-verdict rule, raising an error on conflicting labels.
* **Dedicated Polarity Summary:**
  * Added `polarity_summary` in `compute_metrics()`, tracking `preservation_rate`, `flip_rate`, and `loss_rate` broken down by pipeline, budget, and round.
* **Dynamic Pipeline Visualization:**
  * Generalized `src/phase2/visualize.py` to automatically detect and render available pipelines (`rolling`, `sliding`, `hierarchical`).

### Verification & Testing
* Added 5 new unit tests in `tests/phase2/test_phase2.py`:
  * `test_sliding_progressive_turn_feeding_calls_model_across_rounds`
  * `test_polarity_parser_strict_matching`
  * `test_polarity_parser_rejects_ambiguous`
  * `test_compute_arc_clustered_bootstrap`
  * `test_metrics_polarity_summary`
* Ran full test suite via `pytest`: **all 21 unit tests passed** in 2.50s.
* Regenerated visualization figures via `python -m src.phase2.visualize`.
