# Pilot Protocol Confounds and Methodological Fixes

**Project:** Consolidation Drift  
**Author:** Research Team  
**Scope:** Phase 2 Experimental Methodology & Diagnostic Fixes  

---

## 1. Overview

Analysis of the 12-item pilot run (720 evaluations across `sliding` and `hierarchical` pipelines at `c0`, `c8`, and `c20`) surfaced three critical experimental confounds. These confounds risked distorting metric interpretations unless resolved at the protocol level.

---

## 2. Confound 1: Forced-Choice LLM Judge & Hallucinated Grades

### The Problem
In the initial implementation of `src/phase2/evaluate.py:grade()`, the evaluator prompt instructed:
```text
The memory belongs to this 1-5 scale for '{attribute}':
{scale}
Return ONLY the integer 1, 2, 3, 4, or 5.
```
When memory consolidation drops the target attribute completely—especially common in `hierarchical|c20` where the target is diluted across 20 neutral turns—the LLM judge was forced to select an integer between 1 and 5. LLM evaluators under forced-choice conditions default to mid-scale ratings (level 3) or hallucinate grades based on spurious cues. 

This artificially inflated central convergence, creating an apparent $b = -1.0$ attractor effect that was actually an **attrition artifact** rather than iterative semantic drift.

### The Fix
1. **Introduction of the `ATTRIBUTE_NOT_PRESENT` Gate:**
   The prompt in `src/phase2/evaluate.py` has been updated to explicitly offer an escape token:
   ```text
   If the consolidated memory does not mention, retain, or reflect this attribute at all, return ONLY 'NOT_PRESENT'.
   Otherwise, return ONLY the integer 1, 2, 3, 4, or 5.
   ```
2. **Handling in Evaluator & Grader:**
   * When `NOT_PRESENT` is encountered, `evaluate.grade()` returns `None`.
   * `grade.py` tags the record with `present: False` and `level: None`.
3. **Decoupled Metric Tracking:**
   * **Attrition Rate:** Computed as $\frac{\text{count}(\text{NOT\_PRESENT})}{\text{total evaluations}}$ per (pipeline, context_level, round).
   * **Semantic Drift & ARC:** Computed strictly over surviving memories (`present: True`).

---

## 3. Confound 2: Hierarchical Pipeline Degeneration Across Rounds

### The Problem
In `src/phase2/pipelines.py:hierarchical_step()`, global memory was decomposed into seed statements for the next round using naive string splitting:
```python
next_statements = [s.strip() for s in glob.replace("\n", " ").split(". ") if s.strip()]
```
Across 10 rounds of consolidation, this naive split caused severe text degradation:
* Numbers and times containing periods (e.g., `10 a.m.`, `3.5 hours`, `Step 1.`) were split into nonsensical fragments (`"10 a"`, `"m"`, `"3"`, `"5 hours"`).
* Enumerations and bullet lists were fragmented into isolated numbers and single words.
* By round 10, the input statements became fragmented noise, accelerating artificial failure.

### The Fix
1. **Prompt Clarification:**
   The fusion prompt was modified to instruct the model to produce output structured as concise bullet points or clean standalone statements.
2. **Robust Multi-Stage Statement Extraction:**
   * Lines are extracted individually from the model response.
   * Markdown list markers, bullets, and numbering prefixes (`- `, `* `, `1. `, `• `) are cleanly stripped using regular expressions.
   * Any fragment shorter than 10 characters is filtered out.
   * If line-based parsing yields no valid statements, a fallback regex sentence splitter (`(?<=[.!?])\s+`) is applied with the same length threshold.
   * Tests in `tests/phase2/test_phase2.py` verify that valid multi-sentence and bulleted structures are preserved intact without fragmentation.

---

## 4. Confound 3: `sliding | c0` as an Inert No-Op Control

### The Problem
In the `sliding` pipeline:
* Window size is fixed at $W = 4$ turns.
* At context level `c0`, the target item contains only 1 turn.
* Because $\text{len}(\text{window}) \le 4$, the while-loop in `sliding_step` is never entered.
* `tail_add` remains empty, and **zero LLM calls are made**.
* The memory output at round 10 is identical character-for-character to round 0.

### The Methodological Reading
Treating `sliding|c0` as an active consolidation pipeline produces misleading comparisons. However, it serves as an **invaluable negative control**:
* Because no rewriting occurs, true semantic drift is identically zero.
* ARC computed on `sliding|c0` must equal exactly $0.000$.
* As verified in the pilot audit, `sliding|c0` produced $\text{ARC} = +0.0000$ ($p = 1.000$), serving as a rigorous verification that the ARC calculation introduces no algorithmic bias of its own.

---

## 5. Summary of Protocol Enhancements

```text
+-------------------------------------------------------------------------------+
| Raw Memory State m_k                                                          |
+-------------------------------------------------------------------------------+
                                       |
                                       v
                     [ Does memory contain target attribute? ]
                                    /     \
                                  No       Yes
                                 /           \
                                v             v
                   NOT_PRESENT (level=None)   Level 1-5 assigned
                                |                     |
                                v                     v
                   Feeds Attrition Metric     Feeds Signed/Abs Drift & ARC
                   (Loss / Dropped Rate)      (Pure Semantic Distortion Rate)
```

With these fixes deployed, future runs will accurately isolate genuine semantic drift and attractor dynamics from target attrition and parser degradation.
