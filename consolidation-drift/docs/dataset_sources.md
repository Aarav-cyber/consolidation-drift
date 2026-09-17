Accessed 2026-09-17. Research use.

### LongMemEval-S cleaned (filler pool — USED in v1)
- HF dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Files staged in `data/raw/` (gitignored, ~280 MB total):
  - `longmemeval_s_cleaned.json` (265 MB, 500 instances, ~45-53 sessions each)
  - `longmemeval_oracle.json` (15 MB, evidence-only reference)
- Instance fields: `question_id, question_type, question, answer,
  answer_session_ids, haystack_session_ids, haystack_sessions[{role, content, has_answer?}]`
- Why S-cleaned: 40-session haystacks with noisy sessions removed; M (500
  sessions / ~1.5 M tokens) skipped as too large for v1.

### LoCoMo (REJECTED for v1 filler — documented alternative)
- https://github.com/snap-research/locomo (`locomo10.json`)
- Natural persona dialogues, but smaller pool (10 conversations) than
  LongMemEval-S; kept as fallback if filler diversity proves insufficient.

## Built artifacts (DriftBench v1, 2026-09-17)

- `data/processed/filler_pool.jsonl` — 77,431 neutral turns filtered from
  LongMemEval-S (evidence turns, answer sessions, and graded-keyword turns
  excluded; 10-280 chars, ASCII). Regenerate: `python3 src/dataset/build_filler_pool.py`
- `data/atomic/atomic_v1.jsonl` — 200 graded targets (v0 20 kept verbatim +
  180 new). Per-category levels L1:4 L2:8 L3:12 L4:8 L5:8; negation 24F/16T.
  Regenerate: `python3 src/dataset/expand_atomic.py`
- `data/embedded/embedded_v1.jsonl` — 600 rows (200 items x c0/c8/c20).
  Filler frozen per item (`md5(item_id)` seed), c8 is a prefix of c20.
  Regenerate: `python3 src/dataset/build_embedded.py`
- `data/embedded/validation_report.json` — drift-readiness proof (prompts
  render, scales check, no filler/target leakage). Regenerate:
  `python3 src/dataset/validate_drift_ready.py`
- Tests: `pytest tests/dataset/ -v` (6 tests, all passing).

Atomic dataset:
Synthetic items created specifically for Consolidation Drift,
following the five attribute categories defined in our proposal.

Categories:
1. Preference intensity
2. Epistemic certainty
3. Obligation strength
4. Numeric magnitude
5. Polarity / negation