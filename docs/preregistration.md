# Pre-registration: Stage A confirmation run

Written on 2026-11-01, before any Stage A data exists. The 12-item pilot gave a hint, not a result. This file fixes the test in advance so that the follow-up cannot be tuned to the data.

## Hypothesis

**H1-strengthen.** When the hierarchical pipeline rewrites memory repeatedly, statements about obligation and preference are graded **stronger** after the rewrite than before (positive mean signed drift). Certainty and numeric statements are not expected to shift and act as controls.

Where it comes from: in the Qwen pilot (12 items, 10 rounds), hierarchical showed mean signed drift of +0.31 at 0 filler turns and +0.23 at 20 filler turns, driven by 4 of 12 items (3 obligation, 1 preference). Neither passed correction for six comparisons.

## Data

- **Items:** 12 per graded category, picked at random (seed 2026) from items **not used in the pilot**. The list is in `validation/stage_a_items.json`. 24 of them are obligation or preference items. 24 are controls (certainty, number).
- **Pipeline:** hierarchical only. Sliding showed no drift in the pilot.
- **Filler:** 0, 8 and 20 turns.
- **Rounds:** 20.
- **Writer:** one model, stated in the results. **Grader:** `gpt-oss-120b`, a different model from the writer.
- **Chains:** 48 items x 3 filler levels = 144.

## Primary test (one test)

Mean signed drift for obligation and preference items in the hierarchical pipeline, pooled over the three filler levels and all 20 rounds. Each item is averaged first, then items are resampled 5000 times (item-level bootstrap). Two-sided, alpha = 0.05. The effect counts as confirmed only if the 95% interval excludes 0 **and** the human check below passes.

## Secondary tests (Holm-corrected)

- The same test at each filler level separately.
- The same test on the 24 control items (expected near 0).
- The primary test again with `NOT_PRESENT` grades excluded and again with them counted, to show they do not drive the result.

## Human check of the grader

A person labels the 100-sample sheet in `validation/label_sheet.txt` without seeing the judge's answers. The grader is trusted for this claim only if, among samples the judge called "too strong", the human also says the memory differs from the true level in at least 80% of them.

## Rules fixed in advance

- A chain with any empty memory is excluded and reported, not repaired.
- Nothing about the items, the primary test or the cut-offs changes after Stage A data is seen. Anything added later is labelled as exploratory.
- If the primary test fails, the report says so: no confirmed strengthening.
