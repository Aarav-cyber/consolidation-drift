import json
from collections import Counter
from pathlib import Path

ATOMIC = Path("data/atomic/atomic_v1.jsonl")
GRADED = ["preference_intensity", "epistemic_certainty",
          "obligation_strength", "numeric_magnitude"]


def test_atomic_v1_balance():
    rows = [json.loads(line) for line in ATOMIC.open(encoding="utf-8") if line.strip()]
    assert len(rows) >= 200
    cats = Counter(r["category"] for r in rows)
    for c in GRADED + ["polarity_negation"]:
        assert cats[c] >= 36, (c, cats[c])
    for c in GRADED:
        lv = Counter(r["true_level"] for r in rows if r["category"] == c)
        assert set(lv.keys()) == {1, 2, 3, 4, 5}, (c, lv)
        assert lv[3] >= lv[1] and lv[3] >= lv[5], (c, lv)


def test_atomic_v1_keeps_v0_ids():
    v0_ids = [json.loads(line)["id"]
              for line in open("data/atomic/atomic_v0.jsonl", encoding="utf-8") if line.strip()]
    v1_ids = {json.loads(line)["id"]
              for line in ATOMIC.open(encoding="utf-8") if line.strip()}
    assert set(v0_ids) <= v1_ids
