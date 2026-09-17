import json
from pathlib import Path

EMB = Path("data/embedded/embedded_v1.jsonl")


def _rows():
    return [json.loads(line) for line in EMB.open(encoding="utf-8") if line.strip()]


def test_embedded_row_count_and_levels():
    rows = _rows()
    assert len(rows) == 600
    by_item = {}
    for r in rows:
        assert "target" in r and "context_turns" in r and "context_level" in r
        assert "true_level" in r["target"] or r["target"]["category"] == "polarity_negation"
        by_item.setdefault(r["item_id"], []).append(r["context_level"])
    assert all(sorted(v) == ["c0", "c20", "c8"] for v in by_item.values())


def test_embedded_prefix_property():
    rows = _rows()
    r8 = next(r for r in rows if r["item_id"] == "pref_001" and r["context_level"] == "c8")
    r20 = next(r for r in rows if r["item_id"] == "pref_001" and r["context_level"] == "c20")
    assert len(r8["context_turns"]) == 8 and len(r20["context_turns"]) == 20
    assert ([t["filler_id"] for t in r8["context_turns"]]
            == [t["filler_id"] for t in r20["context_turns"]][:8])
