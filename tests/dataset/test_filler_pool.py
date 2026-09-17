import json
from pathlib import Path

POOL = Path("data/processed/filler_pool.jsonl")
BANNED = ["prefer", "require", "must", "should", "probably", "definitely",
          "allergic", "budget"]


def test_filler_pool_has_no_graded_leakage():
    rows = [json.loads(line) for line in POOL.open(encoding="utf-8") if line.strip()]
    assert len(rows) >= 2000, f"only {len(rows)} fillers"
    for r in rows:
        assert "filler_id" in r and "content" in r and "source" in r
        low = r["content"].lower()
        assert not any(b in low for b in BANNED), r["filler_id"]
