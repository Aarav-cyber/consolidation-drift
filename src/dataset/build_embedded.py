"""Wrap each atomic target with frozen LongMemEval-S filler at 3 context levels.

Levels: c0 (0 turns, pure atomic - reproduces the v0 experiment),
        c8 (8 turns), c20 (20 turns). 200 items x 3 = 600 rows.

Determinism: filler sampled with random.Random(md5(item_id)) from the
filler pool sorted by filler_id, so the assignment is frozen. c8 is a
prefix of c20, so budget pressure is monotonic within an item.

Output: data/embedded/embedded_v1.jsonl, one JSON per line:
  {item_id, context_level, target:{...atomic row...},
   context_turns:[{role, content, filler_id}], expected_metric}
"""
import hashlib
import json
import random
from pathlib import Path

ATOMIC_PATH = Path("data/atomic/atomic_v1.jsonl")
FILLER_PATH = Path("data/processed/filler_pool.jsonl")
OUT_PATH = Path("data/embedded/embedded_v1.jsonl")

LEVELS = [("c0", 0), ("c8", 8), ("c20", 20)]


def seed_for(item_id: str) -> int:
    return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)


def main() -> None:
    atomic = [json.loads(line) for line in ATOMIC_PATH.open(encoding="utf-8") if line.strip()]
    filler = sorted(
        (json.loads(line) for line in FILLER_PATH.open(encoding="utf-8") if line.strip()),
        key=lambda r: r["filler_id"],
    )
    assert len(filler) >= 20, "filler pool too small"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in atomic:
            rng = random.Random(seed_for(row["id"]))
            pool = rng.sample(filler, 20)
            turns = [
                {"role": t["role"], "content": t["content"], "filler_id": t["filler_id"]}
                for t in pool
            ]
            for level, count in LEVELS:
                f.write(
                    json.dumps(
                        {
                            "item_id": row["id"],
                            "context_level": level,
                            "target": row,
                            "context_turns": turns[:count],
                            "expected_metric": "signed_drift",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    print(f"wrote {n} rows ({len(atomic)} items x {len(LEVELS)} levels) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
