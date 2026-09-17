"""Build a neutral filler pool from LongMemEval-S cleaned haystack sessions.

Filler turns supply the unrelated natural conversation wrapped around each
graded target item. They are NEVER scored; they only create budget pressure
and ecological validity for the consolidator.

Filtering rules:
- Skip evidence turns (has_answer is True).
- Skip sessions listed in answer_session_ids.
- Skip turns containing graded keywords (would leak a second gradable
  attribute into the context and contaminate phi scoring).
- Keep length 10-280 chars, ASCII only.

Output: data/processed/filler_pool.jsonl, one JSON per line:
  {filler_id, role, content, source}
"""
import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw/longmemeval_s_cleaned.json")
OUT_PATH = Path("data/processed/filler_pool.jsonl")

BANNED = re.compile(
    r"prefer|require|must|should|need|probably|definitely|certain|"
    r"allergic|available|budget|deadline|rupees|\bGB\b|vegetarian|"
    r"license|obligation|confident|allergy",
    re.IGNORECASE,
)


def main() -> None:
    raw = json.load(RAW_PATH.open(encoding="utf-8"))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_seen = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for inst in raw:
            qid = inst["question_id"]
            ans_sess = set(map(str, inst.get("answer_session_ids", [])))
            for si, sess in enumerate(inst["haystack_sessions"]):
                sess_ids = inst.get("haystack_session_ids", [])
                sess_label = str(sess_ids[si]) if si < len(sess_ids) else f"sess{si}"
                if sess_label in ans_sess:
                    continue
                for ti, turn in enumerate(sess):
                    if turn.get("has_answer"):
                        continue
                    text = (turn.get("content") or "").strip()
                    n_seen += 1
                    if not (10 <= len(text) <= 280):
                        continue
                    if not text.isascii():
                        continue
                    if BANNED.search(text):
                        continue
                    f.write(
                        json.dumps(
                            {
                                "filler_id": f"lme_s_{qid}_s{si}_t{ti}",
                                "role": turn.get("role", "user"),
                                "content": text,
                                "source": f"longmemeval_s_cleaned:{qid}:{sess_label}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_kept += 1
    print(f"seen={n_seen} kept={n_kept} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
