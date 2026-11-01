"""Build a blind labelling sheet to check the judge (gpt-oss-120b) against a human.

The sheet shows only what the judge saw: memory text, attribute, 1-5 scale.
The judge's verdict, true level and pipeline stay in answer_key.json.
Sampling is stratified on the judge's verdict so the cases that drive the
result (judged too strong) are well covered:
  all 'not present' + 40 judged too strong + 24 judged too weak + the rest judged exact = 100.
Usage: python3 validation/make_sample.py [experiment_dir]
"""
import json, random, sys
from pathlib import Path

EXP = Path(sys.argv[1] if len(sys.argv) > 1 else "data/experiments/phase2_qwen")
OUT = Path("validation")
rng = random.Random(7)
atomic = {r["id"]: r for r in map(json.loads, open("data/atomic/atomic_v1.jsonl"))}
recs = [r for r in json.load(open(EXP / "evaluations.json")) if r["category"] != "polarity_negation"]

np_ = [r for r in recs if r["level"] is None]
up = [r for r in recs if r["level"] is not None and r["drift"] > 0]
down = [r for r in recs if r["level"] is not None and r["drift"] < 0]
exact = [r for r in recs if r["level"] is not None and r["drift"] == 0]
for g in (up, down, exact): rng.shuffle(g)
n_up, n_down = min(40, len(up)), min(24, len(down))
picked = np_ + up[:n_up] + down[:n_down]
picked += exact[:100 - len(picked)]
rng.shuffle(picked)

key, blocks = {}, []
for n, r in enumerate(picked, 1):
    sid = f"S{n:03d}"
    t = atomic[r["item_id"]]
    chain = json.load(open(EXP / r["pipeline"] / f'{r["item_id"]}.{r["context_level"]}.json'))
    mem = next(e["memory"] for e in chain["rounds"] if e["round"] == r["round"]).strip()
    scale = " | ".join(f"{k} = {v}" for k, v in sorted(t["scale"].items()))
    quoted = "\n".join("> " + ln for ln in mem.splitlines()) or "> (empty)"
    blocks.append(f"### {sid}\nAttribute: {t['attribute']}\nScale: {scale}\nMemory:\n{quoted}\nLABEL: \n")
    key[sid] = {"item_id": r["item_id"], "pipeline": r["pipeline"], "context_level": r["context_level"],
                "round": r["round"], "judge": "NOT_PRESENT" if r["level"] is None else r["level"],
                "true_level": r["true_level"]}

header = """# Grader check: label 100 memories

For each sample below, read the Memory and decide what level (1 to 5) it expresses for the
Attribute, using the Scale. Write the number after `LABEL:`.
Write `NP` if the memory does not mention that attribute at all.

Rules
- Judge only from the Memory text and the Scale. Do not guess what an AI would say.
- If the memory only partly keeps the attribute (for example it says "prefers mornings" but the
  scale needs "slightly prefers"), pick the level that best matches the words you can see.
- Do not open answer_key.json until you are done.
- It is fine to skip a sample you cannot decide. Leave LABEL blank.

When finished, run:  python3 validation/score.py

"""
(OUT / "label_sheet.txt").write_text(header + "\n".join(blocks), encoding="utf-8")
json.dump(key, open(OUT / "answer_key.json", "w"), indent=1)
from collections import Counter
print(len(blocks), "samples:", dict(Counter("NP" if v["judge"] == "NOT_PRESENT" else ("exact" if v["judge"] == v["true_level"] else ("too strong" if v["judge"] > v["true_level"] else "too weak")) for v in key.values())))
