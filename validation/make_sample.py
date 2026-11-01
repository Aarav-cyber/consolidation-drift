"""Build a blind labelling sheet to check the judge (gpt-oss-120b) against a human.

Sheet shows only what the judge saw: memory text + attribute + 1-5 scale.
Judge verdict, true level and pipeline are kept in answer_key.json, not the sheet.
"""
import csv, json, random
from collections import defaultdict
from pathlib import Path

EXP = Path("data/experiments/phase2_v2")
OUT = Path("validation")
rng = random.Random(7)

atomic = {r["id"]: r for r in map(json.loads, open("data/atomic/atomic_v1.jsonl"))}
recs = [r for r in json.load(open(EXP / "evaluations.json")) if r["category"] != "polarity_negation"]

np_by, pr_by = defaultdict(list), defaultdict(list)
for r in recs:
    key = (r["pipeline"], r["context_level"])
    (np_by if r["level"] is None else pr_by)[key].append(r)

# 40 judged NOT_PRESENT (spread over the cells that have them), 60 judged graded
def take(pool_by, n):
    cells = [k for k in sorted(pool_by) if pool_by[k]]
    out, i = [], 0
    for k in cells:
        rng.shuffle(pool_by[k])
    while len(out) < n and any(pool_by[k] for k in cells):
        k = cells[i % len(cells)]
        if pool_by[k]:
            out.append(pool_by[k].pop())
        i += 1
    return out

picked = take(np_by, 40) + take(pr_by, 60)
rng.shuffle(picked)

rows, key = [], {}
for n, r in enumerate(picked, 1):
    sid = f"S{n:03d}"
    t = atomic[r["item_id"]]
    chain = json.load(open(EXP / r["pipeline"] / f'{r["item_id"]}.{r["context_level"]}.json'))
    mem = next(e["memory"] for e in chain["rounds"] if e["round"] == r["round"])
    scale = " | ".join(f"{k}={v}" for k, v in sorted(t["scale"].items()))
    rows.append({"sample_id": sid, "attribute": t["attribute"], "scale": scale,
                 "memory": mem.replace("\r", ""), "human_label": ""})
    key[sid] = {"item_id": r["item_id"], "pipeline": r["pipeline"], "context_level": r["context_level"],
                "round": r["round"], "judge": "NOT_PRESENT" if r["level"] is None else r["level"],
                "true_level": r["true_level"]}

with open(OUT / "label_sheet.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
json.dump(key, open(OUT / "answer_key.json", "w"), indent=1)
print(len(rows), "rows;", sum(1 for v in key.values() if v["judge"] == "NOT_PRESENT"), "judged NOT_PRESENT")
