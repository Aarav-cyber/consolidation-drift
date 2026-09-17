"""Phase 2 grader: grade saved chains and compute drift metrics.

Usage:
  python3 -m src.phase2.grade --dir data/experiments/phase2

Reads every {pipeline}/{item}.{level}.json chain, grades rounds 1..N
against atomic_v1 ground truth, writes:
  {dir}/evaluations.json  (per-round records)
  {dir}/drift_metrics.json (metrics.compute_metrics output)
Prints a compact mean-signed-drift table.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .llm import DEFAULT_MODEL, get_client
from .evaluate import grade, polarity
from .metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/experiments/phase2")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    d = Path(args.dir)
    targets = {json.loads(line)["id"]: json.loads(line) for line in
               open("data/atomic/atomic_v1.jsonl", encoding="utf-8") if line.strip()}
    files = sorted(d.glob("*/*.json"))
    files = [f for f in files if f.name not in ("evaluations.json", "drift_metrics.json")]
    client = get_client()
    records = []
    for f in files:
        chain = json.loads(f.read_text(encoding="utf-8"))
        t = targets[chain["item_id"]]
        print(f"grading {chain['pipeline']} {chain['item_id']}.{chain['context_level']}",
              flush=True)
        for entry in chain["rounds"]:
            if entry["round"] == 0:
                continue
            if t["category"] == "polarity_negation":
                records.append({
                    "item_id": t["id"], "category": t["category"],
                    "pipeline": chain["pipeline"],
                    "context_level": chain["context_level"],
                    "round": entry["round"],
                    "verdict": polarity(client, t, entry["memory"],
                                        model=args.model, delay=args.delay),
                })
            else:
                lv = grade(client, t, entry["memory"], model=args.model, delay=args.delay)
                records.append({
                    "item_id": t["id"], "category": t["category"],
                    "true_level": t["true_level"], "pipeline": chain["pipeline"],
                    "context_level": chain["context_level"],
                    "round": entry["round"], "level": lv,
                    "drift": lv - t["true_level"],
                })
    (d / "evaluations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    metrics = compute_metrics(records)
    (d / "drift_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nMean signed drift by (pipeline, level, round):")
    agg = defaultdict(list)
    for r in records:
        if "drift" in r:
            agg[(r["pipeline"], r["context_level"], r["round"])].append(r["drift"])
    for key in sorted(agg):
        ds = agg[key]
        print(f"  {key[0]:12s} {key[1]:3s} round {key[2]:2d}: {sum(ds)/len(ds):+7.3f} (n={len(ds)})")
    print("\nGrowth exponents (err ~ k^b):")
    for name, s in sorted(metrics["series"].items()):
        print(f"  {name}: b={s['growth_exponent']:+.3f}")


if __name__ == "__main__":
    main()
