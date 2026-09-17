"""Phase 2 runner: push embedded rows through consolidation pipelines.

Usage:
  python3 -m src.phase2.run --pilot --pipelines sliding,hierarchical --rounds 10
  python3 -m src.phase2.run --items pref_001,cert_001 --levels c0 --pipelines rolling --rounds 3

Resume-aware: chains already on disk under --out are skipped.
Output: {out}/{pipeline}/{item_id}.{level}.json with per-round memories.
"""
import argparse
import json
from pathlib import Path

from .llm import DEFAULT_MODEL, format_turns, get_client
from . import pipelines as P

PILOT_IDS = [
    "pref_001", "pref_002", "pref_003",
    "cert_001", "cert_002", "cert_003",
    "obl_001", "obl_002", "obl_003",
    "num_001", "num_003", "num_025",
]
LEVELS = ["c0", "c8", "c20"]


def run_chain(client, pipeline, target_text, turns, rounds, model, delay):
    mems = [{"round": 0, "memory": format_turns(target_text, turns)}]
    if pipeline == "rolling":
        mem = mems[0]["memory"]
        for k in range(1, rounds + 1):
            mem = P.rolling_step(client, mem, model, delay)
            mems.append({"round": k, "memory": mem})
    elif pipeline == "sliding":
        state = P.sliding_init(target_text, turns)
        for k in range(1, rounds + 1):
            state, rendered = P.sliding_step(client, state, model, delay)
            mems.append({"round": k, "memory": rendered})
    elif pipeline == "hierarchical":
        stmts = P.hierarchical_init(target_text, turns)
        for k in range(1, rounds + 1):
            glob, stmts = P.hierarchical_step(client, stmts, model, delay)
            mems.append({"round": k, "memory": glob})
    else:
        raise ValueError(f"unknown pipeline {pipeline}")
    return mems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--items", default="")
    ap.add_argument("--levels", default="c0,c8,c20")
    ap.add_argument("--pipelines", default="sliding,hierarchical")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--out", default="data/experiments/phase2")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            open("data/embedded/embedded_v1.jsonl", encoding="utf-8") if line.strip()]
    want_items = set(PILOT_IDS if args.pilot else
                     (args.items.split(",") if args.items else None) or [])
    want_levels = set(args.levels.split(","))
    pipes = args.pipelines.split(",")
    out = Path(args.out)
    client = get_client()

    done, skipped = 0, 0
    for r in rows:
        if want_items and r["item_id"] not in want_items:
            continue
        if r["context_level"] not in want_levels:
            continue
        target_text = r["target"]["original_text"]
        for pipe in pipes:
            dest = out / pipe / f"{r['item_id']}.{r['context_level']}.json"
            if dest.exists():
                skipped += 1
                continue
            print(f"[{pipe}] {r['item_id']}.{r['context_level']}", flush=True)
            mems = run_chain(client, pipe, target_text, r["context_turns"],
                             args.rounds, args.model, args.delay)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({
                "item_id": r["item_id"], "context_level": r["context_level"],
                "pipeline": pipe, "model": args.model, "rounds": mems,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            done += 1
    print(f"done={done} skipped={skipped}")


if __name__ == "__main__":
    main()
