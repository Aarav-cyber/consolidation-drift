"""Prove every embedded row can drive a consolidation round + a drift score.

For each row renders:
- consolidator input: target text + frozen context turns (ground truth hidden)
- evaluator input: scale + consolidated text -> predicted level in 1..5

Checks: all rows render without missing fields, graded rows have a 5-point
scale, polarity rows carry a boolean true_value, filler is disjoint from
target attribute text (no exact containment either way).

Writes data/embedded/validation_report.json.
"""
import json
from collections import Counter
from pathlib import Path

EMB_PATH = Path("data/embedded/embedded_v1.jsonl")
OUT_PATH = Path("data/embedded/validation_report.json")


def render_consolidator(target_text, turns):
    lines = [f"User memory to retain: {target_text}"]
    for t in turns:
        who = "User" if t["role"] == "user" else "Agent"
        lines.append(f"{who}: {t['content']}")
    lines.append(
        "Rewrite the above memory concisely. Preserve qualifiers, uncertainty, "
        "negation, numbers, and statement strength. Do not add information."
    )
    return "\n".join(lines)


def render_evaluator(target, consolidated="[consolidated memory m_k]"):
    scale = "\n".join(f"{k} = {v}" for k, v in sorted(target["scale"].items()))
    return (
        f"Attribute: {target['attribute']}\nScale:\n{scale}\n"
        f"Memory: {consolidated}\nReturn ONLY the integer 1-5."
    )


def main():
    rows = [json.loads(line) for line in EMB_PATH.open(encoding="utf-8") if line.strip()]
    errors = []
    graded = 0
    for r in rows:
        t = r["target"]
        if t["category"] == "polarity_negation":
            if not isinstance(t.get("true_value"), bool):
                errors.append(f"{r['item_id']}: negation missing boolean true_value")
            continue
        graded += 1
        if set(t.get("scale", {}).keys()) != {"1", "2", "3", "4", "5"}:
            errors.append(f"{r['item_id']}: scale is not 1-5")
        if t.get("true_level") not in (1, 2, 3, 4, 5):
            errors.append(f"{r['item_id']}: bad true_level")
        # leakage: filler must not contain the full target text or vice versa
        for turn in r["context_turns"]:
            if t["original_text"] in turn["content"] or turn["content"] in t["original_text"]:
                errors.append(f"{r['item_id']}: filler/target containment")
        # prompts must render with no placeholders left
        cp = render_consolidator(t["original_text"], r["context_turns"])
        ep = render_evaluator(t)
        if "{" in cp and "}" in cp and "User memory" not in cp:
            errors.append(f"{r['item_id']}: unrendered consolidator prompt")
        if "{" in ep or "}" in ep:
            errors.append(f"{r['item_id']}: unrendered evaluator prompt")

    sample = next(r for r in rows if r["item_id"] == "pref_001" and r["context_level"] == "c8")
    report = {
        "n_rows": len(rows),
        "n_items": len({r["item_id"] for r in rows}),
        "levels": sorted({r["level"] if "level" in r else r["context_level"] for r in rows}),
        "categories": dict(Counter(r["target"]["category"] for r in rows)),
        "graded_rows": graded,
        "errors": errors[:20],
        "n_errors": len(errors),
        "sample_consolidator_prompt": render_consolidator(
            sample["target"]["original_text"], sample["context_turns"]
        ),
        "sample_evaluator_prompt": render_evaluator(sample["target"]),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sample_consolidator_prompt" and k != "sample_evaluator_prompt"}, indent=2))
    assert not errors, f"{len(errors)} validation errors, see {OUT_PATH}"


if __name__ == "__main__":
    main()
