import json
from collections import defaultdict

input_path = "data\experiments\evaluations.json"
atomic_path = "data/atomic/atomic_v0.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(atomic_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

categories = {
    item["id"]: item["category"]
    for item in items
}

print("Mean signed drift by category")
print("=" * 60)

for category in sorted(set(categories.values())):
    print(f"\n{category}")

    for round_num in range(1, 11):
        drifts = []

        for item_id, results in data.items():
            if categories[item_id] != category:
                continue

            for result in results:
                if result["round"] == round_num:
                    drifts.append(result["drift"])
                    break

        if drifts:
            mean_drift = sum(drifts) / len(drifts)

            print(
                f"  Round {round_num:2d}: "
                f"{mean_drift:+.3f}"
            )