import json
from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

item_ids = [
    "pref_001",
    "pref_002",
    "pref_003",
    "pref_004",
    "cert_001",
    "cert_002",
    "cert_003",
    "cert_004",
    "obl_001",
    "obl_002",
    "obl_003",
    "obl_004",
    "num_001",
    "num_002",
    "num_003",
    "num_004"
]

with open("data/atomic/atomic_v0.jsonl", "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

true_levels = {
    item["id"]: item.get("true_level")
    for item in items
}

all_results = {}

for item_id in item_ids:
    result_path = f"data/experiments/rolling_summary/{item_id}.json"

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    original = results[0]["memory"]
    true_level = true_levels[item_id]

    all_results[item_id] = []

    print(f"\n{'=' * 60}")
    print(f"Item: {item_id}")
    print(f"True level: {true_level}")
    print(f"Original: {original}")

    for result in results:
        round_num = result["round"]
        memory = result["memory"]

        if round_num == 0:
            level = true_level
            drift = 0

            all_results[item_id].append({
                "round": round_num,
                "level": level,
                "true_level": true_level,
                "drift": drift
            })

            print(
                f"Round {round_num}: "
                f"level={level}, "
                f"drift={drift:+d}"
            )

            continue

        prompt = f"""
You are evaluating the strength of a user's statement.

Original memory:
{original}

Consolidated memory:
{memory}

The original memory belongs to a 1–5 scale.

For preference intensity:
1 = no preference
2 = slight preference
3 = moderate preference
4 = strong preference
5 = extreme/absolute preference

For epistemic certainty:
1 = very uncertain
2 = somewhat uncertain
3 = moderate confidence
4 = fairly confident
5 = certain

For obligation strength:
1 = optional
2 = weak obligation
3 = moderate obligation
4 = strong obligation
5 = mandatory/required

For numeric magnitude:
Determine which level best preserves the original numeric magnitude.

Return ONLY the integer 1, 2, 3, 4, or 5.
"""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        except Exception as e:
            print(
                f"API error on {item_id}, "
                f"round {round_num}: {e}"
            )
            break

        level = int(response.choices[0].message.content.strip())

        drift = level - true_level

        all_results[item_id].append({
            "round": round_num,
            "level": level,
            "true_level": true_level,
            "drift": drift
        })

        print(
            f"Round {round_num}: "
            f"level={level}, "
            f"drift={drift:+d}"
        )

        time.sleep(2)

output_path = "data/experiments/rolling_summary/evaluations.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"\nSaved evaluations to: {output_path}")