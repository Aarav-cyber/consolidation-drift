from dotenv import load_dotenv
import os
import time
import json
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

input_path = "data/atomic/atomic_v0.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(items)} items")

prompt_template = """
You are a memory consolidation system.

Rewrite the following memory into a concise form.

Preserve:
- the original meaning
- qualifiers
- uncertainty
- negation
- numerical values
- strength or intensity of statements

Do not add information.
Do not infer information that is not explicitly stated.

Memory:
{memory}
"""

for item in items:
    memory = item["original_text"]
    item_id = item["id"]

    print(f"\nProcessing: {item_id}")
    print(f"Original: {memory}")

    results = [
        {
            "round": 0,
            "memory": memory
        }
    ]

    for round_num in range(1, 11):
        prompt = prompt_template.format(memory=memory)

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
            print(f"API error on {item_id}, round {round_num}: {e}")
            break

        memory = response.choices[0].message.content.strip()

        results.append({
            "round": round_num,
            "memory": memory
        })

        print(f"\nRound {round_num}:")
        print(memory)

        time.sleep(2)

    output_path = f"data/experiments/rolling_summary/{item_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")