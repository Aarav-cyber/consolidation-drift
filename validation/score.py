"""Score the judge against the human labels in label_sheet.csv.
human_label: 1-5, or NP for not present.   Run: python3 validation/score.py
"""
import csv, json
from collections import Counter

key = json.load(open("validation/answer_key.json"))
rows = list(csv.DictReader(open("validation/label_sheet.csv", encoding="utf-8")))
norm = lambda s: "NP" if s.strip().upper() in ("NP", "NOT_PRESENT") else int(s)
pairs = [(norm(r["human_label"]), "NP" if key[r["sample_id"]]["judge"] == "NOT_PRESENT" else key[r["sample_id"]]["judge"])
         for r in rows if r["human_label"].strip()]
n = len(pairs)
print(f"labelled: {n}/{len(rows)}")
if not n: raise SystemExit
agree = sum(h == j for h, j in pairs)
print(f"exact agreement: {agree}/{n} = {agree/n:.0%}")
both = [(h, j) for h, j in pairs if h != "NP" and j != "NP"]
if both:
    print(f"both graded: {len(both)}  exact {sum(h==j for h,j in both)/len(both):.0%}  within 1 {sum(abs(h-j)<=1 for h,j in both)/len(both):.0%}  mean(judge-human) {sum(j-h for h,j in both)/len(both):+.2f}")
jn = [(h, j) for h, j in pairs if j == "NP"]; hn = [(h, j) for h, j in pairs if h == "NP"]
if jn: print(f"judge said NOT_PRESENT: {len(jn)}, human agrees on {sum(h=='NP' for h,_ in jn)} ({sum(h=='NP' for h,_ in jn)/len(jn):.0%})")
if hn: print(f"human said NOT_PRESENT: {len(hn)}, judge also said so on {sum(j=='NP' for _,j in hn)} ({sum(j=='NP' for _,j in hn)/len(hn):.0%})")
ch, cj = Counter(h for h, _ in pairs), Counter(j for _, j in pairs)
pe = sum(ch[c] * cj[c] for c in set(ch) | set(cj)) / n**2
print(f"Cohen kappa: {(agree/n - pe)/(1-pe):.2f}" if pe < 1 else "kappa n/a")
print("confusion (human -> judge):")
for (h, j), c in sorted(Counter(pairs).items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
    if h != j: print(f"  human {h} judge {j}: {c}")
