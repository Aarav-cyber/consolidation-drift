"""Score the judge against the human labels in label_sheet.txt.
Label: 1-5, or NP for not present.   Run: python3 validation/score.py
"""
import json, re
from collections import Counter, defaultdict

key = json.load(open("validation/answer_key.json"))
text = open("validation/label_sheet.txt", encoding="utf-8").read()
labels = {}
for m in re.finditer(r"^### (S\d+)\n.*?^LABEL:[ \t]*(\S*)", text, flags=re.S | re.M):
    if m.group(2): labels[m.group(1)] = m.group(2)
norm = lambda s: "NP" if s.strip().upper() in ("NP", "NOT_PRESENT") else int(s)
pairs = []
for sid, lab in labels.items():
    k = key[sid]; j = "NP" if k["judge"] == "NOT_PRESENT" else k["judge"]
    pairs.append((sid, norm(lab), j, k["true_level"]))
n = len(pairs)
print(f"labelled: {n}/{len(key)}")
if not n: raise SystemExit
agree = sum(h == j for _, h, j, _ in pairs)
print(f"exact agreement with judge: {agree}/{n} = {agree/n:.0%}")
both = [(h, j) for _, h, j, _ in pairs if h != "NP" and j != "NP"]
if both:
    print(f"both graded: {len(both)}  exact {sum(h==j for h,j in both)/len(both):.0%}  within 1 {sum(abs(h-j)<=1 for h,j in both)/len(both):.0%}  mean(judge-human) {sum(j-h for h,j in both)/len(both):+.2f}")
print("\nBy the judge's verdict (this is what matters for the drift result):")
grp = defaultdict(list)
for sid, h, j, t in pairs:
    cls = "judge: not present" if j == "NP" else ("judge: too strong" if j > t else "judge: too weak" if j < t else "judge: exact")
    grp[cls].append((h, j, t))
for cls, xs in sorted(grp.items()):
    ok = sum(h == j for h, j, _ in xs)
    extra = ""
    if cls in ("judge: too strong", "judge: too weak"):
        real = sum(h != "NP" and h != t for h, j, t in xs)
        extra = f" | human also says it differs from the true level: {real}/{len(xs)}"
    print(f"  {cls:20s} n={len(xs):3d}  human agrees with judge's level: {ok}/{len(xs)}{extra}")
ch, cj = Counter(h for _, h, _, _ in pairs), Counter(j for _, _, j, _ in pairs)
pe = sum(ch[c] * cj[c] for c in set(ch) | set(cj)) / n**2
print(f"\nCohen kappa: {(agree/n - pe)/(1-pe):.2f}" if pe < 1 else "kappa n/a")
