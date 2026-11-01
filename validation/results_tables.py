"""Print markdown result tables for one experiment dir.  python3 validation/results_tables.py <dir>"""
import json, sys
from collections import defaultdict
d = sys.argv[1].rstrip("/") + "/"
rec = [r for r in json.load(open(d + "evaluations.json")) if r["category"] != "polarity_negation"]
m = json.load(open(d + "drift_metrics.json"))
cells = sorted({(r["pipeline"], r["context_level"]) for r in rec})
print("| pipeline | filler turns | records | not present | mean signed drift (all rounds) | mean signed drift (round 10) | present but wrong (round 10) | ARC slope [95% CI] | ARC p (clustered) |")
print("|---|---|---|---|---|---|---|---|---|")
for p, c in cells:
    rs = [r for r in rec if r["pipeline"] == p and r["context_level"] == c]
    pres = [r for r in rs if r["level"] is not None]
    r10 = [r for r in pres if r["round"] == 10]
    sd = lambda xs: f"{sum(x['drift'] for x in xs)/len(xs):+.2f} (n={len(xs)})" if xs else "n/a"
    wrong = [r for r in r10 if r["drift"] != 0]
    w = f"{len(wrong)}/{len(r10)}" if r10 else "n/a"
    a = m["arc_pooled"].get(f"{p}|{c}", {})
    ci = a.get("ci_95")
    arc = f"{a['slope']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if a.get("slope") is not None and ci else (f"{a['slope']:+.2f}" if a.get("slope") is not None else "n/a")
    pc = a.get("p_value_clustered")
    print(f"| {p} | {c[1:]} | {len(rs)} | {len(rs)-len(pres)} | {sd(pres)} | {sd(r10)} | {w} | {arc} | {'n/a' if pc is None else f'{pc:.2f}'} |")
