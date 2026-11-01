"""Item-level bootstrap of mean signed drift per (pipeline, filler) cell.
Rounds of one item are not independent, so each item is averaged first and items are resampled.
python3 validation/mean_drift_bootstrap.py <experiment_dir>
"""
import json, random, sys
from collections import defaultdict
rec = [x for x in json.load(open(sys.argv[1].rstrip("/") + "/evaluations.json")) if x.get("drift") is not None]
rng = random.Random(11)
def item_means(rs):
    d = defaultdict(list)
    for x in rs: d[x["item_id"]].append(x["drift"])
    return {k: sum(v) / len(v) for k, v in d.items()}
def boot(rs, B=5000):
    im = item_means(rs); ks = list(im); out = []
    for _ in range(B):
        s = [im[rng.choice(ks)] for _ in ks]; out.append(sum(s) / len(s))
    out.sort(); m = sum(im.values()) / len(im)
    p = min(1.0, 2 * min(sum(o <= 0 for o in out), sum(o >= 0 for o in out)) / B)
    return m, out[int(.025 * B)], out[int(.975 * B)], p, im
print("| pipeline | filler turns | mean signed drift (item level) | 95% CI | p | items drifting up / flat / down |")
print("|---|---|---|---|---|---|")
for p in sorted({x["pipeline"] for x in rec}):
    for c in ("c0", "c8", "c20"):
        rs = [x for x in rec if x["pipeline"] == p and x["context_level"] == c]
        if not rs: continue
        m, lo, hi, pv, im = boot(rs)
        up = sum(v > .05 for v in im.values()); dn = sum(v < -.05 for v in im.values())
        print(f"| {p} | {c[1:]} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {pv:.3f} | {up} / {len(im)-up-dn} / {dn} |")
m, lo, hi, pv, _ = boot(rec)
print(f"| all cells pooled | | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} | |")
