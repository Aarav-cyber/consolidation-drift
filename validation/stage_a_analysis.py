"""Pre-registered Stage A analysis (docs/preregistration.md).
python3 validation/stage_a_analysis.py [data/experiments/stage_a]
"""
import glob, json, random, sys
from collections import Counter, defaultdict

D = (sys.argv[1] if len(sys.argv) > 1 else "data/experiments/stage_a").rstrip("/") + "/"
rec = [r for r in json.load(open(D + "evaluations.json")) if r["category"] != "polarity_negation"]
TARGET = {"obligation_strength", "preference_intensity"}
CONTROL = {"epistemic_certainty", "numeric_magnitude"}
B = 5000


def item_means(rs):
    d = defaultdict(list)
    for r in rs:
        if r.get("drift") is not None:
            d[r["item_id"]].append(r["drift"])
    return {k: sum(v) / len(v) for k, v in d.items()}


def boot(im, seed=11):
    rng = random.Random(seed); ks = list(im); out = []
    for _ in range(B):
        s = [im[rng.choice(ks)] for _ in ks]; out.append(sum(s) / len(s))
    out.sort(); m = sum(im.values()) / len(im)
    p = min(1.0, 2 * min(sum(o <= 0 for o in out), sum(o >= 0 for o in out)) / B)
    return m, out[int(.025 * B)], out[int(.975 * B)], p


def sel(arm, cats, level=None):
    return [r for r in rec if r["pipeline"] == arm and r["category"] in cats and (level is None or r["context_level"] == level)]


def fmt(t):
    m, lo, hi, p = t
    return f"{m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}"


print("== PRIMARY (pre-registered): hierarchical, obligation+preference, all filler, all rounds ==")
prim = boot(item_means(sel("hierarchical", TARGET)))
print("  ", fmt(prim), "| n items =", len(item_means(sel("hierarchical", TARGET))))
print("   confirmed (CI excludes 0):", prim[1] > 0 or prim[2] < 0)

print("\n== SECONDARY (Holm-corrected) ==")
sec = [(f"filler {l[1:]}", boot(item_means(sel("hierarchical", TARGET, l)))) for l in ("c0", "c8", "c20")]
sec.append(("controls (certainty+number)", boot(item_means(sel("hierarchical", CONTROL)))))
order = sorted(range(len(sec)), key=lambda i: sec[i][1][3])
adj = [None] * len(sec); running = 0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(sec) - rank) * sec[i][1][3])); adj[i] = running
for (name, t), a in zip(sec, adj):
    print(f"   {name:28s} {fmt(t)}  Holm p={a:.4f}")

print("\n== AVR arm, same test ==")
avr = boot(item_means(sel("hierarchical_avr", TARGET)))
print("   obligation+preference:", fmt(avr))
for l in ("c0", "c8", "c20"):
    print(f"   filler {l[1:]:3s}:", fmt(boot(item_means(sel("hierarchical_avr", TARGET, l)))))
print("   controls:", fmt(boot(item_means(sel("hierarchical_avr", CONTROL)))))

print("\n== Paired effect of AVR (baseline minus AVR, per item, obligation+preference) ==")
b, a = item_means(sel("hierarchical", TARGET)), item_means(sel("hierarchical_avr", TARGET))
print("  ", fmt(boot({k: b[k] - a[k] for k in b if k in a})))
babs = defaultdict(list); aabs = defaultdict(list)
for r in sel("hierarchical", TARGET | CONTROL):
    if r.get("drift") is not None: babs[r["item_id"]].append(abs(r["drift"]))
for r in sel("hierarchical_avr", TARGET | CONTROL):
    if r.get("drift") is not None: aabs[r["item_id"]].append(abs(r["drift"]))
dabs = {k: sum(babs[k]) / len(babs[k]) - sum(aabs[k]) / len(aabs[k]) for k in babs if k in aabs}
print("   mean absolute error, all 48 items, baseline minus AVR:", fmt(boot(dabs)))

print("\n== Attrition: 'not present' share ==")
for arm in ("hierarchical", "hierarchical_avr"):
    for l in ("c0", "c8", "c20"):
        rs = [r for r in rec if r["pipeline"] == arm and r["context_level"] == l]
        npn = sum(r.get("level") is None for r in rs)
        print(f"   {arm:17s} filler {l[1:]:3s}: {npn}/{len(rs)} = {npn/len(rs):.1%}")

print("\n== Exact-match rate (graded level equals the true level) ==")
for arm in ("hierarchical", "hierarchical_avr"):
    rs = [r for r in rec if r["pipeline"] == arm]
    ok = sum(r.get("drift") == 0 for r in rs)
    print(f"   {arm:17s} {ok}/{len(rs)} = {ok/len(rs):.1%}")

print("\n== AVR mechanics ==")
items = {x["id"]: x for x in map(json.loads, open("data/atomic/atomic_v1.jsonl"))}
anch, filler_anch, recall, checks, restores, base_calls = 0, 0, 0, 0, 0, 0
for f in glob.glob(D + "hierarchical_avr/*.json"):
    c = json.load(open(f)); a0 = c["rounds"][0]["anchors"]; tgt = items[c["item_id"]]["original_text"]
    anch += len(a0); filler_anch += sum(x != tgt for x in a0); recall += tgt in a0
    for r in c["rounds"][1:]:
        checks += len(r["avr"]["verdicts"]); restores += len(r["avr"]["restored"])
n = len(glob.glob(D + "hierarchical_avr/*.json"))
print(f"   target statement anchored in {recall}/{n} chains; filler statements anchored: {filler_anch} ({filler_anch/n:.1f} per chain)")
print(f"   anchor checks: {checks}, restorations: {restores} ({restores/max(checks,1):.1%} of checks)")
