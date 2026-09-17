"""Phase 2 drift metrics (pure Python, no extra deps).

Inputs: list of eval records, each:
  {item_id, category, true_level, pipeline, context_level, round, level}
  (negation rows carry verdict instead of level and are excluded from
  signed-drift means; they feed survival-vs-truth only.)

Outputs dict with:
- mean signed drift + mean absolute drift per (pipeline, level, category, round)
- survival-vs-truth: fraction of present-but-wrong graded rows per round
- growth exponent b from error(k) ~ a*k^b fit per (pipeline, level),
  via log-log least squares over k >= 1
"""
import math
from collections import defaultdict


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _loglog_exponent(points):
    """points: [(k, err)]. Returns b in err ~ a*k^b. 0.0 if un fittable."""
    pts = [(math.log(k), math.log(e)) for k, e in points if k > 0 and e > 0]
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0
    return (n * sxy - sx * sy) / den


def compute_metrics(records):
    graded = [r for r in records if "level" in r and r.get("true_level") is not None]
    by_plc = defaultdict(list)  # (pipeline, level, category) -> records
    for r in graded:
        by_plc[(r["pipeline"], r["context_level"], r["category"])].append(r)

    series = {}
    for key, rows in sorted(by_plc.items()):
        by_round = defaultdict(list)
        for r in rows:
            by_round[r["round"]].append(r["level"] - r["true_level"])
        rounds = {}
        for k in sorted(by_round):
            ds = by_round[k]
            rounds[k] = {
                "mean_signed": _mean(ds),
                "mean_abs": _mean([abs(d) for d in ds]),
                "n": len(ds),
            }
        errs = [(k, v["mean_abs"]) for k, v in rounds.items() if k >= 1]
        series["|".join(key)] = {
            "pipeline": key[0], "context_level": key[1], "category": key[2],
            "rounds": rounds,
            "growth_exponent": _loglog_exponent(errs),
        }

    # survival-vs-truth per (pipeline, level, round): present-but-wrong share
    svt = {}
    by_plr = defaultdict(list)
    for r in graded:
        by_plr[(r["pipeline"], r["context_level"], r["round"])].append(r)
    for key, rows in sorted(by_plr.items()):
        wrong = sum(1 for r in rows if r["level"] != r["true_level"])
        svt["|".join(map(str, key))] = {
            "pipeline": key[0], "context_level": key[1], "round": key[2],
            "present_but_wrong": wrong / len(rows), "n": len(rows),
        }

    return {"series": series, "survival_vs_truth": svt}
