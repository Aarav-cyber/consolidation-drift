"""Phase 2 drift metrics (pure Python with optional scipy for p-values).

Inputs: list of eval records, each:
  {item_id, category, true_level, pipeline, context_level, round, level, present (optional), verdict (optional)}
  - Records where level is None (ATTRIBUTE_NOT_PRESENT) feed attrition tracking.
  - Negation rows carry verdict instead of level and feed survival-vs-truth / polarity metrics.

Outputs dict with:
- mean signed drift + mean absolute drift per (pipeline, level, category, round)
- survival-vs-truth: fraction of present-but-wrong graded rows per round
- growth exponent b from error(k) ~ a*k^b fit per (pipeline, level, category)
- attractor regression coefficient (ARC): OLS slope of drift y on distance from midpoint x (true_level - 3)
- attrition: loss/omission rates where target attribute was not retained
"""
import math
from collections import defaultdict

try:
    import scipy.stats as stats
except ImportError:
    stats = None


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _loglog_exponent(points):
    """points: [(k, err)]. Returns b in err ~ a*k^b. 0.0 if unfittable."""
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


def compute_arc(rows):
    """Compute Attractor Regression Coefficient (ARC) for a set of graded records.
    
    Fits OLS: y = a + b * x
    where x_i = true_level_i - 3 (centering distance from midpoint 3)
          y_i = level_i - true_level_i (signed drift)
    
    Returns dict with n, slope (ARC), intercept, r2, se, t_stat, p_value.
    """
    valid = [r for r in rows if r.get("level") is not None and r.get("true_level") is not None]
    n = len(valid)
    if n < 2:
        return {
            "n": n, "slope": 0.0, "intercept": 0.0,
            "r2": 0.0, "se": 0.0, "t_stat": 0.0, "p_value": 1.0,
        }
    xs = [r["true_level"] - 3 for r in valid]
    ys = [r["level"] - r["true_level"] for r in valid]
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    ss_xx = sum((x - x_bar) ** 2 for x in xs)
    ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    ss_yy = sum((y - y_bar) ** 2 for x, y in zip(xs, ys))
    if ss_xx == 0:
        return {
            "n": n, "slope": 0.0, "intercept": y_bar,
            "r2": 0.0, "se": 0.0, "t_stat": 0.0, "p_value": 1.0,
        }
    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = max(0.0, 1.0 - (ss_res / ss_yy)) if ss_yy > 0 else 0.0
    se = math.sqrt((ss_res / (n - 2)) / ss_xx) if n > 2 and ss_res >= 0 else 0.0
    t_stat = slope / se if se > 0 else 0.0
    p_val = float(stats.t.sf(abs(t_stat), df=n - 2) * 2) if stats and n > 2 and se > 0 else None

    return {
        "n": n,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2": round(r2, 4),
        "se": round(se, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 6) if p_val is not None else None,
    }


def compute_metrics(records):
    # Separate numeric/level evaluations from negation/polarity evaluations
    total_evals_by_plr = defaultdict(int)
    not_present_by_plr = defaultdict(int)
    for r in records:
        plr_key = (r["pipeline"], r["context_level"], r["round"])
        total_evals_by_plr[plr_key] += 1
        if r.get("present") is False or r.get("level") is None and "verdict" not in r:
            not_present_by_plr[plr_key] += 1
        elif r.get("verdict") == "LOST":
            not_present_by_plr[plr_key] += 1

    attrition = {}
    for key, total in sorted(total_evals_by_plr.items()):
        lost = not_present_by_plr[key]
        attrition["|".join(map(str, key))] = {
            "pipeline": key[0], "context_level": key[1], "round": key[2],
            "total": total, "lost": lost,
            "attrition_rate": lost / total if total > 0 else 0.0,
        }

    # Graded records that have an intact valid level
    graded = [r for r in records if r.get("level") is not None and r.get("true_level") is not None]

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
            "arc": compute_arc(rows),
        }

    # ARC pooled per (pipeline, context_level)
    by_pl = defaultdict(list)
    for r in graded:
        by_pl[(r["pipeline"], r["context_level"])].append(r)
    arc_pooled = {}
    for key, rows in sorted(by_pl.items()):
        arc_pooled["|".join(key)] = compute_arc(rows)

    # ARC trajectory per round k: (pipeline, context_level, round)
    by_plr_graded = defaultdict(list)
    for r in graded:
        by_plr_graded[(r["pipeline"], r["context_level"], r["round"])].append(r)
    arc_by_round = {}
    for key, rows in sorted(by_plr_graded.items()):
        arc_by_round["|".join(map(str, key))] = compute_arc(rows)

    # survival-vs-truth per (pipeline, level, round): present-but-wrong share
    svt = {}
    by_plr = defaultdict(list)
    for r in graded:
        by_plr[(r["pipeline"], r["context_level"], r["round"])].append(r)
    for key, rows in sorted(by_plr.items()):
        wrong = sum(1 for r in rows if r["level"] != r["true_level"])
        svt["|".join(map(str, key))] = {
            "pipeline": key[0], "context_level": key[1], "round": key[2],
            "present_but_wrong": wrong / len(rows) if rows else 0.0, "n": len(rows),
        }

    return {
        "series": series,
        "arc_pooled": arc_pooled,
        "arc_by_round": arc_by_round,
        "survival_vs_truth": svt,
        "attrition": attrition,
    }
