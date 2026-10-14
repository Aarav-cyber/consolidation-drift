"""Offline tests for Phase 2 (fake LLM client, no API calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2 import pipelines as P
from src.phase2 import metrics as M
from src.phase2 import evaluate as E


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeCompletions:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def create(self, model, messages):
        self.calls.append(messages[0]["content"])
        text = self.texts.pop(0) if self.texts else "ok summary."
        return _Resp(text)


class FakeChat:
    def __init__(self, texts):
        self.completions = FakeCompletions(texts)


class FakeClient:
    def __init__(self, texts):
        self.chat = FakeChat(texts)


def test_rolling_step_returns_text():
    c = FakeClient(["Some summary."])
    out = P.rolling_step(c, "I slightly prefer mornings.", "m", delay=0)
    assert out == "Some summary."


def test_sliding_pops_window_and_summarizes_tail():
    c = FakeClient(["Tail merged."])
    state = P.sliding_init("TARGET", [{"role": "user", "content": f"f{i}"} for i in range(6)])
    assert len(state["window"]) == 7
    new_state, rendered = P.sliding_step(c, state, "m", delay=0, window_size=4)
    assert len(new_state["window"]) == 4
    assert new_state["summary"] == "Tail merged."
    assert "Tail merged." in rendered


def test_sliding_no_summary_when_window_fits():
    c = FakeClient([])
    state = {"summary": "", "window": ["a", "b"]}
    new_state, rendered = P.sliding_step(c, state, "m", delay=0, window_size=4)
    assert new_state["summary"] == ""
    assert len(c.chat.completions.calls) == 0
    assert "a" in rendered and "b" in rendered


def test_hierarchical_fuses_chunks():
    # 5 statements -> note groups [4,1] -> 2 notes, 1 chunk, 1 global
    c = FakeClient(["g1.", "g2.", "c1.", "GLOBAL MEMORY."])
    glob, nxt = P.hierarchical_step(c, ["t1", "t2", "t3", "t4", "t5"], "m", delay=0)
    assert glob == "GLOBAL MEMORY."
    assert nxt == ["GLOBAL MEMORY."]


def test_grade_parses_int():
    c = FakeClient(["The answer is 4."])
    target = {"attribute": "x", "scale": {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e"}}
    assert E.grade(c, target, "mem", "m", delay=0) == 4


def test_metrics_signed_and_exponent():
    recs = []
    for k in range(1, 6):
        for i in range(4):
            recs.append({"item_id": f"a{i}", "category": "c", "true_level": 2,
                         "pipeline": "sliding", "context_level": "c8",
                         "round": k, "level": 2 + k})  # drift grows 1..5
    out = M.compute_metrics(recs)
    s = out["series"]["sliding|c8|c"]
    assert s["rounds"][5]["mean_signed"] == 5.0
    assert s["rounds"][5]["mean_abs"] == 5.0
    assert s["growth_exponent"] > 0.9  # linear growth -> b ~= 1
    svt = out["survival_vs_truth"]["sliding|c8|5"]
    assert svt["present_but_wrong"] == 1.0


def test_grade_not_present():
    c = FakeClient(["NOT_PRESENT"])
    target = {"attribute": "x", "scale": {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e"}}
    assert E.grade(c, target, "mem with no relevant attribute", "m", delay=0) is None


def test_compute_arc_attractor():
    # Perfect attractor towards center (3): all levels collapse to 3
    # true_level 1 (x=-2) -> level 3 (drift y=+2)
    # true_level 2 (x=-1) -> level 3 (drift y=+1)
    # true_level 3 (x=0)  -> level 3 (drift y=0)
    # true_level 4 (x=+1) -> level 3 (drift y=-1)
    # true_level 5 (x=+2) -> level 3 (drift y=-2)
    # Slope of y on x must be exactly -1.0
    rows = [
        {"true_level": lvl, "level": 3}
        for lvl in [1, 2, 3, 4, 5]
    ]
    arc = M.compute_arc(rows)
    assert arc["slope"] == -1.0
    assert arc["r2"] == 1.0
    assert arc["n"] == 5


def test_compute_arc_zero_drift():
    # No drift: level == true_level for all
    rows = [
        {"true_level": lvl, "level": lvl}
        for lvl in [1, 2, 3, 4, 5]
    ]
    arc = M.compute_arc(rows)
    assert arc["slope"] == 0.0
    assert arc["n"] == 5


def test_metrics_handles_attrition():
    recs = [
        {"item_id": "i1", "category": "c", "true_level": 2, "pipeline": "hierarchical",
         "context_level": "c20", "round": 1, "level": None, "present": False},
        {"item_id": "i2", "category": "c", "true_level": 4, "pipeline": "hierarchical",
         "context_level": "c20", "round": 1, "level": 3, "present": True},
    ]
    out = M.compute_metrics(recs)
    att = out["attrition"]["hierarchical|c20|1"]
    assert att["total"] == 2
    assert att["lost"] == 1
    assert att["attrition_rate"] == 0.5
    # The graded series must only calculate drift on present records (i2: drift 3 - 4 = -1)
    s = out["series"]["hierarchical|c20|c"]
    assert s["rounds"][1]["n"] == 1
    assert s["rounds"][1]["mean_signed"] == -1.0


def test_sliding_progressive_turn_feeding_calls_model_across_rounds():
    """Verify that feeding turns incrementally across rounds calls the model repeatedly."""
    c = FakeClient(["Summary 1", "Summary 2", "Summary 3", "Summary 4", "Summary 5"])
    state = P.sliding_init("Target fact")
    # Feed 2 turns per round for 3 rounds with window size 2
    for r in range(1, 4):
        new_turns = [{"content": f"filler turn {r}.1"}, {"content": f"filler turn {r}.2"}]
        state, rendered = P.sliding_step(c, state, "model", delay=0, window_size=2, new_turns=new_turns)
        assert len(state["window"]) == 2
        assert state["summary"] != ""
    assert len(c.chat.completions.calls) == 3


def test_polarity_parser_strict_matching():
    target = {"original_text": "I like cats.", "true_value": True}
    
    # 1. Standard PRESERVED
    c1 = FakeClient(["PRESERVED"])
    assert E.polarity(c1, target, "User likes cats", "m", delay=0) == "PRESERVED"
    
    # 2. Standard FLIPPED
    c2 = FakeClient(["FLIPPED"])
    assert E.polarity(c2, target, "User hates cats", "m", delay=0) == "FLIPPED"
    
    # 3. Explicit "NOT PRESERVED" must NOT return PRESERVED
    c3 = FakeClient(["The fact was NOT PRESERVED."])
    assert E.polarity(c3, target, "Irrelevant text", "m", delay=0) == "LOST"


def test_polarity_parser_rejects_ambiguous():
    import pytest
    target = {"original_text": "I like cats.", "true_value": True}
    # Ambiguous output with both PRESERVED and FLIPPED
    c = FakeClient(["It was PRESERVED and also FLIPPED.", "Still ambiguous PRESERVED or LOST."])
    with pytest.raises(RuntimeError):
        E.polarity(c, target, "mem", "m", delay=0)


def test_compute_arc_clustered_bootstrap():
    # 3 items observed across 4 rounds (12 observations)
    rows = []
    for item_idx in [1, 2, 3]:
        true_lvl = item_idx + 1 # 2, 3, 4
        for r in range(1, 5):
            # item 1 (true=2) drifts +1 -> level 3
            # item 2 (true=3) drifts 0  -> level 3
            # item 3 (true=4) drifts -1 -> level 3
            rows.append({
                "item_id": f"item_{item_idx}",
                "true_level": true_lvl,
                "round": r,
                "level": 3
            })
    arc = M.compute_arc(rows, cluster_bootstrap=True, n_boot=100)
    assert arc["slope"] == -1.0
    assert arc["n"] == 12
    assert arc["n_items"] == 3
    assert arc["ci_95"] is not None
    # All bootstrap samples will yield -1.0 because all items collapsed to 3
    assert arc["ci_95"][0] == -1.0
    assert arc["ci_95"][1] == -1.0


def test_metrics_polarity_summary():
    recs = [
        {"item_id": "p1", "category": "polarity_negation", "pipeline": "sliding",
         "context_level": "c8", "round": 1, "verdict": "PRESERVED"},
        {"item_id": "p2", "category": "polarity_negation", "pipeline": "sliding",
         "context_level": "c8", "round": 1, "verdict": "FLIPPED"},
        {"item_id": "p3", "category": "polarity_negation", "pipeline": "sliding",
         "context_level": "c8", "round": 1, "verdict": "LOST"},
    ]
    out = M.compute_metrics(recs)
    pol = out["polarity_summary"]["sliding|c8|1"]
    assert pol["total"] == 3
    assert pol["preserved"] == 1
    assert pol["flipped"] == 1
    assert pol["lost"] == 1
    assert pol["flip_rate"] == round(1/3, 4)
    assert pol["preservation_rate"] == round(1/3, 4)
    assert pol["loss_rate"] == round(1/3, 4)
