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
