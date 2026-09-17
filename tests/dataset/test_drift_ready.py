import json
from pathlib import Path

REP = Path("data/embedded/validation_report.json")


def test_validation_report_exists():
    rep = json.loads(REP.read_text(encoding="utf-8"))
    assert rep["n_rows"] == 600 and rep["n_items"] >= 200
    assert set(rep["levels"]) == {"c0", "c8", "c20"}
    assert rep["n_errors"] == 0
    assert "sample_consolidator_prompt" in rep and "sample_evaluator_prompt" in rep
    assert "{target_text}" not in rep["sample_consolidator_prompt"]
