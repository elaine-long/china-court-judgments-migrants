import json

import pytest

from src import llm_extract as L


def test_parse_json_plain_and_fenced():
    assert L.parse_json('{"a": 1}') == {"a": 1}
    assert L.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert L.parse_json('好的：{"a": {"value": "1"}} 以上') == {"a": {"value": "1"}}
    with pytest.raises(ValueError):
        L.parse_json("not json")


def _full(**over):
    obj = {f: {"value": "NA", "evidence": ""} for f in L.FIELDS}
    obj.update({"ever_bail": {"value": "0", "evidence": ""}, "plea_formal": {"value": "0", "evidence": ""},
                "jiejie": {"value": "0", "evidence": ""}, "counsel_any": {"value": "1", "evidence": "辩护人"},
                "penalty_type": {"value": "拘役", "evidence": ""}, "probation": {"value": "0", "evidence": ""}})
    obj.update(over)
    return obj


def test_validate_ok():
    vals, evid, errs = L.validate(_full(term_months={"value": "5", "evidence": ""},
                                        status_at_judgment={"value": "在押", "evidence": "现羁押于"}))
    assert errs == [] and vals["term_months"] == "5.0" and vals["status_at_judgment"] == "在押"
    assert evid["counsel_any"] == "辩护人"


def test_validate_rejects_values_outside_the_table():
    vals, _, errs = L.validate(_full(procedure={"value": "独任审判", "evidence": ""},
                                     probation={"value": "是", "evidence": ""}))
    assert vals["procedure"] is None and vals["probation"] is None
    assert "invalid:procedure" in errs and "invalid:probation" in errs


def test_validate_normalises_types_and_missing():
    obj = _full(term_months={"value": 18, "evidence": ""}, ever_bail={"value": True, "evidence": ""})
    del obj["jiejie"]
    vals, _, errs = L.validate(obj)
    assert vals["term_months"] == "18.0" and vals["ever_bail"] == "1"
    assert "missing:jiejie" in errs and vals["jiejie"] is None
    vals, _, _ = L.validate(_full(origin_text={"value": None, "evidence": ""}))
    assert vals["origin_text"] == "NA"


def test_cache_hit_does_not_call_again(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LLM_DIR", tmp_path)
    calls = []

    def fake(system, user):
        calls.append(user)
        return json.dumps(_full()), {"in_hit": 10, "in_miss": 100, "out": 50, "finish": "stop"}

    led = L.Ledger(budget=1.0)
    r1 = L.call_cached(L.MODEL_A, "extract_v1", "输入文本", ledger=led, caller=fake)
    r2 = L.call_cached(L.MODEL_A, "extract_v1", "输入文本", ledger=led, caller=fake)
    assert len(calls) == 1 and not r1["cached"] and r2["cached"]
    assert r2["raw"] == r1["raw"] and led.calls == 1
    L.call_cached(L.MODEL_A, "extract_v1", "另一份输入", ledger=led, caller=fake)
    assert len(calls) == 2


def test_failed_call_not_cached_and_budget_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LLM_DIR", tmp_path)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)

    def boom(system, user):
        raise TimeoutError("timeout")

    rec = L.call_cached(L.MODEL_A, "extract_v1", "x", caller=boom, retries=2)
    assert rec["error"].startswith("TimeoutError") and not list(tmp_path.rglob("*.json"))
    led = L.Ledger(budget=0.0)
    with pytest.raises(RuntimeError):
        L.call_cached(L.MODEL_A, "extract_v1", "y", ledger=led, caller=boom)


def test_cost_and_peak():
    u = {"in_hit": 1_000_000, "in_miss": 1_000_000, "out": 1_000_000}
    assert L.cost_usd(L.MODEL_A, u, peak=False) == pytest.approx(0.003 + 0.15 + 0.60)
    assert L.cost_usd(L.MODEL_A, u, peak=True) == pytest.approx(0.006 + 0.30 + 1.20)
    import calendar
    assert L.is_peak(calendar.timegm((2026, 9, 30, 2, 0, 0))) is True     # Wed 02:00 UTC
    assert L.is_peak(calendar.timegm((2026, 9, 30, 5, 0, 0))) is False
    assert L.is_peak(calendar.timegm((2026, 10, 3, 2, 0, 0))) is False    # Saturday


def test_keys_not_in_source():
    import os
    from pathlib import Path
    from dotenv import dotenv_values
    env = dotenv_values(Path(L.paths.ROOT) / ".env")
    # only secrets: API keys and the Cloud project id (GEMINI_BACKEND / LOCATION values are plain words)
    secrets = [v for k, v in env.items() if v and ("KEY" in k or k == "GOOGLE_CLOUD_PROJECT") and len(v) >= 6]
    for p in list(Path(L.paths.ROOT, "src").glob("*.py")) + list(Path(L.paths.ROOT, "scripts").glob("*.py")):
        s = p.read_text(encoding="utf-8")
        assert not any(k in s for k in secrets), p
