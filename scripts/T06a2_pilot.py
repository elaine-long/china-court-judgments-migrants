"""T06a2 §3-4: re-run the pilot with prompt v2 (rules v2 as the comparison).

Usage:
    python scripts/T06a2_pilot.py b_p3        # model B (Vertex) independent extraction on P3, prompt v2
    python scripts/T06a2_pilot.py a_p1        # model A on P1, prompt v2 -- refuses to run in DeepSeek peak hours
    python scripts/T06a2_pilot.py adjudicate  # model B adjudicates rule-v2 vs A-v2 disagreements
    python scripts/T06a2_pilot.py report

Only the v2 semantic fields are compared/adjudicated; word-match fields are rules only (T06a2 decision 1).
Budget: Gemini <= US$1 in this task (ledger at list price).
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import llm_extract as L  # noqa: E402
from src import paths  # noqa: E402
from src.io import load_texts  # noqa: E402
from T06a_pilot import _run_jobs, agree, geo, kappa, parse_records, rule_values, snippet  # noqa: E402

OUT = paths.DATA / "llm" / "pilot"
T = paths.TABLES
PROMPT = "extract_v2"
V2_COMPARE = ["nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail", "penalty_type",
              "term_months", "probation"]
CALLS = OUT / "calls_v2.csv"


def _sample():
    s = pd.read_csv(OUT / "pilot_sample.csv")
    return s[s.set == "P1"].doc_id.tolist(), s[s.in_P3].doc_id.tolist()


def _inputs(ids):
    txt = load_texts(ids, columns=["judgment"]).set_index("doc_id").judgment
    return {d: L.build_input(txt[d], "core")[0] for d in ids}


def _append(rows):
    df = pd.DataFrame(rows)
    if CALLS.exists():
        df = pd.concat([pd.read_csv(CALLS), df])
    df.drop_duplicates(["tag", "doc_id"], keep="last").to_csv(CALLS, index=False)


def b_p3():
    _, p3 = _sample()
    inp = _inputs(p3)
    led = L.Ledger(budget=1.0)
    _append(_run_jobs([(L.MODEL_B, PROMPT, d, inp[d], "P3_B_v2") for d in p3], led, "P3 model B v2"))


def a_p1():
    if L.is_peak():
        sys.exit("DeepSeek peak hours (UTC Mon-Fri 01-04, 06-10): not running model A now.")
    p1, _ = _sample()
    inp = _inputs(p1)
    led = L.Ledger(budget=1.0)
    _append(_run_jobs([(L.MODEL_A, PROMPT, d, inp[d], "P1_A_v2") for d in p1], led, "P1 model A v2"))


def _model_vals(M, d, r):
    m = M.loc[d]
    out = {f: m.get(f"v_{f}") for f in V2_COMPARE}
    out["nonlocal_origin"] = geo(m.get("v_origin_text"), r.court_province, r.court_prefecture, "origin")
    out["residence_local"] = geo(m.get("v_residence_text"), r.court_province, r.court_prefecture, "res")
    return out


def _recs(tag):
    c = pd.read_csv(CALLS)
    recs = c[c.tag == tag].to_dict("records")
    for r in recs:
        r["error"] = None if pd.isna(r["error"]) else r["error"]
    return recs


def adjudicate():
    p1, _ = _sample()
    A = parse_records(_recs("P1_A_v2"), L.FIELDS_V2).set_index("doc_id")
    R = rule_values(p1)
    inp = _inputs(p1)
    jobs, meta = [], []
    for d in p1:
        if d not in A.index or A.at[d, "status"] == "call_error":
            continue
        r, av = R.loc[d], _model_vals(A, d, R.loc[d])
        items = []
        for f in V2_COMPARE:
            if av[f] is None or pd.isna(av[f]) or agree(f, r[f], av[f]):
                continue
            af, rv, aval = {"nonlocal_origin": ("origin_text", r.origin_text, A.at[d, "v_origin_text"]),
                            "residence_local": ("residence_text", r.residence_text, A.at[d, "v_residence_text"])
                            }.get(f, (f, r[f], av[f]))
            items.append((af, rv, aval, f))
        if not items:
            continue
        rnd = random.Random(f"{paths.SEED}-{d}-v2")
        lines = []
        for af, rv, aval, f in items:
            order = ["rule", "A"]
            rnd.shuffle(order)
            ans = {"rule": rv, "A": aval}
            lines.append(f"- {af}：答案1 = {ans[order[0]]}；答案2 = {ans[order[1]]}")
            meta.append({"doc_id": d, "field": f, "adj_field": af, "rule": rv, "A": aval,
                         "ans1": order[0], "ans2": order[1]})
        jobs.append((L.MODEL_B, "adjudicate_v1", d, inp[d] + "\n\n【待复核字段】\n" + "\n".join(lines), "P1_adj_v2"))
    pd.DataFrame(meta).to_csv(OUT / "adjudication_items_v2.csv", index=False)
    led = L.Ledger(budget=1.0)
    _append(_run_jobs(jobs, led, "P1 adjudication v2 (model B)"))


# ----------------------------------------------------------------------------- error types (v1 vs v2)
_STATUS_CUES = re.compile(r"羁押|押于|在押|拘留|逮捕|取保|监视居住|现在家")
_ORIGIN_CUES = re.compile(r"户籍|户口|出生|生于|籍贯")
_RES_CUES = re.compile(r"住")


def _header_of(inp):
    m = re.search(r"【当事人段】\n(.*?)(?:\n\n【|$)", inp, re.S)
    return m.group(1) if m else inp


def status_boilerplate_error(val, inp):
    """'在押' although the party block has no status sentence and no measure event: can only come from
    the sentence-credit boilerplate in the verdict."""
    return val == "在押" and not _STATUS_CUES.search(_header_of(inp))


def residence_as_origin_error(val, inp):
    """origin_text taken from a residence phrase: the text occurs in the party block only after a 住 cue
    and never after a hukou / birth cue, and is not a native-place '…人'."""
    if not isinstance(val, str) or val in ("NA", "") or pd.isna(val):
        return False
    h = _header_of(inp)
    pos = [m.start() for m in re.finditer(re.escape(val), h)]
    if not pos:
        return False
    for p in pos:
        ctx = h[max(0, p - 8):p]
        if _ORIGIN_CUES.search(ctx) or h[p + len(val):p + len(val) + 1] == "人":
            return False
    return any(_RES_CUES.search(h[max(0, p - 6):p]) for p in pos)


def report():
    p1, p3 = _sample()
    R = rule_values(p1)
    inp = _inputs(p1)
    old_calls = pd.read_csv(OUT / "calls.csv").to_dict("records")
    for r in old_calls:
        r["error"] = None if pd.isna(r["error"]) else r["error"]
    A1 = parse_records([r for r in old_calls if r["tag"] == "P1_A"], L.FIELDS).set_index("doc_id")
    A2 = parse_records(_recs("P1_A_v2"), L.FIELDS_V2).set_index("doc_id")
    ok = [d for d in p1 if d in A2.index and A2.at[d, "status"] != "call_error" and d in A1.index]
    V1 = pd.DataFrame({d: _model_vals(A1, d, R.loc[d]) for d in ok}).T
    V2 = pd.DataFrame({d: _model_vals(A2, d, R.loc[d]) for d in ok}).T
    rows = []
    for f in V2_COMPARE:
        r = R.loc[ok, f].astype(str)
        row = {"field": f, "n": len(ok), "rule_v2_NA": (r == "NA").mean()}
        for lab, V in (("A_v1", V1), ("A_v2", V2)):
            m = V[f].astype(str)
            row[f"agree_rule_{lab}"] = np.mean([agree(f, x, y) for x, y in zip(r, m)])
            row[f"kappa_rule_{lab}"] = kappa(list(r), list(m)) if f != "term_months" else np.nan
            row[f"{lab}_NA"] = (m == "NA").mean()
        rows.append(row)
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(T / "T06a2_rule_vs_A_v1_v2.csv", index=False)
    print(cmp_.round(3).to_string(index=False))

    err = pd.DataFrame([{
        "error": "status 在押 inferred from sentence-credit boilerplate",
        "A_v1": int(sum(status_boilerplate_error(A1.at[d, "v_status_at_judgment"], inp[d]) for d in ok)),
        "A_v2": int(sum(status_boilerplate_error(A2.at[d, "v_status_at_judgment"], inp[d]) for d in ok)),
    }, {
        "error": "residence used as origin_text",
        "A_v1": int(sum(residence_as_origin_error(A1.at[d, "v_origin_text"], inp[d]) for d in ok)),
        "A_v2": int(sum(residence_as_origin_error(A2.at[d, "v_origin_text"], inp[d]) for d in ok)),
    }])
    err["n_docs"] = len(ok)
    err.to_csv(T / "T06a2_error_types.csv", index=False)
    print(err.to_string(index=False))

    # adjudication
    ip = OUT / "adjudication_items_v2.csv"
    if ip.exists() and CALLS.exists():
        from T06a_pilot import parse_adjudication
        adj = parse_adjudication(_recs("P1_adj_v2") and [dict(r, tag="P1_adj") for r in _recs("P1_adj_v2")],
                                 pd.read_csv(ip))
        adj.to_csv(T / "T06a2_adjudication.csv", index=False)
        s = adj.groupby("field").agg(n=("doc_id", "size"), side_rule=("side", lambda x: (x == "rule").mean()),
                                     side_A=("side", lambda x: (x == "A").mean()),
                                     third=("side", lambda x: (x == "third").mean()),
                                     failed=("side", lambda x: (x == "failed").mean()),
                                     low_conf=("confidence", lambda x: (x == "低").mean())).reset_index()
        s.loc[len(s)] = {"field": "ALL", "n": len(adj), "side_rule": (adj.side == "rule").mean(),
                         "side_A": (adj.side == "A").mean(), "third": (adj.side == "third").mean(),
                         "failed": (adj.side == "failed").mean(), "low_conf": (adj.confidence == "低").mean()}
        s.to_csv(T / "T06a2_adjudication_summary.csv", index=False)
        print(s.round(3).to_string(index=False))
        print(f"docs with >=1 disagreement: {adj.doc_id.nunique()}/{len(ok)}; field-level rate "
              f"{len(adj) / (len(ok) * len(V2_COMPARE)):.3f}")
        txt = load_texts(adj.doc_id.unique(), columns=["judgment"]).set_index("doc_id").judgment
        sp = adj.sample(min(40, len(adj)), random_state=paths.SEED).copy()
        sp["A_evidence"] = [A2.at[d, f"e_{f}"] if f"e_{f}" in A2 else "" for d, f in zip(sp.doc_id, sp.adj_field)]
        sp["snippet"] = [snippet(txt[d], f, a, r) for d, f, a, r in zip(sp.doc_id, sp.adj_field, sp.A, sp.rule)]
        sp[["doc_id", "field", "adj_field", "rule", "A", "B_value", "side", "confidence", "B_evidence",
            "A_evidence", "snippet"]].to_csv(T / "T06a2_disagreements_40.csv", index=False)

    # P3: A v2 vs B v2 (independent)
    B2 = parse_records(_recs("P3_B_v2"), L.FIELDS_V2).set_index("doc_id")
    p3ok = [d for d in p3 if d in B2.index and B2.at[d, "status"] != "call_error" and d in ok]
    if p3ok:
        BV = pd.DataFrame({d: _model_vals(B2, d, R.loc[d]) for d in p3ok}).T
        rows = [{"field": f, "n": len(p3ok),
                 "agree_A_B": np.mean([agree(f, x, y) for x, y in zip(V2.loc[p3ok, f].astype(str), BV[f].astype(str))]),
                 "agree_rule_B": np.mean([agree(f, x, y) for x, y in zip(R.loc[p3ok, f].astype(str), BV[f].astype(str))]),
                 "agree_rule_A": np.mean([agree(f, x, y) for x, y in zip(R.loc[p3ok, f].astype(str), V2.loc[p3ok, f].astype(str))])}
                for f in V2_COMPARE]
        p3t = pd.DataFrame(rows)
        p3t.to_csv(T / "T06a2_P3_A_vs_B.csv", index=False)
        print(p3t.round(3).to_string(index=False))
        err_b = pd.DataFrame([{
            "error": "status 在押 inferred from boilerplate (model B v2, P3)",
            "n": int(sum(status_boilerplate_error(B2.at[d, "v_status_at_judgment"], inp[d]) for d in p3ok))},
            {"error": "residence used as origin_text (model B v2, P3)",
             "n": int(sum(residence_as_origin_error(B2.at[d, "v_origin_text"], inp[d]) for d in p3ok))}])
        err_b["n_docs"] = len(p3ok)
        err_b.to_csv(T / "T06a2_error_types_B.csv", index=False)
        print(err_b.to_string(index=False))

    c = pd.read_csv(CALLS)
    cost = c.groupby(["tag", "model"]).agg(
        calls=("doc_id", "size"), cached=("cached", "sum"), errors=("error", lambda x: x.notna().sum()),
        in_hit=("u_in_hit", "sum"), in_miss=("u_in_miss", "sum"), out=("u_out", "sum"),
        cost_usd=("cost_usd", "sum"), peak_calls=("peak", "sum"), latency_mean=("latency_s", "mean")).reset_index()
    cost.to_csv(T / "T06a2_cost.csv", index=False)
    print(cost.round(4).to_string(index=False))
    fails = pd.concat([A2.status.value_counts().rename("P1_A_v2"), B2.status.value_counts().rename("P3_B_v2")],
                      axis=1).fillna(0).astype(int)
    fails.to_csv(T / "T06a2_failures.csv")
    print(fails.to_string())


if __name__ == "__main__":
    {"b_p3": b_p3, "a_p1": a_p1, "adjudicate": adjudicate, "report": report}[sys.argv[1]]()
