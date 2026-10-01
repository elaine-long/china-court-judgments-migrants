"""T06a pilot: rules vs model A (DeepSeek) vs model B (Gemini) on P1/P2/P3. Never runs the full data.

Usage:
    python scripts/T06a_pilot.py sample      # draw P1 (200), P2 (100), P3 (50 of P1)
    python scripts/T06a_pilot.py run         # API calls (cached, resumable, stops at US$1)
    python scripts/T06a_pilot.py report      # tables for the report
"""
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import llm_extract as L  # noqa: E402
from src import paths  # noqa: E402
from src.extract_rules import _migrant, normalize  # noqa: E402
from src.gazetteer import resolve_place  # noqa: E402
from src.io import load_texts  # noqa: E402
from src.segment import find_spans  # noqa: E402
from T04_sample import load_frame  # noqa: E402

OUT = paths.DATA / "llm" / "pilot"
T = paths.TABLES
ANN = paths.ROOT / "annotation"
N_P1_PER, N_P2, N_P3 = 25, 100, 50
THREADS = 6
COMPARE = ["nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail", "plea_formal", "jiejie",
           "procedure", "counsel_any", "penalty_type", "term_months", "probation"]
lock = threading.Lock()


# ----------------------------------------------------------------------------- sample
def sample():
    f = load_frame()
    excl = set(pd.read_csv(ANN / "sample_main.csv").doc_id) | set(pd.read_csv(ANN / "sample_pilot.csv").doc_id)
    f = f[~f.doc_id.isin(excl)]
    p1 = pd.concat([g.sample(N_P1_PER, random_state=paths.SEED) for _, g in f.groupby("stratum")])
    raw = f[["d1_hukou_raw", "d1_birthplace_raw", "d1_native_raw", "d1_residence_raw"]]
    p2 = f[raw.isna().all(axis=1) & ~f.doc_id.isin(p1.doc_id)].sample(N_P2, random_state=paths.SEED)
    p3 = p1.sample(N_P3, random_state=paths.SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.concat([p1.assign(set="P1"), p2.assign(set="P2")])[["doc_id", "set", "stratum"]]
    s["in_P3"] = s.doc_id.isin(p3.doc_id) & (s.set == "P1")
    s.to_csv(OUT / "pilot_sample.csv", index=False)
    print(s.groupby(["set", "stratum"]).size().to_string(), "\nP3:", int(s.in_P3.sum()))


# ----------------------------------------------------------------------------- run
def _job(model, prompt, doc_id, user, ledger, tag):
    rec = L.call_cached(model, prompt, user, ledger=ledger)
    with lock:
        return {"doc_id": doc_id, "tag": tag, "model": model, "prompt": prompt, "key": rec["key"],
                "cached": rec["cached"], "error": rec["error"], "latency_s": rec["latency_s"],
                "cost_usd": rec["cost_usd"], "peak": rec["peak"], "in_chars": len(user),
                **{f"u_{k}": v for k, v in (rec["usage"] or {}).items()}}


def _run_jobs(jobs, ledger, label):
    t0, rows = time.time(), []
    with ThreadPoolExecutor(THREADS) as ex:
        futs = [ex.submit(_job, model, prompt, doc_id, user, ledger, tag)
                for model, prompt, doc_id, user, tag in jobs]
        for k, fu in enumerate(futs):
            try:
                rows.append(fu.result())
            except RuntimeError as e:  # budget reached
                print("STOP:", e)
                ex.shutdown(cancel_futures=True)
                break
            if (k + 1) % 50 == 0:
                print(f"  {label}: {k + 1}/{len(jobs)}  spent ${ledger.spent:.4f}  {time.time() - t0:.0f}s", flush=True)
    print(f"{label}: {len(rows)} calls, spent so far ${ledger.spent:.4f}, {time.time() - t0:.0f}s", flush=True)
    return rows


def rule_values(ids):
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("ids", pd.DataFrame({"doc_id": ids}))
    r = con.execute(f"""SELECT f.*, t.court_province_final FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
                        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
                        WHERE doc_id IN (SELECT doc_id FROM ids)""").df().set_index("doc_id")
    yn = lambda s: s.map(lambda v: "NA" if pd.isna(v) else ("1" if bool(v) else "0"))  # noqa: E731
    pen = r.d1_penalty_type.map(lambda v: "NA" if pd.isna(v) else
                                (v if v in ("有期徒刑", "拘役", "管制", "单处罚金", "免予刑事处罚") else "其他"))
    out = pd.DataFrame({
        "nonlocal_origin": yn(r.d1_migrant_city_nores), "residence_local": yn(r.d1_local_residence),
        "status_at_judgment": r.d1_pretrial_status_at_judgment.map({"在押": "在押", "取保": "取保", "rsl": "监视居住"}).fillna("NA"),
        "ever_bail": yn(r.d1_bail), "plea_formal": yn(r.plea_formal), "jiejie": yn(r.jiejie),
        "procedure": r.procedure.fillna("NA"), "counsel_any": yn(r.counsel), "penalty_type": pen,
        "term_months": r.d1_term_months.map(lambda v: "NA" if pd.isna(v) else str(float(v))),
        "probation": yn(r.d1_probation),
        "origin_text": r[["d1_hukou_raw", "d1_birthplace_raw", "d1_native_raw"]].bfill(axis=1).iloc[:, 0].fillna("NA"),
        "residence_text": r.d1_residence_raw.fillna("NA"),
        "court_province": r.court_province_final, "court_prefecture": r.court_prefecture_g,
    }, index=r.index)
    return out


def geo(text, prov, pref, kind):
    """nonlocal_origin / residence_local from a place text with the project gazetteer."""
    if text in (None, "NA") or pd.isna(text):
        return "NA"
    p, f, _, _ = resolve_place(text, prov, pref)
    m = _migrant(p, f, prov, pref, "city")
    if m is None:
        return "NA"
    return ("1" if m else "0") if kind == "origin" else ("0" if m else "1")


def parse_records(recs, spec):
    rows = []
    for r in recs:
        path = L.cache_path(r["model"], r["key"])
        row = {"doc_id": r["doc_id"], "tag": r["tag"], "status": "call_error" if r["error"] else "ok"}
        if not r["error"] and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
            try:
                vals, evid, errs = L.validate(L.parse_json(raw), spec)
                row.update({f"v_{k}": v for k, v in vals.items()})
                row.update({f"e_{k}": v for k, v in evid.items()})
                row["errors"] = ";".join(errs)
                row["status"] = "invalid_value" if errs else "ok"
            except (ValueError, AttributeError) as e:
                row["status"], row["errors"] = "json_error", str(e)[:100]
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    s = pd.read_csv(OUT / "pilot_sample.csv")
    if L.is_peak():
        print("WARNING: DeepSeek peak hours now (UTC Mon-Fri 01-04, 06-10); prices are doubled.")
    ledger = L.Ledger()
    txt = load_texts(s.doc_id, columns=["judgment"]).set_index("doc_id").judgment
    inputs = {d: L.build_input(txt[d], "core")[0] for d in s.doc_id}
    p1 = s[s.set == "P1"].doc_id.tolist()
    p2 = s[s.set == "P2"].doc_id.tolist()
    p3 = s[s.in_P3].doc_id.tolist()
    calls = []
    calls += _run_jobs([(L.MODEL_A, "extract_v1", d, inputs[d], "P1_A") for d in p1], ledger, "P1 model A")
    orig_in = {d: L.build_input(txt[d], "origin")[0] for d in p2}
    calls += _run_jobs([(L.MODEL_A, "extract_origin_v1", d, orig_in[d], "P2_A") for d in p2], ledger, "P2 model A")
    calls += _run_jobs([(L.MODEL_B, "extract_v1", d, inputs[d], "P3_B") for d in p3], ledger, "P3 model B")

    # disagreements between rules and model A on P1 -> model B adjudication
    A = parse_records([c for c in calls if c["tag"] == "P1_A"], L.FIELDS).set_index("doc_id")
    R = rule_values(p1)
    adj_jobs, adj_meta = [], []
    for d in p1:
        if d not in A.index or A.at[d, "status"] == "call_error":
            continue
        a = A.loc[d]
        r = R.loc[d]
        a_geo = {"nonlocal_origin": geo(a.get("v_origin_text"), r.court_province, r.court_prefecture, "origin"),
                 "residence_local": geo(a.get("v_residence_text"), r.court_province, r.court_prefecture, "res")}
        items = []
        for f in COMPARE:
            av = a_geo.get(f, a.get(f"v_{f}"))
            if av is None or pd.isna(av):
                continue
            if agree(f, r[f], av):
                continue
            # place fields are adjudicated on the underlying text, not on the local/non-local judgment
            af, rv, aval = {"nonlocal_origin": ("origin_text", r.origin_text, a.get("v_origin_text")),
                            "residence_local": ("residence_text", r.residence_text, a.get("v_residence_text"))
                            }.get(f, (f, r[f], av))
            items.append((af, rv, aval, f))
        if not items:
            continue
        rnd = random.Random(f"{paths.SEED}-{d}")
        lines, meta = [], []
        for af, rv, aval, f in items:
            order = ["rule", "A"]
            rnd.shuffle(order)
            ans = {"rule": rv, "A": aval}
            lines.append(f"- {af}：答案1 = {ans[order[0]]}；答案2 = {ans[order[1]]}")
            meta.append({"doc_id": d, "field": f, "adj_field": af, "rule": rv, "A": aval,
                         "ans1": order[0], "ans2": order[1]})
        user = inputs[d] + "\n\n【待复核字段】\n" + "\n".join(lines)
        adj_jobs.append((L.MODEL_B, "adjudicate_v1", d, user, "P1_adj"))
        adj_meta += meta
    pd.DataFrame(adj_meta).to_csv(OUT / "adjudication_items.csv", index=False)
    calls += _run_jobs(adj_jobs, ledger, "P1 adjudication by model B")
    pd.DataFrame(calls).to_csv(OUT / "calls.csv", index=False)
    print(f"TOTAL spent (paid-rate accounting): ${ledger.spent:.4f} in {ledger.calls} new calls")


def agree(f, a, b):
    if f == "term_months" and a != "NA" and b != "NA":
        return abs(float(a) - float(b)) < 0.5
    return str(a) == str(b)


# ----------------------------------------------------------------------------- report
def kappa(a, b):
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return np.nan
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((np.mean([x == c for x in a]) * np.mean([y == c for y in b])) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def report():
    s = pd.read_csv(OUT / "pilot_sample.csv")
    calls = pd.read_csv(OUT / "calls.csv")
    meta = pd.read_csv(OUT / "adjudication_items.csv") if (OUT / "adjudication_items.csv").exists() else pd.DataFrame()
    recs = calls.to_dict("records")
    for r in recs:
        r["error"] = None if pd.isna(r["error"]) else r["error"]
    A = parse_records([r for r in recs if r["tag"] == "P1_A"], L.FIELDS).set_index("doc_id")
    B3 = parse_records([r for r in recs if r["tag"] == "P3_B"], L.FIELDS).set_index("doc_id")
    P2 = parse_records([r for r in recs if r["tag"] == "P2_A"], L.ORIGIN_FIELDS).set_index("doc_id")
    p1 = s[s.set == "P1"].doc_id.tolist()
    R = rule_values(p1)

    def model_vals(M, d, r):
        m = M.loc[d]
        out = {f: m.get(f"v_{f}") for f in COMPARE}
        out["nonlocal_origin"] = geo(m.get("v_origin_text"), r.court_province, r.court_prefecture, "origin")
        out["residence_local"] = geo(m.get("v_residence_text"), r.court_province, r.court_prefecture, "res")
        out["nonlocal_origin_self"] = m.get("v_nonlocal_origin_model")
        out["residence_local_self"] = m.get("v_residence_local_model")
        return out

    okA = [d for d in p1 if d in A.index and A.at[d, "status"] in ("ok", "invalid_value")]
    AV = pd.DataFrame({d: model_vals(A, d, R.loc[d]) for d in okA}).T

    # 1. rules vs model A
    rows = []
    for f in COMPARE:
        a, m = R.loc[okA, f].astype(str), AV[f].astype(str)
        valid = m.notna() & (m != "None")
        ag = [agree(f, x, y) for x, y in zip(a[valid], m[valid])]
        rows.append({"field": f, "n": int(valid.sum()), "agree": np.mean(ag),
                     "kappa": kappa(list(a[valid]), list(m[valid])) if f != "term_months" else np.nan,
                     "rule_NA": (a == "NA").mean(), "A_NA": (m[valid] == "NA").mean(),
                     "A_invalid": (~valid).mean()})
    for f, base in (("nonlocal_origin_self", "nonlocal_origin"), ("residence_local_self", "residence_local")):
        m = AV[f].astype(str)
        rows.append({"field": f + " (model's own judgment) vs rule", "n": len(m),
                     "agree": np.mean([x == y for x, y in zip(R.loc[okA, base].astype(str), m)]),
                     "kappa": kappa(list(R.loc[okA, base].astype(str)), list(m)),
                     "A_NA": (m == "NA").mean()})
        rows.append({"field": f + " vs gazetteer on model text", "n": len(m),
                     "agree": np.mean([x == y for x, y in zip(AV[base].astype(str), m)]),
                     "kappa": kappa(list(AV[base].astype(str)), list(m))})
    cmp1 = pd.DataFrame(rows)
    cmp1.to_csv(T / "T06a_rule_vs_A.csv", index=False)
    print(cmp1.round(3).to_string(index=False))

    # 2. adjudication
    adj = parse_adjudication(recs, meta)
    if len(adj):
        adj.to_csv(T / "T06a_adjudication.csv", index=False)
        summ = adj.groupby("field").agg(n=("doc_id", "size"), side_rule=("side", lambda x: (x == "rule").mean()),
                                        side_A=("side", lambda x: (x == "A").mean()),
                                        third=("side", lambda x: (x == "third").mean()),
                                        failed=("side", lambda x: (x == "failed").mean()),
                                        low_conf=("confidence", lambda x: (x == "低").mean())).reset_index()
        tot = adj.agg({"doc_id": "size"}).rename({"doc_id": "n"})
        summ.loc[len(summ)] = {"field": "ALL", "n": len(adj), "side_rule": (adj.side == "rule").mean(),
                               "side_A": (adj.side == "A").mean(), "third": (adj.side == "third").mean(),
                               "failed": (adj.side == "failed").mean(), "low_conf": (adj.confidence == "低").mean()}
        summ.to_csv(T / "T06a_adjudication_summary.csv", index=False)
        print(summ.round(3).to_string(index=False))
        n_docs_dis = adj.doc_id.nunique()
        print(f"docs with >=1 disagreement: {n_docs_dis}/{len(okA)}; field-level disagreement rate "
              f"{len(adj) / (len(okA) * len(COMPARE)):.3f}")

    # 3. P3 model A vs model B (independent)
    p3 = [d for d in s[s.in_P3].doc_id if d in B3.index and B3.at[d, "status"] != "call_error" and d in AV.index]
    BV = pd.DataFrame({d: model_vals(B3, d, R.loc[d]) for d in p3}).T
    rows = []
    for f in COMPARE + ["nonlocal_origin_self", "residence_local_self"]:
        a, b = AV.loc[p3, f].astype(str), BV[f].astype(str)
        rows.append({"field": f, "n": len(p3), "agree_A_B": np.mean([agree(f, x, y) for x, y in zip(a, b)]),
                     "agree_rule_B": np.mean([agree(f, x, y) for x, y in zip(R.loc[p3, f].astype(str), b)])
                     if f in COMPARE else np.nan})
    p3t = pd.DataFrame(rows)
    p3t.to_csv(T / "T06a_P3_A_vs_B.csv", index=False)
    print(p3t.round(3).to_string(index=False))

    # 4. P2 origin hit rate and location
    p2ids = s[s.set == "P2"].doc_id.tolist()
    txt = load_texts(p2ids, columns=["judgment"]).set_index("doc_id").judgment
    rows = []
    for d in p2ids:
        if d not in P2.index:
            continue
        v = P2.at[d, "v_origin_text"] if "v_origin_text" in P2 else None
        t = normalize(txt[d])
        sp = find_spans(t)
        where = "NA"
        if v and v != "NA" and not pd.isna(v):
            where = "not_found_in_text"
            for k in ("header", "charge", "facts", "reasoning", "verdict"):
                if sp.get(k) and v in t[sp[k][0]:sp[k][1]]:
                    where = k
                    break
        p, f, _, st = resolve_place(v) if v and v != "NA" and not pd.isna(v) else (None, None, None, "NA")
        rows.append({"doc_id": d, "origin_text": v, "model_segment": P2.at[d, "v_segment"] if "v_segment" in P2 else None,
                     "found_in": where, "resolve_status": st, "evidence": P2.at[d, "e_origin_text"] if "e_origin_text" in P2 else None,
                     "residence_text": P2.at[d, "v_residence_text"] if "v_residence_text" in P2 else None,
                     "status": P2.at[d, "status"]})
    p2t = pd.DataFrame(rows)
    p2t.to_csv(T / "T06a_P2_origin.csv", index=False)
    hit = p2t.origin_text.notna() & (p2t.origin_text != "NA")
    print(f"P2 hit rate {hit.mean():.3f} ({hit.sum()}/{len(p2t)}); found_in:", p2t[hit].found_in.value_counts().to_dict(),
          "resolve:", p2t[hit].resolve_status.value_counts().to_dict())

    # 5. 40 disagreements for spot check
    if len(adj):
        txt1 = load_texts(adj.doc_id.unique(), columns=["judgment"]).set_index("doc_id").judgment
        sp40 = adj.sample(min(40, len(adj)), random_state=paths.SEED).copy()
        sp40["A_evidence"] = [A.at[d, f"e_{f}"] if f"e_{f}" in A else "" for d, f in zip(sp40.doc_id, sp40.adj_field)]
        sp40["snippet"] = [snippet(txt1[d], f, a, r) for d, f, a, r in zip(sp40.doc_id, sp40.adj_field, sp40.A, sp40.rule)]
        sp40[["doc_id", "field", "adj_field", "rule", "A", "B_value", "side", "confidence", "B_evidence",
              "A_evidence", "snippet"]].to_csv(T / "T06a_disagreements_40.csv", index=False)

    # 6-7. cost, tokens, failures
    calls["tag2"] = calls.tag
    cost = calls.groupby(["tag2", "model"]).agg(
        calls=("doc_id", "size"), cached=("cached", "sum"), errors=("error", lambda x: x.notna().sum()),
        in_hit=("u_in_hit", "sum"), in_miss=("u_in_miss", "sum"), out=("u_out", "sum"),
        cost_usd=("cost_usd", "sum"), latency_mean=("latency_s", "mean"), peak_calls=("peak", "sum"),
        in_chars_mean=("in_chars", "mean")).reset_index()
    cost.to_csv(T / "T06a_cost.csv", index=False)
    print(cost.round(4).to_string(index=False))
    fails = pd.concat([A.status.value_counts().rename("P1_A"), P2.status.value_counts().rename("P2_A"),
                       B3.status.value_counts().rename("P3_B")], axis=1).fillna(0).astype(int)
    fails.to_csv(T / "T06a_failures.csv")
    print(fails.to_string())
    projection(cost)


def parse_adjudication(recs, meta):
    if meta.empty:
        return pd.DataFrame()
    out = []
    by_doc = {r["doc_id"]: r for r in recs if r["tag"] == "P1_adj"}
    for d, g in meta.groupby("doc_id"):
        r = by_doc.get(d)
        obj = {}
        if r and not r["error"]:
            try:
                obj = L.parse_json(json.loads(L.cache_path(r["model"], r["key"]).read_text(encoding="utf-8"))["raw"])
            except (ValueError, FileNotFoundError):
                obj = {}
        for m in g.to_dict("records"):
            item = obj.get(m["adj_field"]) or {}
            choice = str(item.get("choice", "")).strip()
            side = {"1": m["ans1"], "2": m["ans2"]}.get(choice, "third" if choice == "other" else "failed")
            side = {"rule": "rule", "A": "A"}.get(side, side)
            out.append({**m, "B_choice": choice, "B_value": item.get("value"), "side": side,
                        "confidence": item.get("confidence"), "B_evidence": item.get("evidence")})
    return pd.DataFrame(out)


def snippet(text, field, a, r):
    t = normalize(text)
    for key in [x for x in (a, r) if isinstance(x, str) and len(x) >= 2 and x != "NA"] + \
               {"status_at_judgment": ["现羁押", "现取保", "现在家", "现押"], "procedure": ["程序"],
                "counsel_any": ["辩护"], "ever_bail": ["取保"], "jiejie": ["具结"], "plea_formal": ["认罪认罚"],
                "term_months": ["判处"], "penalty_type": ["判处"], "probation": ["缓刑"],
                "origin_text": ["户籍", "出生"], "residence_text": ["住"]}.get(field, []):
        i = t.find(key)
        if i >= 0:
            return t[max(0, i - 60):i + 80]
    return t[:140]


def projection(cost):
    """Full run estimate: V (5,000 docs, core prompt) + R (rule-missing/uncertain docs, header-only)."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    r = con.execute(f"""SELECT count(*) n_theft,
        sum((d1_origin_status IN ('unresolved','ambiguous'))::INT) r_origin,
        sum((d1_pretrial_status_at_judgment IS NULL)::INT) r_status,
        sum((procedure IS NULL)::INT) r_procedure,
        sum((NOT coalesce(d1_outcomes_ok, false))::INT) r_outcome,
        sum(((d1_origin_status IN ('unresolved','ambiguous')) OR d1_pretrial_status_at_judgment IS NULL
             OR procedure IS NULL OR NOT coalesce(d1_outcomes_ok, false))::INT) r_any
        FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_theft AND s.universe""").df().iloc[0]
    a = cost[cost.tag2 == "P1_A"].iloc[0]
    per_in_miss = a.in_miss / a.calls
    per_in_hit = a.in_hit / a.calls
    per_out = a.out / a.calls
    pa = L.PRICES[L.MODEL_A]["offpeak"]
    v_cost = 5000 * (per_in_miss * pa["miss"] + per_in_hit * pa["hit"] + per_out * pa["out"]) / 1e6
    # R: header-only (<=800 chars) and a short field subset: input ~ 40% of the core input, output ~ 30%
    r_cost = r.r_any * (per_in_miss * 0.4 * pa["miss"] + per_in_hit * pa["hit"] + per_out * 0.3 * pa["out"]) / 1e6
    # model B: token profile from its *successful* calls only (failed quota calls carry no usage);
    # adjudication input = the core input plus the answers, so the extraction profile is a close proxy
    calls = pd.read_csv(OUT / "calls.csv")
    okb = calls[(calls.model == L.MODEL_B) & calls.error.isna()]
    adj_docs = cost[cost.tag2 == "P1_adj"]
    share_adj = (adj_docs.iloc[0].calls / a.calls) if len(adj_docs) else 0
    pb = L.PRICES[L.MODEL_B]["batch"]
    if len(okb):
        b_per = (((okb.u_in_hit + okb.u_in_miss).mean() + 150) * pb["in"] + okb.u_out.mean() * pb["out"]) / 1e6
        b_cost = 5000 * share_adj * b_per
    else:
        b_cost = np.nan
    fx = 7.2
    proj = pd.DataFrame([
        {"item": "V model A (5,000 docs, off-peak)", "docs": 5000, "usd": v_cost},
        {"item": "R model A (rule-missing/uncertain, header only)", "docs": int(r.r_any), "usd": r_cost},
        {"item": f"V model B adjudication (Batch, paid; {share_adj:.0%} of docs)", "docs": int(5000 * share_adj), "usd": b_cost},
    ])
    proj.loc[len(proj)] = {"item": "TOTAL", "docs": np.nan, "usd": proj.usd.sum()}
    proj["cny"] = proj.usd * fx
    proj.to_csv(T / "T06a_projection.csv", index=False)
    pd.DataFrame([r]).to_csv(T / "T06a_R_size.csv", index=False)
    print(r.to_string())
    print(proj.round(2).to_string(index=False))


if __name__ == "__main__":
    {"sample": sample, "run": run, "report": report}[sys.argv[1]]()
