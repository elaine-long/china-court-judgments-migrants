"""T06b production run: validation sample V (A + B independent + B adjudication) and fill-in set R (A + B).

Usage (each stage is cached and resumable):
    python scripts/T06b_run.py sample_v      # data/V_sample.parquet (8 strata × 625)
    python scripts/T06b_run.py a_v           # model A on V  (refuses DeepSeek peak hours)
    python scripts/T06b_run.py b_v           # model B on V  (Vertex)
    python scripts/T06b_run.py adj_v         # model B adjudicates rule vs A disagreements on V
    python scripts/T06b_run.py r_prep        # data/R_set.parquet (fields to fill per doc)
    python scripts/T06b_run.py a_r | b_r     # models on R
    python scripts/T06b_run.py labels        # data/V_labels.parquet, data/R_fill.parquet

Budgets (T06b): DeepSeek ≤ ¥50, Vertex ≤ $60, tracked across stages in data/llm/T06b_ledger.json
(list prices; stops the stage when reached).
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
from T06a_pilot import agree, parse_records  # noqa: E402

OUT = paths.DATA / "llm" / "T06b"
LEDGER_DIR = paths.DATA / "llm"  # one ledger file per model, so parallel A and B processes never clash
FX = 7.2
BUDGET = {L.MODEL_A: 50 / FX, L.MODEL_B: 60.0}
N_PER, THREADS = 625, 16
PROMPT = "extract_v2"
FIELDS7 = ["origin_text", "residence_text", "status_at_judgment", "ever_bail", "penalty_type", "term_months", "probation"]
COMPARE = ["nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail", "penalty_type",
           "term_months", "probation"]
lock = threading.Lock()


# ----------------------------------------------------------------------------- budget ledger
def _ledger_path(model, tag=None):
    return LEDGER_DIR / (f"T06b_ledger_{model}__{tag}.json" if tag else f"T06b_ledger_{model}.json")


def spent_so_far(model, exclude=None):
    """Task spend for a model = sum over its per-stage ledger files (parallel stages never clash)."""
    total = 0.0
    for p in LEDGER_DIR.glob(f"T06b_ledger_{model}*.json"):
        if exclude is not None and p == exclude:
            continue
        total += json.loads(p.read_text())["spent_usd"]
    return total


class StageLedger(L.Ledger):
    """Ledger whose budget is what is left of the task budget for this model."""

    def __init__(self, model, tag):
        self.model, self.path = model, _ledger_path(model, tag)
        self.prev = json.loads(self.path.read_text())["spent_usd"] if self.path.exists() else 0.0
        self.others = spent_so_far(model, exclude=self.path)
        self.start = self.others + self.prev
        super().__init__(budget=BUDGET[model] - self.start)

    def save(self):
        """Each stage file holds only that stage's own (cumulative) spend."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model": self.model, "spent_usd": self.prev + self.spent,
                                         "budget_usd": BUDGET[self.model]}, indent=1))


def run_jobs(jobs, model, label):
    """jobs: list of (doc_id, prompt, user, tag). Returns call rows; writes calls_{tag}.csv."""
    led = StageLedger(model, jobs[0][3] if jobs else "none")
    rows, t0 = [], time.time()

    def one(doc_id, prompt, user, tag):
        rec = L.call_cached(model, prompt, user, ledger=led)
        with lock:
            return {"doc_id": doc_id, "tag": tag, "model": model, "prompt": prompt, "key": rec["key"],
                    "cached": rec["cached"], "error": rec["error"], "latency_s": rec["latency_s"],
                    "cost_usd": rec["cost_usd"], "peak": rec["peak"], "in_chars": len(user),
                    **{f"u_{k}": v for k, v in (rec["usage"] or {}).items()}}

    stopped = None
    with ThreadPoolExecutor(THREADS) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for k, f in enumerate(futs):
            try:
                rows.append(f.result())
            except RuntimeError as e:
                stopped = str(e)
                ex.shutdown(cancel_futures=True)
                break
            if (k + 1) % 500 == 0:
                errs = sum(r["error"] is not None for r in rows)
                print(f"  {label}: {k + 1}/{len(jobs)}  new-spend ${led.spent:.3f}  errors {errs}  "
                      f"{time.time() - t0:.0f}s", flush=True)
                led.save()
    led.save()
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = jobs[0][3] if jobs else "none"
    cp = OUT / f"calls_{tag}.csv"          # one file per stage: parallel processes never clash
    if cp.exists():
        df = pd.concat([pd.read_csv(cp), df]).drop_duplicates(["tag", "doc_id"], keep="last")
    df.to_csv(cp, index=False)
    n_err = int(pd.DataFrame(rows).error.notna().sum()) if rows else 0
    print(f"{label}: {len(rows)} calls, errors {n_err}, stage spend ${led.spent:.3f}, "
          f"task total {model} ${led.start + led.spent:.3f} of ${BUDGET[model]:.2f}, {time.time() - t0:.0f}s")
    if stopped:
        print("STOPPED:", stopped)
    return rows


# ----------------------------------------------------------------------------- V
def sample_v():
    f = load_frame()
    excl = set()
    for p in (paths.ROOT / "annotation").glob("sample_*.csv"):
        excl |= set(pd.read_csv(p).get("doc_id", pd.Series(dtype=int)))
    rm = paths.ROOT / "annotation" / "_private" / "remap.csv"
    if rm.exists():
        excl |= set(pd.read_csv(rm).doc_id)
    f = f[~f.doc_id.isin(excl)]
    pop = f.stratum.value_counts()
    v = pd.concat([g.sample(N_PER, random_state=paths.SEED) for _, g in f.groupby("stratum")])
    v = v[["doc_id", "stratum", "period", "pilot_city"]].copy()
    v["N_stratum"] = v.stratum.map(pop)
    v["weight"] = v.N_stratum / N_PER
    v.to_parquet(paths.DATA / "V_sample.parquet", index=False)
    print(f"V: {len(v)} docs; excluded {len(excl)} annotation docs; frame {len(f):,}")
    print(v.groupby("stratum").agg(N=("N_stratum", "first"), n=("doc_id", "size"), w=("weight", "first")).to_string())


def _inputs(ids, mode="core"):
    txt = load_texts(ids, columns=["judgment"]).set_index("doc_id").judgment
    return {d: L.build_input(txt[d], mode)[0] for d in ids}


def a_v():
    if L.is_peak():
        sys.exit("DeepSeek peak hours now; not running model A.")
    ids = pd.read_parquet(paths.DATA / "V_sample.parquet").doc_id.tolist()
    inp = _inputs(ids)
    run_jobs([(d, PROMPT, inp[d], "V_A") for d in ids], L.MODEL_A, "V model A")


def b_v():
    ids = pd.read_parquet(paths.DATA / "V_sample.parquet").doc_id.tolist()
    inp = _inputs(ids)
    run_jobs([(d, PROMPT, inp[d], "V_B") for d in ids], L.MODEL_B, "V model B")


# ----------------------------------------------------------------------------- rule values & geo
def rule_values(ids):
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("ids", pd.DataFrame({"doc_id": list(ids)}))
    r = con.execute(f"""SELECT f.*, t.court_province_final FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        WHERE doc_id IN (SELECT doc_id FROM ids)""").df().set_index("doc_id")
    yn = lambda s: s.map(lambda v: "NA" if pd.isna(v) else ("1" if bool(v) else "0"))  # noqa: E731
    pen = r.d1_penalty_type.map(lambda v: "NA" if pd.isna(v) else
                                (v if v in ("有期徒刑", "拘役", "管制", "单处罚金", "免予刑事处罚") else "其他"))
    return pd.DataFrame({
        "nonlocal_origin": yn(r.d1_migrant_city_nores), "residence_local": yn(r.d1_local_residence),
        "status_at_judgment": r.d1_pretrial_status_at_judgment.map({"在押": "在押", "取保": "取保", "rsl": "监视居住"}).fillna("NA"),
        "ever_bail": yn(r.d1_bail), "penalty_type": pen,
        "term_months": r.d1_term_months.map(lambda v: "NA" if pd.isna(v) else str(float(v))),
        "probation": yn(r.d1_probation),
        "origin_text": r[["d1_hukou_raw", "d1_birthplace_raw", "d1_native_raw"]].bfill(axis=1).iloc[:, 0].fillna("NA"),
        "residence_text": r.d1_residence_raw.fillna("NA"),
        "court_province": r.court_province_final, "court_prefecture": r.court_prefecture_g,
        "d1_origin_status": r.d1_origin_status, "d1_outcomes_ok": r.d1_outcomes_ok,
        "n_defendants": r.n_defendants,
    }, index=r.index)


def geo(text, prov, pref, kind):
    if text is None or (isinstance(text, float) and np.isnan(text)) or text == "NA":
        return "NA"
    p, f, _, _ = resolve_place(text, prov, pref)
    m = _migrant(p, f, prov, pref, "city")
    if m is None:
        return "NA"
    return ("1" if m else "0") if kind == "origin" else ("0" if m else "1")


def model_values(M, d, r):
    m = M.loc[d]
    out = {f: m.get(f"v_{f}") for f in FIELDS7}
    out["nonlocal_origin"] = geo(m.get("v_origin_text"), r.court_province, r.court_prefecture, "origin")
    out["residence_local"] = geo(m.get("v_residence_text"), r.court_province, r.court_prefecture, "res")
    return out


def _calls(tag):
    c = pd.read_csv(OUT / f"calls_{tag}.csv")
    recs = c[c.tag == tag].to_dict("records")
    for r in recs:
        r["error"] = None if pd.isna(r["error"]) else r["error"]
    return recs


def adj_v():
    v = pd.read_parquet(paths.DATA / "V_sample.parquet")
    ids = v.doc_id.tolist()
    R = rule_values(ids)
    A = parse_records(_calls("V_A"), L.FIELDS_V2).set_index("doc_id")
    inp = _inputs(ids)
    jobs, meta = [], []
    for d in ids:
        if d not in A.index or A.at[d, "status"] == "call_error" or d not in R.index:
            continue
        r, av = R.loc[d], model_values(A, d, R.loc[d])
        items = []
        for f in COMPARE:
            if av[f] is None or pd.isna(av[f]) or agree(f, r[f], av[f]):
                continue
            af, rv, aval = {"nonlocal_origin": ("origin_text", r.origin_text, A.at[d, "v_origin_text"]),
                            "residence_local": ("residence_text", r.residence_text, A.at[d, "v_residence_text"])
                            }.get(f, (f, r[f], av[f]))
            items.append((af, rv, aval, f))
        if not items:
            continue
        rnd = random.Random(f"{paths.SEED}-{d}-T06b")
        lines = []
        for af, rv, aval, f in items:
            order = ["rule", "A"]
            rnd.shuffle(order)
            ans = {"rule": rv, "A": aval}
            lines.append(f"- {af}：答案1 = {ans[order[0]]}；答案2 = {ans[order[1]]}")
            meta.append({"doc_id": d, "field": f, "adj_field": af, "rule": rv, "A": aval,
                         "ans1": order[0], "ans2": order[1]})
        jobs.append((d, "adjudicate_v1", inp[d] + "\n\n【待复核字段】\n" + "\n".join(lines), "V_adj"))
    pd.DataFrame(meta).to_csv(OUT / "adjudication_items.csv", index=False)
    print(f"adjudication: {len(jobs)} docs, {len(meta)} fields")
    run_jobs(jobs, L.MODEL_B, "V adjudication (model B)")


# ----------------------------------------------------------------------------- R
def r_prep():
    """Fields to fill per document (status excluded, T06b decision)."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    f = con.execute(f"""SELECT f.doc_id, f.d1_origin_status, f.d1_residence_raw, f.d1_local_residence,
            f.d1_outcomes_ok, f.n_defendants
        FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_theft AND s.universe""").df()
    need = pd.DataFrame({"doc_id": f.doc_id})
    need["origin"] = f.d1_origin_status.isin(["unresolved", "ambiguous"]).values
    need["residence"] = (f.d1_residence_raw.notna() & f.d1_local_residence.isna()).values
    need["penalty"] = (~f.d1_outcomes_ok.fillna(False).astype(bool)).values
    need["no_defendant"] = (f.n_defendants == 0).values
    need = need[need[["origin", "residence", "penalty", "no_defendant"]].any(axis=1)]
    need.to_parquet(paths.DATA / "R_set.parquet", index=False)
    print(need[["origin", "residence", "penalty", "no_defendant"]].sum().to_string(), f"\nR docs: {len(need):,}")


def _r_inputs(need):
    """Party block ≤ 800 chars; when the sentence is missing (penalty) also the verdict ≤ 600 chars
    (the penalty is not in the party block; T06b deviation, see report)."""
    txt = load_texts(need.doc_id, columns=["judgment"]).set_index("doc_id").judgment
    out = {}
    for d, pen in zip(need.doc_id, need.penalty | need.no_defendant):
        t = normalize(txt[d])
        sp = dict(find_spans(t))
        if sp.get("header"):
            nxt = [v[0] for k, v in sp.items() if v and k != "header" and v[0] >= sp["header"][1]]
            head = t[:min(nxt) if nxt else len(t)][:800]
        else:
            head = t[:800]
        body = f"【当事人段】\n{head}"
        if pen and sp.get("verdict"):
            body += f"\n\n【判决段】\n{t[sp['verdict'][0]:sp['verdict'][1]][:600]}"
        out[d] = body
    return out


def a_r():
    if L.is_peak():
        sys.exit("DeepSeek peak hours now; not running model A.")
    need = pd.read_parquet(paths.DATA / "R_set.parquet")
    inp = _r_inputs(need)
    run_jobs([(d, PROMPT, inp[d], "R_A") for d in need.doc_id], L.MODEL_A, "R model A")


def b_r():
    need = pd.read_parquet(paths.DATA / "R_set.parquet")
    inp = _r_inputs(need)
    run_jobs([(d, PROMPT, inp[d], "R_B") for d in need.doc_id], L.MODEL_B, "R model B")


# ----------------------------------------------------------------------------- outputs
def _adjudication(meta_path, recs):
    from T06a_pilot import parse_adjudication
    meta = pd.read_csv(meta_path)
    return parse_adjudication([dict(r, tag="P1_adj") for r in recs], meta)


def labels():
    """data/V_labels.parquet (one row per doc × field) and data/R_fill.parquet."""
    v = pd.read_parquet(paths.DATA / "V_sample.parquet")
    ids = v.doc_id.tolist()
    R = rule_values(ids)
    A = parse_records(_calls("V_A"), L.FIELDS_V2).set_index("doc_id")
    B = parse_records(_calls("V_B"), L.FIELDS_V2).set_index("doc_id")
    adj = _adjudication(OUT / "adjudication_items.csv", _calls("V_adj")) \
        if (OUT / "calls_V_adj.csv").exists() else pd.DataFrame(columns=["doc_id", "field"])
    adj_ix = adj.set_index(["doc_id", "field"]) if len(adj) else None
    rows = []
    for d in ids:
        if d not in R.index:
            continue
        r = R.loc[d]
        av = model_values(A, d, r) if d in A.index and A.at[d, "status"] != "call_error" else {}
        bv = model_values(B, d, r) if d in B.index and B.at[d, "status"] != "call_error" else {}
        for f in COMPARE + ["origin_text", "residence_text"]:
            row = {"doc_id": d, "field": f, "rule": r[f], "A": av.get(f), "B": bv.get(f),
                   "A_evidence": A.at[d, f"e_{f}"] if d in A.index and f"e_{f}" in A else None,
                   "B_evidence": B.at[d, f"e_{f}"] if d in B.index and f"e_{f}" in B else None}
            if adj_ix is not None and (d, f) in adj_ix.index:
                a = adj_ix.loc[(d, f)]
                a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
                row.update(B_adj_value=a["B_value"], B_adj_side=a["side"], B_adj_conf=a["confidence"],
                           B_adj_evidence=a["B_evidence"], adj_field=a["adj_field"])
            rows.append(row)
    vl = pd.DataFrame(rows)
    vl = vl.merge(v[["doc_id", "stratum", "weight"]], on="doc_id", how="left")
    for c in ("rule", "A", "B", "B_adj_value"):
        if c in vl:
            vl[c] = vl[c].astype("string")
    vl.to_parquet(paths.DATA / "V_labels.parquet", index=False)
    print(f"V_labels: {len(vl):,} rows, {vl.doc_id.nunique():,} docs")

    # ------------------------------------------------------------- R fills (A = B only)
    if not (OUT / "calls_R_B.csv").exists() or not (OUT / "calls_R_A.csv").exists():
        print("R calls not complete yet; R_fill skipped")
        return
    need = pd.read_parquet(paths.DATA / "R_set.parquet").set_index("doc_id")
    RR = rule_values(need.index)
    AR = parse_records(_calls("R_A"), L.FIELDS_V2).set_index("doc_id")
    BR = parse_records(_calls("R_B"), L.FIELDS_V2).set_index("doc_id")
    per_need = {"origin": ["nonlocal_origin"], "residence": ["residence_local"],
                "penalty": ["penalty_type", "term_months", "probation"],
                "no_defendant": ["nonlocal_origin", "residence_local", "penalty_type", "term_months", "probation"]}
    rows = []
    for d, nrow in need.iterrows():
        if d not in RR.index:
            continue
        r = RR.loc[d]
        fields = sorted({f for k, fs in per_need.items() if nrow[k] for f in fs})
        a = model_values(AR, d, r) if d in AR.index and AR.at[d, "status"] != "call_error" else None
        b = model_values(BR, d, r) if d in BR.index and BR.at[d, "status"] != "call_error" else None
        for f in fields:
            av = a.get(f) if a else None
            bv = b.get(f) if b else None
            ok = av not in (None, "NA") and bv not in (None, "NA") and not pd.isna(av) and not pd.isna(bv) \
                and agree(f, av, bv)
            rows.append({"doc_id": d, "field": f, "rule": r[f], "A": av, "B": bv,
                         "A_text": AR.at[d, "v_origin_text"] if (a and f == "nonlocal_origin") else None,
                         "B_text": BR.at[d, "v_origin_text"] if (b and f == "nonlocal_origin") else None,
                         "filled": av if ok else None,
                         "filled_by": "A=B" if ok else (
                             "llm_missing" if not (a and b) else
                             "both_NA" if (av in (None, "NA") or pd.isna(av)) and (bv in (None, "NA") or pd.isna(bv))
                             else "unresolved_llm")})
    rf = pd.DataFrame(rows)
    for c in ("rule", "A", "B", "filled"):
        rf[c] = rf[c].astype("string")
    rf.to_parquet(paths.DATA / "R_fill.parquet", index=False)
    s = rf.groupby("field").agg(n=("doc_id", "size"), filled=("filled_by", lambda x: (x == "A=B").sum()),
                                both_NA=("filled_by", lambda x: (x == "both_NA").sum()),
                                unresolved=("filled_by", lambda x: (x == "unresolved_llm").sum()),
                                missing=("filled_by", lambda x: (x == "llm_missing").sum())).reset_index()
    s["fill_rate"] = s.filled / s.n
    s.to_csv(paths.TABLES / "T06b_R_fill_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    {"sample_v": sample_v, "a_v": a_v, "b_v": b_v, "adj_v": adj_v, "r_prep": r_prep,
     "a_r": a_r, "b_r": b_r, "labels": labels}[sys.argv[1]]()
