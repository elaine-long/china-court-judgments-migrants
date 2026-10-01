"""T06b §6: V metrics (rule / A / B agreement, kappa, adjudication), R fill summary, cost and failures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import paths  # noqa: E402
from T06a_pilot import agree, kappa  # noqa: E402

T = paths.TABLES
OUT = paths.DATA / "llm" / "T06b"
FIELDS = ["nonlocal_origin", "residence_local", "status_at_judgment", "ever_bail", "penalty_type",
          "term_months", "probation"]


def _pair(x, y, f, w):
    ok = x.notna() & y.notna()
    x, y, w = x[ok].astype(str), y[ok].astype(str), w[ok]
    a = np.array([agree(f, p, q) for p, q in zip(x, y)])
    return {"n": int(ok.sum()), "agree": a.mean(), "agree_w": (a * w).sum() / w.sum(),
            "kappa": kappa(list(x), list(y)) if f != "term_months" else np.nan}


def v_metrics():
    vl = pd.read_parquet(paths.DATA / "V_labels.parquet")
    rows = []
    for f in FIELDS:
        g = vl[vl.field == f]
        for lab, a, b in (("rule_vs_A", "rule", "A"), ("rule_vs_B", "rule", "B"), ("A_vs_B", "A", "B")):
            rows.append({"field": f, "pair": lab, **_pair(g[a], g[b], f, g.weight)})
    m = pd.DataFrame(rows)
    na = vl[vl.field.isin(FIELDS)].groupby("field").agg(
        rule_NA=("rule", lambda x: (x == "NA").mean()), A_NA=("A", lambda x: (x == "NA").mean()),
        B_NA=("B", lambda x: (x == "NA").mean()), A_missing=("A", lambda x: x.isna().mean()),
        B_missing=("B", lambda x: x.isna().mean())).reset_index()
    wide = m.pivot_table(index="field", columns="pair", values=["agree", "kappa"]).round(3)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(na, on="field")
    wide.to_csv(T / "T06b_V_agreement.csv", index=False)
    m.to_csv(T / "T06b_V_agreement_long.csv", index=False)
    print(wide.round(3).to_string(index=False))
    # three-way pattern on V: all agree / rule=A only / rule=B only / A=B only / all differ
    pat = []
    for f in FIELDS:
        g = vl[(vl.field == f) & vl.A.notna() & vl.B.notna()]
        ra = np.array([agree(f, p, q) for p, q in zip(g.rule.astype(str), g.A.astype(str))])
        rb = np.array([agree(f, p, q) for p, q in zip(g.rule.astype(str), g.B.astype(str))])
        ab = np.array([agree(f, p, q) for p, q in zip(g.A.astype(str), g.B.astype(str))])
        pat.append({"field": f, "n": len(g), "all_agree": (ra & rb).mean(), "rule_A_only": (ra & ~rb).mean(),
                    "rule_B_only": (rb & ~ra).mean(), "A_B_only": (ab & ~ra).mean(),
                    "all_differ": (~ra & ~rb & ~ab).mean()})
    pat = pd.DataFrame(pat)
    pat.to_csv(T / "T06b_V_pattern.csv", index=False)
    print(pat.round(3).to_string(index=False))
    # adjudication
    adj = vl[vl.B_adj_side.notna()] if "B_adj_side" in vl else vl.iloc[0:0]
    s = adj.groupby("field").agg(n=("doc_id", "size"), side_rule=("B_adj_side", lambda x: (x == "rule").mean()),
                                 side_A=("B_adj_side", lambda x: (x == "A").mean()),
                                 third=("B_adj_side", lambda x: (x == "third").mean()),
                                 failed=("B_adj_side", lambda x: (x == "failed").mean()),
                                 low_conf=("B_adj_conf", lambda x: (x == "低").mean())).reset_index()
    s.loc[len(s)] = {"field": "ALL", "n": len(adj), "side_rule": (adj.B_adj_side == "rule").mean(),
                     "side_A": (adj.B_adj_side == "A").mean(), "third": (adj.B_adj_side == "third").mean(),
                     "failed": (adj.B_adj_side == "failed").mean(), "low_conf": (adj.B_adj_conf == "低").mean()}
    # does B's adjudication agree with B's own independent extraction?
    post = np.where(adj.B_adj_side == "rule", adj.rule, np.where(adj.B_adj_side == "A", adj.A, adj.B_adj_value))
    s["adj_consistent_with_B_indep"] = np.nan
    s.loc[s.field == "ALL", "adj_consistent_with_B_indep"] = np.mean(
        [agree(f, str(p), str(b)) for f, p, b in zip(adj.field, post, adj.B)])
    s.to_csv(T / "T06b_V_adjudication.csv", index=False)
    print(s.round(3).to_string(index=False))
    n_docs = vl.doc_id.nunique()
    print(f"V docs {n_docs}; docs with >=1 rule-A disagreement {adj.doc_id.nunique()} "
          f"({adj.doc_id.nunique() / n_docs:.3f}); field-level rate {len(adj) / (n_docs * len(FIELDS)):.4f}")


def cost():
    rows = []
    for p in sorted(OUT.glob("calls_*.csv")):
        c = pd.read_csv(p)
        rows.append({"stage": p.stem.replace("calls_", ""), "model": c.model.iloc[0], "calls": len(c),
                     "cached": int(c.cached.sum()), "errors": int(c.error.notna().sum()),
                     "in_hit": c.get("u_in_hit", pd.Series(0)).sum(), "in_miss": c.get("u_in_miss", pd.Series(0)).sum(),
                     "out": c.get("u_out", pd.Series(0)).sum(), "cost_usd": c.cost_usd.sum(),
                     "peak_calls": int(c.peak.sum()), "latency_mean": c.latency_s.mean()})
    cs = pd.DataFrame(rows)
    cs.to_csv(T / "T06b_cost.csv", index=False)
    print(cs.round(3).to_string(index=False))
    by = cs.groupby("model").cost_usd.sum()
    print("total by model (USD):", by.round(3).to_dict(), "| DeepSeek CNY", round(by.get("deepseek-flash", 0) * 7.2, 1))


if __name__ == "__main__":
    v_metrics()
    cost()
