"""T06a2 §1: coverage before/after the rule fixes, and the new R (fill-in set) by field.

Before = data/archive_T03/feat_rules.parquet (T03 rules); after = data/feat_rules.parquet.
Sample: theft documents in the universe, first defendant.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402

T = paths.TABLES


def load(path):
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con.execute(f"""SELECT f.* FROM read_parquet('{path}') f
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_theft AND s.universe""").df().set_index("doc_id")


def summary(f, counsel_col):
    return {
        "n": len(f),
        "hukou_raw": f.d1_hukou_raw.notna().mean(),
        "residence_raw": f.d1_residence_raw.notna().mean(),
        "local_residence defined": f.d1_local_residence.notna().mean(),
        "migrant_city_nores defined": f.d1_migrant_city_nores.notna().mean(),
        "migrant_city defined": f.d1_migrant_city.notna().mean(),
        "counsel = 1": f[counsel_col].astype(float).mean(),
        "procedure NA": f.procedure.isna().mean(),
        "procedure = 普通": (f.procedure == "普通").mean(),
        "procedure = 简易": (f.procedure == "简易").mean(),
        "procedure = 速裁": (f.procedure == "速裁").mean(),
        "panel_trial": f.panel_trial.mean() if "panel_trial" in f else float("nan"),
        "n_defendants = 0": (f.n_defendants == 0).mean(),
        "status NA": f.d1_pretrial_status_at_judgment.isna().mean(),
    }


def r_sizes(f):
    """R: fields the LLM would fill (v2 prompt fields only; word-match fields are rules only)."""
    parts = {
        "origin (unresolved / ambiguous)": f.d1_origin_status.isin(["unresolved", "ambiguous"]),
        "residence (text present, place unresolved)": f.d1_residence_raw.notna() & f.d1_local_residence.isna(),
        "status_at_judgment NA": f.d1_pretrial_status_at_judgment.isna(),
        "penalty / term incomplete (not outcomes_ok)": ~f.d1_outcomes_ok.fillna(False).astype(bool),
        "no defendant parsed": f.n_defendants == 0,
    }
    rows = [{"field": k, "n_docs": int(v.sum()), "share": v.mean()} for k, v in parts.items()]
    anyr = pd.concat(parts.values(), axis=1).any(axis=1)
    rows.append({"field": "ANY (union)", "n_docs": int(anyr.sum()), "share": anyr.mean()})
    no_status = pd.concat([v for k, v in parts.items() if not k.startswith("status")], axis=1).any(axis=1)
    rows.append({"field": "ANY excluding status", "n_docs": int(no_status.sum()), "share": no_status.mean()})
    return pd.DataFrame(rows)


def main():
    # before: T03 archive by default; T06b passes the T06a2 archive and an output prefix
    before = sys.argv[1] if len(sys.argv) > 1 else "archive_T03"
    tag = sys.argv[2] if len(sys.argv) > 2 else "T06a2"
    old = load(paths.DATA / before / "feat_rules.parquet")
    new = load(paths.DATA / "feat_rules.parquet")
    old_counsel = "d1_counsel" if "d1_counsel" in old else "counsel"
    cov = pd.DataFrame({"before": summary(old, old_counsel), "after": summary(new, "d1_counsel")})
    cov["change"] = cov.after - cov.before
    cov.to_csv(T / f"{tag}_coverage_before_after.csv")
    print(cov.round(4).to_string())
    both = old.index.intersection(new.index)
    chg = pd.DataFrame([{
        "field": "counsel changed (first defendant)",
        "n": int((old.loc[both, old_counsel].fillna(False).astype(bool) != new.loc[both, "d1_counsel"].fillna(False).astype(bool)).sum())},
        {"field": "procedure changed", "n": int((old.loc[both, "procedure"].fillna("NA") != new.loc[both, "procedure"].fillna("NA")).sum())},
        {"field": "residence_raw newly found", "n": int((old.loc[both, "d1_residence_raw"].isna() & new.loc[both, "d1_residence_raw"].notna()).sum())},
        {"field": "hukou_raw changed", "n": int((old.loc[both, "d1_hukou_raw"].fillna("") != new.loc[both, "d1_hukou_raw"].fillna("")).sum())},
        {"field": "first defendant newly found (n_defendants 0 -> >0)", "n": int(((old.loc[both, "n_defendants"] == 0) & (new.loc[both, "n_defendants"] > 0)).sum())},
    ])
    chg.to_csv(T / f"{tag}_changes.csv", index=False)
    print(chg.to_string(index=False))
    r_old, r_new = r_sizes(old), r_sizes(new)
    r = r_old.merge(r_new, on="field", suffixes=("_before", "_after"))
    r.to_csv(T / f"{tag}_R_size.csv", index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
