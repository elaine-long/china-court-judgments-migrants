"""T09a §1: analysis dataset from rule outputs (PAP v1 §1-3) -> data/analysis_rules.parquet.

Sample (PAP §1): theft, universe, first-instance criminal, outcomes_ok, judgment date 2013-01-01..2019-12-31,
first defendant, migrant_city_nores defined; court city resolved. dup_pseudonym is kept (flag) and only
excluded in robustness checks.
Missing controls (not specified by PAP): zero-filled with a missing indicator (see docs/variables.md, issues).
"""
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.treatment import NATIONAL_DATE, PILOT_DATE, PROVINCIAL_CAPITALS, SUB_PROVINCIAL  # noqa: E402

SPEEDY_DATE = pd.Timestamp("2014-08-26")
START, END = pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")


def load_all():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con.execute(f"""
        SELECT f.*, t.judgment_date_parsed AS jdate, t.pilot_city, t.court_city, t.court_province_final,
               m.court_name, m.case_number, m.doc_kind, s.universe
        FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        JOIN read_parquet('{paths.META}') m USING (doc_id)
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_theft AND s.universe AND m.doc_kind = '刑初'
          AND t.judgment_date_parsed BETWEEN DATE '2013-01-01' AND DATE '2019-12-31'""").df()


def build(df):
    d = pd.DataFrame({"doc_id": df.doc_id})
    d["jdate"] = pd.to_datetime(df.jdate)
    d["q"] = d.jdate.dt.to_period("Q").astype(str)
    d["year"] = d.jdate.dt.year
    d["period"] = np.select([d.jdate < SPEEDY_DATE, d.jdate < PILOT_DATE, d.jdate < NATIONAL_DATE],
                            ["P0", "P1", "P2"], default="P3")
    d["city"] = df.court_prefecture_g
    d["court_name"] = df.court_name
    d["province"] = df.court_province_final
    d["pilot"] = df.pilot_city.astype(int)
    d["capital_or_subprov"] = df.court_city.isin(PROVINCIAL_CAPITALS + SUB_PROVINCIAL).astype(int)
    # migrant definitions (main + robustness)
    for c in ("migrant_city_nores", "migrant_city", "migrant_prov", "local_residence"):
        d[c] = df[f"d1_{c}"].astype("float")
    d["migrant"] = d["migrant_city_nores"]
    # outcomes (PAP §3)
    d["probation"] = df.d1_probation.astype("float")
    st = df.d1_pretrial_status_at_judgment
    d["bail_at_judgment"] = (st == "取保").astype(float).where(st.notna())
    term_ok = df.d1_penalty_type.isin(["有期徒刑", "拘役"])
    d["term_months"] = df.d1_term_months.where(term_ok)
    d["log_term"] = np.log1p(d["term_months"])
    d["leniency_proc"] = df.leniency_proc.astype(float)
    d["speedy"] = df.speedy.astype(float)
    d["plea_formal"] = df.plea_formal.astype(float)
    d["jiejie"] = df.jiejie.astype(float)
    # controls X (PAP §4); missing -> 0 plus indicator
    amt = df.amt_yuan_ctrl
    d["log_amt"] = np.log1p(amt).fillna(0)
    d["amt_missing"] = amt.isna().astype(float)
    for c in ("spec_burglary", "spec_pickpocket", "spec_multiple", "spec_weapon",
              "surrender", "confession", "restitution", "forgiveness"):
        d[c] = df[c].astype(float)
    d["prior_record"] = df.d1_prior_record.astype(float)
    d["recidivist"] = df.d1_recidivist.astype(float)
    age = df.d1_age_at_judgment.astype("float")
    d["age"] = age.fillna(0)
    d["age2"] = (age ** 2).fillna(0) / 100
    d["age_missing"] = age.isna().astype(float)
    d["male"] = (df.d1_gender == "男").astype(float).where(df.d1_gender.notna()).fillna(0)
    d["gender_missing"] = df.d1_gender.isna().astype(float)
    d["multi_def"] = (df.n_defendants > 1).astype(float)
    d["n_charges"] = df.d1_charges.map(lambda x: len(x) if hasattr(x, "__len__") else 0).astype(float)
    d["dup_pseudonym"] = df.dup_pseudonym.astype(bool)
    d["outcomes_ok"] = df.d1_outcomes_ok.fillna(False).astype(bool)
    return d


def main():
    t0 = time.time()
    df = load_all()
    d = build(df)
    steps = [("theft & universe & 刑初 & 2013-2019", len(d))]
    d = d[d.outcomes_ok]
    steps.append(("& outcomes_ok", len(d)))
    d = d[d.city.notna()]
    steps.append(("& court city resolved", len(d)))
    full = d.copy()                   # before the migrant restriction (for coverage tests)
    full.to_parquet(paths.DATA / "analysis_rules_full.parquet", index=False)
    d = d[d.migrant.notna()]
    steps.append(("& migrant_city_nores defined (main sample)", len(d)))
    d.to_parquet(paths.DATA / "analysis_rules.parquet", index=False)
    flow = pd.DataFrame(steps, columns=["step", "n"])
    flow.to_csv(paths.TABLES / "T09a_sample_flow_prelim_rules.csv", index=False)
    print(flow.to_string(index=False))
    print(f"cities {d.city.nunique()} (pilot {d.loc[d.pilot == 1, 'city'].nunique()}); "
          f"migrant share {d.migrant.mean():.3f}; missing: amt {d.amt_missing.mean():.3f}, "
          f"age {d.age_missing.mean():.3f}, bail {d.bail_at_judgment.isna().mean():.3f}, "
          f"term {d.term_months.isna().mean():.3f}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
