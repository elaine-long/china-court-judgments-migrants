"""T09a §3: identification checks (PAP §5.1). Numbers only; no specification changes.

1. 2016Q4 composition: Pilot × Post_2016Q4 on migrant share, origin-defined rate and a case index
   (probation predicted from X), city + quarter FE, clustered by city. Window 2013Q1-2018Q3
   (before the national roll-out, so only pilot cities change status).
2. Publication rate: per court and case-number year, max 刑初 serial = lower bound on filings;
   rate = documents in the data / that bound. Trends pilot vs non-pilot; DiD Pilot × Post (year >= 2017).
3. First-stage event study of leniency_proc, separately for local and migrant defendants.
"""
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import estimate as E  # noqa: E402
from src import paths, plotstyle  # noqa: E402
from src.treatment import NATIONAL_DATE, PILOT_DATE  # noqa: E402

T, FIG = paths.TABLES, paths.FIGURES
TAG = "prelim_rules"
SPEEDY_DATE = pd.Timestamp("2014-08-26")
X = ["log_amt", "amt_missing", "spec_burglary", "spec_pickpocket", "spec_multiple", "spec_weapon",
     "prior_record", "recidivist", "surrender", "confession", "restitution", "forgiveness",
     "age", "age2", "age_missing", "male", "gender_missing", "multi_def"]


def composition(full):
    d = full[(full.q >= "2013Q1") & (full.q <= "2018Q3")].copy()
    d["post16q4"] = (d.jdate >= pd.Timestamp("2016-10-01")).astype(float)
    d["pilot_post"] = d.pilot * d.post16q4
    d["defined"] = d.migrant.notna().astype(float)
    d["city_id"] = d.city.astype("category").cat.codes
    # PAP v1.1 §7.2: case index = LPM of probation on X fitted ONLY on non-pilot cities, 2016Q3 and earlier
    # (no post-treatment outcomes), then predicted for the whole sample
    fit_s = d[(d.pilot == 0) & (d.q <= "2016Q3")]
    fx = pf.feols(f"probation ~ {' + '.join(X)}", data=fit_s)
    d["case_index"] = fx.predict(newdata=d)
    print(f"case index fitted on {fx._N:,} non-pilot cases up to 2016Q3")
    dd = d[d.migrant.notna()].copy()
    dd["city_mig"] = dd.city.astype(str) + "_" + dd.migrant.astype(int).astype(str)
    dd["q_mig"] = dd.q.astype(str) + "_" + dd.migrant.astype(int).astype(str)
    dd["case_index_defined"] = dd.case_index
    # FE: the main specification's α_c + δ_q; where Migrant is not the outcome (case index on the
    # migrant-defined sample) also the Migrant-interacted effects of PAP §4
    specs = (("migrant", dd, "city + q"), ("defined", d, "city + q"), ("case_index", d, "city + q"),
             ("case_index_defined", dd, "city_mig + q_mig"))
    rows = []
    for y, sub, fe in specs:
        f = pf.feols(f"{y} ~ pilot_post | {fe}", data=sub, vcov={"CRV1": "city_id"})
        r = f.tidy().loc["pilot_post"]
        pre = sub[(sub.post16q4 == 0)]
        rows.append({"outcome": y, "fe": fe, "coef": r["Estimate"], "se": r["Std. Error"], "p": r["Pr(>|t|)"],
                     "N": f._N, "clusters": int(f._G[0]),
                     "pre_mean_pilot": pre.loc[pre.pilot == 1, y].mean(),
                     "pre_mean_nonpilot": pre.loc[pre.pilot == 0, y].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(T / f"T09a_composition_2016q4_{TAG}.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


_SERIAL = re.compile(r"(\d+)\s*号\s*$")
_YEAR = re.compile(r"[(（]\s*(\d{4})\s*[)）]")


def publication():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # all first-instance criminal documents of each court (any crime in the data), not only theft
    m = con.execute(f"""
        SELECT m.doc_id, m.case_number, m.court_name, t.pilot_city, t.court_city
        FROM read_parquet('{paths.META}') m
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        WHERE s.universe""").df()
    m["serial"] = pd.to_numeric(m.case_number.str.extract(_SERIAL)[0], errors="coerce")
    m["cy"] = pd.to_numeric(m.case_number.str.extract(_YEAR)[0], errors="coerce")
    n_all = len(m)
    # serials above 20,000 are concatenation errors ("第0025600262号") or block numbering; the 99th
    # percentile of all serials is ~2,900, so they would dominate the court-year maximum
    m = m[m.serial.notna() & (m.serial <= 20000) & m.cy.between(2013, 2019) & m.court_name.notna()]
    print(f"documents used for the publication rate: {len(m):,} of {n_all:,}")
    cy = (m.groupby(["court_name", "cy"])
          .agg(n_docs=("doc_id", "size"), max_serial=("serial", "max"),
               pilot=("pilot_city", "max"), city=("court_city", "first")).reset_index())
    cy = cy[cy.max_serial >= 10]          # tiny serials are unreliable bounds
    cy["rate"] = (cy.n_docs / cy.max_serial).clip(upper=1.0)
    cy["post"] = (cy.cy >= 2017).astype(float)
    cy["pilot_post"] = cy.pilot.astype(float) * cy.post
    cy["city_id"] = cy.city.astype("category").cat.codes
    trend = (cy.groupby(["cy", "pilot"])
             .apply(lambda g: pd.Series({"courts": len(g), "rate_mean": g.rate.mean(),
                                         "rate_weighted": g.n_docs.sum() / g.max_serial.sum()}),
                    include_groups=False).reset_index())
    trend.to_csv(T / f"T09a_publication_trend_{TAG}.csv", index=False)
    print(trend.round(4).to_string(index=False))
    f1 = pf.feols("rate ~ pilot_post | court_name + cy", data=cy, vcov={"CRV1": "city_id"})
    f2 = pf.feols("rate ~ pilot_post | court_name + cy", data=cy, vcov={"CRV1": "city_id"}, weights="max_serial")
    rows = [{"spec": lab, **{k: f.tidy().loc["pilot_post", c] for k, c in
                             (("coef", "Estimate"), ("se", "Std. Error"), ("p", "Pr(>|t|)"))},
             "N_court_years": f._N, "clusters": int(f._G[0])}
            for lab, f in (("unweighted", f1), ("weighted by max serial", f2))]
    did = pd.DataFrame(rows)
    did.to_csv(T / f"T09a_publication_did_{TAG}.csv", index=False)
    print(did.round(4).to_string(index=False))
    plotstyle.use()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=plotstyle.SIZE_1COL)
    for p, (lab, col, ls, mk) in ((1, ("Pilot cities",) + plotstyle.SERIES[0]), (0, ("Non-pilot cities",) + plotstyle.SERIES[1])):
        g = trend[trend.pilot == p]
        ax.plot(g.cy, g.rate_weighted, color=col, ls=ls, marker=mk, label=lab)
    ax.axvline(2016.87, color="0.35", lw=0.6, ls=":")
    ax.set_ylabel("Documents / max case serial")
    ax.set_xlabel("Case-number year")
    ax.legend()
    plotstyle.save(fig, FIG / f"T09a_publication_rate_{TAG}.pdf")
    return did


def first_stage(main):
    plotstyle.use()
    import matplotlib.pyplot as plt
    tabs = []
    fig, ax = plt.subplots(figsize=plotstyle.SIZE_2COL)
    for k, (m, lab) in enumerate(((0, "Local"), (1, "Migrant"))):
        _, tab = E.event_study_simple(main[main.migrant == m], "leniency_proc")
        tab["group"] = lab
        tabs.append(tab)
        x = pd.PeriodIndex(tab.quarter, freq="Q").start_time + pd.Timedelta(days=15 * k)
        col, ls, mk = plotstyle.SERIES[k]
        ax.errorbar(x, tab.beta, yerr=1.96 * tab.se, color=col, ls=ls, marker=mk, capsize=0, lw=0.9, label=lab)
    plotstyle.vlines(ax, (SPEEDY_DATE, PILOT_DATE))
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylabel("Pilot × quarter, leniency_proc")
    ax.legend()
    plotstyle.save(fig, FIG / f"T09a_es_firststage_{TAG}.pdf")
    t = pd.concat(tabs)
    t.to_csv(T / f"T09a_es_firststage_{TAG}.csv", index=False)
    s = t.pivot_table(index="quarter", columns="group", values="beta")
    print(s.loc[["2014Q1", "2014Q4", "2015Q4", "2016Q3", "2017Q2", "2018Q1", "2018Q3"]].round(3).to_string())
    return t


def main():
    t0 = time.time()
    full = pd.read_parquet(paths.DATA / "analysis_rules_full.parquet")
    main_ = pd.read_parquet(paths.DATA / "analysis_rules.parquet")
    print("== composition"); composition(full)
    print("== publication"); publication()
    print("== first stage"); first_stage(main_)
    print(f"TOTAL {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
