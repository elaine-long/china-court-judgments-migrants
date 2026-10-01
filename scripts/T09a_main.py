"""T09a §4: PAP v1.1 §4 main regressions on rule variables (prelim_rules) + quarterly event studies.

4 outcomes × parameters β2_P1, β2_P2 (β1 and P3 reported for completeness). Clustered by court city;
wild cluster bootstrap p-values and Romano–Wolf over the 8 PAP hypotheses.
All outputs carry 'prelim_rules'; results must be re-run once the LLM hybrid variables exist.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import estimate as E  # noqa: E402
from src import paths, plotstyle  # noqa: E402
from src.treatment import PILOT_DATE  # noqa: E402

T, FIG = paths.TABLES, paths.FIGURES
TAG = "prelim_rules"
SPEEDY_DATE = pd.Timestamp("2014-08-26")
X = ["log_amt", "amt_missing", "spec_burglary", "spec_pickpocket", "spec_multiple", "spec_weapon",
     "prior_record", "recidivist", "surrender", "confession", "restitution", "forgiveness",
     "age", "age2", "age_missing", "male", "gender_missing", "multi_def"]
OUTCOMES = [("probation", X), ("bail_at_judgment", X), ("log_term", X + ["n_charges"]), ("leniency_proc", X)]
PARAMS = ["pilot_P1_mig", "pilot_P2_mig"]
BOOT_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 999


def main():
    t0 = time.time()
    d = E.add_design(pd.read_parquet(paths.DATA / "analysis_rules.parquet"))
    print(f"N = {len(d):,}; clusters = {d.city.nunique()}")
    fits = {}
    rows = []
    for y, ctrl in OUTCOMES:
        f = E.triple_diff(d, y, controls=ctrl)
        fits[y] = f
        for p in ["pilot_P1", "pilot_P2", "pilot_P3"] + [f"{x}_mig" for x in ("pilot_P1", "pilot_P2", "pilot_P3")]:
            r = E.coef_row(f, p)
            r["mean_y"] = d[y].mean()
            rows.append(r)
        print(f"  {y}: done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    # wild cluster bootstrap for the PAP parameters
    wb = []
    for y, _ in OUTCOMES:
        for p in PARAMS:
            wb.append({"outcome": y, "param": p, "p_wild": E.wild_bootstrap(fits[y], p, reps=BOOT_REPS)})
    res = res.merge(pd.DataFrame(wb), on=["outcome", "param"], how="left")
    rw = E.romano_wolf([(fits[y], p) for y, _ in OUTCOMES for p in PARAMS], reps=BOOT_REPS)
    res = res.merge(rw[["outcome", "param", "p_romano_wolf"]], on=["outcome", "param"], how="left")
    res["boot_reps"] = BOOT_REPS
    res.to_csv(T / f"T09a_main_{TAG}.csv", index=False)
    show = res[res.param.isin(PARAMS)][["outcome", "param", "coef", "se", "p", "p_wild", "p_romano_wolf", "N", "n_clusters"]]
    print(show.round(4).to_string(index=False))
    tex = pf.etable([fits[y] for y, _ in OUTCOMES], type="tex",
                    keep=["pilot_P1", "pilot_P2", "pilot_P3"],
                    labels={"pilot_P1": "Pilot × P1", "pilot_P2": "Pilot × P2", "pilot_P3": "Pilot × P3",
                            "pilot_P1_mig": "Pilot × P1 × Migrant", "pilot_P2_mig": "Pilot × P2 × Migrant",
                            "pilot_P3_mig": "Pilot × P3 × Migrant", "probation": "Probation",
                            "bail_at_judgment": "Bail at judgment", "log_term": "log(1+term)",
                            "leniency_proc": "Leniency procedure", "city_mig": "City × Migrant",
                            "q_mig": "Quarter × Migrant"},
                    notes="Preliminary estimates on rule-extracted variables (prelim\\_rules). "
                          "SEs clustered by court city. Controls per PAP §4.")
    (T / f"T09a_main_{TAG}.tex").write_text(tex, encoding="utf-8")
    print(f"main done {time.time() - t0:.0f}s", flush=True)

    # PAP v1.1 §7.1 robustness: complete cases (no missing control), without the missing indicators
    miss = ["amt_missing", "age_missing", "gender_missing"]
    cc = d[(d[miss] == 0).all(axis=1)]
    cc_rows = []
    for y, ctrl in OUTCOMES:
        f = E.triple_diff(cc, y, controls=[c for c in ctrl if c not in miss])
        for p in PARAMS:
            cc_rows.append(E.coef_row(f, p))
    ccr = pd.DataFrame(cc_rows)
    ccr.to_csv(T / f"T09a_main_completecase_{TAG}.csv", index=False)
    print(f"complete-case sample: {len(cc):,} of {len(d):,}")
    print(ccr[["outcome", "param", "coef", "se", "p", "N"]].round(4).to_string(index=False))

    # event studies (PAP §4): quarterly Pilot × Migrant coefficients, base 2014Q2, window 2013Q1-2018Q3
    plotstyle.use()
    import matplotlib.pyplot as plt
    es_all = []
    for y, ctrl in OUTCOMES:
        _, tab = E.event_study(d, y, base="2014Q2", window=("2013Q1", "2018Q3"), controls=ctrl)
        tab["outcome"] = y
        es_all.append(tab)
        fig, ax = plt.subplots(figsize=plotstyle.SIZE_2COL)
        x = pd.PeriodIndex(tab.quarter, freq="Q").start_time
        ax.errorbar(x, tab.beta2, yerr=1.96 * tab.se2, color="black", marker="o", ls="-", capsize=0, lw=0.9)
        ax.axhline(0, color="0.6", lw=0.5)
        plotstyle.vlines(ax, (SPEEDY_DATE, PILOT_DATE))
        ax.set_ylabel(f"Pilot × Migrant × quarter: {y}")
        plotstyle.save(fig, FIG / f"T09a_es_{y}_{TAG}.pdf")
    es = pd.concat(es_all)
    es.to_csv(T / f"T09a_es_{TAG}.csv", index=False)
    pre = es[(es.quarter < "2014Q3") & (es.quarter != "2014Q2")]
    print("pre-period Pilot×Migrant coefficients (2013Q1-2014Q1):")
    print(pre.pivot_table(index="quarter", columns="outcome", values="beta2").round(4).to_string())
    print(f"TOTAL {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
