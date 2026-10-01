"""T04 step 1: annotation samples (main 300, pilot 10, recheck 60).

Frame: theft, universe, d1 outcomes_ok, judgment date 2013-01-01..2019-12-31, dup_pseudonym == False.
Strata: period (4) x pilot city (2), 30 each; plus two supplementary strata of 30:
  S1 pilot city, plea-pilot period, leniency_proc == 1
  S2 first defendant's origin unresolved, or all four origin text fields missing
Weights: `weight` = stratum population / draws in the stratum the unit was drawn from.
`weight_combined` = 1 / inclusion probability over the overlapping designs (main stratum + S1/S2).
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.treatment import NATIONAL_DATE, PILOT_DATE  # noqa: E402

SPEEDY_DATE = pd.Timestamp("2014-08-26")
ANN = paths.ROOT / "annotation"
PRIVATE = ANN / "_private"
N_PER_STRATUM, N_SUPP, N_PILOT, N_RECHECK = 30, 30, 10, 60
PERIODS = ["P0_base", "P1_speedy", "P2_plea", "P3_national"]


def load_frame():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    f = con.execute(f"""
        SELECT f.doc_id, f.leniency_proc, f.dup_pseudonym, f.d1_origin_status,
               f.d1_hukou_raw, f.d1_birthplace_raw, f.d1_native_raw, f.d1_residence_raw,
               t.judgment_date_parsed AS jd, t.pilot_city
        FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_theft AND s.universe AND f.d1_outcomes_ok AND NOT f.dup_pseudonym
          AND t.judgment_date_parsed BETWEEN DATE '2013-01-01' AND DATE '2019-12-31'""").df()
    f["period"] = np.select(
        [f.jd < SPEEDY_DATE, f.jd < PILOT_DATE, f.jd < NATIONAL_DATE], PERIODS[:3], default=PERIODS[3])
    f["stratum"] = f.period + "_" + np.where(f.pilot_city, "pilot", "nonpilot")
    raw = f[["d1_hukou_raw", "d1_birthplace_raw", "d1_native_raw", "d1_residence_raw"]]
    f["in_S1"] = f.pilot_city & (f.period == "P2_plea") & f.leniency_proc
    f["in_S2"] = (f.d1_origin_status == "unresolved") | raw.isna().all(axis=1)
    return f


def main():
    rng = np.random.default_rng(paths.SEED)
    f = load_frame()
    print(f"frame: {len(f):,} docs")
    pop = f.stratum.value_counts()
    picks = []
    for st in sorted(pop.index):
        ids = rng.choice(f.loc[f.stratum == st, "doc_id"].to_numpy(), N_PER_STRATUM, replace=False)
        picks += [(i, st, pop[st] / N_PER_STRATUM) for i in ids]
    taken = {p[0] for p in picks}
    supp_pop = {}
    for st, col in (("S1_pilot_plea_leniency", "in_S1"), ("S2_origin_unresolved_or_missing", "in_S2")):
        pool = f.loc[f[col] & ~f.doc_id.isin(taken), "doc_id"].to_numpy()
        supp_pop[st] = int(f[col].sum())
        ids = rng.choice(pool, N_SUPP, replace=False)
        picks += [(i, st, supp_pop[st] / N_SUPP) for i in ids]
        taken |= set(ids)
    main_df = pd.DataFrame(picks, columns=["doc_id", "stratum", "weight"])
    # combined inclusion probability: main stratum draw, plus S1/S2 draw if the unit is in that population
    info = f.set_index("doc_id")
    p_main = main_df.doc_id.map(lambda i: N_PER_STRATUM / pop[info.at[i, "stratum"]])
    p1 = main_df.doc_id.map(lambda i: N_SUPP / supp_pop["S1_pilot_plea_leniency"] if info.at[i, "in_S1"] else 0)
    p2 = main_df.doc_id.map(lambda i: N_SUPP / supp_pop["S2_origin_unresolved_or_missing"] if info.at[i, "in_S2"] else 0)
    main_df["weight_combined"] = 1 / (1 - (1 - p_main) * (1 - p1) * (1 - p2))
    main_df = main_df.sample(frac=1, random_state=paths.SEED).reset_index(drop=True)
    main_df.insert(0, "ann_id", [f"a{i:03d}" for i in range(1, len(main_df) + 1)])

    pool = f.loc[~f.doc_id.isin(taken)]
    pilot = pool.sample(N_PILOT, random_state=paths.SEED)[["doc_id", "stratum"]].reset_index(drop=True)
    pilot.insert(0, "ann_id", [f"p{i:03d}" for i in range(1, N_PILOT + 1)])

    # recheck: proportional by stratum (10 strata x 30 -> 6 each), shuffled and renumbered
    rc = pd.concat([g.sample(round(N_RECHECK * len(g) / len(main_df)), random_state=paths.SEED)
                    for _, g in main_df.groupby("stratum")])
    rc = rc.sample(frac=1, random_state=paths.SEED + 1).reset_index(drop=True)
    rc.insert(0, "rid", [f"rid_{i:03d}" for i in range(1, len(rc) + 1)])

    ANN.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    main_df[["ann_id", "doc_id", "stratum", "weight", "weight_combined"]].to_csv(ANN / "sample_main.csv", index=False)
    pilot.to_csv(ANN / "sample_pilot.csv", index=False)
    rc[["rid"]].to_csv(ANN / "sample_recheck.csv", index=False)
    rc[["rid", "ann_id", "doc_id", "stratum"]].to_csv(PRIVATE / "remap.csv", index=False)

    strata = pd.DataFrame({"population": pop}).rename_axis("stratum").reset_index()
    strata = pd.concat([strata, pd.DataFrame({"stratum": list(supp_pop), "population": list(supp_pop.values())})])
    strata["drawn"] = strata.stratum.map(main_df.stratum.value_counts())
    strata["weight"] = strata.population / strata.drawn
    strata["recheck"] = strata.stratum.map(rc.stratum.value_counts())
    strata.to_csv(paths.TABLES / "T04_strata.csv", index=False)
    print(strata.to_string(index=False))
    assert main_df.doc_id.is_unique and not set(pilot.doc_id) & set(main_df.doc_id)
    assert set(rc.doc_id) <= set(main_df.doc_id) and len(rc) == N_RECHECK


if __name__ == "__main__":
    main()
