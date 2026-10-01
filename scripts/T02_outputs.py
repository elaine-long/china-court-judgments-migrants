"""T02 steps 5-6: comparison/spot-check tables and G1 raw descriptive figures (no regressions)."""
import sys
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import paths  # noqa: E402
from src.extract_rules import normalize  # noqa: E402
from src.io import load_texts  # noqa: E402
from src.segment import find_spans  # noqa: E402
from src.treatment import NATIONAL_DATE, PILOT_DATE  # noqa: E402

T, FIG = paths.TABLES, paths.FIGURES
FIG.mkdir(parents=True, exist_ok=True)


def load():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con.execute(f"""
        SELECT f.*, t.judgment_date_parsed, t.court_city, t.court_city_source, t.pilot_city, t.city_type,
               t.post_pilot, t.post_national, t.post, t.event_time_q, t.court_province_final,
               m.case_number, m.judgment_year, s.universe, o.sentence_months AS old_sentence_months
        FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        JOIN read_parquet('{paths.META}') m USING (doc_id)
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        LEFT JOIN read_parquet('{paths.OLD_FEATURES}') o USING (doc_id)""").df()


def old_vs_new(th):
    both = th[th.old_sentence_months.notna() & th.d1_term_months.notna()]
    o, n = both.old_sentence_months, both.d1_term_months
    rows = [{"subset": "all theft, both non-missing", "n": len(both),
             "agree_abs_lt_0.5": (abs(o - n) < 0.5).mean(), "old_eq_2x_new": (abs(o - 2 * n) < 0.5).mean(),
             "old_gt_new": (o > n + 0.5).mean(), "old_lt_new": (o < n - 0.5).mean()}]
    for yrs, lab in [(both.d1_probation.fillna(False), "probation"), (~both.d1_probation.fillna(False), "no probation")]:
        b = both[yrs]
        rows.append({"subset": lab, "n": len(b), "agree_abs_lt_0.5": (abs(b.old_sentence_months - b.d1_term_months) < 0.5).mean(),
                     "old_eq_2x_new": (abs(b.old_sentence_months - 2 * b.d1_term_months) < 0.5).mean(),
                     "old_gt_new": (b.old_sentence_months > b.d1_term_months + 0.5).mean(),
                     "old_lt_new": (b.old_sentence_months < b.d1_term_months - 0.5).mean()})
    rows.append({"subset": "missing: old only", "n": int((th.old_sentence_months.notna() & th.d1_term_months.isna()).sum())})
    rows.append({"subset": "missing: new only", "n": int((th.old_sentence_months.isna() & th.d1_term_months.notna()).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(T / "T02_old_vs_new.csv", index=False)
    print(out.round(4).to_string(index=False))


COV_FIELDS = ["d1_hukou_raw", "d1_birthplace_raw", "d1_native_raw", "d1_residence_raw", "d1_migrant_v0",
              "d1_pretrial_status_at_judgment", "d1_detained_date", "d1_arrested_date", "d1_bail_date",
              "d1_gender", "d1_birth_date", "d1_education", "d1_occupation_raw", "d1_term_months",
              "d1_penalty_type", "amt_yuan", "procedure"]
RATE_FIELDS = ["counsel", "plea_leniency", "duty_lawyer", "d1_probation", "d1_detained", "d1_arrested",
               "d1_bail", "d1_rsl", "d1_prior_record", "d1_recidivist", "has_header", "has_charge",
               "has_facts", "has_reasoning", "has_verdict", "has_fffd"]


def coverage(th):
    rows = []
    for y, g in [("all", th)] + list(th.groupby("judgment_year")):
        r = {"judgment_year": y, "n": len(g)}
        for c in COV_FIELDS:
            r[f"nonmiss_{c}"] = g[c].notna().mean()
        for c in RATE_FIELDS:
            r[f"rate_{c}"] = g[c].astype("float").mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out = out[(out.judgment_year == "all") | (out.n >= 100)]
    out.to_csv(T / "T02_coverage.csv", index=False)
    print(out.set_index("judgment_year").T.round(3).to_string())


def spotcheck(th):
    rng = np.random.default_rng(paths.SEED)
    picks = []
    for pilot in (True, False):
        for yr in (2015, 2018):
            pool = th[(th.pilot_city == pilot) & (th.judgment_year == yr)].doc_id.to_numpy()
            picks += [(pilot, yr, i) for i in rng.choice(pool, 10, replace=False)]
    ids = [p[2] for p in picks]
    txt = load_texts(ids, columns=["judgment"]).set_index("doc_id")["judgment"]
    cols = ["doc_id", "case_number", "court_city", "pilot_city", "judgment_year", "n_defendants", "charge_main",
            "d1_def_name", "d1_gender", "d1_age_at_judgment", "d1_hukou_raw", "d1_birthplace_raw", "d1_residence_raw",
            "d1_origin_province", "d1_migrant_v0", "d1_prior_record", "d1_recidivist", "d1_detained", "d1_detained_date",
            "d1_arrested", "d1_arrested_date", "d1_bail", "d1_bail_date", "d1_pretrial_status_at_judgment",
            "d1_penalty_type", "d1_term_months", "d1_probation", "d1_probation_months", "d1_fine_yuan",
            "plea_leniency", "procedure", "counsel", "counsel_type", "duty_lawyer", "amt_yuan", "amt_source",
            "spec_burglary", "spec_pickpocket", "surrender", "confession", "restitution", "forgiveness",
            "probation_denial_text"]
    sc = th.set_index("doc_id").loc[ids].reset_index()[cols]
    heads, verds = [], []
    for i in sc.doc_id:
        t = normalize(txt[i])
        sp = find_spans(t)
        heads.append(t[sp["header"][0]:sp["header"][1]][:500] if sp["header"] else t[:500])
        verds.append(t[sp["verdict"][0]:sp["verdict"][1]][:300] if sp["verdict"] else None)
    sc["header_500"], sc["verdict_300"] = heads, verds
    sc.to_csv(T / "T02_spotcheck_40.csv", index=False)


GROUPS = [((True, False), "Pilot, local", "-", "o"), ((True, True), "Pilot, migrant", "-", "s"),
          ((False, False), "Non-pilot, local", "--", "^"), ((False, True), "Non-pilot, migrant", "--", "v")]


def g1(th):
    s = th[th.universe & th.judgment_year.between(2014, 2019) & th.d1_migrant_v0.notna()
           & th.judgment_date_parsed.notna()].copy()
    s["quarter"] = s.judgment_date_parsed.dt.to_period("Q").dt.start_time
    s["migrant"] = s.d1_migrant_v0.astype(bool)
    s["plea"] = s.plea_leniency.astype(float)
    s["probation"] = s.d1_probation.astype("float")
    s["bail_at_judgment"] = (s.d1_pretrial_status_at_judgment == "取保").astype(float).where(
        s.d1_pretrial_status_at_judgment.notna())
    gm = (s.groupby(["quarter", "pilot_city", "migrant"])
          .agg(n=("doc_id", "size"), plea=("plea", "mean"), probation=("probation", "mean"),
               n_status=("bail_at_judgment", "count"), bail_at_judgment=("bail_at_judgment", "mean"))
          .reset_index())
    gm.to_csv(T / "T02_group_means.csv", index=False)
    print(f"G1 sample: {len(s):,} docs; migrant share {s.migrant.mean():.3f}; pilot share {s.pilot_city.mean():.3f}")

    def plot(var, fname, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (p, m), lab, ls, mk in GROUPS:
            g = gm[(gm.pilot_city == p) & (gm.migrant == m)].sort_values("quarter")
            ax.plot(g.quarter, g[var], ls=ls, marker=mk, ms=4, color="black" if not m else "0.5", label=lab)
        for d in (PILOT_DATE, NATIONAL_DATE):
            ax.axvline(d, color="0.3", lw=0.8, ls=":")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    plot("plea", "T02_plea_by_group.png", "Share with plea leniency")
    plot("probation", "T02_probation_by_group.png", "Probation rate (defendant 1)")
    plot("bail_at_judgment", "T02_bail_by_group.png", "Share on bail at judgment (defendant 1)")

    w = gm.pivot_table(index=["quarter", "pilot_city"], columns="migrant", values="probation").reset_index()
    w["gap"] = w[True] - w[False]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, lab, ls, mk in [(True, "Pilot cities", "-", "o"), (False, "Non-pilot cities", "--", "^")]:
        g = w[w.pilot_city == p].sort_values("quarter")
        ax.plot(g.quarter, g.gap, ls=ls, marker=mk, ms=4, color="black", label=lab)
    for d in (PILOT_DATE, NATIONAL_DATE):
        ax.axvline(d, color="0.3", lw=0.8, ls=":")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Probation rate: migrant minus local")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "T02_gap_by_group.png", dpi=150)
    plt.close(fig)

    # yearly readings for the report
    s["year"] = s.judgment_date_parsed.dt.year
    y = (s.groupby(["year", "pilot_city", "migrant"])
         .agg(n=("doc_id", "size"), plea=("plea", "mean"), probation=("probation", "mean"),
              bail=("bail_at_judgment", "mean")).reset_index())
    y.to_csv(T / "T02_group_means_yearly.csv", index=False)
    pv = y.pivot_table(index="year", columns=["pilot_city", "migrant"], values=["plea", "probation"])
    print(pv.round(3).to_string())
    # first stage around the pilot date in pilot cities (half-years)
    pc = s[s.pilot_city].copy()
    pc["half"] = pc.judgment_date_parsed.dt.to_period("Q")
    print(pc.groupby("half").plea.agg(["size", "mean"]).loc["2016Q1":"2017Q4"].round(3).to_string())


def balance(th):
    s = th[th.universe & th.judgment_year.between(2014, 2015)].copy()
    s["log_amt"] = np.log1p(s.amt_yuan)
    s["log_fine"] = np.log1p(s.d1_fine_yuan)
    s["d1_male"] = (s.d1_gender == "男").astype(float).where(s.d1_gender.notna())
    s["d1_bail_at_judgment"] = (s.d1_pretrial_status_at_judgment == "取保").astype(float).where(
        s.d1_pretrial_status_at_judgment.notna())
    s["proc_simple"] = (s.procedure == "简易").astype(float).where(s.procedure.notna())
    vars_ = ["d1_migrant_v0", "d1_probation", "d1_term_months", "log_fine", "log_amt", "d1_prior_record",
             "d1_recidivist", "d1_age_at_judgment", "d1_male", "d1_bail_at_judgment", "n_defendants", "counsel",
             "proc_simple", "spec_burglary", "spec_pickpocket", "spec_multiple", "surrender", "restitution",
             "forgiveness", "has_facts"]
    rows = []
    for v in vars_:
        r = {"variable": v}
        for g, gg in s.groupby("city_type"):
            x = gg[v].astype("float")
            r[f"mean_{g}"] = x.mean()
            r[f"n_{g}"] = int(x.notna().sum())
        rows.append(r)
    rows.append({"variable": "N_docs", **{f"n_{g}": len(gg) for g, gg in s.groupby("city_type")}})
    out = pd.DataFrame(rows)
    out.to_csv(T / "T02_pilot_balance.csv", index=False)
    print(out.round(3).to_string(index=False))


def main():
    t0 = time.time()
    df = load()
    th = df[df.in_theft & df.universe]
    print(f"theft docs {len(th):,}; robbery docs {int((df.in_robbery & df.universe).sum()):,}")
    old_vs_new(th)
    coverage(th)
    spotcheck(th)
    g1(th)
    balance(th)
    print(f"TOTAL {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
