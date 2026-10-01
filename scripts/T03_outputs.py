"""T03: origin coverage & selection diagnostics, LLM queue, G1 figures 2013Q1-2019Q4 (no regressions)."""
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
from src.treatment import NATIONAL_DATE, PILOT_DATE  # noqa: E402

SPEEDY_DATE = pd.Timestamp("2014-08-26")
BASE_END = pd.Timestamp("2014-07-01")  # base period 2013Q1-2014Q2
T, FIG = paths.TABLES, paths.FIGURES
VLINES = (SPEEDY_DATE, PILOT_DATE, NATIONAL_DATE)
GROUPS = [((True, False), "Pilot, local", "-", "o", "black"), ((True, True), "Pilot, migrant", "-", "s", "0.55"),
          ((False, False), "Non-pilot, local", "--", "^", "black"), ((False, True), "Non-pilot, migrant", "--", "v", "0.55")]


def load():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con.execute(f"""
        SELECT f.*, t.judgment_date_parsed, t.pilot_city, m.judgment_year, s.universe
        FROM read_parquet('{paths.DATA / "feat_rules.parquet"}') f
        JOIN read_parquet('{paths.DATA / "treatment.parquet"}') t USING (doc_id)
        JOIN read_parquet('{paths.META}') m USING (doc_id)
        JOIN read_parquet('{paths.DATA / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_theft AND s.universe""").df()


def period(d):
    return np.select([d < BASE_END, (d >= SPEEDY_DATE) & (d < PILOT_DATE),
                      (d >= PILOT_DATE) & (d < NATIONAL_DATE), d >= NATIONAL_DATE],
                     ["0 base 2013Q1-2014Q2", "1 speedy pilot", "2 plea pilot", "3 national"], default="gap")


# ----------------------------------------------------------------------------- section 4
def diagnostics(th):
    th = th.copy()
    th["quarter"] = th.judgment_date_parsed.dt.to_period("Q").astype(str)
    th["has_city"] = th.d1_migrant_city.notna()
    th["has_v0"] = th.d1_migrant_v0.notna()
    th["has_prov"] = th.d1_migrant_prov.notna()
    yr = pd.concat([th.assign(judgment_year="all"), th]).groupby("judgment_year").agg(
        n=("doc_id", "size"), migrant_city=("has_city", "mean"), migrant_prov=("has_prov", "mean"),
        migrant_v0_T02=("has_v0", "mean"), migrant_city_nores=("d1_migrant_city_nores", lambda x: x.notna().mean()),
        migrant_city_share=("d1_migrant_city", lambda x: x.astype("float").mean()))
    yr = yr[yr.n >= 100]
    yr.to_csv(T / "T03_coverage_by_year.csv")
    print(yr.round(3).to_string())

    q = (th[th.judgment_year.between(2013, 2019)].groupby(["quarter", "pilot_city"])
         .agg(n=("doc_id", "size"), cov_city=("has_city", "mean"), cov_v0=("has_v0", "mean")).reset_index())
    q.to_csv(T / "T03_coverage_quarter_pilot.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, lab, ls, mk in [(True, "Pilot cities", "-", "o"), (False, "Non-pilot cities", "--", "^")]:
        g = q[q.pilot_city == p]
        x = pd.PeriodIndex(g.quarter, freq="Q").start_time
        ax.plot(x, g.cov_city, ls=ls, marker=mk, ms=4, color="black", label=f"{lab}: migrant_city")
        ax.plot(x, g.cov_v0, ls=ls, marker=mk, ms=3, color="0.6", label=f"{lab}: migrant_v0 (T02)")
    for d in VLINES:
        ax.axvline(d, color="0.3", lw=0.8, ls=":")
    ax.set_ylabel("Share of theft docs with origin defined")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "T03_coverage_by_group.png", dpi=150)
    plt.close(fig)
    # pilot minus non-pilot coverage by year, for the report
    th["year"] = th.judgment_date_parsed.dt.year
    cy = th[th.year.between(2013, 2019)].pivot_table(index="year", columns="pilot_city", values="has_city")
    cy["pilot_minus_nonpilot"] = cy[True] - cy[False]
    cy.to_csv(T / "T03_coverage_pilot_gap_by_year.csv")
    print(cy.round(3).to_string())

    st = pd.concat([
        th.d1_origin_status.fillna("no_origin_text").value_counts().rename("n_status"),
        th.d1_origin_status.fillna("no_origin_text").value_counts(normalize=True).rename("share_status")], axis=1)
    src = pd.concat([th.d1_origin_source.fillna("none").value_counts().rename("n_source"),
                     th.d1_origin_source.fillna("none").value_counts(normalize=True).rename("share_source")], axis=1)
    st.to_csv(T / "T03_origin_status.csv")
    src.to_csv(T / "T03_origin_source.csv")
    print(st.round(4).to_string(), "\n", src.round(4).to_string())

    both = th[th.d1_migrant_v0.notna() & th.d1_migrant_city.notna()]
    bp = th[th.d1_migrant_v0.notna() & th.d1_migrant_prov.notna()]
    agr = pd.DataFrame([
        {"compare": "migrant_city vs migrant_v0", "n_both": len(both),
         "agree": (both.d1_migrant_city == both.d1_migrant_v0).mean(),
         "city_T_v0_F": ((both.d1_migrant_city == True) & (both.d1_migrant_v0 == False)).mean(),  # noqa: E712
         "city_F_v0_T": ((both.d1_migrant_city == False) & (both.d1_migrant_v0 == True)).mean()},  # noqa: E712
        {"compare": "migrant_prov vs migrant_v0", "n_both": len(bp),
         "agree": (bp.d1_migrant_prov == bp.d1_migrant_v0).mean(),
         "city_T_v0_F": ((bp.d1_migrant_prov == True) & (bp.d1_migrant_v0 == False)).mean(),  # noqa: E712
         "city_F_v0_T": ((bp.d1_migrant_prov == False) & (bp.d1_migrant_v0 == True)).mean()}])  # noqa: E712
    agr.to_csv(T / "T03_agreement_v0.csv", index=False)
    print(agr.round(4).to_string(index=False))

    flags = pd.DataFrame([{
        "n_theft_docs": len(th), "dup_pseudonym_docs": int(th.dup_pseudonym.sum()),
        "dup_pseudonym_share": th.dup_pseudonym.mean(),
        "dup_pseudonym_share_multi_def": th.loc[th.n_defendants > 1, "dup_pseudonym"].mean(),
        "d1_name_repeated_docs": int(th.d1_name_repeated.fillna(False).sum()),
        "amt_outlier": int(th.amt_outlier.sum()), "d1_fine_outlier": int(th.d1_fine_outlier.fillna(False).sum()),
        "d1_outcomes_ok": th.d1_outcomes_ok.fillna(False).mean(),
        "speedy": th.speedy.mean(), "plea_formal": th.plea_formal.mean(), "jiejie": th.jiejie.mean(),
        "leniency_proc": th.leniency_proc.mean(), "plea_leniency_T02": th.plea_leniency.mean(),
        "jiejie_without_plea_formal": (th.jiejie & ~th.plea_formal).sum(),
        "d1_status_rsl": (th.d1_pretrial_status_at_judgment == "rsl").mean()}])
    flags.to_csv(T / "T03_flags.csv", index=False)
    print(flags.T.round(4).to_string())


def llm_queue(th):
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    F = con.execute(f"""SELECT doc_id, def_idx, origin_status, hukou_raw, birthplace_raw, native_raw, residence_raw
                        FROM read_parquet('{paths.DATA / "defendants_rules.parquet"}')""").df()
    F = F[F.doc_id.isin(th.doc_id)]
    raw_missing = F[["hukou_raw", "birthplace_raw", "native_raw", "residence_raw"]].isna().all(axis=1)
    reason = np.select([raw_missing, F.origin_status == "unresolved", F.origin_status == "ambiguous"],
                       ["no_origin_text", "unresolved", "ambiguous"], default="")
    q = F.assign(reason=reason)[lambda x: x.reason != ""][["doc_id", "def_idx", "reason"]]
    no_def = pd.DataFrame({"doc_id": th.loc[th.n_defendants == 0, "doc_id"], "def_idx": -1,
                           "reason": "no_defendant_parsed"})
    q = pd.concat([q, no_def]).sort_values(["doc_id", "def_idx"]).reset_index(drop=True)
    q.to_parquet(paths.DATA / "llm_queue_origin.parquet", index=False)
    summ = q.groupby("reason").agg(n_defendants=("doc_id", "size"), n_docs=("doc_id", "nunique"))
    summ.loc["total"] = [len(q), q.doc_id.nunique()]
    d1 = q[q.def_idx <= 0]
    summ["n_docs_first_defendant"] = d1.groupby("reason").doc_id.nunique()
    summ.loc["total", "n_docs_first_defendant"] = d1.doc_id.nunique()
    summ.to_csv(T / "T03_llm_queue_summary.csv")
    print(summ.to_string())


# ----------------------------------------------------------------------------- section 5
def g1_sample(th, mig):
    s = th[th.judgment_year.between(2013, 2019) & th[mig].notna() & th.d1_outcomes_ok.fillna(False)
           & th.judgment_date_parsed.notna()].copy()
    s["quarter"] = s.judgment_date_parsed.dt.to_period("Q").dt.start_time
    s["migrant"] = s[mig].astype(bool)
    s["probation"] = s.d1_probation.astype("float")
    st = s.d1_pretrial_status_at_judgment
    s["bail"] = (st == "取保").astype(float).where(st.notna())
    s["term"] = s.d1_term_months.where(s.d1_penalty_type.isin(["有期徒刑", "拘役"]))
    for c in ["speedy", "plea_formal", "leniency_proc"]:
        s[c] = s[c].astype(float)
    return s


OUTCOMES = ["speedy", "plea_formal", "leniency_proc", "probation", "bail", "term"]


def group_means(s):
    agg = {"n": ("doc_id", "size")}
    for c in OUTCOMES:
        agg[c] = (c, "mean")
        agg[f"n_{c}"] = (c, "count")
        agg[f"sd_{c}"] = (c, "std")
    return s.groupby(["quarter", "pilot_city", "migrant"]).agg(**agg).reset_index()


def lines(ax, gm, var):
    for (p, m), lab, ls, mk, col in GROUPS:
        g = gm[(gm.pilot_city == p) & (gm.migrant == m)].sort_values("quarter")
        ax.plot(g.quarter, g[var], ls=ls, marker=mk, ms=3, color=col, label=lab)
    for d in VLINES:
        ax.axvline(d, color="0.3", lw=0.8, ls=":")


def gaps(gm, var):
    w = gm.pivot_table(index=["quarter", "pilot_city"], columns="migrant",
                       values=[var, f"n_{var}", f"sd_{var}"]).reset_index()
    out = pd.DataFrame({"quarter": w["quarter"], "pilot_city": w["pilot_city"]})
    out["gap"] = w[(var, True)] - w[(var, False)]
    out["se"] = np.sqrt(w[(f"sd_{var}", True)] ** 2 / w[(f"n_{var}", True)]
                        + w[(f"sd_{var}", False)] ** 2 / w[(f"n_{var}", False)])
    return out


def plot_gaps(gm, fname, title_var):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (var, ylab) in zip(axes, [("probation", "Probation: migrant - local"),
                                      ("bail", "On bail at judgment: migrant - local"),
                                      ("term", "Term (months): migrant - local")]):
        g = gaps(gm, var)
        for p, lab, ls, mk in [(True, "Pilot cities", "-", "o"), (False, "Non-pilot cities", "--", "^")]:
            x = g[g.pilot_city == p].sort_values("quarter")
            ax.plot(x.quarter, x.gap, ls=ls, marker=mk, ms=3, color="black" if p else "0.45", label=lab)
            ax.fill_between(x.quarter, x.gap - 1.96 * x.se, x.gap + 1.96 * x.se,
                            color="0.2" if p else "0.6", alpha=0.15, lw=0)
        ax.axhline(0, color="0.6", lw=0.6)
        for d in VLINES:
            ax.axvline(d, color="0.3", lw=0.8, ls=":")
        ax.set_ylabel(ylab, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def period_gaps(s, label):
    s = s.assign(period=period(s.judgment_date_parsed))
    s = s[s.period != "gap"]
    rows = []
    for (per, p), g in s.groupby(["period", "pilot_city"]):
        for var in ["probation", "bail", "term", "leniency_proc"]:
            a, b = g.loc[g.migrant, var].dropna(), g.loc[~g.migrant, var].dropna()
            se = np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 1 and len(b) > 1 else np.nan
            rows.append({"definition": label, "period": per, "pilot_city": p, "outcome": var,
                         "mean_migrant": a.mean(), "mean_local": b.mean(), "gap": a.mean() - b.mean(),
                         "ci_low": a.mean() - b.mean() - 1.96 * se, "ci_high": a.mean() - b.mean() + 1.96 * se,
                         "n_migrant": len(a), "n_local": len(b)})
    return pd.DataFrame(rows)


def g1(th):
    s = g1_sample(th, "d1_migrant_city")
    print(f"G1 sample (migrant_city): {len(s):,}; migrant share {s.migrant.mean():.3f}; pilot {s.pilot_city.mean():.3f}")
    gm = group_means(s)
    gm.assign(definition="migrant_city").to_csv(T / "T03_group_means.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (var, ylab) in zip(axes, [("speedy", "Speedy procedure"), ("plea_formal", "Formal plea leniency (renzui renfa)"),
                                      ("leniency_proc", "Any: speedy / plea leniency / signed pledge")]):
        lines(ax, gm, var)
        ax.set_ylabel(ylab, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "T03_leniency_proc_by_group.png", dpi=150)
    plt.close(fig)
    for var, fname, ylab in [("probation", "T03_probation_by_group.png", "Probation rate (defendant 1)"),
                             ("bail", "T03_bail_by_group.png", "On bail at judgment (defendant 1)"),
                             ("term", "T03_term_by_group.png", "Term in months (fixed-term / criminal detention)")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        lines(ax, gm, var)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)
    plot_gaps(gm, "T03_gaps.png", "city")
    sp = g1_sample(th, "d1_migrant_prov")
    gmp = group_means(sp)
    gmp.assign(definition="migrant_prov").to_csv(T / "T03_group_means_prov.csv", index=False)
    plot_gaps(gmp, "T03_gaps_prov.png", "prov")
    pg = pd.concat([period_gaps(s, "migrant_city"), period_gaps(sp, "migrant_prov")])
    pg.to_csv(T / "T03_gap_periods.csv", index=False)
    show = pg[pg.definition == "migrant_city"].pivot_table(
        index=["outcome", "period"], columns="pilot_city", values="gap")
    print(show.round(3).to_string())
    # first stage readings
    s["year"] = s.judgment_date_parsed.dt.year
    print(s.pivot_table(index="year", columns=["pilot_city", "migrant"], values=["speedy", "leniency_proc"]).round(3).to_string())


def main():
    t0 = time.time()
    th = load()
    print(f"theft docs in universe: {len(th):,}")
    diagnostics(th)
    llm_queue(th)
    g1(th)
    print(f"TOTAL {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
