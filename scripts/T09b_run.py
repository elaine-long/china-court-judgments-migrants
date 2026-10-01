"""T09b: main estimates on hybrid variables, PAP §5 robustness, PAP v1.2 exploratory analyses.

    python scripts/T09b_run.py main      # PAP §4, CRV1 + WCR 9,999 + Romano–Wolf, event studies
    python scripts/T09b_run.py robust    # PAP §5.2-5.9 + complete case / no dup_pseudonym / base without 2013
    python scripts/T09b_run.py explore   # PAP v1.2 (a)-(d), labelled exploratory
    python scripts/T09b_run.py summary   # T09b_summary_for_claude.csv

Main sample: data/analysis_hybrid.parquet (rule value; LLM fill only where rules are missing and models
A = B). No LLM calls; annotation/ is not read.
"""
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import estimate as E  # noqa: E402
from src import paths, plotstyle  # noqa: E402
from src.treatment import NATIONAL_DATE, PILOT_DATE  # noqa: E402

T, FIG, D = paths.TABLES, paths.FIGURES, paths.DATA
SPEEDY_DATE = pd.Timestamp("2014-08-26")
REPS = 9999
X = ["log_amt", "amt_missing", "spec_burglary", "spec_pickpocket", "spec_multiple", "spec_weapon",
     "prior_record", "recidivist", "surrender", "confession", "restitution", "forgiveness",
     "age", "age2", "age_missing", "male", "gender_missing", "multi_def", "prior_missing"]
OUTCOMES = [("probation", X), ("bail_at_judgment", X), ("log_term", X + ["n_charges"]), ("leniency_proc", X)]
PARAMS = ["pilot_P1_mig", "pilot_P2_mig"]
LABELS = {"pilot_P1": "Pilot × P1", "pilot_P2": "Pilot × P2", "pilot_P3": "Pilot × P3",
          "pilot_P1_mig": "Pilot × P1 × Migrant", "pilot_P2_mig": "Pilot × P2 × Migrant",
          "pilot_P3_mig": "Pilot × P3 × Migrant", "probation": "Probation", "bail_at_judgment": "Bail at judgment",
          "log_term": "log(1+term)", "leniency_proc": "Leniency procedure", "city_mig": "City × Migrant",
          "q_mig": "Quarter × Migrant"}


def load(name="analysis_hybrid"):
    """PAP v1.1 §7.1: every control with missing values is zero-filled with a missing indicator.
    prior_record / recidivist are missing only for LLM-filled documents without a parsed defendant
    paragraph (132 rows in the hybrid main sample), so they get one shared indicator here."""
    d = pd.read_parquet(D / f"{name}.parquet")
    d["prior_missing"] = d.prior_record.isna().astype(float)
    for c in ("prior_record", "recidivist"):
        d[c] = d[c].fillna(0.0)
    return d


def period_of(jdate, shift_years=0):
    s = pd.DateOffset(years=shift_years)
    return np.select([jdate < SPEEDY_DATE - s, jdate < PILOT_DATE - s, jdate < NATIONAL_DATE - s],
                     ["P0", "P1", "P2"], default="P3")


def fit_all(d, label, outcomes=OUTCOMES, params=PARAMS, weights=None):
    """Main specification on sample d; returns long rows for the robustness table."""
    d = E.add_design(d)
    rows = []
    for y, ctrl in outcomes:
        ctrl = [c for c in ctrl if c in d and d[c].std() > 0]
        if weights is None:
            f = E.triple_diff(d, y, controls=ctrl)
        else:
            rhs = [f"pilot_{p}" for p in E.PERIODS] + [f"pilot_{p}_mig" for p in E.PERIODS] + ctrl
            f = pf.feols(f"{y} ~ {' + '.join(rhs)} | city_mig + q_mig", data=d, vcov={"CRV1": "city_id"},
                         weights=weights)
        for p in params:
            if p in f.coef().index:
                rows.append({"check": label, **E.coef_row(f, p)})
    return rows


# ----------------------------------------------------------------------------- §1 main
def main():
    t0 = time.time()
    d = E.add_design(load())
    print(f"main sample N = {len(d):,}; clusters {d.city.nunique()}; migrant share {d.migrant.mean():.3f}; "
          f"LLM-filled migrant {int((d.migrant_source == 'llm_AB').sum()):,}")
    fits, rows = {}, []
    for y, ctrl in OUTCOMES:
        f = E.triple_diff(d, y, controls=ctrl)
        fits[y] = f
        for p in ["pilot_P1", "pilot_P2", "pilot_P3", "pilot_P1_mig", "pilot_P2_mig", "pilot_P3_mig"]:
            rows.append({**E.coef_row(f, p), "mean_y": d[y].mean()})
    res = pd.DataFrame(rows)
    wb = [{"outcome": y, "param": p, "p_wild": E.wild_bootstrap(fits[y], p, reps=REPS)} for y, _ in OUTCOMES for p in PARAMS]
    res = res.merge(pd.DataFrame(wb), on=["outcome", "param"], how="left")
    rw = E.romano_wolf([(fits[y], p) for y, _ in OUTCOMES for p in PARAMS], reps=REPS)
    res = res.merge(rw[["outcome", "param", "p_romano_wolf"]], on=["outcome", "param"], how="left")
    res["boot_reps"] = REPS
    res.to_csv(T / "T09b_main.csv", index=False)
    print(res[res.param.isin(PARAMS)][["outcome", "param", "coef", "se", "p", "p_wild", "p_romano_wolf", "N"]]
          .round(4).to_string(index=False))
    tex = pf.etable([fits[y] for y, _ in OUTCOMES], type="tex", keep=["pilot_P1", "pilot_P2", "pilot_P3"],
                    labels=LABELS, notes="Hybrid variables (rules; LLM fills where rules are missing and two "
                                         "models agree). SEs clustered by court city. Controls per PAP §4.")
    (T / "T09b_main.tex").write_text(tex, encoding="utf-8")

    plotstyle.use()
    import matplotlib.pyplot as plt
    es = []
    for y, ctrl in OUTCOMES:
        _, tab = E.event_study(d, y, base="2014Q2", window=("2013Q1", "2018Q3"), controls=ctrl)
        tab["outcome"] = y
        es.append(tab)
        fig, ax = plt.subplots(figsize=plotstyle.SIZE_2COL)
        x = pd.PeriodIndex(tab.quarter, freq="Q").start_time
        ax.errorbar(x, tab.beta2, yerr=1.96 * tab.se2, color="black", marker="o", ls="-", capsize=0, lw=0.9)
        ax.axhline(0, color="0.6", lw=0.5)
        plotstyle.vlines(ax, (SPEEDY_DATE, PILOT_DATE))
        ax.set_ylabel(f"Pilot × Migrant × quarter: {LABELS[y]}")
        plotstyle.save(fig, FIG / f"T09b_es_{y}.pdf")
    pd.concat(es).to_csv(T / "T09b_es.csv", index=False)
    # first stage by group
    fig, ax = plt.subplots(figsize=plotstyle.SIZE_2COL)
    tabs = []
    for k, (m, lab) in enumerate(((0, "Local"), (1, "Migrant"))):
        _, tab = E.event_study_simple(d[d.migrant == m], "leniency_proc")
        tab["group"] = lab
        tabs.append(tab)
        x = pd.PeriodIndex(tab.quarter, freq="Q").start_time + pd.Timedelta(days=15 * k)
        col, ls, mk = plotstyle.SERIES[k]
        ax.errorbar(x, tab.beta, yerr=1.96 * tab.se, color=col, ls=ls, marker=mk, capsize=0, lw=0.9, label=lab)
    plotstyle.vlines(ax, (SPEEDY_DATE, PILOT_DATE))
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylabel("Pilot × quarter: leniency procedure")
    ax.legend()
    plotstyle.save(fig, FIG / "T09b_es_firststage.pdf")
    pd.concat(tabs).to_csv(T / "T09b_es_firststage.csv", index=False)
    print(f"main done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- §2 robustness
def ipw_weights(full):
    """1 / P(migrant defined | X, city, quarter): logit with city and quarter FE, fitted on all cases."""
    f = full.copy()
    f["defined"] = f.migrant.notna().astype(float)
    m = pf.feglm(f"defined ~ {' + '.join(X)} | city + q", data=f, family="logit")
    f["p_def"] = np.clip(m.predict(newdata=f, type="response"), 0.02, 1.0)
    return f.loc[f.migrant.notna(), ["doc_id", "p_def"]]


def high_coverage_courts(full, thr=0.5):
    pre = full[full.jdate < pd.Timestamp("2016-01-01")]
    cov = pre.groupby("court_name").migrant.apply(lambda s: s.notna().mean())
    return set(cov[cov >= thr].index)


def double_selection(d, y, ctrl, target_terms):
    """Belloni-Chernozhukov-Hansen double selection after partialling out the fixed effects.

    Candidates: the PAP controls, their pairwise interactions and further rule variables. Lasso (5-fold CV)
    of the outcome and of each treatment term on the candidates (all FE-demeaned); the union of selected
    controls enters the main specification. Returns (fit, selected).
    """
    from sklearn.linear_model import LassoCV
    base = list(dict.fromkeys(c for c in ctrl if d[c].std() > 0))
    extra = [c for c in ("attempted", "accomplice", "n_charges", "has_facts")
             if c in d and c not in base and d[c].std() > 0]
    cand = d[base + extra].astype(float).copy()
    inter = {}
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            v = cand[a] * cand[b]
            if v.std() > 0:
                inter[f"{a}__x__{b}"] = v
    cand = pd.concat([cand, pd.DataFrame(inter)], axis=1)
    dd = d[["city_mig", "q_mig", y] + target_terms].copy()
    dd = pd.concat([dd, cand], axis=1).dropna(subset=[y])
    cols = [y] + target_terms + list(cand.columns)
    dm = pf.estimation.demean(dd[cols].to_numpy(float),
                              dd[["city_mig", "q_mig"]].apply(lambda s: s.astype("category").cat.codes).to_numpy(),
                              np.ones(len(dd)))[0]
    dm = pd.DataFrame(dm, columns=cols, index=dd.index)
    rng = np.random.default_rng(E.SEED)
    sub = dm.iloc[rng.choice(len(dm), min(120_000, len(dm)), replace=False)]
    Z = (sub[cand.columns] - sub[cand.columns].mean()) / sub[cand.columns].std().replace(0, 1)
    sel = set()
    for target in [y] + target_terms:
        las = LassoCV(cv=5, alphas=30, random_state=E.SEED, max_iter=5000).fit(Z, sub[target])
        sel |= {c for c, b in zip(cand.columns, las.coef_) if abs(b) > 1e-8}
    selected = sorted(sel)
    d2 = d.loc[dd.index].copy()
    for c in selected:
        d2[c] = cand.loc[dd.index, c]
    return E.triple_diff(d2, y, controls=selected), selected


def dml_plr(d, y, ctrl, target):
    """DoubleML partially linear model for one treatment term, FE partialled out first, clustered by city."""
    import doubleml as dml
    from sklearn.linear_model import LassoCV
    others = [f"pilot_{p}" for p in E.PERIODS] + [f"pilot_{p}_mig" for p in E.PERIODS]
    xs = [c for c in ctrl if d[c].std() > 0] + [o for o in others if o != target]
    dd = d[["city_mig", "q_mig", "city_id", y, target] + xs].dropna(subset=[y])
    cols = [y, target] + xs
    dm = pf.estimation.demean(dd[cols].to_numpy(float),
                              dd[["city_mig", "q_mig"]].apply(lambda s: s.astype("category").cat.codes).to_numpy(),
                              np.ones(len(dd)))[0]
    df = pd.DataFrame(dm, columns=cols)
    df["city_id"] = dd.city_id.to_numpy()
    data = dml.DoubleMLData(df, y_col=y, d_cols=target, x_cols=xs, cluster_cols="city_id")
    m = dml.DoubleMLPLR(data, LassoCV(cv=3, alphas=20, max_iter=3000), LassoCV(cv=3, alphas=20, max_iter=3000),
                        n_folds=5)
    np.random.seed(E.SEED)
    m.fit()
    return float(m.coef[0]), float(m.se[0]), float(m.pval[0]), len(df)


def robbery_panel():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from T09a_build_panel import build
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    df = con.execute(f"""
        SELECT f.*, t.judgment_date_parsed AS jdate, t.pilot_city, t.court_city, t.court_province_final,
               m.court_name, m.case_number, m.doc_kind, s.universe
        FROM read_parquet('{D / "feat_rules.parquet"}') f
        JOIN read_parquet('{D / "treatment.parquet"}') t USING (doc_id)
        JOIN read_parquet('{paths.META}') m USING (doc_id)
        JOIN read_parquet('{D / "sample_flags.parquet"}') s USING (doc_id)
        WHERE f.in_robbery AND NOT f.in_theft AND s.universe AND m.doc_kind = '刑初'
          AND t.judgment_date_parsed BETWEEN DATE '2013-01-01' AND DATE '2019-12-31'""").df()
    p = build(df)
    return p[p.outcomes_ok & p.city.notna() & p.migrant.notna()]


def robust():
    t0 = time.time()
    d = load()
    full = load("analysis_hybrid_full")
    rows = fit_all(d, "0 main (hybrid)")
    # 5.2 control group: pilot cities vs non-pilot provincial capitals / sub-provincial cities
    rows += fit_all(d[(d.pilot == 1) | (d.capital_or_subprov == 1)], "5.2 controls = capitals/sub-provincial")
    # 5.3 migrant definitions (rule-based alternatives; LLM fills exist for the main definition only)
    for alt in ("migrant_city", "migrant_prov"):
        a = full[full[alt].notna()].copy()
        a["migrant"] = a[alt]
        rows += fit_all(a, f"5.3 migrant = {alt}")
    # 5.4 courts with pre-2016 coverage >= 50%
    hc = high_coverage_courts(full)
    rows += fit_all(d[d.court_name.isin(hc)], f"5.4 high-coverage courts (n={len(hc)})")
    # 5.5 IPW
    w = ipw_weights(full)
    dw = d.merge(w, on="doc_id", how="left")
    dw["ipw"] = 1 / dw.p_def
    rows += fit_all(dw[dw.ipw.notna()], "5.5 IPW (defined)", weights="ipw")
    print(f"  5.2-5.5 done {time.time() - t0:.0f}s", flush=True)
    # 5.6 double-selection lasso and DML
    dd = E.add_design(d)
    sel_log = {}
    for y, ctrl in OUTCOMES:
        f, sel = double_selection(dd, y, ctrl, ["pilot_P1_mig", "pilot_P2_mig"])
        sel_log[y] = len(sel)
        for p in PARAMS:
            rows.append({"check": "5.6 double-selection lasso", **E.coef_row(f, p), "note": f"{len(sel)} controls"})
        for p in PARAMS:
            c, s, pv, n = dml_plr(dd, y, ctrl, p)
            rows.append({"check": "5.6 DML partially linear (Lasso)", "outcome": y, "param": p, "coef": c, "se": s,
                         "t": c / s, "p": pv, "N": n, "n_clusters": dd.city.nunique()})
    print(f"  5.6 done {time.time() - t0:.0f}s; selected controls {sel_log}", flush=True)
    # 5.7 placebos
    rob = robbery_panel()
    rows += fit_all(rob, f"5.7 placebo: robbery (rules), n={len(rob)}")
    sh = d.copy()
    sh["period"] = period_of(sh.jdate, shift_years=1)
    rows += fit_all(sh, "5.7 placebo: reform dates one year earlier")
    # 5.8 Callaway-Sant'Anna on the migrant-local gap (city × quarter), vs TWFE
    import contextlib
    import io
    cs = d.copy()
    cs["qnum"] = cs.jdate.dt.year * 4 + cs.jdate.dt.quarter
    first = {c: (2016 * 4 + 4 if p == 1 else 2018 * 4 + 4) for c, p in cs.groupby("city").pilot.first().items()}
    cs_rows = []
    for y, _ in OUTCOMES:
        with contextlib.redirect_stdout(io.StringIO()):
            tab, att, se = E.csdid_wrapper(cs, y, first)
        tab["outcome"] = y
        cs_rows.append(tab)
        rows.append({"check": "5.8 Callaway-Sant'Anna (gap, dynamic overall)", "outcome": y,
                     "param": "ATT_overall (≈ P2 cohort)", "coef": att, "se": se, "t": att / se,
                     "p": float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(att / se)))), "N": len(cs)})
    pd.concat(cs_rows).to_csv(T / "T09b_csdid_dynamic.csv", index=False)
    # 5.9 rules variables
    rows += fit_all(load("analysis_rules"), "5.9 rule variables")
    # extra
    miss = ["amt_missing", "age_missing", "gender_missing"]
    cc = d[(d[miss] == 0).all(axis=1)]
    rows += fit_all(cc, "extra: complete cases", outcomes=[(y, [c for c in c_ if c not in miss]) for y, c_ in OUTCOMES])
    rows += fit_all(d[~d.dup_pseudonym], "extra: exclude dup_pseudonym")
    rows += fit_all(d[d.jdate >= pd.Timestamp("2014-01-01")], "extra: base period without 2013")
    out = pd.DataFrame(rows)
    out.to_csv(T / "T09b_robustness.csv", index=False)
    piv = out.assign(est=out.apply(lambda r: f"{r.coef:.4f} ({r.se:.4f})" + ("*" if r.p < 0.05 else ""), axis=1)) \
        .pivot_table(index="check", columns=["outcome", "param"], values="est", aggfunc="first")
    (T / "T09b_robustness.tex").write_text(
        piv.to_latex(escape=True, caption="Robustness (PAP §5). Coefficient (clustered SE); * p<0.05 (CRV1).",
                     label="tab:robust"), encoding="utf-8")
    print(out[["check", "outcome", "param", "coef", "se", "p", "N"]].round(4).to_string(index=False))
    print(f"robust done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- §3 exploratory (v1.2)
def pretrial_vars(ids):
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.register("ids", pd.DataFrame({"doc_id": list(ids)}))
    return con.execute(f"""SELECT doc_id, d1_detained::DOUBLE AS ever_detained, d1_bail::DOUBLE AS ever_bail,
            attempted::DOUBLE AS attempted
        FROM read_parquet('{D / "feat_rules.parquet"}') WHERE doc_id IN (SELECT doc_id FROM ids)""").df()


def gelbach(d, y, groups, base_fe="city + q"):
    """Gelbach (2016): δ_base − δ_full = Σ_k Γ_k'β_k, Γ_k from regressing each covariate on migrant (same FE)."""
    dd = d.dropna(subset=[y])
    # covariates constant in this sample (e.g. prior_missing in P0) are dropped from their group
    groups = {g: [c for c in cols if dd[c].std() > 0] for g, cols in groups.items()}
    allx = [c for g in groups.values() for c in g]
    fb = pf.feols(f"{y} ~ migrant | {base_fe}", data=dd, vcov={"CRV1": "city_id"})
    ff = pf.feols(f"{y} ~ migrant + {' + '.join(allx)} | {base_fe}", data=dd, vcov={"CRV1": "city_id"})
    bfull = ff.coef()
    contrib = {}
    for g, cols in groups.items():
        s = 0.0
        for c in cols:
            gam = pf.feols(f"{c} ~ migrant | {base_fe}", data=dd).coef()["migrant"]
            s += gam * bfull[c]
        contrib[g] = s
    return fb.coef()["migrant"], ff.coef()["migrant"], contrib, len(dd)


def explore():
    t0 = time.time()
    d = load()
    rows = []
    # (a) first stage dynamics: P2 split into P2a (2016Q4-2017Q4) and P2b (2018Q1-2018Q3)
    a = d.copy()
    a["period"] = np.where((a.period == "P2") & (a.jdate < pd.Timestamp("2018-01-01")), "P2a",
                           np.where(a.period == "P2", "P2b", a.period))
    a = E.add_design(a)
    for p in ("P1", "P2a", "P2b", "P3"):
        a[f"pilot_{p}"] = (a.pilot * (a.period == p)).astype(float)
        a[f"pilot_{p}_mig"] = a[f"pilot_{p}"] * a.migrant
    rhs = [f"pilot_{p}" for p in ("P1", "P2a", "P2b", "P3")] + [f"pilot_{p}_mig" for p in ("P1", "P2a", "P2b", "P3")] + X
    for y in ("leniency_proc", "speedy", "plea_formal"):
        f = pf.feols(f"{y} ~ {' + '.join(rhs)} | city_mig + q_mig", data=a, vcov={"CRV1": "city_id"})
        for p in ("pilot_P1_mig", "pilot_P2a_mig", "pilot_P2b_mig", "pilot_P1", "pilot_P2a", "pilot_P2b"):
            rows.append({"analysis": "(a) P2 split", **E.coef_row(f, p)})
    # (b) Pilot × P × Migrant × local_residence: β2 for migrants with / without a local residence
    b = d[d.local_residence.notna()].copy()
    b["grp"] = np.select([b.migrant == 0, b.local_residence == 1], ["local", "mig_res"], default="mig_nores")
    b["city_mig"] = b.city.astype(str) + "_" + b.grp
    b["q_mig"] = b.q.astype(str) + "_" + b.grp
    b["city_id"] = b.city.astype("category").cat.codes
    rhs = []
    for p in E.PERIODS:
        b[f"pilot_{p}"] = (b.pilot * (b.period == p)).astype(float)
        for g in ("mig_res", "mig_nores"):
            b[f"pilot_{p}_{g}"] = b[f"pilot_{p}"] * (b.grp == g)
        rhs += [f"pilot_{p}", f"pilot_{p}_mig_res", f"pilot_{p}_mig_nores"]
    for y, ctrl in OUTCOMES:
        f = pf.feols(f"{y} ~ {' + '.join(rhs + ctrl)} | city_mig + q_mig", data=b, vcov={"CRV1": "city_id"})
        for p in ("pilot_P1_mig_res", "pilot_P1_mig_nores", "pilot_P2_mig_res", "pilot_P2_mig_nores"):
            rows.append({"analysis": "(b) by local residence", **E.coef_row(f, p)})
        # equality test β2(res) = β2(nores) for P2
        try:
            wt = f.wald_test(R=np.array([[1.0 if c == "pilot_P2_mig_res" else -1.0 if c == "pilot_P2_mig_nores" else 0.0
                                          for c in f.coef().index]]))
            rows.append({"analysis": "(b) by local residence", "outcome": y, "param": "P2: res = nores (Wald p)",
                         "p": float(np.asarray(wt["pvalue"]).ravel()[0]), "N": f._N})
        except Exception as e:  # noqa: BLE001
            rows.append({"analysis": "(b) by local residence", "outcome": y, "param": "P2 Wald failed: " + type(e).__name__})
    print(f"  (a)(b) done {time.time() - t0:.0f}s", flush=True)
    # (c) P0 gap decomposition (Gelbach), outcomes probation and bail
    c = d[d.period == "P0"].merge(pretrial_vars(d.doc_id), on="doc_id", how="left")
    # PAP v1.1: missing controls zero-filled with an indicator (pretrial / attempted are missing for the
    # same LLM-filled docs without a parsed defendant paragraph as prior_record -> prior_missing)
    for col in ("ever_detained", "ever_bail", "attempted"):
        c[col] = c[col].fillna(0.0)
    c["city_id"] = c.city.astype("category").cat.codes
    c["lr_missing"] = c.local_residence.isna().astype(float)
    c["lr"] = c.local_residence.fillna(0)
    groups = {"case": ["log_amt", "amt_missing", "spec_burglary", "spec_pickpocket", "spec_multiple", "spec_weapon",
                       "surrender", "confession", "restitution", "forgiveness", "multi_def", "attempted"],
              "prior": ["prior_record", "recidivist"],
              "pretrial": ["ever_detained", "ever_bail", "prior_missing"],
              "local_residence": ["lr", "lr_missing"],
              "other (age, sex)": ["age", "age2", "age_missing", "male", "gender_missing"]}
    dec = []
    for y in ("probation", "bail_at_judgment"):
        base, fullc, contrib, n = gelbach(c, y, groups)
        dec.append({"outcome": y, "component": "raw gap (city+quarter FE)", "value": base, "N": n})
        for g, v in contrib.items():
            dec.append({"outcome": y, "component": f"explained: {g}", "value": v, "share_of_raw": v / base})
        dec.append({"outcome": y, "component": "unexplained (full-model gap)", "value": fullc, "share_of_raw": fullc / base})
        # DML conditional gap
        import doubleml as dml
        from sklearn.linear_model import LassoCV
        xs = [x for g in groups.values() for x in g if c[x].std() > 0]
        cc = c.dropna(subset=[y])
        dm = pf.estimation.demean(cc[[y, "migrant"] + xs].to_numpy(float),
                                  cc[["city", "q"]].apply(lambda s: s.astype("category").cat.codes).to_numpy(),
                                  np.ones(len(cc)))[0]
        df = pd.DataFrame(dm, columns=[y, "migrant"] + xs)
        df["city_id"] = cc.city_id.to_numpy()
        np.random.seed(E.SEED)
        m = dml.DoubleMLPLR(dml.DoubleMLData(df, y_col=y, d_cols="migrant", x_cols=xs, cluster_cols="city_id"),
                            LassoCV(cv=3, alphas=20), LassoCV(cv=3, alphas=20), n_folds=5)
        m.fit()
        dec.append({"outcome": y, "component": "DML conditional gap (Lasso)", "value": float(m.coef[0]),
                    "se": float(m.se[0]), "N": len(df)})
    dec = pd.DataFrame(dec)
    dec.to_csv(T / "T09b_explore_gelbach.csv", index=False)
    print(dec.round(4).to_string(index=False))
    # (d) MDE for β2_P2 (80% power, 5% two-sided) vs the P0 gap
    from scipy.stats import norm
    main_res = pd.read_csv(T / "T09b_main.csv")
    p0 = d[d.period == "P0"]
    mde = []
    for y, _ in OUTCOMES:
        se = main_res[(main_res.outcome == y) & (main_res.param == "pilot_P2_mig")].se.item()
        gap = p0.loc[p0.migrant == 1, y].mean() - p0.loc[p0.migrant == 0, y].mean()
        m_ = (norm.ppf(0.975) + norm.ppf(0.8)) * se
        mde.append({"outcome": y, "se_beta2_P2": se, "MDE_80pct": m_, "P0_raw_gap": gap,
                    "MDE_over_abs_gap": m_ / abs(gap) if gap else np.nan})
    mde = pd.DataFrame(mde)
    mde.to_csv(T / "T09b_explore_mde.csv", index=False)
    print(mde.round(4).to_string(index=False))
    out = pd.DataFrame(rows)
    out["label"] = "exploratory"
    out.to_csv(T / "T09b_explore.csv", index=False)
    print(out[["analysis", "outcome", "param", "coef", "se", "p", "N"]].round(4).to_string(index=False))
    print(f"explore done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- summary
def summary():
    m = pd.read_csv(T / "T09b_main.csv")
    r = pd.read_csv(T / "T09b_robustness.csv")
    r = r[r.check != "0 main (hybrid)"]
    rows = []
    for (y, p), g in r[r.param.isin(PARAMS)].groupby(["outcome", "param"]):
        mm = m[(m.outcome == y) & (m.param == p)].iloc[0]
        rows.append({"outcome": y, "param": p, "main_coef": mm.coef, "main_se": mm.se, "main_p": mm.p,
                     "main_p_wild": mm.p_wild, "main_p_rw": mm.p_romano_wolf,
                     "robust_min": g.coef.min(), "robust_max": g.coef.max(), "n_checks": len(g),
                     "n_sig_05": int((g.p < 0.05).sum()), "checks_sig": "; ".join(g.loc[g.p < 0.05, "check"])})
    s = pd.DataFrame(rows)
    s.to_csv(T / "T09b_summary_for_claude.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    {"main": main, "robust": robust, "explore": explore, "summary": summary}[sys.argv[1]]()
