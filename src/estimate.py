"""Estimation module (PAP v1 §4-6), built on pyfixest.

Data contract (one row per case, first defendant):
    city     court prefecture (cluster and fixed effect)
    q        judgment quarter as a string '2016Q4'
    pilot    1 if the court is in one of the 18 pilot cities
    period   'P0' | 'P1' | 'P2' | 'P3' (PAP §2 date cut-offs)
    migrant  1 / 0 (migrant_city_nores)
"""
import numpy as np
import pandas as pd
import pyfixest as pf

PERIODS = ["P1", "P2", "P3"]
SEED = 20260929


def add_design(df):
    """Add the treatment interactions and the Migrant-interacted fixed-effect identifiers."""
    d = df.copy()
    for p in PERIODS:
        d[f"pilot_{p}"] = (d["pilot"] * (d["period"] == p)).astype(float)
        d[f"pilot_{p}_mig"] = d[f"pilot_{p}"] * d["migrant"]
    # integer cluster id: the wildboottest numba kernels cannot type object (string) arrays
    d["city_id"] = d["city"].astype("category").cat.codes.astype("int64")
    d["city_mig"] = d["city"].astype(str) + "_" + d["migrant"].astype(int).astype(str)
    d["q_mig"] = d["q"].astype(str) + "_" + d["migrant"].astype(int).astype(str)
    return d


def triple_diff(df, y, controls=(), cluster="city_id"):
    """PAP §4 main regression.

    y = Σ_p [β1_p Pilot·P_p + β2_p Pilot·P_p·Migrant] + γ'X + α_c×Migrant + δ_q×Migrant + α_c + δ_q + ε
    (α_c, δ_q are nested in the Migrant-interacted effects). Returns a pyfixest Feols object.
    """
    d = df if "city_mig" in df else add_design(df)
    rhs = [f"pilot_{p}" for p in PERIODS] + [f"pilot_{p}_mig" for p in PERIODS] + list(controls)
    return pf.feols(f"{y} ~ {' + '.join(rhs)} | city_mig + q_mig", data=d, vcov={"CRV1": cluster})


def _qkey(q):
    return str(q).replace("Q", "q")


def event_study(df, y, base="2014Q2", window=("2013Q1", "2018Q3"), controls=(), cluster="city_id"):
    """Quarterly event study: Pilot×1[q=k] and Pilot×1[q=k]×Migrant for every k in the window except base.

    Returns (fit, table) with table columns quarter, beta1, se1, beta2, se2 (base quarter = 0).
    """
    d = df[(df["q"] >= window[0]) & (df["q"] <= window[1])].copy()
    d = d if "city_mig" in d else add_design(d)
    qs = sorted(d["q"].unique())
    rhs = []
    for k in qs:
        if k == base:
            continue
        d[f"es1_{_qkey(k)}"] = ((d["q"] == k) & (d["pilot"] == 1)).astype(float)
        d[f"es2_{_qkey(k)}"] = d[f"es1_{_qkey(k)}"] * d["migrant"]
        rhs += [f"es1_{_qkey(k)}", f"es2_{_qkey(k)}"]
    fit = pf.feols(f"{y} ~ {' + '.join(rhs + list(controls))} | city_mig + q_mig", data=d, vcov={"CRV1": cluster})
    co, se = fit.coef(), fit.se()
    rows = []
    for k in qs:
        if k == base:
            rows.append({"quarter": k, "beta1": 0.0, "se1": 0.0, "beta2": 0.0, "se2": 0.0})
            continue
        rows.append({"quarter": k, "beta1": co.get(f"es1_{_qkey(k)}", np.nan), "se1": se.get(f"es1_{_qkey(k)}", np.nan),
                     "beta2": co.get(f"es2_{_qkey(k)}", np.nan), "se2": se.get(f"es2_{_qkey(k)}", np.nan)})
    return fit, pd.DataFrame(rows)


def event_study_simple(df, y, base="2014Q2", window=("2013Q1", "2018Q3"), cluster="city_id"):
    """Two-way event study Pilot×1[q=k] with city and quarter FE (used for the first stage by group)."""
    d = df[(df["q"] >= window[0]) & (df["q"] <= window[1])].copy()
    if "city_id" not in d:
        d["city_id"] = d["city"].astype("category").cat.codes.astype("int64")
    qs = sorted(d["q"].unique())
    rhs = []
    for k in qs:
        if k != base:
            d[f"es_{_qkey(k)}"] = ((d["q"] == k) & (d["pilot"] == 1)).astype(float)
            rhs.append(f"es_{_qkey(k)}")
    fit = pf.feols(f"{y} ~ {' + '.join(rhs)} | city + q", data=d, vcov={"CRV1": cluster})
    co, se = fit.coef(), fit.se()
    tab = pd.DataFrame([{"quarter": k, "beta": 0.0 if k == base else co.get(f"es_{_qkey(k)}", np.nan),
                         "se": 0.0 if k == base else se.get(f"es_{_qkey(k)}", np.nan)} for k in qs])
    return fit, tab


def csdid_wrapper(df, y, first_treated, unit="city", time="qnum", gap=True, control_group="notyettreated",
                  est_method="reg", seed=SEED):
    """Callaway–Sant'Anna on a city × quarter panel.

    Cases are aggregated to city-quarter cells. With gap=True the cell outcome is the migrant-minus-local
    mean (the triple-difference analogue); otherwise the cell mean. `first_treated` maps unit -> first
    treated time index (pilot cities: 2016Q4; others: 2018Q4). Returns the dynamic aggregation object.
    """
    from csdid.att_gt import ATTgt
    g = df.groupby([unit, time, "migrant"])[y].mean().unstack("migrant")
    if gap:
        cell = (g[1] - g[0]).rename("yy").dropna().reset_index()
    else:
        cell = df.groupby([unit, time])[y].mean().rename("yy").reset_index()
    cell["gvar"] = cell[unit].map(first_treated).fillna(0).astype(int)
    cell["uid"] = cell[unit].astype("category").cat.codes + 1
    np.random.seed(seed)
    att = ATTgt(yname="yy", tname=time, idname="uid", gname="gvar", data=cell, control_group=control_group,
                allow_unbalanced_panel=True, biters=999)
    att.fit(est_method=est_method, bstrap=True)
    res = att.aggte(typec="dynamic", na_rm=True).atte
    tab = pd.DataFrame({"event_time": res["egt"], "att": res["att_egt"],
                        "se": np.asarray(res["se_egt"]).ravel()})
    return tab, float(res["overall_att"]), float(np.asarray(res["overall_se"]).ravel()[0])


def _wcr(fit, param, reps, seed, chunk=64):
    """Wild cluster restricted bootstrap (Rademacher, H0: β_param = 0) on the FE-demeaned data (FWL).

    pyfixest's own .wildboottest() returns a NaN / wrong observed t once high-dimensional fixed effects
    are absorbed (seen in T09a), so the bootstrap is done here. The observed t and every bootstrap t use
    the same CRV1 formula with pyfixest's small-sample factor, so the observed t equals pyfixest's t.
    Weights are drawn per *global* cluster id, so fits sharing the cluster coding get identical draws.
    Returns (t_obs, t_boot[reps]).
    """
    from scipy import sparse
    X = np.asarray(fit._X, dtype=float)
    y = np.asarray(fit._Y, dtype=float).ravel()
    names = [str(c) for c in fit._coefnames]
    k = names.index(param)
    cl = np.asarray(fit._cluster_df.iloc[:, 0])
    g_ids, g_inv = np.unique(cl, return_inverse=True)
    C = sparse.csr_matrix((np.ones(len(y)), (g_inv, np.arange(len(y)))), shape=(len(g_ids), len(y)))
    ssc = float(np.asarray(fit._ssc).ravel()[0])
    B = np.linalg.inv(X.T @ X)
    w = X @ B[k]                                     # β_k = w'y
    beta = B @ (X.T @ y)
    u = y - X @ beta
    t_obs = beta[k] / np.sqrt(ssc * np.sum((C @ (w * u)) ** 2))
    # restricted fit (β_k = 0)
    Xr = np.delete(X, k, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    yhat_r, u_r = Xr @ br, y - Xr @ br
    rng = np.random.default_rng(seed)
    n_global = int(np.max(g_ids)) + 1 if np.issubdtype(g_ids.dtype, np.integer) else len(g_ids)
    V = rng.choice([-1.0, 1.0], size=(n_global, reps))  # one draw per global cluster id
    Vg = V[g_ids.astype(int)] if np.issubdtype(g_ids.dtype, np.integer) else V
    t_boot = np.empty(reps)
    for s in range(0, reps, chunk):
        v = Vg[:, s:s + chunk][g_inv]                # n × r
        ys = yhat_r[:, None] + u_r[:, None] * v
        bs = B @ (X.T @ ys)                          # k × r
        us = ys - X @ bs
        se = np.sqrt(ssc * np.asarray((C @ (w[:, None] * us)) ** 2).sum(axis=0))
        t_boot[s:s + chunk] = bs[k] / se
    return float(t_obs), t_boot


def wild_bootstrap(fit, param, reps=9999, seed=SEED):
    """Wild cluster restricted bootstrap p-value (two-sided, symmetric) for one coefficient."""
    t_obs, tb = _wcr(fit, param, reps, seed)
    return float(np.mean(np.abs(tb) >= abs(t_obs)))


def romano_wolf(fits_params, reps=4999, seed=SEED):
    """Romano–Wolf step-down over (fit, param) pairs, using wild-cluster-bootstrap t statistics.

    Every test uses the same seed and global cluster ids, so the bootstrap draws are common across
    hypotheses (the dependence the step-down exploits). Returns a DataFrame with t, p_unadj, p_rw.
    """
    t_obs, t_boot, labels = [], [], []
    for fit, param in fits_params:
        t, tb = _wcr(fit, param, reps, seed)
        t_obs.append(t)
        t_boot.append(tb)
        labels.append((fit._depvar, param))
    t_obs = np.abs(np.array(t_obs))
    tb = np.abs(np.vstack(t_boot))                      # hypotheses × reps
    p_unadj = (tb >= t_obs[:, None]).mean(axis=1)
    order = np.argsort(-t_obs)
    p_rw = np.empty(len(t_obs))
    prev = 0.0
    for i, j in enumerate(order):
        maxstat = tb[order[i:], :].max(axis=0)
        p = max((maxstat >= t_obs[j]).mean(), prev)
        p_rw[j] = prev = p
    return pd.DataFrame({"outcome": [l[0] for l in labels], "param": [l[1] for l in labels],
                         "abs_t": t_obs, "p_unadj_boot": p_unadj, "p_romano_wolf": p_rw})


def coef_row(fit, param):
    """Coefficient, clustered SE, t, p and N for one parameter."""
    t = fit.tidy()
    r = t.loc[param]
    return {"outcome": fit._depvar, "param": param, "coef": r["Estimate"], "se": r["Std. Error"],
            "t": r["t value"], "p": r["Pr(>|t|)"], "N": fit._N, "n_clusters": int(fit._G[0])}
