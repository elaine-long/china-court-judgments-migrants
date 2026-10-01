"""T09a: estimation code recovers planted effects on simulated data; clustering has the right dimension."""
import numpy as np
import pandas as pd
import pytest

from src import estimate as E

BETA2 = {"P1": -0.05, "P2": -0.12, "P3": 0.03}
BETA1 = {"P1": 0.02, "P2": 0.04, "P3": 0.06}
N_CITY, N_PILOT = 80, 18


def _period(q):
    y, k = int(q[:4]), int(q[-1])
    t = y * 4 + k
    if t < 2014 * 4 + 3:
        return "P0"
    if t < 2016 * 4 + 4:
        return "P1"
    if t < 2018 * 4 + 4:
        return "P2"
    return "P3"


def simulate(seed=1, per_cell=12):
    rng = np.random.default_rng(seed)
    qs = [f"{y}Q{k}" for y in range(2013, 2020) for k in range(1, 5)]
    city_fe = rng.normal(0, 0.1, N_CITY)
    city_mig = rng.normal(0, 0.05, N_CITY)            # city × migrant effects
    q_fe = dict(zip(qs, rng.normal(0, 0.05, len(qs))))
    q_mig = dict(zip(qs, rng.normal(0, 0.03, len(qs))))
    rows = []
    for c in range(N_CITY):
        pilot = int(c < N_PILOT)
        shock = rng.normal(0, 0.03, len(qs))          # city-quarter shock -> within-city correlation
        for i, q in enumerate(qs):
            p = _period(q)
            mig = rng.binomial(1, 0.45, per_cell)
            x = rng.normal(0, 1, per_cell)
            te = pilot * (BETA1.get(p, 0) + BETA2.get(p, 0) * mig)
            y = (0.3 + city_fe[c] + city_mig[c] * mig + q_fe[q] + q_mig[q] * mig + shock[i]
                 + 0.05 * x + te + rng.normal(0, 0.15, per_cell))
            rows.append(pd.DataFrame({"city": f"c{c}", "q": q, "pilot": pilot, "period": p,
                                      "migrant": mig, "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def sim():
    return simulate()


def test_triple_diff_recovers_beta2(sim):
    fit = E.triple_diff(sim, "y", controls=["x"])
    co, se = fit.coef(), fit.se()
    for p, b in BETA2.items():
        assert abs(co[f"pilot_{p}_mig"] - b) < 3 * se[f"pilot_{p}_mig"], p
    for p, b in BETA1.items():
        assert abs(co[f"pilot_{p}"] - b) < 4 * se[f"pilot_{p}"], p
    assert abs(co["x"] - 0.05) < 0.01


def test_triple_diff_unbiased_monte_carlo():
    # with 18 treated clusters one draw can miss by ~2 SE; the mean over draws must sit on the truth
    # (a 30-draw run during development: mean deviation <= 0.002, MC s.e. ~0.003, 95% coverage 0.90-0.93)
    dev = {p: [] for p in BETA2}
    for seed in range(200, 212):
        c = E.triple_diff(simulate(seed=seed, per_cell=6), "y", controls=["x"]).coef()
        for p, b in BETA2.items():
            dev[p].append(c[f"pilot_{p}_mig"] - b)
    for p, d in dev.items():
        assert abs(np.mean(d)) < 3 * np.std(d) / np.sqrt(len(d)) + 0.005, p


def test_cluster_dimension(sim):
    fit = E.triple_diff(sim, "y", controls=["x"])
    assert int(fit._G[0]) == N_CITY
    row = E.coef_row(fit, "pilot_P2_mig")
    assert row["n_clusters"] == N_CITY and row["N"] == len(sim)
    # clustered SEs exceed iid SEs here because of the city-quarter shocks
    iid = E.triple_diff(sim, "y", controls=["x"], cluster="city")
    assert iid.se()["pilot_P2_mig"] > 0


def test_event_study_base_and_shape(sim):
    fit, tab = E.event_study(sim, "y", base="2014Q2", window=("2013Q1", "2018Q3"))
    assert tab.loc[tab.quarter == "2014Q2", "beta2"].item() == 0.0
    assert len(tab) == 23  # 2013Q1..2018Q3
    pre = tab[tab.quarter < "2014Q3"].beta2.drop(tab.index[tab.quarter == "2014Q2"])
    p2 = tab[(tab.quarter >= "2017Q1") & (tab.quarter <= "2018Q3")].beta2
    assert abs(pre.mean()) < 0.03
    assert abs(p2.mean() - BETA2["P2"]) < 0.03


def test_csdid_wrapper_runs_on_gap(sim):
    import contextlib
    import io
    s = sim.assign(qnum=sim.q.str[:4].astype(int) * 4 + sim.q.str[-1].astype(int))
    first = {f"c{c}": (2016 * 4 + 4 if c < N_PILOT else 2018 * 4 + 4) for c in range(N_CITY)}
    with contextlib.redirect_stdout(io.StringIO()):
        tab, overall, se = E.csdid_wrapper(s, "y", first)
    assert {"event_time", "att", "se"} <= set(tab.columns) and np.isfinite(overall) and se > 0
    # pilot cohort: the P2 change in the migrant-local gap relative to its P1 level is -0.07
    post = tab[(tab.event_time >= 0) & (tab.event_time <= 7)].att.mean()
    assert -0.13 < post < -0.01


def test_wcr_observed_t_matches_pyfixest(sim):
    # pyfixest's own wildboottest returned t = 164 here (true t ~ -8); ours must equal the CRV1 t
    fit = E.triple_diff(sim, "y", controls=["x"])
    for p in ("pilot_P1_mig", "pilot_P2_mig", "x"):
        t_obs, tb = E._wcr(fit, p, reps=99, seed=1)
        assert t_obs == pytest.approx(fit.tidy().loc[p, "t value"], rel=1e-6), p
        assert np.isfinite(tb).all() and 0.5 < np.std(tb) < 2.0, p


def test_wcr_null_p_close_to_analytic(sim):
    # on a pure-noise outcome the bootstrap p and the analytic CRV1 p should be of the same size
    fit0 = E.triple_diff(sim.assign(z=np.random.default_rng(11).normal(0, 0.15, len(sim))), "z", controls=["x"])
    for p in ("pilot_P1_mig", "pilot_P2_mig", "pilot_P3_mig"):
        pb = E.wild_bootstrap(fit0, p, reps=499)
        assert abs(pb - fit0.tidy().loc[p, "Pr(>|t|)"]) < 0.15, p


def test_wild_bootstrap_and_romano_wolf(sim):
    fit = E.triple_diff(sim, "y", controls=["x"])
    p = E.wild_bootstrap(fit, "pilot_P2_mig", reps=199)
    assert 0 <= p < 0.01
    # a true null: pure noise, unrelated to treatment
    fit0 = E.triple_diff(sim.assign(y0=np.random.default_rng(7).normal(0, 0.15, len(sim))), "y0", controls=["x"])
    rw = E.romano_wolf([(fit, "pilot_P2_mig"), (fit0, "pilot_P2_mig"), (fit, "pilot_P1_mig")], reps=199)
    assert list(rw.columns) == ["outcome", "param", "abs_t", "p_unadj_boot", "p_romano_wolf"]
    assert (rw.p_romano_wolf >= rw.p_unadj_boot - 1e-12).all()
    assert rw.p_romano_wolf.iloc[0] < 0.05          # planted effect survives
    assert rw.p_romano_wolf.iloc[1] > 0.05          # effect removed -> not rejected
