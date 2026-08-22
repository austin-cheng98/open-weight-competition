"""Produce every number quoted in the paper and write them to results.json."""
import json, pathlib

import numpy as np
import pandas as pd

import capability as cap
import counterfactual as cfm
import demand
import estimate as est
import event_study as es
import supply
from event_study import absorb, cluster_ols

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, OUT = ROOT / "data" / "derived", ROOT / "results.json"
BETA, SIGMA = -1.0, 0.4          # preferred demand parameters


def coverage(R):
    xs = pd.read_csv(DER / "models_cross_section.csv")
    u = xs[xs.tokens.notna()].copy()
    u["spend"] = (u.prompt_tokens * u.p_in + u.completion_tokens * u.p_out) / 1e6
    imp, _ = cap.impute(u)
    for c in cap.DIMS:
        imp[c] = imp[c + "_imp"]
    scored = imp.dropna(subset=cap.DIMS)
    R["n_catalogue"] = int(len(xs))
    R["n_active"] = int(len(u))
    dly = pd.read_csv(DER / "usage_daily.csv")
    live = set(dly.permaslug)
    R["n_daily_models"] = int(len(live))
    R["n_daily_open"] = int(xs[xs.permaslug.isin(live)].open_weight.sum())
    R["n_open"] = int(u.open_weight.sum())
    R["n_scored"] = int(len(scored))
    R["token_coverage_scored"] = float(scored.tokens.sum() / u.tokens.sum())
    R["open_token_share"] = float(u.loc[u.open_weight, "tokens"].sum() / u.tokens.sum())
    R["open_spend_share"] = float(u.loc[u.open_weight, "spend"].sum() / u.spend.sum())
    for lab, col in (("token", "tokens"), ("spend", "spend")):
        s = u.groupby("author")[col].sum()
        s = s / s.sum()
        R[f"hhi_provider_{lab}"] = float(1e4 * (s ** 2).sum())
        R[f"top3_{lab}"] = float(s.nlargest(3).sum())
        R[f"top3_{lab}_names"] = list(s.nlargest(3).index)
    # capability dimensionality, on models scored on every dimension directly
    obs = u.dropna(subset=cap.DIMS)
    R["n_scored_direct"] = int(len(obs))
    Z = ((obs[cap.DIMS] - obs[cap.DIMS].mean()) / obs[cap.DIMS].std(ddof=0)).to_numpy()
    _, sv, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    R["capability_pc1"] = float((sv ** 2 / (sv ** 2).sum())[0])
    wide = u.dropna(subset=cap.DIMS_WIDE)
    Zw = ((wide[cap.DIMS_WIDE] - wide[cap.DIMS_WIDE].mean())
          / wide[cap.DIMS_WIDE].std(ddof=0)).to_numpy()
    _, sw, _ = np.linalg.svd(Zw - Zw.mean(0), full_matrices=False)
    R["capability_pc1_wide"] = float((sw ** 2 / (sw ** 2).sum())[0])
    R["n_scored_wide"] = int(len(wide))
    return R


def stickiness(R):
    pp = pd.read_csv(DER / "price_panel.csv", parse_dates=["snapshot"])
    pp = pp[pp.p_in > 0].sort_values(["permaslug", "snapshot"])
    g = pp.groupby("permaslug")
    pp["dlp"] = g.p_in.transform(lambda s: np.log(s).diff())
    pp["gap"] = g.snapshot.transform(lambda s: s.diff().dt.days)
    n = g.size()
    sub = pp[pp.permaslug.isin(n[n >= 3].index)]
    ch = sub.dlp.abs() > 1e-6
    R["price_panel_models"] = int(sub.permaslug.nunique())
    R["price_panel_snapshots"] = int(pp.snapshot.nunique())
    R["price_panel_start"] = str(pp.snapshot.min().date())
    R["price_panel_end"] = str(pp.snapshot.max().date())
    R["price_change_rate"] = float(ch.mean())
    R["price_median_gap_days"] = float(sub.gap.median())
    R["share_never_repriced"] = float(sub.groupby("permaslug").dlp
                                      .apply(lambda s: (s.abs() > 1e-6).sum() == 0).mean())
    R["price_cut_share"] = float((sub.loc[ch, "dlp"] < 0).mean())
    R["price_change_median"] = float(sub.loc[ch, "dlp"].median())
    R["n_price_changes"] = int(ch.sum())
    R["n_price_transitions"] = int(sub.dlp.notna().sum())
    import fig_prices
    surv = fig_prices.survival(pp)
    grid = np.arange(0, 391, 30)
    km = fig_prices.km(surv, grid)
    R["price_survival_180"] = float(km[grid == 180][0])
    R["price_survival_360"] = float(km[grid == 360][0])
    # monthly frequency of a price change, the comparable statistic in the
    # price-setting literature
    pp2 = pp.copy()
    pp2["month"] = pp2.snapshot.dt.to_period("M")
    pp2["chg"] = pp2.groupby("permaslug").p_in.transform(
        lambda x: np.log(x).diff().abs() > 1e-6)
    mm = pp2.groupby(["permaslug", "month"]).agg(any_chg=("chg", "max"), n=("chg", "size"))
    mm = mm[mm.n >= 2]
    R["monthly_change_freq"] = float(mm.any_chg.mean())
    R["price_duration_months"] = float(-1 / np.log(1 - mm.any_chg.mean()))
    return R


def elasticity(R):
    d = pd.read_csv(DER / "panel_long.csv", parse_dates=["date"])
    d["day"] = d.date.astype(str)
    R["long_panel_obs"] = int(len(d))
    R["long_panel_models"] = int(d.permaslug.nunique())
    R["long_panel_start"] = str(d.date.min().date())
    R["long_panel_end"] = str(d.date.max().date())
    chg = d.sort_values(["permaslug", "date"]).groupby("permaslug").ln_p.transform(
        lambda x: x.diff().abs() > 1e-6)
    R["n_revisions_panel"] = int(chg.sum())
    for lab, y in (("tokens", "ln_tokens"), ("requests", "ln_req")):
        yv, Xv = absorb(d, y, ["ln_p"], ["permaslug", "day"])
        b, se = cluster_ols(yv, Xv, d.permaslug.factorize()[0])
        R[f"within_elasticity_{lab}"] = float(b[0])
        R[f"within_elasticity_{lab}_se"] = float(se[0])
    yv, Xv = absorb(d, "ln_tokens", ["ln_p"], ["day"])
    b, se = cluster_ols(yv, Xv, d.permaslug.factorize()[0])
    R["cross_elasticity"] = float(b[0])
    R["cross_elasticity_se"] = float(se[0])
    for tag, mask in (("open", d.open_weight == True), ("closed", d.open_weight == False)):
        sub = d[mask]
        yv, Xv = absorb(sub, "ln_tokens", ["ln_p"], ["permaslug", "day"])
        b, se = cluster_ols(yv, Xv, sub.permaslug.factorize()[0])
        R[f"within_elasticity_{tag}"] = float(b[0])
        R[f"within_elasticity_{tag}_se"] = float(se[0])
    import fig_prices
    es, n_ev = fig_prices.event_study(d)
    R["n_price_events"] = int(n_ev)
    R["price_event_pre"] = float(es.loc[es.k.between(-3, -2), "beta"].mean())
    R["price_event_post"] = float(es.loc[es.k >= 3, "beta"].mean())
    return R


def growth(R):
    u = pd.read_csv(DER / "usage_daily.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs["created_at"] = pd.to_datetime(xs["created_at"], format="mixed", utc=True).dt.tz_localize(None)
    u = u.merge(xs[["permaslug", "created_at", "open_weight"]], on="permaslug", how="left")
    days = sorted(u.date.unique())
    d0, d1 = days[0], days[-2]
    T0 = u[u.date == d0].tokens.sum()
    T1 = u[u.date == d1].tokens.sum()
    new = u.created_at > d0
    e1 = u[new & (u.date == d1)].tokens.sum()
    i1 = u[~new & (u.date == d1)].tokens.sum()
    R["daily_start"], R["daily_end"] = str(pd.Timestamp(d0).date()), str(pd.Timestamp(d1).date())
    R["market_growth"] = float(T1 / T0 - 1)
    R["entrant_share_final"] = float(e1 / T1)
    R["incumbent_growth"] = float(i1 / T0 - 1)
    R["entry_share_of_growth"] = float(e1 / (T1 - T0))
    R["entrant_open_share"] = float(u[new & (u.date == d1) & (u.open_weight == True)].tokens.sum() / e1)
    R["daily_tokens_T"] = float(T1 / 1e12)
    return R


def diversion(R):
    u, xs, scored = es.load()
    S, _, lam = es.similarity_table(scored)
    p, ev = es.build_pairs(u, scored, S)
    tier = demand.build_sample().groupby("permaslug").first().tier
    p["tier_j"] = p.permaslug.map(tier)
    p["tier_k"] = p.event.map(tier)
    p = p.dropna(subset=["tier_j", "tier_k"])
    p["event_tau"] = p.event + "|" + p.tau.astype(str)
    p["same_nest"] = (p.tier_j == p.tier_k).astype(float)
    p["post_same"] = p.post * p.same_nest
    p["post_threat"] = p.post * p.threat
    p["post_q"] = p.post * p.q_j
    p["post_lp"] = p.post * p.logp_j
    R["n_events"] = int(len(ev))
    R["n_events_open"] = int(ev.open_weight.sum())
    R["n_pairs"] = int(p.pair.nunique())
    R["n_incumbents"] = int(p.permaslug.nunique())
    for lab, terms in (("same_nest", ["post_same", "post_q", "post_lp"]),
                       ("threat", ["post_threat", "post_q", "post_lp"])):
        yv, Xv = absorb(p, "log_tokens", terms, ["pair", "event_tau"])
        b, se = cluster_ols(yv, Xv, p.permaslug.factorize()[0])
        R[f"diversion_{lab}"] = float(b[0])
        R[f"diversion_{lab}_se"] = float(se[0])
        R[f"diversion_{lab}_lower"] = float(b[0] - 1.96 * se[0])
    return R


def structure(R):
    df = demand.build_sample()
    d = cfm.cross_section(df)
    R["cf_date"] = str(d.date.iloc[0].date())
    R["cf_models"] = int(len(d))
    R["cf_outside_share"] = float(d.share_out.iloc[0])
    R["cf_open_token_share"] = float(d.loc[d.open_weight, "share"].sum() / d.share.sum())
    grid = []
    for sg in (0.0, 0.2, 0.4, 0.6, 0.7):
        base, cf, _ = cfm.run(d, BETA, sg)
        dg = cfm.diagnostics(d, BETA, sg)
        tok = df[df.date == d.date.iloc[0]].tokens.sum()
        grid.append(dict(sigma=sg,
                         price_base=base["price_paid"], price_cf=cf["price_paid"],
                         d_price=cf["price_paid"] / base["price_paid"] - 1,
                         cs_base=base["cs"], cs_cf=cf["cs"],
                         d_cs=cf["cs"] / base["cs"] - 1,
                         cs_loss_daily=(base["cs"] - cf["cs"]) * tok / 1e6,
                         hhi_base=base["hhi"], hhi_cf=cf["hhi"],
                         own_el=dg["median_own_elasticity"],
                         share_inelastic=dg["share_inelastic"]))
    R["counterfactual"] = grid
    main = [g for g in grid if abs(g["sigma"] - SIGMA) < 1e-9][0]
    R.update({f"cf_{k}": v for k, v in main.items()})
    R["cf_beta"], R["cf_sigma"] = BETA, SIGMA
    R["daily_tokens_cf"] = float(df[df.date == d.date.iloc[0]].tokens.sum() / 1e12)
    return R


def frontier(R):
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs = xs[xs.tokens.notna()].copy()
    xs, _ = cap.impute(xs)
    for c in cap.DIMS:
        xs[c] = xs[c + "_imp"]
    xs = xs.dropna(subset=cap.DIMS + ["p_blend"])
    xs = xs[xs.p_blend > 0]
    xs["q_index"], _ = cap.quality_index(xs)
    xs["release"] = pd.to_datetime(xs.created_at, format="mixed", utc=True).dt.tz_localize(None)
    grid = np.linspace(xs.q_index.quantile(0.02), xs.q_index.max(), 120)
    ratio = []
    for g in grid:
        c = xs[(xs.q_index >= g) & (~xs.open_weight)].p_blend.min()
        o = xs[(xs.q_index >= g) & (xs.open_weight)].p_blend.min()
        if np.isfinite(c) and np.isfinite(o):
            ratio.append(c / o)
    R["price_ratio_at_capability_median"] = float(np.median(ratio))
    # like-for-like comparison: proprietary price against the median open-weight
    # price among models within two points of the same capability
    rows = []
    for _, j in xs[~xs.open_weight].iterrows():
        o = xs[xs.open_weight & ((xs.q_index - j.q_index).abs() <= 2)]
        if len(o):
            rows.append((j.p_blend / o.p_blend.median(), j.tokens))
    m = pd.DataFrame(rows, columns=["ratio", "tokens"])
    R["matched_price_ratio_median"] = float(m.ratio.median())
    R["matched_price_ratio_weighted"] = float(np.average(m.ratio, weights=m.tokens))
    R["n_matched"] = int(len(m))
    xs["spend"] = (xs.prompt_tokens * xs.p_in + xs.completion_tokens * xs.p_out) / 1e6
    for openw, tag in ((True, "open"), (False, "closed")):
        s_ = xs[xs.open_weight == openw]
        R[f"realised_price_{tag}"] = float(s_.spend.sum() / (s_.tokens.sum() / 1e6))
    days = pd.date_range(xs.release.quantile(0.05), xs.release.max(), freq="7D")
    fo = np.array([xs.loc[xs.open_weight & (xs.release <= d), "q_index"].max() for d in days], float)
    fc = np.array([xs.loc[~xs.open_weight & (xs.release <= d), "q_index"].max() for d in days], float)
    lags = []
    for i, d in enumerate(days):
        reach = [days[j] for j in range(len(days)) if fc[j] >= fo[i]]
        if reach and np.isfinite(fo[i]):
            lags.append((d - reach[0]).days)
    R["open_frontier_lag_days"] = float(np.median(lags))
    R["open_frontier_lag_days_recent"] = float(np.median(lags[-52:]))
    sup = pd.read_csv(DER / "supply_side.csv").merge(
        xs[["permaslug", "open_weight", "tokens"]], on="permaslug")
    for openw, tag in ((True, "open"), (False, "closed")):
        s = sup[sup.open_weight == openw]
        R[f"hosts_weighted_{tag}"] = float(np.average(s.n_hosts, weights=s.tokens))
        R[f"hosts_median_{tag}"] = float(s.n_hosts.median())
    return R


def long_horizon(R):
    import robustness
    d = robustness.displacement_long()
    R["n_major_releases"] = int(len(d))
    R["incumbent_growth_median"] = float(d.incumbent_growth.median())
    R["incumbent_decline_share"] = float((d.incumbent_growth < 0).mean())
    R["market_growth_median_long"] = float(d.market_growth.median())
    r1 = d[d.event == "DeepSeek: R1"]
    if len(r1):
        R["r1_incumbent_growth"] = float(r1.incumbent_growth.iloc[0])
        R["r1_market_growth"] = float(r1.market_growth.iloc[0])
    sc, share = robustness.successor_check()
    R["successor_share"] = float(share)
    R["elasticity_ex_successor"] = float(sc.loc[2, "beta"])
    R["elasticity_ex_successor_se"] = float(sc.loc[2, "se"])
    return R


def open_share_history(R):
    """Open-weight share of tokens in the archived leaderboards. Levels are not
    comparable with the daily panel, which is built from a different endpoint."""
    lu = pd.read_csv(DER / "usage_long.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv").set_index("permaslug")
    lu["open"] = lu.permaslug.map(xs.open_weight)
    R["open_history_matched"] = float(lu[lu.open.notna()].tokens.sum() / lu.tokens.sum())
    lu = lu.dropna(subset=["open"])
    q = lu.groupby([lu.date.dt.to_period("Q").astype(str), "open"]).tokens.sum().unstack(fill_value=0)
    share = (q[True] / (q[True] + q[False])).round(3)
    R["open_share_by_quarter"] = {k: float(v) for k, v in share.items()}
    R["open_share_2024q4"] = float(share.get("2024Q4", np.nan))
    R["open_share_2025q2"] = float(share.get("2025Q2", np.nan))
    R["open_share_2026q2"] = float(share.get("2026Q2", np.nan))
    return R


def tasks(R):
    t = pd.read_csv(DER / "task_markets.csv")
    sp = t[t.measure == "spend"]
    w = sp.assign(w=sp.share * sp.task_share_of_total)
    by = w.groupby(["macro", "open_weight"])["w"].sum().unstack(fill_value=0)
    by = (by.T / by.sum(axis=1)).T
    R["open_spend_share_by_task"] = {k: float(v) for k, v in by[True].items()}
    R["n_tasks"] = int(sp.task.nunique())
    R["task_inside_share"] = float(sp.groupby("task").share.sum().mean())
    conc = sp.groupby("task")["share"].apply(lambda s: 1e4 * ((s / s.sum()) ** 2).sum())
    R["task_hhi_median"] = float(conc.median())
    return R


def instruments(R):
    d = est.instrument_diagnostic().set_index("spec")
    R["iv_ols"] = float(d.loc["OLS", "beta_p"])
    R["iv_params"] = float(d.loc["parameter count", "beta_p"])
    R["iv_params_se"] = float(d.loc["parameter count", "se"])
    R["iv_params_F"] = float(d.loc["parameter count", "first_stage_F"])
    R["iv_diff"] = float(d.loc["differentiation IVs", "beta_p"])
    hf = pd.read_csv(DER / "hf_meta.csv")
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs, _ = cap.impute(xs[xs.tokens.notna()].copy())
    for c in cap.DIMS:
        xs[c] = xs[c + "_imp"]
    xs = xs.dropna(subset=cap.DIMS)
    xs["q_index"], _ = cap.quality_index(xs)
    j = xs.merge(hf[["hf_slug", "params"]], on="hf_slug").dropna(subset=["params"])
    j = j[j.params > 0]
    R["params_capability_corr"] = float(np.corrcoef(np.log(j.params), j.q_index)[0, 1])
    return R


def task_counterfactual(R):
    import task_counterfactual as tc
    d = tc.run(tc.load(), BETA, SIGMA)
    R["task_cf_price"] = float(np.average(d.d_price, weights=d.weight))
    R["task_cf_cs"] = float(np.average(d.d_cs, weights=d.weight))
    R["task_cf_slope"] = float(np.polyfit(d.open_share, d.d_price, 1)[0])
    R["task_cf_by_macro"] = {
        m: {"open_share": float(np.average(g.open_share, weights=g.weight)),
            "d_price": float(np.average(g.d_price, weights=g.weight)),
            "d_cs": float(np.average(g.d_cs, weights=g.weight)),
            "weight": float(g.weight.sum()),
            "n": int(len(g))}
        for m, g in d.groupby("macro")}
    return R


def main():
    R = {}
    for step in (coverage, stickiness, elasticity, growth, diversion, structure,
                 frontier, tasks, open_share_history, long_horizon,
                 instruments, task_counterfactual):
        R = step(R)
        print(f"  {step.__name__} done")
    OUT.write_text(json.dumps(R, indent=1, default=float))
    print(f"\nwrote {OUT} with {len(R)} entries")


if __name__ == "__main__":
    main()
