"""Demand estimation. Price and capability coefficients come from instrumental
variables; the substitution parameter is selected by how well the implied
diversion matches what actually happened at observed model releases."""
import pathlib

import numpy as np
import pandas as pd

import capability as cap
import demand

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"

CHARS = ["q", "log_ctx", "multimodal", "reasoning", "log_age", "log_latency"]
SIGMA_GRID = np.round(np.arange(0.0, 0.96, 0.05), 2)


def add_cost_shifters(df):
    xs = pd.read_csv(DER / "models_cross_section.csv")[["permaslug", "hf_slug"]]
    hf = pd.read_csv(DER / "hf_meta.csv")[["hf_slug", "params"]]
    spd = pd.read_csv(DER / "model_speed.csv")
    d = df.merge(xs, on="permaslug", how="left").merge(hf, on="hf_slug", how="left")
    d = d.merge(spd, on="permaslug", how="left")
    d["log_latency"] = np.log(d.latency_ms.fillna(d.latency_ms.median()))
    d["log_thr"] = np.log(d.throughput_tps.fillna(d.throughput_tps.median()))
    d["log_params"] = np.log(d.params.where(d.params > 0))
    d["has_params"] = d.log_params.notna().astype(float)
    d["log_params"] = d.log_params.fillna(0.0)
    return d


def instruments(df):
    z = demand.diff_ivs(df)
    z["iv_log_params"] = df.log_params.to_numpy()
    z["iv_has_params"] = df.has_params.to_numpy()
    z["iv_log_thr"] = df.log_thr.to_numpy()
    return z


def estimate_beta(df, Z, sigma):
    """IV estimate of the linear parameters holding the nesting parameter fixed."""
    X = demand.design(df)
    y = df.y.to_numpy(float) + sigma * df.log_share_within.to_numpy(float)
    Zf = pd.concat([X.drop(columns=["price"]), Z], axis=1)
    b, se, e = demand.tsls(y, X, Zf, cluster=df.permaslug.factorize()[0])
    return pd.Series(b, index=X.columns), pd.Series(se, index=X.columns), e


def shares_from_delta(delta, nest, sigma):
    """Nested-logit shares implied by mean utilities (outside good normalised)."""
    lam = 1 - sigma
    d = pd.DataFrame({"delta": delta, "nest": nest})
    d["e"] = np.exp(d.delta / lam)
    D = d.groupby("nest")["e"].transform("sum")
    inc = d.groupby("nest")["e"].sum() ** lam
    denom = 1 + inc.sum()
    s_g = (inc / denom).reindex(d.nest).to_numpy()
    s_jg = (d.e / D).to_numpy()
    return s_jg * s_g, s_jg


def elasticities(df_t, beta_p, sigma):
    """Matrix of d log s_j / d log p_k within one market."""
    s = df_t.share.to_numpy()
    s_jg = df_t.share_within.to_numpy()
    nest = df_t.tier.to_numpy()
    n = len(s)
    same = nest[:, None] == nest[None, :]
    E = np.empty((n, n))
    lam = 1 - sigma
    E[:] = beta_p * s[None, :]
    E += np.where(same, beta_p * (sigma / lam) * s_jg[None, :], 0.0)
    own = beta_p * (1 / lam - (sigma / lam) * s_jg - s)
    np.fill_diagonal(E, own)
    return E


def instrument_diagnostic():
    """Why we do not instrument: the available cost shifters fail the exclusion
    restriction, driving the price coefficient toward zero despite a strong first
    stage."""
    df = add_cost_shifters(demand.build_sample())
    X = demand.design(df)
    exog = X.drop(columns=["price"])
    cluster = df.permaslug.factorize()[0]
    sets = {"OLS": None,
            "differentiation IVs": demand.diff_ivs(df),
            "parameter count": pd.DataFrame({"lp": df.log_params, "hp": df.has_params}),
            "both": pd.concat([demand.diff_ivs(df),
                               pd.DataFrame({"lp": df.log_params,
                                             "hp": df.has_params})], axis=1)}
    out = []
    for name, Z in sets.items():
        if Z is None:
            b = np.linalg.lstsq(X.to_numpy(float), df.y.to_numpy(float), rcond=None)[0]
            out.append(dict(spec=name, beta_p=b[0], se=np.nan, first_stage_F=np.nan))
            continue
        b, se, _ = demand.tsls(df.y, X, pd.concat([exog, Z], axis=1), cluster=cluster)
        F = demand.first_stage_F(np.log(df.p_blend.to_numpy()), exog.to_numpy(float),
                                 np.nan_to_num(Z.to_numpy(float)))
        out.append(dict(spec=name, beta_p=b[0], se=se[0], first_stage_F=F))
    return pd.DataFrame(out)


if __name__ == "__main__":
    print(instrument_diagnostic().round(3).to_string(index=False))
