"""Differentiated-products demand for inference: logit and nested-logit IV, with
differentiation instruments for price and the within-nest share."""
import pathlib

import numpy as np
import pandas as pd

from analysis import capability as cap

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
TIERS = 3


def build_sample():
    u = pd.read_csv(DER / "usage_daily.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs["created_at"] = pd.to_datetime(xs["created_at"], format="mixed", utc=True)
    sup = pd.read_csv(DER / "supply_side.csv")

    active = xs[xs.tokens.notna()].copy()
    active, _ = cap.impute(active)
    imp = [c + "_imp" for c in cap.DIMS]
    active = active.dropna(subset=imp)
    for c, i in zip(cap.DIMS, imp):
        active[c] = active[i]
    q, _ = cap.quality_index(active)
    active = active.assign(q_index=q)
    keep = active[(active.p_blend > 0) & active.p_blend.notna()]

    df = u.merge(keep[["permaslug", "name", "author", "open_weight", "p_in", "p_out",
                       "p_blend", "q_index", "q_intelligence", "q_coding", "q_agentic",
                       "context_length", "multimodal", "created_at", "supports_reasoning",
                       "n_scored"]],
                 on="permaslug", how="inner")
    df = df.merge(sup[["permaslug", "n_hosts", "latency_median", "throughput_median"]],
                  on="permaslug", how="left")

    size = u.groupby("date")["tokens"].sum().rename("market_tokens")
    df = df.merge(size, on="date")
    df = df[df.tokens > 0]
    df["share"] = df.tokens / df.market_tokens
    inside = df.groupby("date")["share"].sum().rename("inside")
    df = df.merge(inside, on="date")
    df["share_0"] = 1 - df.inside
    df["age_days"] = (df.date - df.created_at.dt.tz_localize(None)).dt.days.clip(lower=0)
    df["context_length"] = df.context_length.fillna(df.context_length.median())
    df["log_ctx"] = np.log(df.context_length.clip(lower=2048))
    df["n_hosts"] = df.n_hosts.fillna(1)
    df["tier"] = pd.qcut(df.q_index, TIERS, labels=False)
    df["y"] = np.log(df.share) - np.log(df.share_0)
    within = df.groupby(["date", "tier"])["share"].transform("sum")
    df["share_within"] = df.share / within
    df["log_share_within"] = np.log(df.share_within)
    return df


def diff_ivs(df, chars=("q_index", "log_ctx")):
    """Gandhi-Houde style differentiation instruments plus rival-count terms."""
    out = {}
    for c in chars:
        same, other, cnt = [], [], []
        for _, g in df.groupby("date"):
            v = g[c].to_numpy()
            d = v[:, None] - v[None, :]
            np.fill_diagonal(d, np.nan)
            tier = g.tier.to_numpy()
            m_same = tier[:, None] == tier[None, :]
            sq = d ** 2
            same.append(pd.Series(np.nansum(np.where(m_same, sq, np.nan), 1), index=g.index))
            other.append(pd.Series(np.nansum(np.where(~m_same, sq, np.nan), 1), index=g.index))
            cnt.append(pd.Series(np.nansum(np.abs(d) < 0.5, 1), index=g.index))
        out[f"iv_{c}_same"] = pd.concat(same)
        out[f"iv_{c}_other"] = pd.concat(other)
        out[f"iv_{c}_close"] = pd.concat(cnt)
    ivs = pd.DataFrame(out).loc[df.index]
    ivs["iv_n_rivals"] = df.groupby("date")["permaslug"].transform("size")
    ivs["iv_n_tier"] = df.groupby(["date", "tier"])["permaslug"].transform("size")
    ivs["iv_hosts"] = df["n_hosts"].fillna(1)
    return ivs


def _clean(A):
    A = np.array(A, dtype=float, copy=True)
    A[~np.isfinite(A)] = 0.0
    return A


def tsls(y, X, Z, cluster=None):
    """2SLS with cluster-robust variance."""
    y, X, Z = _clean(y), _clean(X), _clean(Z)
    scale = np.where(np.std(Z, axis=0) > 0, np.std(Z, axis=0), 1.0)
    Z = Z / scale
    ZZ_inv = np.linalg.pinv(Z.T @ Z)
    P = Z @ ZZ_inv @ Z.T
    XPX_inv = np.linalg.pinv(X.T @ P @ X)
    b = XPX_inv @ (X.T @ P @ y)
    e = y - X @ b
    Xh = P @ X
    if cluster is None:
        V = XPX_inv * (e @ e) / (len(y) - X.shape[1])
    else:
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in np.unique(cluster):
            m = cluster == c
            u = Xh[m].T @ e[m]
            meat += np.outer(u, u)
        V = XPX_inv @ meat @ XPX_inv
    return b, np.sqrt(np.diag(V)), e


def first_stage_F(endog, exog, instr, cluster=None):
    X = np.column_stack([instr, exog])
    b = np.linalg.lstsq(X, endog, rcond=None)[0]
    r = endog - X @ b
    Xr = exog
    br = np.linalg.lstsq(Xr, endog, rcond=None)[0]
    rr = endog - Xr @ br
    k = instr.shape[1]
    n, p = X.shape
    return ((rr @ rr - r @ r) / k) / (r @ r / (n - p))


def design(df, extra=()):
    d = pd.get_dummies(df["date"].dt.strftime("%m%d"), prefix="d", drop_first=True).astype(float)
    base = pd.DataFrame({
        "price": np.log(df.p_blend.to_numpy()),
        "q": df.q_index.to_numpy(),
        "log_ctx": df.log_ctx.to_numpy(),
        "multimodal": df.multimodal.astype(float).to_numpy(),
        "reasoning": df.supports_reasoning.astype(float).to_numpy(),
        "log_age": np.log1p(df.age_days).to_numpy(),
        "log_latency": df.log_latency.to_numpy() if "log_latency" in df else 0.0,
    }, index=df.index)
    for c in extra:
        base[c] = df[c].to_numpy()
    return pd.concat([base, d], axis=1)


def estimate(df):
    ivs = diff_ivs(df)
    X = design(df)
    exog = X.drop(columns=["price"])
    res = {}

    Z = pd.concat([exog, ivs], axis=1)
    b, se, e = tsls(df.y, X, Z, cluster=df.permaslug.factorize()[0])
    res["logit_iv"] = dict(zip(X.columns[:6], zip(b[:6], se[:6])))
    res["F_price"] = first_stage_F(df.p_blend.to_numpy(), exog.to_numpy(), ivs.to_numpy())

    Xo = X.copy()
    bo = np.linalg.lstsq(Xo.to_numpy(float), df.y.to_numpy(float), rcond=None)[0]
    res["logit_ols"] = dict(zip(X.columns[:6], bo[:6]))

    Xn = X.copy()
    Xn.insert(1, "log_share_within", df.log_share_within.to_numpy())
    exog_n = Xn.drop(columns=["price", "log_share_within"])
    Zn = pd.concat([exog_n, ivs], axis=1)
    bn, sen, en = tsls(df.y, Xn, Zn, cluster=df.permaslug.factorize()[0])
    res["nested_iv"] = dict(zip(Xn.columns[:7], zip(bn[:7], sen[:7])))
    res["xi_nested"] = pd.Series(en, index=df.index)
    res["b_nested"] = pd.Series(bn, index=Xn.columns)
    return res


if __name__ == "__main__":
    df = build_sample()
    print(f"sample: {len(df):,} model-days, {df.permaslug.nunique()} models, "
          f"{df.date.nunique()} days")
    print(f"inside share: mean {df.groupby('date').inside.first().mean():.3f}")
    r = estimate(df)
    print(f"\nfirst-stage F on price: {r['F_price']:.1f}")
    print("\nlogit OLS:", {k: round(v, 4) for k, v in r["logit_ols"].items()})
    print("\nlogit IV:")
    for k, (b, s) in r["logit_iv"].items():
        print(f"  {k:12s} {b: .4f} ({s:.4f})")
    print("\nnested logit IV:")
    for k, (b, s) in r["nested_iv"].items():
        print(f"  {k:16s} {b: .4f} ({s:.4f})")
