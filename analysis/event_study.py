"""Incumbent-by-entrant event studies: does a model release pull volume away from
the incumbents that are technologically closest to it?"""
import pathlib

import numpy as np
import pandas as pd

from analysis import capability as cap

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
WINDOW = 7          # days each side of a release
MIN_PRE_TOKENS = 1e8


def load():
    u = pd.read_csv(DER / "usage_daily.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs["created_at"] = pd.to_datetime(xs["created_at"], format="mixed", utc=True).dt.tz_localize(None)
    xs["q_index"], _ = cap.quality_index(xs)
    scored = xs.dropna(subset=cap.DIMS + ["p_blend"]).query("p_blend > 0")
    return u, xs, scored


def similarity_table(scored):
    d, Z = cap.capability_matrix(scored)
    S, lam = cap.similarity(cap.whitened_distance(Z))
    S = pd.DataFrame(S, index=d.permaslug, columns=d.permaslug)
    alt = {}
    for name, fn in (("euclid", cap.euclidean_distance), ("cosine", cap.cosine_distance)):
        A, _ = cap.similarity(fn(Z))
        alt[name] = pd.DataFrame(A, index=d.permaslug, columns=d.permaslug)
    return S, alt, lam


def events(u, scored, start, end):
    """Releases that fall inside the usable part of the daily panel."""
    lo = u.date.min() + pd.Timedelta(days=WINDOW)
    hi = u.date.max() - pd.Timedelta(days=WINDOW)
    e = scored[(scored.created_at >= max(lo, start)) & (scored.created_at <= min(hi, end))]
    return e[["permaslug", "name", "author", "open_weight", "created_at", "q_index",
              "p_blend"]].sort_values("created_at")


def build_pairs(u, scored, S):
    ev = events(u, scored, u.date.min(), u.date.max())
    pre_tok = (u[u.date <= u.date.min() + pd.Timedelta(days=3)]
               .groupby("permaslug")["tokens"].mean())
    rows = []
    for _, k in ev.iterrows():
        T = k.created_at.normalize()
        inc = scored[(scored.created_at < T - pd.Timedelta(days=WINDOW)) &
                     (scored.permaslug != k.permaslug)]
        inc = inc[inc.permaslug.isin(pre_tok[pre_tok > MIN_PRE_TOKENS].index)]
        inc = inc[inc.permaslug.isin(S.index)]
        if k.permaslug not in S.index or inc.empty:
            continue
        sub = u[(u.date >= T - pd.Timedelta(days=WINDOW)) &
                (u.date <= T + pd.Timedelta(days=WINDOW)) &
                (u.permaslug.isin(inc.permaslug))].copy()
        sub["event"] = k.permaslug
        sub["tau"] = (sub.date - T).dt.days
        sub["post"] = (sub.tau >= 0).astype(float)
        sub["sim"] = S.loc[k.permaslug, sub.permaslug].to_numpy()
        sub["entrant_open"] = float(k.open_weight)
        sub["same_author"] = (sub.permaslug.map(scored.set_index("permaslug").author)
                              == k.author).astype(float)
        rows.append(sub)
    p = pd.concat(rows, ignore_index=True)
    p["pair"] = p.event + "|" + p.permaslug
    p["log_tokens"] = np.log(p.tokens.clip(lower=1))
    p["log_requests"] = np.log(p.requests.clip(lower=1))
    meta = scored.set_index("permaslug")
    p["q_j"] = p.permaslug.map(meta.q_index)
    p["logp_j"] = np.log(p.permaslug.map(meta.p_blend))
    p["open_j"] = p.permaslug.map(meta.open_weight).astype(float)
    p["author_j"] = p.permaslug.map(meta.author)
    ent = ev.set_index("permaslug")
    p["q_k"] = p.event.map(ent.q_index)
    p["logp_k"] = np.log(p.event.map(ent.p_blend))
    # entrant's value-for-money advantage over the incumbent, in logs
    p["rel_value"] = (np.log(p.q_k) - p.logp_k) - (np.log(p.q_j) - p.logp_j)
    p["threat"] = p.sim * p.rel_value
    return p, ev


def absorb(df, y, X, fes):
    """Within transformation on a set of fixed effects (alternating projections)."""
    Y = df[y].to_numpy(float).copy()
    M = df[X].to_numpy(float).copy()
    A = np.column_stack([Y, M])
    codes = [df[f].factorize()[0] for f in fes]
    for _ in range(60):
        prev = A.copy()
        for c in codes:
            means = np.zeros((c.max() + 1, A.shape[1]))
            counts = np.bincount(c, minlength=c.max() + 1)[:, None]
            np.add.at(means, c, A)
            A -= (means / counts)[c]
        if np.max(np.abs(A - prev)) < 1e-10:
            break
    return A[:, 0], A[:, 1:]


def cluster_ols(y, X, cluster):
    XX = np.linalg.pinv(X.T @ X)
    b = XX @ (X.T @ y)
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cluster):
        m = cluster == c
        u = X[m].T @ e[m]
        meat += np.outer(u, u)
    V = XX @ meat @ XX
    return b, np.sqrt(np.diag(V))


TRENDS = ["post_q_j", "post_logp_j", "post_open_j"]


def did(p, y="log_tokens", terms=("post_threat",), fes=("pair", "event_tau"),
        trends=True):
    d = p.copy()
    d["event_tau"] = d.event + "|" + d.tau.astype(str)
    d["post_sim"] = d.post * d.sim
    d["post_threat"] = d.post * d.threat
    d["post_threat_open"] = d.post * d.threat * d.entrant_open
    d["post_sim_open"] = d.post * d.sim * d.entrant_open
    d["post_open"] = d.post * d.entrant_open
    d["post_same_author"] = d.post * d.same_author
    d["post_q_j"] = d.post * d.q_j
    d["post_logp_j"] = d.post * d.logp_j
    d["post_open_j"] = d.post * d.open_j
    terms = list(terms) + (TRENDS if trends else [])
    yv, Xv = absorb(d, y, list(terms), list(fes))
    b, se = cluster_ols(yv, Xv, d.permaslug.factorize()[0])
    return dict(zip(terms, zip(b, se))), len(d)


def event_time(p, y="log_tokens", fes=("pair", "event_tau"), key="threat"):
    d = p.copy()
    d["event_tau"] = d.event + "|" + d.tau.astype(str)
    taus = [t for t in sorted(d.tau.unique()) if t != -1]
    for t in taus:
        d[f"k{t}"] = (d.tau == t) * d[key]
        d[f"c{t}"] = (d.tau == t) * d.q_j
        d[f"p{t}"] = (d.tau == t) * d.logp_j
    cols = [f"k{t}" for t in taus] + [f"c{t}" for t in taus] + [f"p{t}" for t in taus]
    yv, Xv = absorb(d, y, cols, list(fes))
    b, se = cluster_ols(yv, Xv, d.permaslug.factorize()[0])
    n = len(taus)
    return pd.DataFrame({"tau": taus, "beta": b[:n], "se": se[:n]})


if __name__ == "__main__":
    u, xs, scored = load()
    S, alt, lam = similarity_table(scored)
    p, ev = build_pairs(u, scored, S)
    print(f"{len(ev)} release events ({int(ev.open_weight.sum())} open-weight)")
    print(ev[["name", "author", "open_weight", "created_at", "q_index", "p_blend"]]
          .to_string(index=False))
    print(f"\npair-days: {len(p):,}  incumbents: {p.permaslug.nunique()}  "
          f"pairs: {p.pair.nunique()}")
    for terms in [("post_sim",), ("post_threat",), ("post_sim", "post_threat"),
                  ("post_threat", "post_threat_open")]:
        r, n = did(p, "log_tokens", terms=terms)
        print("  " + "   ".join(f"{k}={b: .3f} ({se:.3f})" for k, (b, se) in r.items()
                                if k not in TRENDS) + f"   n={n:,}")
