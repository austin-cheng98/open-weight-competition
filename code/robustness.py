"""Robustness checks reported in the appendix."""
import json, pathlib

import numpy as np
import pandas as pd

import capability as cap
import event_study as es
from event_study import absorb, cluster_ols

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"


def elasticity_variants():
    d = pd.read_csv(DER / "panel_long.csv", parse_dates=["date"])
    d["day"] = d.date.astype(str)
    d["ln_p_in"] = np.log(d.p_in)
    d["ln_p_blend"] = np.log(d.p_blend)
    d["ln_p_out"] = np.log(d.p_out)
    rows = []
    specs = [("fixed-weight price", "ln_p", "ln_tokens", d),
             ("input price", "ln_p_in", "ln_tokens", d),
             ("output price", "ln_p_out", "ln_tokens", d),
             ("realised blended price", "ln_p_blend", "ln_tokens", d),
             ("requests", "ln_p", "ln_req", d),
             ("weekly captures only", "ln_p", "ln_tokens", d[d.weekly == 1]),
             ("trailing captures only", "ln_p", "ln_tokens", d[d.weekly == 0]),
             ("open-weight models", "ln_p", "ln_tokens", d[d.open_weight]),
             ("proprietary models", "ln_p", "ln_tokens", d[~d.open_weight]),
             ("2025 onwards", "ln_p", "ln_tokens", d[d.date >= "2025-01-01"])]
    for lab, x, y, s in specs:
        if len(s) < 200:
            continue
        yv, Xv = absorb(s, y, [x], ["permaslug", "day"])
        b, se = cluster_ols(yv, Xv, s.permaslug.factorize()[0])
        rows.append(dict(spec=lab, beta=b[0], se=se[0], n=len(s),
                         models=s.permaslug.nunique()))
    return pd.DataFrame(rows)


def successor_check():
    """Price cuts may coincide with a provider launching a replacement model."""
    d = pd.read_csv(DER / "panel_long.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs["created"] = pd.to_datetime(xs["created_at"], format="mixed", utc=True).dt.tz_localize(None)
    d["author"] = d.permaslug.map(xs.set_index("permaslug").author)
    d["day"] = d.date.astype(str)
    rel = xs.dropna(subset=["author", "created"])[["author", "created"]]
    flag = np.zeros(len(d), bool)
    for a, g in d.groupby("author"):
        r = rel.loc[rel.author == a, "created"].to_numpy()
        if not len(r):
            continue
        dt = g.date.to_numpy()[:, None] - r[None, :]
        flag[g.index] = ((dt >= np.timedelta64(0, "D")) &
                         (dt <= np.timedelta64(28, "D"))).any(1)
    d["successor"] = flag.astype(float)
    rows = []
    for lab, spec, sub in (("baseline", ["ln_p"], d),
                           ("with successor control", ["ln_p", "successor"], d),
                           ("excluding successor windows", ["ln_p"], d[d.successor == 0])):
        yv, Xv = absorb(sub, "ln_tokens", spec, ["permaslug", "day"])
        b, se = cluster_ols(yv, Xv, sub.permaslug.factorize()[0])
        rows.append(dict(spec=lab, beta=b[0], se=se[0], n=len(sub)))
    return pd.DataFrame(rows), float(d.successor.mean())


def displacement_long():
    """Incumbent volume around the largest open-weight releases in the archived panel."""
    d = pd.read_csv(DER / "panel_long.csv", parse_dates=["date"])
    d = d[d.weekly == 1]
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs["created"] = pd.to_datetime(xs["created_at"], format="mixed", utc=True).dt.tz_localize(None)
    d["created"] = d.permaslug.map(xs.set_index("permaslug").created)
    dates = np.array(sorted(d.date.unique()))
    peak = d.groupby("permaslug").share.max()
    cand = xs[xs.open_weight & xs.permaslug.isin(peak.index)].copy()
    cand["peak"] = cand.permaslug.map(peak)
    W = 8
    ev = cand[(cand.peak >= 0.01) & (cand.created >= dates[W]) &
              (cand.created <= dates[-W - 1])].sort_values("created")
    rows, seen = [], set()
    for _, k in ev.iterrows():
        i = int(np.searchsorted(dates, k.created))
        if i in seen:
            continue
        seen.add(i)
        pre = d[(d.date >= dates[i - W]) & (d.date < dates[i])]
        post = d[(d.date > dates[i]) & (d.date <= dates[i + W])]
        inc = set(pre.loc[pre.created < k.created, "permaslug"])
        a = pre[pre.permaslug.isin(inc)].groupby("date").tokens.sum().mean()
        b = post[post.permaslug.isin(inc)].groupby("date").tokens.sum().mean()
        ta = pre.groupby("date").tokens.sum().mean()
        tb = post.groupby("date").tokens.sum().mean()
        rows.append(dict(event=k["name"], date=str(k.created.date()),
                         incumbent_growth=np.log(b / a), market_growth=np.log(tb / ta)))
    return pd.DataFrame(rows)


def outside_option():
    """The size of the outside option is a modelling choice; this varies it."""
    import counterfactual as cfm
    import demand
    df = demand.build_sample()
    d = cfm.cross_section(df)
    tokens = df[df.date == d.date.iloc[0]].tokens.sum()
    rows = []
    for s0 in (float(d.share_out.iloc[0]), 0.10, 0.25, 0.50):
        dd = d.copy()
        dd["share"] = dd.share * (1 - s0) / dd.share.sum()
        dd["share_out"] = s0
        for sg in (0.0, 0.4, 0.7):
            base, cf, _ = cfm.run(dd, -1.0, sg)
            rows.append(dict(outside_share=s0, sigma=sg,
                             d_price=cf["price_paid"] / base["price_paid"] - 1,
                             d_cs=cf["cs"] / base["cs"] - 1,
                             loss_m_per_day=(base["cs"] - cf["cs"]) * tokens / 1e12))
    return pd.DataFrame(rows)


def diversion_variants():
    u, xs, scored = es.load()
    d, Z = cap.capability_matrix(scored)
    rows = []
    for lab, fn in (("Mahalanobis", cap.whitened_distance),
                    ("Euclidean", cap.euclidean_distance),
                    ("cosine", cap.cosine_distance)):
        S, _ = cap.similarity(fn(Z))
        S = pd.DataFrame(S, index=d.permaslug, columns=d.permaslug)
        p, ev = es.build_pairs(u, scored, S)
        p["event_tau"] = p.event + "|" + p.tau.astype(str)
        p["post_threat"] = p.post * p.threat
        p["post_sim"] = p.post * p.sim
        p["post_q"] = p.post * p.q_j
        p["post_lp"] = p.post * p.logp_j
        for key in ("post_threat", "post_sim"):
            yv, Xv = absorb(p, "log_tokens", [key, "post_q", "post_lp"],
                            ["pair", "event_tau"])
            b, se = cluster_ols(yv, Xv, p.permaslug.factorize()[0])
            rows.append(dict(distance=lab, exposure=key, beta=b[0], se=se[0],
                             n=len(p)))
    return pd.DataFrame(rows)


def main():
    e = elasticity_variants()
    print("Elasticity variants\n" + e.round(3).to_string(index=False))
    v = diversion_variants()
    print("\nDiversion variants\n" + v.round(3).to_string(index=False))
    sc, share = successor_check()
    print(f"\nSuccessor releases (share of observations {share:.3f})\n"
          + sc.round(3).to_string(index=False))
    oo = outside_option()
    print("\nOutside-option sensitivity\n" + oo.round(3).to_string(index=False))
    dl = displacement_long()
    print("\nIncumbent volume around major open-weight releases\n"
          + dl.round(3).to_string(index=False))
    (ROOT / "robustness.json").write_text(json.dumps(
        {"elasticity": e.to_dict("records"), "diversion": v.to_dict("records"),
         "successor": sc.to_dict("records"), "successor_share": share,
         "displacement_long": dl.to_dict("records"),
         "outside_option": oo.to_dict("records")}, indent=1, default=float))


if __name__ == "__main__":
    main()
