"""Price rigidity and the dynamics of volume around a posted-price revision."""
import numpy as np
import pandas as pd

from analysis.event_study import absorb, cluster_ols

LEADS, LAGS = 4, 6


def survival(pp):
    """Share of models whose posted price is still at its initial level after t days."""
    pp = pp.sort_values(["permaslug", "snapshot"])
    out = []
    for slug, g in pp.groupby("permaslug"):
        if len(g) < 3:
            continue
        p0 = g.p_in.iloc[0]
        t0 = g.snapshot.iloc[0]
        changed = g.snapshot[(g.p_in - p0).abs() / p0 > 1e-6]
        end = (changed.iloc[0] - t0).days if len(changed) else np.nan
        out.append(((g.snapshot.iloc[-1] - t0).days, end))
    return pd.DataFrame(out, columns=["observed", "change_at"])


def km(d, grid):
    """Kaplan-Meier survival of the initial posted price, evaluated on `grid`."""
    s, surv = 1.0, [1.0]
    for lo, hi in zip(grid[:-1], grid[1:]):
        at_risk = ((d.observed >= lo) & ((d.change_at >= lo) | d.change_at.isna())).sum()
        events = ((d.change_at >= lo) & (d.change_at < hi)).sum()
        if at_risk > 0:
            s *= 1 - events / at_risk
        surv.append(s)
    return np.array(surv)


def event_study(panel):
    """Log volume around a posted price change, in event time. Restricted to the
    captures that report non-overlapping weekly buckets, so the dynamics are not
    an artefact of a trailing aggregation window."""
    d = panel[panel.weekly == 1].sort_values(["permaslug", "date"]).copy()
    g = d.groupby("permaslug")
    d["dlp"] = g.ln_p.transform(lambda s: s.diff())
    events = d[(d.dlp.abs() > 0.02)][["permaslug", "date", "dlp"]]
    rows = []
    for _, e in events.iterrows():
        sub = d[(d.permaslug == e.permaslug)].copy()
        sub["k"] = sub.groupby("permaslug").cumcount() - \
            sub.index.get_indexer([sub.index[sub.date == e.date][0]])[0]
        sub = sub[(sub.k >= -LEADS) & (sub.k <= LAGS)]
        if sub.k.min() > -LEADS or sub.k.max() < 1:
            continue
        sub = sub.assign(event=f"{e.permaslug}|{e.date.date()}", dlp=e.dlp)
        rows.append(sub)
    if not rows:
        return pd.DataFrame(), 0
    ev = pd.concat(rows)
    ev["day"] = ev.date.astype(str)
    ks = [k for k in sorted(ev.k.unique()) if k != -1]
    for k in ks:
        ev[f"e{k}"] = (ev.k == k) * ev.dlp
    yv, Xv = absorb(ev, "ln_tokens", [f"e{k}" for k in ks], ["event", "day"])
    b, se = cluster_ols(yv, Xv, ev.permaslug.factorize()[0])
    return pd.DataFrame({"k": ks, "beta": b, "se": se}), ev.event.nunique()
