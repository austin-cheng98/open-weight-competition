"""Figure 2: posted prices are almost never revised, and volume responds weakly
when they are."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from event_study import absorb, cluster_ols

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, FIG = ROOT / "data" / "derived", ROOT / "figures"
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


def main():
    pp = pd.read_csv(DER / "price_panel.csv", parse_dates=["snapshot"])
    pp = pp[pp.p_in > 0]
    surv = survival(pp)
    grid = np.arange(0, 391, 30)
    s_open = km(surv, grid)

    panel = pd.read_csv(DER / "panel_long.csv", parse_dates=["date"])
    es, n_ev = event_study(panel)

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 1.6))
    ax = axes[0]
    ax.step(grid, s_open, where="post", color=style.INK, lw=1.6)
    ax.fill_between(grid, 0, s_open, step="post", color=style.GREY, alpha=0.14, lw=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days since a model first appears in the catalogue")
    ax.set_ylabel("Share still at its launch price")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])

    ax2 = axes[1]
    if len(es):
        ax2.axhline(0, color=style.GREY, lw=0.7)
        ax2.axvline(-0.5, color=style.GREY, lw=0.7, ls=(0, (3, 3)))
        ax2.errorbar(es.k, es.beta, yerr=1.96 * es.se, fmt="o", ms=3.2,
                     color=style.BLUE, ecolor=style.BLUE, elinewidth=0.9, capsize=1.6)
        ax2.set_xlabel("Observations since the price change")
        ax2.set_ylabel("Response of log volume")
    fig.tight_layout(w_pad=1.6)
    style.save(fig, FIG / "fig5_prices.pdf")
    print(f"price-change events used: {n_ev}")
    if len(es):
        print(es.round(3).to_string(index=False))
    print(f"survival at 180 days: {s_open[grid == 180][0]:.3f}; "
          f"at 360: {s_open[grid == 360][0]:.3f}")


if __name__ == "__main__":
    main()
