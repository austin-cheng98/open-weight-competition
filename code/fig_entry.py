"""Figure 3: entrants take a large share of volume within weeks, yet incumbent
volume keeps growing and no diversion toward close rivals is detectable."""
import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import demand
import event_study as es
import style
from event_study import absorb, cluster_ols

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, FIG = ROOT / "data" / "derived", ROOT / "figures"


def volume_panel(ax):
    u = pd.read_csv(DER / "usage_daily.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs["created_at"] = pd.to_datetime(xs["created_at"], format="mixed", utc=True).dt.tz_localize(None)
    u = u.merge(xs[["permaslug", "created_at", "open_weight"]], on="permaslug", how="left")
    u = u[u.date <= u.date.max() - pd.Timedelta(days=1)]
    d0 = u.date.min()
    u["group"] = np.where(u.created_at <= d0, "incumbent",
                          np.where(u.open_weight.fillna(False), "entrant_open", "entrant_closed"))
    piv = (u.groupby(["date", "group"]).tokens.sum().unstack(fill_value=0) / 1e12)
    order = ["incumbent", "entrant_closed", "entrant_open"]
    labels = ["Incumbents", "Entrants (proprietary)", "Entrants (open-weight)"]
    colours = [style.GREY, style.CLOSED, style.OPEN]
    ax.stackplot(piv.index, [piv[c] for c in order], colors=colours, alpha=0.9,
                 labels=labels, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Tokens served per day (trillions)")
    ax.set_ylim(0, 1.42 * piv.sum(axis=1).max())
    ax.legend(loc="upper left", handlelength=1.1, handletextpad=0.5,
              labelspacing=0.25, ncol=2, columnspacing=0.9, fontsize=6.6,
              framealpha=0.0)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.margins(x=0)


def diversion_panel(ax):
    u, xs, scored = es.load()
    S, _, _ = es.similarity_table(scored)
    p, ev = es.build_pairs(u, scored, S)
    p["event_tau"] = p.event + "|" + p.tau.astype(str)
    p["post_q"] = p.post * p.q_j
    p["post_lp"] = p.post * p.logp_j
    taus = [t for t in sorted(p.tau.unique()) if t != -1]
    for t in taus:
        p[f"k{t}"] = (p.tau == t) * p.threat
        p[f"c{t}"] = (p.tau == t) * p.q_j
        p[f"d{t}"] = (p.tau == t) * p.logp_j
    cols = [f"k{t}" for t in taus] + [f"c{t}" for t in taus] + [f"d{t}" for t in taus]
    yv, Xv = absorb(p, "log_tokens", cols, ["pair", "event_tau"])
    b, se = cluster_ols(yv, Xv, p.permaslug.factorize()[0])
    n = len(taus)
    r = pd.DataFrame({"tau": taus, "beta": b[:n], "se": se[:n]})
    r = pd.concat([r, pd.DataFrame({"tau": [-1], "beta": [0.0], "se": [0.0]})]).sort_values("tau")
    ax.axhline(0, color=style.GREY, lw=0.7)
    ax.axvline(-0.5, color=style.GREY, lw=0.7, ls=(0, (3, 3)))
    ax.fill_between(r.tau, r.beta - 1.96 * r.se, r.beta + 1.96 * r.se,
                    color=style.BLUE, alpha=0.16, lw=0)
    ax.plot(r.tau, r.beta, color=style.BLUE, lw=1.4, marker="o", ms=2.6)
    ax.set_xlabel("Days since the release")
    ax.set_ylabel("Incumbent log volume")
    return r, ev


def main():
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 1.62))
    volume_panel(axes[0])
    r, ev = diversion_panel(axes[1])
    fig.tight_layout(w_pad=1.6)
    style.save(fig, FIG / "fig5_entry.pdf")
    post = r[r.tau >= 0]
    print(f"{len(ev)} events; mean post-release coefficient "
          f"{post.beta.mean():.3f}, mean 95% half-width {1.96 * post.se.mean():.3f}")


if __name__ == "__main__":
    main()
