"""Figure 1: the price of capability. Where open and proprietary models sit on the
price-capability plane, and the cost of the cheapest model at each capability level."""
import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import capability as cap
import style

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, FIG = ROOT / "data" / "derived", ROOT / "figures"


def frontier(d, grid):
    """Cheapest price available at capability >= each grid point."""
    out = []
    for g in grid:
        s = d[d.q_index >= g]
        out.append(s.p_blend.min() if len(s) else np.nan)
    return np.array(out)


def main():
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs = xs[xs.tokens.notna()].copy()
    xs, _ = cap.impute(xs)
    for c in cap.DIMS:
        xs[c] = xs[c + "_imp"]
    xs = xs.dropna(subset=cap.DIMS + ["p_blend"])
    xs = xs[xs.p_blend > 0]
    xs["q_index"], _ = cap.quality_index(xs)

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 1.65),
                             gridspec_kw={"width_ratios": [1.32, 1]})
    ax = axes[0]
    tot = xs.tokens.sum()
    for openw, colour, label in ((False, style.CLOSED, "Proprietary"),
                                 (True, style.OPEN, "Open-weight")):
        s = xs[xs.open_weight == openw]
        ax.scatter(s.q_index, s.p_blend, s=8 + 260 * (s.tokens / tot) ** 0.5,
                   c=colour, alpha=0.55, linewidths=0.4, edgecolors="white",
                   label=label, zorder=3)
    grid = np.linspace(xs.q_index.quantile(0.02), xs.q_index.max(), 120)
    for openw, colour in ((False, style.CLOSED), (True, style.OPEN)):
        f = frontier(xs[xs.open_weight == openw], grid)
        ax.plot(grid, f, color=colour, lw=1.5, alpha=0.9, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Capability index")
    ax.set_ylabel("Price ($ per million tokens)")
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.2,
              labelspacing=0.25)
    for name, dx, dy, ha in (("Claude Opus 5", -8, 2.1, "center"),
                             ("DeepSeek V4 Flash 0731", -5.5, 0.21, "center")):
        m = xs[xs.name.str.contains(name.split(" 0")[0], regex=False)]
        if len(m):
            m = m.iloc[0]
            ax.annotate(name, (m.q_index, m.p_blend), fontsize=6, color=style.MUTED,
                        xytext=(m.q_index + dx, m.p_blend * dy), ha=ha, zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=style.GREY,
                                        shrinkA=1, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=0.85))

    ax2 = axes[1]
    xs["release"] = pd.to_datetime(xs.created_at, format="mixed", utc=True).dt.tz_localize(None)
    days = pd.date_range(xs.release.quantile(0.05), xs.release.max(), freq="7D")
    fr = {}
    for openw, colour, label in ((False, style.CLOSED, "Proprietary"),
                                 (True, style.OPEN, "Open-weight")):
        s = xs[xs.open_weight == openw]
        f = [s.loc[s.release <= d, "q_index"].max() for d in days]
        fr[openw] = np.array(f, float)
        ax2.plot(days, f, color=colour, lw=1.6, label=label)
    ax2.set_ylabel("Best capability available")
    ax2.set_xlabel("")
    ax2.fill_between(days, fr[True], fr[False], color=style.GREY, alpha=0.16, lw=0)
    ax2.legend(loc="lower right", handletextpad=0.4)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    fig.tight_layout(w_pad=1.6)
    style.save(fig, FIG / "fig1_price_capability.pdf")

    lags = []
    for i, d in enumerate(days):
        q_open = fr[True][i]
        reach = [days[j] for j in range(len(days)) if fr[False][j] >= q_open]
        if reach and np.isfinite(q_open):
            lags.append((d - reach[0]).days)
    top = xs.sort_values("q_index").tail(1).iloc[0]
    print(f"frontier model: {top['name']} q={top.q_index:.1f} p={top.p_blend:.2f}")
    ratio = []
    for g in grid:
        c = xs[(xs.q_index >= g) & (~xs.open_weight)].p_blend.min()
        o = xs[(xs.q_index >= g) & (xs.open_weight)].p_blend.min()
        if np.isfinite(c) and np.isfinite(o):
            ratio.append(c / o)
    print(f"median proprietary/open price ratio at equal capability: {np.median(ratio):.2f}x")
    print(f"open-weight capability lag: median {np.median(lags):.0f} days "
          f"(last year {np.median(lags[-52:]):.0f} days)")


if __name__ == "__main__":
    main()
