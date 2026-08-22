"""Open-weight penetration over time, and the gap between volume shares and
revenue shares across providers."""
import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, FIG = ROOT / "data" / "derived", ROOT / "figures"


def open_share_series():
    lu = pd.read_csv(DER / "usage_long.csv", parse_dates=["date"])
    xs = pd.read_csv(DER / "models_cross_section.csv").set_index("permaslug")
    lu["open"] = lu.permaslug.map(xs.open_weight)
    lu = lu.dropna(subset=["open"])
    q = lu.groupby([lu.date.dt.to_period("Q"), "open"]).tokens.sum().unstack(fill_value=0)
    share = q[True] / (q[True] + q[False])
    share.index = share.index.to_timestamp()
    u = pd.read_csv(DER / "usage_daily.csv", parse_dates=["date"])
    u["open"] = u.permaslug.map(xs.open_weight)
    u = u.dropna(subset=["open"])
    recent = u[u.open].tokens.sum() / u.tokens.sum()
    return share, recent, u.date.max()


def provider_shares(top=10):
    xs = pd.read_csv(DER / "models_cross_section.csv")
    u = xs[xs.tokens.notna()].copy()
    u["spend"] = (u.prompt_tokens * u.p_in + u.completion_tokens * u.p_out) / 1e6
    g = u.groupby("author").agg(tokens=("tokens", "sum"), spend=("spend", "sum"),
                                open=("open_weight", "mean"))
    g["tok_share"] = g.tokens / g.tokens.sum()
    g["spend_share"] = g.spend / g.spend.sum()
    return g.nlargest(top, "tok_share").sort_values("tok_share")


def main():
    share, recent, last = open_share_series()
    g = provider_shares()

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 1.65),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    ax.plot(share.index, 100 * share.values, color=style.OPEN, lw=1.6,
            marker="o", ms=2.8)
    ax.set_ylabel("Open-weight share of tokens (%)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))

    ax2 = axes[1]
    y = np.arange(len(g))
    ax2.barh(y - 0.19, 100 * g.tok_share, height=0.36, color=style.BLUE,
             label="Tokens")
    ax2.barh(y + 0.19, 100 * g.spend_share, height=0.36, color=style.ORANGE,
             label="Spending")
    ax2.set_yticks(y)
    ax2.set_yticklabels(g.index, fontsize=6.4)
    ax2.set_xlabel("Share of platform total (%)")
    ax2.legend(loc="lower right", handlelength=1.1, handletextpad=0.5)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.5)
    style.save(fig, FIG / "fig2_structure.pdf")
    print(f"open share: {100 * share.iloc[0]:.0f}% -> {100 * share.min():.0f}% -> "
          f"{100 * share.iloc[-1]:.0f}%; daily panel {100 * recent:.0f}%")
    print(g[["tok_share", "spend_share"]].round(3).to_string())


if __name__ == "__main__":
    main()
