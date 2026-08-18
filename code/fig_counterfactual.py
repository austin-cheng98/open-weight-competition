"""Figure 4: what happens to prices, surplus and shares when open-weight models
leave the choice set."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import counterfactual as cfm
import demand
import style
import task_counterfactual as tc

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
BETA = cfm.__dict__.get("BETA", -1.0)


def main():
    df = demand.build_sample()
    d = cfm.cross_section(df)
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.35),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    tk = tc.run(tc.load(), BETA, 0.4).sort_values("open_share")
    ax = axes[0]
    colours = {"agent": style.BLUE, "code": style.ORANGE, "data": style.AQUA,
               "general": style.GREY}
    for macro, g in tk.groupby("macro"):
        ax.scatter(100 * g.open_share, 100 * g.d_price, s=14 + 900 * g.weight,
                   c=colours[macro], alpha=0.75, linewidths=0.4, edgecolors="white",
                   label=macro.capitalize(), zorder=3)
    z = np.polyfit(tk.open_share, tk.d_price, 1)
    xs_ = np.linspace(tk.open_share.min(), tk.open_share.max(), 20)
    ax.plot(100 * xs_, 100 * np.polyval(z, xs_), color=style.INK, lw=1.1,
            ls=(0, (4, 2)), zorder=2)
    ax.set_xlabel("Open-weight share of task spend (%)")
    ax.set_ylabel("Price increase without open weights (%)")
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2, labelspacing=0.2,
              ncol=2, columnspacing=0.8)

    ax2 = axes[1]
    sig = np.round(np.arange(0.0, 0.75, 0.05), 2)
    tokens = df[df.date == d.date.iloc[0]].tokens.sum()
    dp, dcs = [], []
    for i, (beta, shade) in enumerate(zip((-0.5, -1.0, -2.0),
                                          (style.SEQ[2], style.SEQ[4], style.SEQ[6]))):
        loss = []
        for sg in sig:
            base, cf, _ = cfm.run(d, beta, sg)
            loss.append((base["cs"] - cf["cs"]) * tokens / 1e12)
            if beta == BETA:
                dp.append(100 * (cf["price_paid"] / base["price_paid"] - 1))
                dcs.append(100 * (cf["cs"] / base["cs"] - 1))
        ax2.plot(sig, loss, color=shade, lw=1.6,
                 label=f"$\\beta_p={beta:.1f}$")
    ax2.set_xlabel("Within-tier substitution parameter $\\sigma$")
    ax2.set_ylabel("Surplus lost (\\$m per day)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", handlelength=1.2, handletextpad=0.5, labelspacing=0.25)
    fig.tight_layout(w_pad=1.5)
    style.save(fig, FIG / "fig4_counterfactual.pdf")
    print(f"aggregate: price +{dp[8]:.0f}% and surplus {dcs[8]:.0f}% at sigma=0.40")
    print(f"task level: price +{100 * np.average(tk.d_price, weights=tk.weight):.1f}%, "
          f"surplus {100 * np.average(tk.d_cs, weights=tk.weight):.1f}%")
    print(f"slope of price effect on open share: {z[0]:.2f}")


if __name__ == "__main__":
    main()
