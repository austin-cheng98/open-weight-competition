"""Who serves a model. Open weights are supplied by many hosts at prices that
converge; proprietary models have one seller."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER, FIG = ROOT / "data" / "derived", ROOT / "figures"


def main():
    sup = pd.read_csv(DER / "supply_side.csv")
    xs = pd.read_csv(DER / "models_cross_section.csv")
    d = sup.merge(xs[["permaslug", "open_weight", "tokens"]], on="permaslug")
    d = d[d.tokens.notna()]
    d["w"] = d.tokens / d.tokens.sum()

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 1.7))
    ax = axes[0]
    bins = np.array([1, 2, 3, 5, 9, 17])
    for openw, colour, label, off in ((False, style.CLOSED, "Proprietary", -0.18),
                                      (True, style.OPEN, "Open-weight", 0.18)):
        s = d[d.open_weight == openw]
        idx = np.clip(np.searchsorted(bins, s.n_hosts, side="right") - 1, 0, len(bins) - 1)
        w = np.bincount(idx, weights=s.w, minlength=len(bins))
        w = w / w.sum()
        ax.bar(np.arange(len(bins)) + off, 100 * w, width=0.36, color=colour,
               label=label)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(["1", "2", "3-4", "5-8", "9-16", "17+"], fontsize=6.6)
    ax.set_xlabel("Independent hosting providers")
    ax.set_ylabel("Share of tokens (%)")
    ax.set_ylim(0, 1.28 * 100 * 0.42)
    ax.legend(loc="upper center", handlelength=1.1, handletextpad=0.5,
              ncol=2, columnspacing=0.9, fontsize=6.8)
    ax.grid(axis="x", visible=False)

    ax2 = axes[1]
    m = d[(d.n_hosts > 1) & d.price_ratio_p90_min.notna()]
    for openw, colour, label in ((False, style.CLOSED, "Proprietary"),
                                 (True, style.OPEN, "Open-weight")):
        s = m[m.open_weight == openw].sort_values("price_ratio_p90_min")
        if len(s) < 3:
            continue
        ax2.plot(s.price_ratio_p90_min, np.linspace(0, 100, len(s)), color=colour,
                 lw=1.6, label=label)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 20)
    ax2.set_xticks([1, 2, 5, 10, 20])
    ax2.set_xticklabels(["1x", "2x", "5x", "10x", "20x"])
    ax2.set_xlabel("Ratio of 90th-percentile to cheapest host price")
    ax2.set_ylabel("Cumulative share of models (%)")
    ax2.legend(loc="lower right", handlelength=1.1, handletextpad=0.5)
    fig.tight_layout(w_pad=1.5)
    style.save(fig, FIG / "fig3_supply.pdf")

    for openw, tag in ((True, "open"), (False, "closed")):
        s = d[d.open_weight == openw]
        print(f"{tag}: volume-weighted hosts {np.average(s.n_hosts, weights=s.w):.1f}, "
              f"median {s.n_hosts.median():.0f}, share with one host "
              f"{np.average(s.n_hosts == 1, weights=s.w):.2f}")
    print("median p90/min among multi-host models:",
          m.groupby("open_weight").price_ratio_p90_min.median().round(2).to_dict())


if __name__ == "__main__":
    main()
