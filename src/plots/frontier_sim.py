"""How capable and how cheap a hypothetical open-weight model has to be before it
moves the proprietary side of the market."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import counterfactual as cfm
from src.analysis import demand
from src.plots import style
from src.analysis import supply

FIG = pathlib.Path(__file__).resolve().parents[2] / "figures"
BETA, SIGMA = -1.0, 0.4


def simulate(d, rho, price, beta=BETA, sigma=SIGMA):
    """Insert one open-weight entrant priced at `price` with capability
    (1 - rho) times the frontier, and return the proprietary response."""
    delta = cfm.invert(d, sigma).to_numpy()
    q = d.q_index.to_numpy()
    p = d.p_blend.to_numpy()
    nest = d.tier.to_numpy()
    prop = ~d.open_weight.to_numpy().astype(bool)
    base_s, _ = supply.nested_shares(delta, nest, sigma)

    # the entrant's mean utility, read off the fitted capability-price gradient
    X = np.column_stack([np.ones(len(q)), q, np.log(p)])
    coef = np.linalg.lstsq(X, delta, rcond=None)[0]
    q_new = (1 - rho) * q.max()
    d_new = coef[0] + coef[1] * q_new + coef[2] * np.log(price)
    tier_new = int(np.clip(np.digitize(q_new, np.quantile(q, [1 / 3, 2 / 3])), 0, 2))

    delta2 = np.append(delta, d_new)
    nest2 = np.append(nest, tier_new)
    p2 = np.append(p, price)
    s2, _ = supply.nested_shares(delta2, nest2, sigma)
    prop2 = np.append(prop, False)
    return dict(
        prop_share=float(s2[prop2].sum() / base_s[prop].sum() - 1),
        entrant_share=float(s2[-1]),
        price_paid=float((s2 * p2).sum() / s2.sum() /
                         ((base_s * p).sum() / base_s.sum()) - 1),
        cs=float(supply.consumer_surplus(delta2, nest2, sigma, beta) /
                 supply.consumer_surplus(delta, nest, sigma, beta) - 1))


def main():
    d = cfm.cross_section()
    frontier_p = float(d.loc[d.q_index.idxmax(), "p_blend"])
    rhos = np.linspace(0.0, 0.20, 21)
    prices = np.geomspace(0.02, frontier_p, 25)
    Z = np.array([[simulate(d, r, p)["prop_share"] for p in prices] for r in rhos])

    fig, ax = plt.subplots(figsize=(style.WIDTH * 0.55, 1.9))
    cs = ax.contourf(prices, 100 * rhos, 100 * Z, levels=12, cmap="Blues_r")
    line = ax.contour(prices, 100 * rhos, 100 * Z, levels=[-10, -5, -2],
                      colors=style.INK, linewidths=0.7)
    ax.clabel(line, fmt="%d%%", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Price of the hypothetical open model (\\$ per million tokens)")
    ax.set_ylabel("Capability shortfall vs frontier (%)")
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("Change in proprietary token share (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    style.save(fig, FIG / "fig8_frontier.pdf")

    for rho in (0.0, 0.05, 0.10, 0.20):
        row = [simulate(d, rho, p) for p in (0.05, 0.5, frontier_p)]
        print(f"rho={rho:.2f}  " + "  ".join(
            f"p={p:5.2f}: share {100 * r['prop_share']:+5.1f}%, entrant {100 * r['entrant_share']:4.1f}%"
            for p, r in zip((0.05, 0.5, frontier_p), row)))
    print(f"frontier price used: {frontier_p:.2f}")


if __name__ == "__main__":
    main()
