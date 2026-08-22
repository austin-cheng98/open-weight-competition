"""The own-price elasticity across specifications."""
import json, pathlib

import matplotlib.pyplot as plt
import numpy as np

import style

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

ORDER = [("Fixed-weight price (baseline)", "fixed-weight price"),
         ("Input price", "input price"),
         ("Output price", "output price"),
         ("Realised blended price", "realised blended price"),
         ("Requests, not tokens", "requests"),
         ("Weekly captures only", "weekly captures only"),
         ("Trailing captures only", "trailing captures only"),
         ("Open-weight models", "open-weight models"),
         ("Proprietary models", "proprietary models"),
         ("2025 onwards", "2025 onwards")]


def main():
    rb = json.loads((ROOT / "robustness.json").read_text())
    e = {r["spec"]: r for r in rb["elasticity"]}
    lo, hi = rb["leave_one_out"]["min"], rb["leave_one_out"]["max"]
    rows = [(lab, e[k]["beta"], e[k]["se"]) for lab, k in ORDER][::-1]
    y = np.arange(len(rows))
    base = e["fixed-weight price"]["beta"]

    fig, ax = plt.subplots(figsize=(style.WIDTH * 0.66, 2.15))
    ax.axvspan(lo, hi, color=style.BLUE, alpha=0.10, lw=0)
    ax.axvline(base, color=style.GREY, lw=0.7, ls=(0, (3, 3)))
    ax.axvline(0, color=style.INK, lw=0.7)
    for i, (lab, b, se) in enumerate(rows):
        colour = style.INK if i == len(rows) - 1 else style.BLUE
        ax.errorbar(b, i, xerr=1.96 * se, fmt="o", ms=3.2, color=colour,
                    ecolor=colour, elinewidth=1.0, capsize=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlabel("Own-price elasticity of token volume")
    ax.set_xlim(-1.0, 0.05)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.06)
    fig.tight_layout()
    style.save(fig, FIG / "fig4_elasticity.pdf")
    print(f"baseline {base:.3f}; leave-one-out band [{lo:.3f}, {hi:.3f}]")
    print(f"range across specifications: {min(r[1] for r in rows):.3f} to "
          f"{max(r[1] for r in rows):.3f}")


if __name__ == "__main__":
    main()
