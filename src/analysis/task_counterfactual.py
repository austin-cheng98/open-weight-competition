"""The open-weight counterfactual computed separately in each task market."""
import pathlib

import numpy as np
import pandas as pd

from src.analysis import capability as cap
from src.analysis import supply

DER = pathlib.Path(__file__).resolve().parents[2] / "data" / "derived"


def load(measure="spend"):
    t = pd.read_csv(DER / "task_markets.csv")
    t = t[t.measure == measure].copy()
    xs = pd.read_csv(DER / "models_cross_section.csv")
    imp, _ = cap.impute(xs[xs.tokens.notna()].copy())
    for c in cap.DIMS:
        imp[c] = imp[c + "_imp"]
    imp = imp.dropna(subset=cap.DIMS)
    imp["q_index"], _ = cap.quality_index(imp)
    t = t.merge(imp[["permaslug", "q_index"]], on="permaslug", how="left")
    t = t.dropna(subset=["q_index", "p_blend"])
    t = t[t.p_blend > 0]
    t["open_weight"] = t.open_weight.astype(bool)
    t["tier"] = pd.qcut(t.q_index, 3, labels=False)
    return t


def run(t, beta, sigma):
    out = []
    for task, g in t.groupby("task"):
        g = g.copy()
        s = g.share.to_numpy()
        s0 = max(1 - s.sum(), 1e-3)
        within = g.groupby("tier")["share"].transform("sum").to_numpy()
        delta = np.log(s) - np.log(s0) - sigma * np.log(s / within)
        nest, price = g.tier.to_numpy(), g.p_blend.to_numpy()
        base_s, _ = supply.nested_shares(delta, nest, sigma)
        keep = ~g.open_weight.to_numpy()
        if keep.sum() < 2 or keep.all():
            continue
        cf_s, _ = supply.nested_shares(delta[keep], nest[keep], sigma)
        out.append(dict(
            task=task, macro=g.macro.iloc[0], weight=g.task_share_of_total.iloc[0],
            open_share=float(s[~keep].sum() / s.sum()),
            price_base=float((base_s * price).sum() / base_s.sum()),
            price_cf=float((cf_s * price[keep]).sum() / cf_s.sum()),
            cs_base=supply.consumer_surplus(delta, nest, sigma, beta),
            cs_cf=supply.consumer_surplus(delta[keep], nest[keep], sigma, beta)))
    d = pd.DataFrame(out)
    d["d_price"] = d.price_cf / d.price_base - 1
    d["d_cs"] = d.cs_cf / d.cs_base - 1
    return d


if __name__ == "__main__":
    t = load()
    d = run(t, -1.0, 0.4)
    by = (d.assign(w=d.weight)
            .groupby("macro")
            .apply(lambda x: pd.Series({
                "open_share": np.average(x.open_share, weights=x.w),
                "d_price": np.average(x.d_price, weights=x.w),
                "d_cs": np.average(x.d_cs, weights=x.w),
                "weight": x.w.sum()}), include_groups=False))
    print(by.round(3).to_string())
    print(f"\nspend-weighted overall: price {np.average(d.d_price, weights=d.weight):+.3f}, "
          f"surplus {np.average(d.d_cs, weights=d.weight):+.3f}")
