"""What the market looks like without open-weight models. Counterfactual A removes
them from the choice set at posted prices; counterfactual B additionally lets
proprietary providers reprice under Bertrand competition."""
import pathlib

import numpy as np
import pandas as pd

from src.analysis import demand
from src.analysis import supply

DER = pathlib.Path(__file__).resolve().parents[2] / "data" / "derived"


def cross_section(df=None):
    """The most recent day on which the panel captures the whole leaderboard."""
    df = demand.build_sample() if df is None else df
    inside = df.groupby("date").inside.first()
    good = inside[inside >= 0.9 * inside.median()].index
    d = df[df.date == max(good)].copy()
    d["share_out"] = 1 - d.share.sum()
    return d.reset_index(drop=True)


def invert(d, sigma):
    """Mean utilities that reproduce the observed shares."""
    return np.log(d.share) - np.log(d.share_out) - sigma * np.log(d.share_within)


def run(d, beta_p, sigma, reprice=False):
    delta = invert(d, sigma).to_numpy()
    price = d.p_blend.to_numpy()
    nest = d.tier.to_numpy()
    firm = d.author.to_numpy()
    openw = d.open_weight.to_numpy().astype(bool)
    s0, s0_jg = supply.nested_shares(delta, nest, sigma)

    base = dict(price_paid=float((s0 * price).sum() / s0.sum()),
                cs=supply.consumer_surplus(delta, nest, sigma, beta_p),
                hhi=supply.hhi(s0), inside=float(s0.sum()))

    keep = ~openw
    if not reprice:
        p_cf, s_cf, d_cf, live = supply.equilibrium(
            delta, price, price, firm, nest, np.ones(len(price), bool),
            sigma, beta_p, keep=keep)
    else:
        mc, mk, _ = supply.markups(s0, s0_jg, nest, price, firm, openw, sigma, beta_p)
        p_cf, s_cf, d_cf, live = supply.equilibrium(
            delta, price, mc, firm, nest, openw, sigma, beta_p, keep=keep)

    cf = dict(price_paid=float((s_cf * p_cf).sum() / s_cf.sum()),
              cs=supply.consumer_surplus(d_cf, nest[live], sigma, beta_p),
              hhi=supply.hhi(s_cf), inside=float(s_cf.sum()))
    return base, cf, (p_cf, price[live], live)


def diagnostics(d, beta_p, sigma):
    delta = invert(d, sigma).to_numpy()
    s, s_jg = supply.nested_shares(delta, d.tier.to_numpy(), sigma)
    mc, markup, J = supply.markups(s, s_jg, d.tier.to_numpy(), d.p_blend.to_numpy(),
                                   d.author.to_numpy(), d.open_weight.to_numpy(),
                                   sigma, beta_p)
    own = np.diag(J) * d.p_blend.to_numpy() / s
    prop = ~d.open_weight.to_numpy()
    return dict(median_own_elasticity=float(np.median(own[prop])),
                share_inelastic=float((np.abs(own[prop]) < 1).mean()),
                share_negative_mc=float((mc[prop] < 0).mean()),
                median_lerner=float(np.median((markup[prop] / d.p_blend.to_numpy()[prop])[mc[prop] > 0]))
                if (mc[prop] > 0).any() else np.nan)


if __name__ == "__main__":
    d = cross_section()
    print(f"cross-section: {len(d)} models on {d.date.iloc[0].date()}, "
          f"outside share {d.share_out.iloc[0]:.3f}, open-weight token share "
          f"{d.loc[d.open_weight, 'share'].sum() / d.share.sum():.3f}")
    rows = []
    for beta in (-0.5, -1.0, -2.0):
        for sigma in (0.0, 0.4, 0.7):
            base, cf, _ = run(d, beta, sigma)
            dg = diagnostics(d, beta, sigma)
            rows.append(dict(beta=beta, sigma=sigma,
                             d_price=cf["price_paid"] / base["price_paid"] - 1,
                             d_cs=cf["cs"] / base["cs"] - 1,
                             d_hhi=cf["hhi"] - base["hhi"],
                             own_el=dg["median_own_elasticity"],
                             neg_mc=dg["share_negative_mc"]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
