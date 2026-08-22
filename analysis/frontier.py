"""How capable and how cheap a hypothetical open-weight model has to be before it
moves the proprietary side of the market."""
import numpy as np

from analysis import counterfactual as cfm
from analysis import supply

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
