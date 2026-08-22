"""Supply side: Bertrand-Nash markups for proprietary models and counterfactual
equilibria when the open-weight choice set changes."""
import numpy as np
import pandas as pd


def nested_shares(delta, nest, sigma):
    """Shares implied by mean utilities under the nested logit."""
    lam = 1 - sigma
    e = np.exp(delta / lam)
    key = pd.Series(nest)
    tot = pd.Series(e).groupby(key).sum()
    D = key.map(tot).to_numpy()
    inc = tot ** lam
    s_g = key.map(inc / (1 + inc.sum())).to_numpy()
    s_jg = e / D
    return s_jg * s_g, s_jg


def jacobian(s, s_jg, nest, sigma, beta_p, price):
    """J[j, k] = d s_j / d p_k, for log price entering utility."""
    n = len(s)
    same = (nest[:, None] == nest[None, :]).astype(float)
    lam = 1 - sigma
    E = beta_p * np.tile(s, (n, 1))
    E += same * beta_p * (sigma / lam) * np.tile(s_jg, (n, 1))
    np.fill_diagonal(E, beta_p * (1 / lam - (sigma / lam) * s_jg - s))
    return (E * s[:, None]) / price[None, :]


def _markup(J, s, firm, fixed):
    own = (firm[:, None] == firm[None, :]) & ~fixed[:, None] & ~fixed[None, :]
    Delta = -(J.T * own)
    idx = ~fixed
    m = np.zeros(len(s))
    m[idx] = np.linalg.solve(Delta[np.ix_(idx, idx)], s[idx])
    return m


def markups(s, s_jg, nest, price, firm, competitive, sigma, beta_p):
    """Marginal costs implied by the pricing first-order condition. Models flagged
    `competitive` are treated as priced at cost by their hosts."""
    J = jacobian(s, s_jg, nest, sigma, beta_p, price)
    m = _markup(J, s, firm, competitive)
    return m, price - m, J


def equilibrium(delta0, price0, mc, firm, nest, fixed, sigma, beta_p,
                keep=None, step=0.4, tol=1e-9, iters=2000):
    """Bertrand equilibrium prices. `fixed` products keep their observed price;
    `keep` optionally restricts the choice set."""
    live = np.ones(len(price0), bool) if keep is None else keep
    d0, p0 = delta0[live], price0[live]
    mc_l, firm_l, nest_l, fix_l = mc[live], firm[live], nest[live], fixed[live]
    p = p0.copy()
    for _ in range(iters):
        d = d0 + beta_p * (np.log(p) - np.log(p0))
        s, s_jg = nested_shares(d, nest_l, sigma)
        J = jacobian(s, s_jg, nest_l, sigma, beta_p, p)
        target = np.where(fix_l, p, np.clip(mc_l + _markup(J, s, firm_l, fix_l), 1e-4, None))
        p_new = np.exp((1 - step) * np.log(p) + step * np.log(target))
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new
    d = d0 + beta_p * (np.log(p) - np.log(p0))
    s, s_jg = nested_shares(d, nest_l, sigma)
    return p, s, d, live


def consumer_surplus(delta, nest, sigma, beta_p):
    """Log-sum expected utility in dollars per million tokens."""
    lam = 1 - sigma
    inc = pd.Series(np.exp(delta / lam)).groupby(pd.Series(nest)).sum() ** lam
    return np.log(1 + inc.sum()) / abs(beta_p)


def hhi(shares):
    s = np.asarray(shares, float)
    s = s / s.sum()
    return 1e4 * float((s ** 2).sum())
