"""Capability vectors, pairwise technological similarity, and the quality-adjusted
open-weight exposure index."""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"

DIMS = ["q_intelligence", "q_coding", "q_agentic"]
DIMS_WIDE = DIMS + ["q_frontend", "q_appbuild"]
SHRINK = 0.10   # ridge on the capability covariance before whitening


def impute(df, dims=DIMS, aux=("q_frontend", "q_appbuild"), min_obs=1):
    """Fill missing capability dimensions by ridge regression on the dimensions a
    model does have. Returns the filled frame and leave-one-out fit quality."""
    cols = list(dims) + [c for c in aux if c in df]
    Z = df[cols].astype(float)
    mu, sd = Z.mean(), Z.std(ddof=0)
    Zs = (Z - mu) / sd
    have = Zs.notna()
    out = Zs.copy()
    diag = {}
    for target in dims:
        miss = Zs[target].isna()
        if not miss.any():
            continue
        for pattern, idx in Zs[miss].notna().groupby(list(Zs.columns), sort=False).groups.items():
            preds = [c for c, ok in zip(Zs.columns, pattern) if ok and c != target]
            if not preds:
                continue
            train = Zs.dropna(subset=[target] + preds)
            A = train[preds].to_numpy()
            b = np.linalg.solve(A.T @ A + 1e-2 * np.eye(len(preds)), A.T @ train[target].to_numpy())
            out.loc[idx, target] = Zs.loc[idx, preds].to_numpy() @ b
            r = train[target].to_numpy() - A @ b
            diag[(target, tuple(preds))] = (1 - r.var() / train[target].var(), len(idx))
    filled = out * sd + mu
    keep = have[list(dims)].sum(axis=1) + have[[c for c in aux if c in df]].sum(axis=1)
    filled = filled[keep >= min_obs]
    res = df.copy()
    for c in dims:
        res.loc[filled.index, c + "_imp"] = filled[c]
    res["n_scored"] = have.sum(axis=1)
    return res, diag


def capability_matrix(df, dims=DIMS):
    """Standardised capability vectors for models scored on every dimension."""
    d = df.dropna(subset=dims).copy()
    Z = (d[dims] - d[dims].mean()) / d[dims].std(ddof=0)
    return d, Z.to_numpy()


def whitened_distance(Z, shrink=SHRINK):
    """Mahalanobis distance with a ridge-shrunk covariance."""
    S = np.cov(Z, rowvar=False)
    S = (1 - shrink) * S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    Si = np.linalg.inv(S)
    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt(np.einsum("ijk,kl,ijl->ij", diff, Si, diff))


def euclidean_distance(Z):
    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def cosine_distance(Z):
    n = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return 1 - n @ n.T


def similarity(D, lam=None):
    """S = exp(-lambda d); lambda normalised so the median pair maps to exp(-1)."""
    off = D[~np.eye(len(D), dtype=bool)]
    lam = lam if lam is not None else 1.0 / np.median(off)
    return np.exp(-lam * D), lam


def quality_index(df, dims=DIMS):
    """First principal component of the capability block, rescaled to the
    intelligence score's units."""
    d = df.dropna(subset=dims)
    Z = ((d[dims] - d[dims].mean()) / d[dims].std(ddof=0)).to_numpy()
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    w = vt[0] * np.sign(vt[0].sum())
    pc = Z @ w
    out = pd.Series(np.nan, index=df.index)
    scale = d["q_intelligence"].std(ddof=0) / pc.std(ddof=0)
    out.loc[d.index] = pc * scale + d["q_intelligence"].mean()
    return out, w


def exposure(S, models, open_flag, quality, price, avail, eta=1.0):
    """C_j = sum over available open-weight rivals of S_jk (Q_k / P_k)^eta."""
    A = np.where(open_flag & avail, (quality / price) ** eta, 0.0)
    np.fill_diagonal(S, 0.0)
    return pd.Series(S @ A, index=models)


if __name__ == "__main__":
    xs = pd.read_csv(DER / "models_cross_section.csv")
    xs = xs[xs.tokens.notna()]
    d, Z = capability_matrix(xs)
    D = whitened_distance(Z)
    S, lam = similarity(D)
    q, w = quality_index(xs)
    print(f"{len(d)} scored models; lambda={lam:.3f}; PC1 weights={np.round(w, 3)}")
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"similarity: mean {off.mean():.3f}  p10 {np.quantile(off, .1):.3f}  "
          f"p90 {np.quantile(off, .9):.3f}")
    names = d.name.to_numpy()
    i = int(np.argmax(d.name.str.contains("Claude Opus 5", na=False).to_numpy()))
    order = np.argsort(-S[i])[:6]
    print(f"nearest to {names[i]}: " + ", ".join(f"{names[o]} ({S[i, o]:.2f})" for o in order))
