"""Merge the archived usage series with the archived price catalogue to obtain a
model-by-date panel with within-model price variation."""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
MAX_GAP_DAYS = 10


def build():
    u = pd.read_csv(DER / "usage_long.csv", parse_dates=["date"])
    u["date"] = pd.to_datetime(u["date"], utc=True).dt.tz_localize(None).dt.normalize()
    p = pd.read_csv(DER / "price_panel.csv", parse_dates=["snapshot"])
    p = p[(p.p_in > 0) & (p.p_out > 0)][["permaslug", "snapshot", "p_in", "p_out",
                                         "open_weight", "context_length"]]
    d = pd.merge_asof(u.sort_values("date"), p.sort_values("snapshot"),
                      left_on="date", right_on="snapshot", by="permaslug",
                      direction="nearest", tolerance=pd.Timedelta(days=MAX_GAP_DAYS))
    d = d.dropna(subset=["p_in"])
    tot = d.groupby("date")["tokens"].sum().rename("market")
    d = d.merge(tot, on="date")
    d["share"] = d.tokens / d.market
    w = d.prompt_tokens / (d.prompt_tokens + d.completion_tokens).replace(0, np.nan)
    d["mix_in"] = w
    # fixed-weight price index: posted prices only, no feedback from the usage mix
    wbar = w.mean()
    d["p_fix"] = wbar * d.p_in + (1 - wbar) * d.p_out
    d["p_blend"] = w * d.p_in + (1 - w) * d.p_out
    d["ln_p"] = np.log(d.p_fix)
    d["ln_s"] = np.log(d.share)
    d["ln_tokens"] = np.log(d.tokens.clip(lower=1))
    d["ln_req"] = np.log(d.requests.clip(lower=1))
    d["month"] = d.date.dt.to_period("M").astype(str)
    # openness from the current catalogue where available: the field is absent from
    # the earliest archived catalogues
    xs = pd.read_csv(DER / "models_cross_section.csv").set_index("permaslug")
    d["open_weight"] = d.permaslug.map(xs.open_weight).fillna(d.open_weight).astype(bool)
    return d.sort_values(["permaslug", "date"])


if __name__ == "__main__":
    d = build()
    print(f"panel: {len(d):,} model-dates, {d.permaslug.nunique()} models, "
          f"{d.date.nunique()} dates ({d.date.min().date()} to {d.date.max().date()})")
    g = d.groupby("permaslug")
    chg = g.ln_p.transform(lambda s: s.diff().abs() > 1e-6)
    print(f"observations following a posted-price change: {int(chg.sum()):,} "
          f"({chg.mean():.1%}); models with a change: "
          f"{d.loc[chg, 'permaslug'].nunique()}")
    within = g.ln_p.std()
    print(f"within-model sd of log price: median {within.median():.3f}, "
          f"share with any variation {(within > 1e-6).mean():.2f}")
    d.to_csv(DER / "panel_long.csv", index=False)
    import json
    stats = json.loads((DER.parent.parent / "results.json").read_text()) \
        if (DER.parent.parent / "results.json").exists() else {}
    stats["n_revisions_panel"] = int(chg.sum())
    (DER.parent.parent / "results.json").write_text(json.dumps(stats, indent=1))
