"""Provider-side competition for each model: how many independent hosts serve it
and how dispersed their prices are."""
import json, pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data" / "raw", ROOT / "data" / "derived"
M = 1e6


def main():
    rows = []
    for f in sorted((RAW / "endpoints").glob("*.json")):
        d = json.loads(f.read_text())
        b = d.get("data", d)
        permaslug = f.stem.replace("__", "/")
        for e in b.get("endpoints", []):
            p = e.get("pricing", {})
            try:
                p_in = float(p.get("prompt", "nan")) * M
                p_out = float(p.get("completion", "nan")) * M
            except (TypeError, ValueError):
                continue
            rows.append(dict(permaslug=permaslug, provider=e.get("provider_name"),
                             p_in=p_in, p_out=p_out,
                             quantization=e.get("quantization"),
                             context_length=e.get("context_length"),
                             uptime_1d=e.get("uptime_last_1d"),
                             latency=e.get("latency_last_30m"),
                             throughput=e.get("throughput_last_30m")))
    ep = pd.DataFrame(rows)
    ep.to_csv(DER / "endpoints.csv", index=False)

    pos = ep[ep.p_in > 0]
    g = pos.groupby("permaslug").agg(
        n_hosts=("provider", "nunique"),
        p_in_min=("p_in", "min"), p_in_median=("p_in", "median"),
        p_in_p90=("p_in", lambda s: s.quantile(0.9)),
        latency_median=("latency", "median"),
        throughput_median=("throughput", "median")).reset_index()
    g["price_ratio_p90_min"] = g.p_in_p90 / g.p_in_min
    g.to_csv(DER / "supply_side.csv", index=False)

    xs = pd.read_csv(DER / "models_cross_section.csv")
    j = g.merge(xs[["permaslug", "open_weight", "tokens"]], on="permaslug", how="left")
    j = j[j.tokens.notna()]
    print(j.groupby("open_weight")[["n_hosts", "price_ratio_p90_min"]]
           .agg(["median", "mean", "count"]).round(2).to_string())
    print("\nshare of models with >1 host:")
    print(j.assign(multi=j.n_hosts > 1).groupby("open_weight")["multi"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
