"""Serving speed per model: request-weighted median latency and throughput across
the providers that serve it."""
import json, pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data" / "raw", ROOT / "data" / "derived"


def main():
    rows = []
    for f in (RAW / "model_stats").glob("*.json"):
        slug = f.stem.replace("__", "/")
        for s in json.loads(f.read_text()):
            rows.append(dict(slug=slug, latency=s.get("p50_latency"),
                             throughput=s.get("p50_throughput"),
                             n=s.get("request_count") or 1))
    d = pd.DataFrame(rows).dropna(subset=["latency", "throughput"])
    g = (d.groupby("slug")
           .apply(lambda x: pd.Series({
               "latency_ms": np.average(x.latency, weights=x.n),
               "throughput_tps": np.average(x.throughput, weights=x.n)}),
                  include_groups=False)
           .reset_index())
    fe = json.loads((RAW / "models_frontend.json").read_text())
    fe = fe["data"] if isinstance(fe, dict) else fe
    m = pd.DataFrame([(x["slug"], x["permaslug"]) for x in fe],
                     columns=["slug", "permaslug"]).drop_duplicates("slug")
    g = g.merge(m, on="slug", how="inner")[["permaslug", "latency_ms", "throughput_tps"]]
    g.to_csv(DER / "model_speed.csv", index=False)
    print(f"{len(g)} models with speed data")


if __name__ == "__main__":
    main()
