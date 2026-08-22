"""Assemble the model-level cross-section: characteristics, prices, capability
scores and current usage."""
import json, pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data" / "raw", ROOT / "data" / "derived"
M = 1e6  # prices are quoted per token; report per million


def load(name):
    d = json.loads((RAW / f"{name}.json").read_text())
    return d["data"] if isinstance(d, dict) and "data" in d else d


def catalog():
    fe = pd.DataFrame(load("models_frontend"))
    fe = fe[["permaslug", "slug", "name", "author", "hf_slug", "created_at",
             "context_length", "input_modalities", "supports_reasoning", "group"]]
    fe["hf_slug"] = fe["hf_slug"].replace("", None)
    fe["open_weight"] = fe["hf_slug"].notna()
    fe["created_at"] = pd.to_datetime(fe["created_at"], format="mixed", utc=True)
    fe["multimodal"] = fe["input_modalities"].apply(lambda x: len(x) > 1)
    return fe.drop_duplicates("permaslug")


def prices():
    v1 = pd.DataFrame(load("models_v1"))
    p = pd.json_normalize(v1["pricing"])
    out = pd.DataFrame({
        "permaslug": v1["canonical_slug"],
        "p_in": pd.to_numeric(p["prompt"], errors="coerce") * M,
        "p_out": pd.to_numeric(p["completion"], errors="coerce") * M,
        "p_cache": pd.to_numeric(p.get("input_cache_read"), errors="coerce") * M,
    })
    return out.drop_duplicates("permaslug")


def usage():
    rk = pd.DataFrame(load("rankings_models"))
    rk = rk[rk["variant"].isin(["standard", "free", "thinking", "batch"])]
    g = rk.groupby(["model_permaslug", "variant"], as_index=False).agg(
        prompt_tokens=("total_prompt_tokens", "sum"),
        completion_tokens=("total_completion_tokens", "sum"),
        requests=("count", "sum"),
        tool_calls=("total_tool_calls", "sum"),
        date=("date", "max"))
    tot = g.groupby("model_permaslug", as_index=False).agg(
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        requests=("requests", "sum"),
        tool_calls=("tool_calls", "sum"))
    tot["has_free_variant"] = tot["model_permaslug"].isin(
        g.loc[g["variant"] == "free", "model_permaslug"])
    return tot.rename(columns={"model_permaslug": "permaslug"})


def capability():
    bm = load("rankings_benchmarks")
    frames = []
    for cat in ("intelligence", "coding", "agentic"):
        df = pd.DataFrame(bm["aaData"][cat])[["permaslug", "score"]]
        frames.append(df.groupby("permaslug", as_index=False)["score"].max()
                        .rename(columns={"score": f"q_{cat}"}))
    aa = frames[0]
    for f in frames[1:]:
        aa = aa.merge(f, on="permaslug", how="outer")

    # design-arena Bradley-Terry scores, grouped into two capability blocks
    blocks = {"q_frontend": [k for k in bm["daData"] if any(
                  s in k for s in ("website", "uicomponent", "dataviz", "svg",
                                   "graphicdesign", "logo", "image"))],
              "q_appbuild": [k for k in bm["daData"] if any(
                  s in k for s in ("fullstack", "webapps", "mobileapps", "gamedev",
                                   "androidnative", "slides", "3d"))]}
    da_rows = []
    for name, keys in blocks.items():
        recs = []
        for k in keys:
            for r in bm["daData"][k]:
                if r.get("permaslug") and r.get("score"):
                    recs.append((r["permaslug"], r["score"]))
        d = pd.DataFrame(recs, columns=["permaslug", name])
        da_rows.append(d.groupby("permaslug", as_index=False)[name].mean())
    for d in da_rows:
        aa = aa.merge(d, on="permaslug", how="outer")

    eff = pd.DataFrame({
        "permaslug": list(bm["weightedInputPrices"]),
        "p_in_weighted": list(bm["weightedInputPrices"].values())})
    cpr = pd.DataFrame({
        "permaslug": list(bm["costPerRequest"]),
        "cost_per_request": list(bm["costPerRequest"].values())})
    return aa.merge(eff, on="permaslug", how="outer").merge(cpr, on="permaslug", how="outer")


def main():
    DER.mkdir(parents=True, exist_ok=True)
    df = (catalog().merge(prices(), on="permaslug", how="left")
                   .merge(usage(), on="permaslug", how="left")
                   .merge(capability(), on="permaslug", how="left"))
    tot = df["prompt_tokens"].fillna(0) + df["completion_tokens"].fillna(0)
    w_in = np.where(tot > 0, df["prompt_tokens"].fillna(0) / tot.replace(0, np.nan), np.nan)
    df["share_input_tokens"] = w_in
    df["p_blend"] = w_in * df["p_in"] + (1 - w_in) * df["p_out"]
    df["tokens"] = tot.replace(0, np.nan)
    df["age_days"] = (pd.Timestamp.now("UTC") - df["created_at"]).dt.days
    df.to_csv(DER / "models_cross_section.csv", index=False)

    obs = df[df["tokens"].notna()]
    print(f"models: {len(df)}   with usage: {len(obs)}   open: {obs['open_weight'].sum()}")
    print(f"priced: {df['p_in'].notna().sum()}   AA-scored: {df['q_intelligence'].notna().sum()}")
    print(f"usage+price+quality: {len(obs.dropna(subset=['p_blend', 'q_intelligence']))}")


if __name__ == "__main__":
    main()
