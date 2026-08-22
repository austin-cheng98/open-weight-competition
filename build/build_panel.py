"""Two panels: (i) daily model-level usage for the trailing month, (ii) the
model-level price/availability panel reconstructed from archived catalogues."""
import json, pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, DER = ROOT / "data" / "raw", ROOT / "data" / "derived"
M = 1e6

USE_COLS = ["date", "model_permaslug", "variant", "total_prompt_tokens",
            "total_completion_tokens", "count", "total_tool_calls",
            "total_native_tokens_reasoning"]


def daily_usage():
    rows = []
    for f in sorted((RAW / "model_pages").glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if "model_permaslug" not in r:
                continue
            rows.append({c: r.get(c) for c in USE_COLS})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(["date", "model_permaslug", "variant"])
    g = df.groupby(["date", "model_permaslug"], as_index=False).agg(
        prompt_tokens=("total_prompt_tokens", "sum"),
        completion_tokens=("total_completion_tokens", "sum"),
        requests=("count", "sum"),
        tool_calls=("total_tool_calls", "sum"),
        reasoning_tokens=("total_native_tokens_reasoning", "sum"),
        n_variants=("variant", "nunique"))
    g["tokens"] = g.prompt_tokens + g.completion_tokens
    g = g.rename(columns={"model_permaslug": "permaslug"})
    # the first and last calendar days of the scrape are partial
    n = g.groupby("date")["permaslug"].nunique()
    v = g.groupby("date")["tokens"].sum()
    full = n[(n >= 0.5 * n.median()) & (v >= 0.5 * v.median())].index
    return g[g.date.isin(full)]


LONG_COLS = ["date", "model_permaslug", "model", "variant",
             "total_prompt_tokens", "total_completion_tokens", "count"]


def long_usage():
    """Weekly model-level usage recovered from archived rankings captures."""
    rows = []
    for f in sorted((RAW / "usage_history").glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            slug = r.get("model_permaslug") or r.get("model")
            if not slug or "total_prompt_tokens" not in r:
                continue
            rows.append(dict(date=r["date"], permaslug=slug,
                             variant=r.get("variant", "standard"),
                             prompt_tokens=r.get("total_prompt_tokens") or 0,
                             completion_tokens=r.get("total_completion_tokens") or 0,
                             requests=r.get("count") or 0,
                             spend=r.get("volume"),
                             # captures carrying spend are non-overlapping weekly
                             # buckets; the rest are trailing-window snapshots
                             weekly=int("volume" in r)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = (pd.to_datetime(df["date"], format="mixed", utc=True)
                    .dt.tz_localize(None).dt.normalize())
    df = df.drop_duplicates(["date", "permaslug", "variant"])
    g = df.groupby(["date", "permaslug"], as_index=False).agg(
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        requests=("requests", "sum"),
        spend=("spend", "sum"),
        weekly=("weekly", "max"))
    g["tokens"] = g.prompt_tokens + g.completion_tokens
    n = g.groupby("date")["permaslug"].nunique()
    return g[g.date.isin(n[n >= 25].index)]


def price_panel():
    recs = []
    for f in sorted((RAW / "price_history").glob("*.json")):
        ts = f.stem
        try:
            data = json.loads(f.read_text())["data"]
        except Exception:
            continue
        for m in data:
            p = m.get("pricing", {})
            try:
                p_in = float(p.get("prompt", "nan")) * M
                p_out = float(p.get("completion", "nan")) * M
            except (TypeError, ValueError):
                continue
            recs.append(dict(
                snapshot=pd.Timestamp(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"),
                permaslug=m.get("canonical_slug") or m.get("id"),
                slug=m.get("id"), p_in=p_in, p_out=p_out,
                context_length=m.get("context_length"),
                hf_id=m.get("hugging_face_id"),
                created=m.get("created")))
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df.drop_duplicates(["snapshot", "permaslug"])
    df["hf_id"] = df["hf_id"].replace("", None)
    df["open_weight"] = df["hf_id"].notna()
    df["release"] = pd.to_datetime(df["created"], unit="s", errors="coerce")
    return df.sort_values(["permaslug", "snapshot"])


def main():
    DER.mkdir(parents=True, exist_ok=True)
    u = daily_usage()
    u.to_csv(DER / "usage_daily.csv", index=False)
    tot = u.groupby("date")["tokens"].sum()
    print(f"usage panel: {len(u):,} rows, {u.permaslug.nunique()} models, "
          f"{u.date.nunique()} days ({u.date.min().date()} to {u.date.max().date()})")

    lu = long_usage()
    if not lu.empty:
        lu.to_csv(DER / "usage_long.csv", index=False)
        print(f"long usage panel: {len(lu):,} rows, {lu.permaslug.nunique()} models, "
              f"{lu.date.nunique()} dates ({lu.date.min().date()} to {lu.date.max().date()})")

    p = price_panel()
    if not p.empty:
        p.to_csv(DER / "price_panel.csv", index=False)
        print(f"price panel: {len(p):,} rows, {p.permaslug.nunique()} models, "
              f"{p.snapshot.nunique()} snapshots ({p.snapshot.min().date()} to {p.snapshot.max().date()})")
        n = p.groupby("snapshot")["permaslug"].nunique()
        print(f"models per snapshot: median {n.median():.0f}, last {n.iloc[-1]:.0f}")


if __name__ == "__main__":
    main()
