"""Reconstruct the model price/characteristic panel from Wayback captures of the
OpenRouter catalog endpoint."""
import json, pathlib, sys

from src.collect.wayback import cdx, snapshot

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = RAW / "price_history"
URL = "https://openrouter.ai/api/v1/models"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = RAW / "cdx_models_api.json"
    if not cache.exists():
        cache.write_text(json.dumps(cdx("openrouter.ai/api/v1/models", limit=400)))
    stamps = json.loads(cache.read_text())
    todo = [t for t in stamps if not (OUT / f"{t}.json").exists()]
    print(f"{len(stamps)} captures, {len(todo)} remaining", flush=True)
    for ts in todo:
        r = snapshot(ts, URL)
        if r is None:
            print(f"{ts} failed", flush=True)
            continue
        try:
            d = r.json()
        except Exception:
            print(f"{ts} unparseable", flush=True)
            continue
        (OUT / f"{ts}.json").write_text(json.dumps(d))
        print(f"{ts} {len(d.get('data', []))}", flush=True)


if __name__ == "__main__":
    main()
