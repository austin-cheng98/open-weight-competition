"""Daily usage series and provider-level endpoint data, one live request per model.
Each OpenRouter model page embeds its own 30-day usage panel in the RSC stream."""
import json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests

from src.collect.flight import endpoint_stats, usage_rows

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = RAW / "model_pages"
STATS_DIR = RAW / "model_stats"
WORKERS = 6


def slugs():
    fe = json.loads((RAW / "models_frontend.json").read_text())
    fe = fe["data"] if isinstance(fe, dict) else fe
    rk = json.loads((RAW / "rankings_models.json").read_text())["data"]
    active = {r["model_permaslug"] for r in rk}
    return sorted({m["slug"] for m in fe if m["permaslug"] in active})


def grab(slug):
    dest = OUT / (slug.replace("/", "__") + ".jsonl")
    if dest.exists() and (STATS_DIR / (slug.replace("/", "__") + ".json")).exists():
        return slug, -1
    for attempt in range(3):
        try:
            r = requests.get(f"https://openrouter.ai/{slug}", timeout=90,
                             headers={"User-Agent": "research"})
            if r.status_code == 200:
                rows = usage_rows(r.text)
                with dest.open("w") as fh:
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")
                stats = endpoint_stats(r.text)
                if stats:
                    (STATS_DIR / (slug.replace("/", "__") + ".json")).write_text(
                        json.dumps(stats))
                return slug, len(rows)
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    return slug, 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    todo = slugs()
    print(f"{len(todo)} models", flush=True)
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for slug, n in ex.map(grab, todo):
            done += 1
            if n == 0:
                print(f"  empty: {slug}", flush=True)
            if done % 50 == 0:
                print(f"{done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
