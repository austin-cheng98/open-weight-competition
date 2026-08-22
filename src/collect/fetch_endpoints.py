"""Per-provider endpoints for each model: prices, quantization, latency, uptime."""
import json, pathlib, time
from concurrent.futures import ThreadPoolExecutor

import requests

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = RAW / "endpoints"
WORKERS = 6


def targets():
    v1 = json.loads((RAW / "models_v1.json").read_text())["data"]
    return sorted({m["canonical_slug"] for m in v1})


def grab(permaslug):
    dest = OUT / (permaslug.replace("/", "__") + ".json")
    if dest.exists():
        return 0
    url = f"https://openrouter.ai/api/v1/models/{permaslug}/endpoints"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "research"})
            if r.status_code == 200:
                dest.write_text(json.dumps(r.json()))
                return 1
            if r.status_code == 404:
                return 0
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = targets()
    print(f"{len(t)} models", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        got = sum(ex.map(grab, t))
    print(f"fetched {got}")


if __name__ == "__main__":
    main()
