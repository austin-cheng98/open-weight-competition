"""Download the live OpenRouter public endpoints used in the analysis."""
import json, pathlib, time

import requests

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
BASE = "https://openrouter.ai"

ENDPOINTS = {
    "models_v1": "/api/v1/models",
    "models_frontend": "/api/frontend/v1/models",
    "rankings_models": "/api/frontend/v1/rankings/models",
    "rankings_market_share": "/api/frontend/v1/rankings/market-share",
    "rankings_benchmarks": "/api/frontend/v1/rankings/benchmarks",
    "rankings_task_spend": "/api/frontend/v1/rankings/task-spend",
    "rankings_session_cost": "/api/frontend/v1/rankings/session-cost",
    "rankings_tools": "/api/frontend/v1/rankings/tools",
    "rankings_apps": "/api/frontend/v1/rankings/apps",
}


def get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "research"})
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in ENDPOINTS.items():
        d = get(BASE + path)
        (RAW / f"{name}.json").write_text(json.dumps(d))
        body = d.get("data", d) if isinstance(d, dict) else d
        n = len(body) if hasattr(body, "__len__") else 0
        print(f"{name}: {n}")
        time.sleep(1)


if __name__ == "__main__":
    main()
