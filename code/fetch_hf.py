"""Hugging Face metadata for open-weight models: parameter counts (a serving-cost
shifter), architecture, and downloads."""
import json, pathlib, time
from concurrent.futures import ThreadPoolExecutor

import requests

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = RAW / "hf"
FIELDS = "expand[]=safetensors&expand[]=config&expand[]=downloads&expand[]=createdAt&expand[]=likes"


def targets():
    fe = json.loads((RAW / "models_frontend.json").read_text())
    fe = fe["data"] if isinstance(fe, dict) else fe
    return sorted({m["hf_slug"] for m in fe if m.get("hf_slug")})


def grab(slug):
    dest = OUT / (slug.replace("/", "__") + ".json")
    if dest.exists():
        return 0
    for attempt in range(3):
        try:
            r = requests.get(f"https://huggingface.co/api/models/{slug}?{FIELDS}",
                             timeout=45, headers={"User-Agent": "research"})
            if r.status_code == 200:
                dest.write_text(json.dumps(r.json()))
                return 1
            if r.status_code in (401, 404):
                return 0
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = targets()
    print(f"{len(t)} repos")
    with ThreadPoolExecutor(6) as ex:
        print(f"fetched {sum(ex.map(grab, t))}")


if __name__ == "__main__":
    main()
