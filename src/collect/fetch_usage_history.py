"""Longer-run usage panel from archived rankings pages (biweekly captures)."""
import json, pathlib, sys
from datetime import datetime

from src.collect.flight import usage_rows
from src.collect.wayback import cdx, snapshot

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
OUT = RAW / "usage_history"
URL = "https://openrouter.ai/rankings"
GAP_DAYS = 14


def thin(stamps, gap):
    kept, last = [], None
    for ts in sorted(stamps):
        d = datetime.strptime(ts[:8], "%Y%m%d")
        if last is None or (d - last).days >= gap:
            kept.append(ts)
            last = d
    return kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = RAW / "cdx_rankings.json"
    if not cache.exists():
        cache.write_text(json.dumps(cdx("openrouter.ai/rankings")))
    stamps = thin(json.loads(cache.read_text()), GAP_DAYS)
    todo = [t for t in stamps if not (OUT / f"{t}.jsonl").exists()]
    print(f"{len(stamps)} captures, {len(todo)} to fetch", flush=True)
    for ts in todo:
        r = snapshot(ts, URL)
        if r is None:
            print(f"{ts} failed", flush=True)
            continue
        rows = usage_rows(r.text)
        with (OUT / f"{ts}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"{ts} {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
