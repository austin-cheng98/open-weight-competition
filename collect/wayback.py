"""Polite Wayback client: fixed spacing between requests plus adaptive backoff
when the archive starts refusing connections."""
import time

import requests

SPACING = 4.0
_last = [0.0]
_penalty = [0.0]


def fetch(url, tries=5, timeout=120):
    for i in range(tries):
        wait = SPACING + _penalty[0] - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "research"})
            if r.status_code == 200:
                _penalty[0] = max(0.0, _penalty[0] - 2.0)
                return r
            if r.status_code in (429, 503, 520):
                _penalty[0] = min(120.0, _penalty[0] * 2 + 10)
        except Exception:
            _penalty[0] = min(120.0, _penalty[0] * 2 + 10)
        time.sleep(5 * (i + 1))
    return None


def cdx(url, limit=3000):
    q = (f"http://web.archive.org/cdx/search/cdx?url={url}&output=json"
         f"&fl=timestamp&filter=statuscode:200&collapse=digest&limit={limit}")
    r = fetch(q, timeout=180)
    return [row[0] for row in r.json()[1:]] if r else []


def snapshot(ts, url):
    return fetch(f"http://web.archive.org/web/{ts}id_/{url}")
