import os
import time
import requests

PROXY = os.getenv("PROXY_BASE", "http://127.0.0.1:8080/proxy").rstrip("/")
URL = f"{PROXY}/esearch.fcgi"

def run(rps: float, seconds: int, use_key: bool) -> None:
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    params = {
        "db": "pubmed",
        "term": "cancer",
        "retmode": "json",
        "retmax": "1",
        # recommended but optional for this test:
        "tool": os.getenv("NCBI_TOOL", "rps-check"),
        "email": os.getenv("NCBI_EMAIL", "example@example.org"),
    }
    if use_key:
        if not api_key:
            raise SystemExit("NCBI_API_KEY not set")
        params["api_key"] = api_key

    interval = 1.0 / rps
    end = time.time() + seconds
    s = requests.Session()

    ok = limited = other = 0

    while time.time() < end:
        t0 = time.time()
        try:
            resp = s.get(URL, params=params, timeout=10)
            body = resp.text or ""

            # NCBI documents a JSON error payload when you exceed the ceiling.
            # In practice, some clients also observe HTTP 429.
            if resp.status_code == 429 or '"API rate limit exceeded"' in body:
                limited += 1
            elif resp.ok:
                ok += 1
            else:
                other += 1
        except Exception:
            other += 1

        dt = time.time() - t0
        sleep_for = interval - dt
        if sleep_for > 0:
            time.sleep(sleep_for)

    total = ok + limited + other
    print(f"use_key={use_key} rps={rps} seconds={seconds} -> total={total} ok={ok} limited={limited} other={other}")

if __name__ == "__main__":
    # Suggested sequence:
    run(rps=5, seconds=10, use_key=False)  # should show "limited" > 0
    run(rps=8, seconds=10, use_key=True)   # should show limited ~ 0
    run(rps=12, seconds=10, use_key=True)  # often shows limited > 0 (exceeds default 10 rps)
