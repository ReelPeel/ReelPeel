import asyncio
import time
import os
import signal
import sqlite3
import hashlib
import json
from typing import Any, Dict

from dotenv import load_dotenv
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request
from aiolimiter import AsyncLimiter
import uvicorn

load_dotenv("/data/home/jak38842/disk/fact_checker/secrets.env", override=True)

# --- CONFIGURATION ---
SHARED_DIR = "/data/home/jak38842/disk/fact_checker/pubmed_db_cache"
DB_FILE = os.path.join(SHARED_DIR, "pubmed_cache.db")

IDLE_TIMEOUT = 300  # 5 minutes
last_request_time = time.time()

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

NCBI_TOOL = os.getenv("NCBI_TOOL", "pubmed-shared-proxy").strip()
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "").strip()
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()

NCBI_COMMON: Dict[str, str] = {}
if NCBI_TOOL:
    NCBI_COMMON["tool"] = NCBI_TOOL
if NCBI_EMAIL:
    NCBI_COMMON["email"] = NCBI_EMAIL
if NCBI_API_KEY:
    NCBI_COMMON["api_key"] = NCBI_API_KEY

RATE = 10 if NCBI_API_KEY else 3
limiter = AsyncLimiter(RATE, 1.1)  # tokens per 1 second


def ensure_shared_dir():
    if not os.path.exists(SHARED_DIR):
        os.makedirs(SHARED_DIR, mode=0o777, exist_ok=True)
        os.chmod(SHARED_DIR, 0o777)


def init_db():
    ensure_shared_dir()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                content BLOB,
                status INTEGER,
                media_type TEXT,
                created_at REAL
            )
            """
        )
        conn.commit()
    try:
        os.chmod(DB_FILE, 0o666)
    except Exception:
        pass


def get_cache(key: str):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.execute("SELECT content, status, media_type FROM cache WHERE key = ?", (key,))
        return cur.fetchone()


def set_cache(key: str, content: bytes, status: int, media_type: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, content, status, media_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (key, content, status, media_type, time.time()),
        )
        conn.commit()


async def monitor_idle():
    while True:
        await asyncio.sleep(10)
        if time.time() - last_request_time > IDLE_TIMEOUT:
            print(f"Proxy idle for {IDLE_TIMEOUT}s. Shutting down...")
            os.kill(os.getpid(), signal.SIGINT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Shared PubMed Proxy...")
    print(f"DATABASE: {DB_FILE}")
    print(f"NCBI key present: {bool(NCBI_API_KEY)}; limiter rate: {RATE}/sec")
    init_db()
    monitor_task = asyncio.create_task(monitor_idle())
    yield
    print("Shutting down PubMed Proxy...")
    monitor_task.cancel()


app = FastAPI(title="PubMed Shared Proxy", lifespan=lifespan)


def generate_key(util: str, params: dict) -> str:
    serialized = json.dumps(dict(sorted(params.items())), separators=(",", ":"), ensure_ascii=True)
    raw_key = f"{util}|{serialized}"
    return hashlib.md5(raw_key.encode()).hexdigest()


@app.get("/health")
def health_check():
    return {"status": "running", "rate_tokens_per_sec": RATE, "has_api_key": bool(NCBI_API_KEY)}


@app.get("/proxy/{util}")
async def proxy_pubmed(util: str, request: Request):
    global last_request_time
    last_request_time = time.time()

    params = dict(request.query_params)

    # Optional cache bypass for testing
    nocache = str(params.pop("_nocache", "")).lower() in ("1", "true", "yes")

    # Do not accept these from callers; inject server-side
    params.pop("api_key", None)
    params.pop("tool", None)
    params.pop("email", None)

    cache_key = generate_key(util, params)

    # 1) Cache
    if not nocache:
        cached = get_cache(cache_key)
        if cached:
            content, status, media = cached
            print(f"DISK HIT: {util}")
            return Response(content=content, status_code=status, media_type=media)

    # 2) Upstream
    target_url = f"{NCBI_BASE}/{util}"
    upstream_params = {**params, **NCBI_COMMON}

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(3):
            async with limiter:
                try:
                    resp = await client.get(target_url, params=upstream_params)

                    if resp.status_code == 429:
                        wait_time = 2 * (attempt + 1)
                        print(f"NCBI 429 Rate Limit. Retry in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status_code == 200 and not nocache:
                        set_cache(cache_key, resp.content, resp.status_code, resp.headers.get("content-type", "application/octet-stream"))

                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type", "application/octet-stream"),
                    )
                except httpx.RequestError as e:
                    print(f"Request Error: {e!r}")
                    return Response(content=str(e), status_code=502)

    return Response(content="NCBI Rate Limit Exceeded", status_code=429)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
