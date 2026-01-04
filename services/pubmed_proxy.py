import asyncio
import time
import os
import signal
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request
from aiolimiter import AsyncLimiter
import uvicorn

# SAFETY CONFIG:
# We limit to 3 requests every 1.5 seconds to account for network bursts.
limiter = AsyncLimiter(3, 1.5)

last_request_time = time.time()
IDLE_TIMEOUT = 300  # 5 minutes
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# SIMPLE IN-MEMORY CACHE
# Key: (util_name, sorted_params_tuple) | Value: (content, status, media_type)
PROXY_CACHE = {}


async def monitor_idle():
    """Background task to shut down the service if no one is using it."""
    while True:
        await asyncio.sleep(10)
        if time.time() - last_request_time > IDLE_TIMEOUT:
            print(f"Proxy idle for {IDLE_TIMEOUT}s. Shutting down...")
            os.kill(os.getpid(), signal.SIGINT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing PubMed Proxy (Safe Mode + Caching)...")
    monitor_task = asyncio.create_task(monitor_idle())
    yield
    print("Shutting down PubMed Proxy...")
    monitor_task.cancel()


app = FastAPI(title="PubMed Shared Proxy", lifespan=lifespan)


@app.get("/proxy/{util}")
async def proxy_pubmed(util: str, request: Request):
    global last_request_time
    last_request_time = time.time()

    target_url = f"{NCBI_BASE}/{util}"
    # Convert query params to a sorted tuple so it can be used as a dictionary key
    # (Sorting ensures 'a=1&b=2' hits the same cache as 'b=2&a=1')
    params = dict(request.query_params)
    cache_key = (util, tuple(sorted(params.items())))

    # 1. CHECK CACHE (Fast Path)
    if cache_key in PROXY_CACHE:
        # Debug print to show it's working
        print(f"CACHE HIT: {util} (Params: {len(params)})")
        content, status, media = PROXY_CACHE[cache_key]
        return Response(content=content, status_code=status, media_type=media)

    # 2. FETCH FROM UPSTREAM (Slow Path)
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            # Only use the rate limiter for actual network calls
            async with limiter:
                try:
                    resp = await client.get(target_url, params=params)

                    if resp.status_code == 429:
                        wait_time = 2 * (attempt + 1)
                        print(f"NCBI 429 Rate Limit hit. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    # 3. SAVE TO CACHE (If successful)
                    if resp.status_code == 200:
                        PROXY_CACHE[cache_key] = (
                            resp.content,
                            resp.status_code,
                            resp.headers.get("content-type")
                        )

                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type")
                    )

                except httpx.RequestError as e:
                    print(f"Request Error: {e}")
                    return Response(content=str(e), status_code=502)

    return Response(content="NCBI Rate Limit Exceeded (Retries failed)", status_code=429)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)