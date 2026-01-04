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
# (Standard is 3/1.0s, but that is often too tight).
limiter = AsyncLimiter(3, 1.5)

last_request_time = time.time()
IDLE_TIMEOUT = 300  # 5 minutes
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def monitor_idle():
    """Background task to shut down the service if no one is using it."""
    while True:
        await asyncio.sleep(10)
        if time.time() - last_request_time > IDLE_TIMEOUT:
            print(f"Proxy idle for {IDLE_TIMEOUT}s. Shutting down...")
            os.kill(os.getpid(), signal.SIGINT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing PubMed Proxy (Safe Mode)...")
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
    params = dict(request.query_params)

    # RETRY LOGIC: Try up to 3 times if we get a 429
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            async with limiter:
                try:
                    resp = await client.get(target_url, params=params)

                    if resp.status_code == 429:
                        # We hit the limit despite our limiter. Wait and retry.
                        wait_time = 2 * (attempt + 1)
                        print(f"⚠️ NCBI 429 Rate Limit hit. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue  # Try loop again

                    # Success (or other error like 404/500 that we shouldn't retry)
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type")
                    )

                except httpx.RequestError as e:
                    print(f"Request Error: {e}")
                    # On network error, maybe just fail or retry? Let's fail for now.
                    return Response(content=str(e), status_code=502)

    # If we exhausted retries
    return Response(content="NCBI Rate Limit Exceeded (Retries failed)", status_code=429)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)