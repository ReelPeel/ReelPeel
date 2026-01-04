# services/pubmed_proxy.py
import asyncio
import time
import os
import signal
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request
from aiolimiter import AsyncLimiter
import uvicorn

# Global limit: 3 requests per second (NCBI default without API key)
# Change to (10, 1) if you add an NCBI API Key
limiter = AsyncLimiter(3, 1)
last_request_time = time.time()
IDLE_TIMEOUT = 300  # 5 minutes

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def monitor_idle():
    """Background task to shut down the service if no one is using it."""
    while True:
        await asyncio.sleep(10)
        # Check if the time since the last request exceeds the timeout
        if time.time() - last_request_time > IDLE_TIMEOUT:
            print(f"Proxy idle for {IDLE_TIMEOUT}s. Shutting down to save resources...")
            os.kill(os.getpid(), signal.SIGINT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: handles startup and shutdown events.
    Replaces the deprecated @app.on_event("startup").
    """
    # --- STARTUP LOGIC ---
    print("Initializing PubMed Proxy and idle monitor...")
    # Start the idle monitor as a background task
    monitor_task = asyncio.create_task(monitor_idle())

    yield  # The application runs here

    # --- SHUTDOWN LOGIC ---
    print("Shutting down PubMed Proxy...")
    monitor_task.cancel()


# Initialize FastAPI with the lifespan handler
app = FastAPI(title="PubMed Shared Proxy", lifespan=lifespan)


@app.get("/proxy/{util}")
async def proxy_pubmed(util: str, request: Request):
    """
    Generic proxy for esearch, esummary, and efetch.
    Example: http://localhost:8080/proxy/esearch.fcgi?db=pubmed&term=...
    """
    global last_request_time
    last_request_time = time.time()

    # We use the shared limiter here to throttle requests globally
    async with limiter:
        async with httpx.AsyncClient() as client:
            target_url = f"{NCBI_BASE}/{util}"

            # Forward the request to NCBI with all original query parameters
            # dict(request.query_params) captures ?db=pubmed&term=xyz...
            resp = await client.get(target_url, params=dict(request.query_params))

            # Return the exact content (XML/JSON) and status code from NCBI
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type")
            )


if __name__ == "__main__":
    # Run uvicorn programmatically
    uvicorn.run(app, host="127.0.0.1", port=8080)