const REMOTE_API_BASE = "http://im-redstone02.hs-regensburg.de:38843";
const LOCAL_API_BASE = "http://127.0.0.1:8765";

function getApiBase(sender, request) {
  if (request?.apiBase) return request.apiBase;

  const pageUrl = sender?.tab?.url || sender?.url || "";
  let offlineDemoOrigin = "";
  try {
    const url = new URL(pageUrl);
    const isOfflineDemo =
      (url.hostname === "127.0.0.1" || url.hostname === "localhost") &&
      /^\/reels?\//.test(url.pathname);
    offlineDemoOrigin = isOfflineDemo ? url.origin : "";
  } catch (error) {
    offlineDemoOrigin = "";
  }

  // Explicit preference from content script takes priority
  if (request?.backendPref === "local") return offlineDemoOrigin || LOCAL_API_BASE;
  if (request?.backendPref === "remote") return REMOTE_API_BASE;

  // Fallback: auto-detect from page URL (offline demo site)
  return offlineDemoOrigin || REMOTE_API_BASE;
}

/**
 * Background service worker: forwards content-script requests to the backend.
 * We keep a legacy message name ("getMedicalScore") for compatibility.
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const type = request?.type;

  if (type === "getReelAnalysis" || type === "getMedicalScore") {
    const reelUrl = request.url || sender?.url || sender?.tab?.url || "";
    const apiBase = getApiBase(sender, request);

    fetch(`${apiBase}/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: reelUrl,
        mock: Boolean(request.mock) || false,
        run_id: request.run_id || null,
      }),
    })
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Process request failed");
        return data;
      })
      .then((data) => sendResponse(data))
      .catch((error) => {
        console.error("Error at API call in background:", error);
        sendResponse({ error: String(error) });
      });

    return true; // Keep message channel open for async response
  }

  if (type === "getEvidenceSummary") {
    const apiBase = getApiBase(sender, request);
    const reelUrl = sender?.tab?.url || sender?.url || "";

    fetch(`${apiBase}/evidence_summary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        statement: request.statement,
        evidence: request.evidence,
        reel_url: reelUrl,
      }),
    })
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Summary request failed");
        sendResponse(data);
      })
      .catch((error) => {
        console.error("Error at summary call in background:", error);
        sendResponse({ error: String(error) });
      });

    return true;
  }

  if (type === "healthCheck") {
    const target = request.target || LOCAL_API_BASE;
    fetch(`${target}/health`, { signal: AbortSignal.timeout(2000) })
      .then((r) => (r.ok ? r.json() : Promise.reject("unhealthy")))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }
});
