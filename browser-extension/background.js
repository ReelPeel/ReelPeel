const API_BASE = "http://im-redstone02.hs-regensburg.de:38843";

/**
 * Background service worker: forwards content-script requests to the backend.
 * We keep a legacy message name ("getMedicalScore") for compatibility.
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const type = request?.type;

  if (type === "getReelAnalysis" || type === "getMedicalScore") {
    const reelUrl = request.url || sender?.url || sender?.tab?.url || "";

    fetch(`${API_BASE}/json`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: reelUrl,
        mock: Boolean(request.mock) || false,
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
    fetch(`${API_BASE}/evidence_summary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        statement: request.statement,
        evidence: request.evidence,
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
});
