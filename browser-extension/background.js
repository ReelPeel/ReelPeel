const API_BASE = "http://im-redstone02.hs-regensburg.de:38843";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("got a message!");

  if (request.type === "getMedicalScore") {
    /* -----------------------------------------
       1.  Work out which reel URL to use.
           – Prefer an explicit URL sent from the
             content script (request.url).
           – Otherwise fall back to the page that
             sent the message (sender.url or
             sender.tab.url).              */
    const reelUrl =
      request.url ||
      sender?.url ||
      sender?.tab?.url ||
      ""; // empty string if nothing found

    console.log("Using reel URL:", reelUrl);

    /* -----------------------------------------
       2.  Forward that URL to your backend.      */
    fetch(`${API_BASE}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: reelUrl, // ⬅️  now dynamic
        mock: false,
      }),
    })
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.detail || "Process request failed");
        }
        return data;
      })
      .then((data) => {
        console.log(data);
        sendResponse(data);
      })
      .catch((error) => {
        console.error("Error at API call in background:", error);
        sendResponse({ error: String(error) });
      });

    return true; // Keep message channel open for async response
  }

  if (request.type === "getEvidenceSummary") {
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
        if (!response.ok) {
          throw new Error(data.detail || "Summary request failed");
        }
        sendResponse(data);
      })
      .catch((error) => {
        console.error("Error at summary call in background:", error);
        sendResponse({ error: String(error) });
      });

    return true;
  }
});
