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
    fetch("http://im-redstone02.hs-regensburg.de:38843/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: reelUrl, // ⬅️  now dynamic
        mock: "true",
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        sendResponse(data);
      })
      .catch((error) =>
        console.error("Error at API call in background:", error)
      );

    return true; // Keep message channel open for async response
  }
});
