let currentPopup = null;
let popupState = null;
let lastAnalyzedUrl = null;
let scrollPauseTimer = null;
let analysisInFlight = false;
let pendingAnalysisUrl = null;
let lastSeenUrl = null;
const scrollPauseDelayMs = 1000;


// Monitor URL changes
let lastUrl = location.href;
const observer = new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    console.log("URL change");
    checkForReel();
  }
});

observer.observe(document, { subtree: true, childList: true });

// Initial check
checkForReel();

window.addEventListener("scroll", scheduleAnalysisAfterPause, { passive: true });

function checkForReel() {
  if (isReelUrl()) {
    showPopup();
  } else {
    removePopup();
  }
}

function isReelUrl() {
  return (
    window.location.pathname.includes("/reels/") ||
    window.location.pathname.includes("/reel/")
  );
}

function getScoreColor(score) {
  if (score <= 30) return "low-score";
  if (score <= 75) return "medium-score";
  return "high-score";
}

function formatScore(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toFixed(2);
}

function formatPubType(pubType) {
  if (!pubType) return "Unknown";
  if (Array.isArray(pubType)) {
    const cleaned = pubType
      .map((item) => (item ? String(item).trim() : ""))
      .filter((item) => item);
    return cleaned.length ? cleaned.join(", ") : "Unknown";
  }
  return String(pubType);
}

function formatEvidenceTitle(evidence) {
  if (!evidence || typeof evidence !== "object") return "Untitled source";
  const title =
    evidence.title || evidence.article_title || evidence.paper_title || "";
  if (title) return String(title);
  if (evidence.pubmed_id) return `PubMed ${evidence.pubmed_id}`;
  if (evidence.epistemonikos_id)
    return `Epistemonikos ${evidence.epistemonikos_id}`;
  if (evidence.chunk_id) return `RAG ${evidence.chunk_id}`;
  return "Untitled source";
}

function getEvidenceUrl(evidence) {
  if (!evidence || typeof evidence !== "object") return null;
  if (evidence.url) return evidence.url;
  if (evidence.pubmed_id)
    return `https://pubmed.ncbi.nlm.nih.gov/${evidence.pubmed_id}/`;
  if (evidence.epistemonikos_id)
    return `https://www.epistemonikos.org/en/documents/${evidence.epistemonikos_id}`;
  return null;
}

function normalizeScore(rawValue) {
  if (rawValue === null || rawValue === undefined || rawValue === "") {
    return null;
  }
  const num = Number(rawValue);
  if (!Number.isFinite(num)) return null;
  let percent = num;
  if (num >= 0 && num <= 1) {
    percent = num * 100;
  }
  if (percent < 0 || percent > 100) {
    return null;
  }
  return Math.round(percent);
}

function getMetaContent(key) {
  const propertyTag = document.querySelector(`meta[property="${key}"]`);
  if (propertyTag && propertyTag.content) return propertyTag.content;
  const nameTag = document.querySelector(`meta[name="${key}"]`);
  if (nameTag && nameTag.content) return nameTag.content;
  return "";
}

function getVisibleArea(rect) {
  const left = Math.max(0, rect.left);
  const right = Math.min(window.innerWidth, rect.right);
  const top = Math.max(0, rect.top);
  const bottom = Math.min(window.innerHeight, rect.bottom);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  return width * height;
}

function getActiveVideoElement() {
  const videos = Array.from(document.querySelectorAll("video"));
  let bestVideo = null;
  let bestArea = 0;

  for (const video of videos) {
    const rect = video.getBoundingClientRect();
    if (!rect.width || !rect.height) continue;
    const area = getVisibleArea(rect);
    if (area > bestArea) {
      bestArea = area;
      bestVideo = video;
    }
  }

  return bestVideo;
}

function getReelContextText(video) {
  const metaDescription =
    getMetaContent("og:description") || getMetaContent("description");
  const metaTitle = getMetaContent("og:title");
  const pageTitle = document.title || "";
  let text = [metaTitle, metaDescription, pageTitle].filter(Boolean).join(" ");

  if (video) {
    const container =
      video.closest("article") ||
      video.closest("div[role='presentation']") ||
      video.parentElement;
    if (container) {
      const containerText = container.innerText || "";
      if (containerText) {
        text = `${text} ${containerText}`.trim();
      }
    }
  }

  if (text.length > 2000) return text.slice(0, 2000);
  return text.trim();
}

function hasAudioTrack(video) {
  if (!video) return null;
  if (typeof video.mozHasAudio === "boolean") return video.mozHasAudio;
  if (video.audioTracks && typeof video.audioTracks.length === "number") {
    return video.audioTracks.length > 0;
  }
  if (typeof video.webkitAudioDecodedByteCount === "number") {
    if (video.webkitAudioDecodedByteCount > 0) return true;
  }
  return null;
}


function shouldSkipPipeline(video) {
  const hasAudio = hasAudioTrack(video);
  if (hasAudio === false) return { skip: true, reason: "no-audio" };
  return { skip: false };
}

function markNewReelSeen(reelUrl) {
  if (!reelUrl || reelUrl === lastSeenUrl) return;
  lastSeenUrl = reelUrl;
  if (reelUrl === lastAnalyzedUrl) return;
  const state = ensurePopup();
  state.resetForNewReel();
}

function queueAnalysisForUrl(reelUrl) {
  if (!reelUrl) return;
  pendingAnalysisUrl = reelUrl;
  maybeStartAnalysis();
}

function maybeStartAnalysis() {
  if (analysisInFlight) return;
  if (!pendingAnalysisUrl) return;
  const nextUrl = pendingAnalysisUrl;
  pendingAnalysisUrl = null;
  runAnalysis(nextUrl);
}

function stanceClassName(label) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.startsWith("support")) return "stance-supports";
  if (normalized.startsWith("refute")) return "stance-refutes";
  if (normalized.startsWith("neutral")) return "stance-neutral";
  return "stance-unknown";
}

function formatStance(evidence) {
  const stance = evidence && evidence.stance;
  if (!stance) return { label: "Unknown", className: "stance-unknown" };
  if (typeof stance === "string") {
    return { label: stance, className: stanceClassName(stance) };
  }

  const label =
    stance.abstract_label || stance.label || stance.abstractLabel || "";
  if (label) {
    return { label, className: stanceClassName(label) };
  }

  const scores = [
    { label: "Supports", value: stance.abstract_p_supports },
    { label: "Refutes", value: stance.abstract_p_refutes },
    { label: "Neutral", value: stance.abstract_p_neutral },
  ].filter((item) => typeof item.value === "number");

  if (!scores.length) {
    return { label: "Unknown", className: "stance-unknown" };
  }

  scores.sort((a, b) => b.value - a.value);
  const top = scores[0];
  return { label: top.label, className: stanceClassName(top.label) };
}

function createPopupState() {
  const medicalScore = 0;
  const needleRotation = medicalScore * 1.8 - 90; // Convert score to degrees (-90 to 90)

  const popup = document.createElement("div");
  popup.className = "reel-alert";
  popup.innerHTML = `
    <div class="dial-container">
      <div class="dial-background">
        <div class="loading-needle dial-needle" style="transform: rotate(${needleRotation}deg);"></div>
      </div>
      <span style="display: none;" id="percent-number" class="score-percentage"><span style="font-size: 14px;">Loading...</span></span>
    </div>
    
    <div class="dropdown-container">
      <div id="statements-view" class="panel-view active">
        <div class="dropdown-content">
          <ul id="dropdown-list"></ul>
        </div>
      </div>
      <div id="evidence-view" class="panel-view">
        <div class="evidence-header">
          <div class="evidence-label">Evidence</div>
          <div id="evidence-title" class="evidence-title"></div>
        </div>
        <div class="evidence-content">
          <ul id="evidence-list"></ul>
        </div>
      </div>
    </div>
    <button id="close-alert">×</button>
  `;

  document.body.appendChild(popup);
  const state = {
    popup,
    dialContainer: popup.querySelector(".dial-container"),
    needle: popup.querySelector(".dial-needle"),
    percentNumber: popup.querySelector("#percent-number"),
    statementsView: popup.querySelector("#statements-view"),
    evidenceView: popup.querySelector("#evidence-view"),
    evidenceTitle: popup.querySelector("#evidence-title"),
    evidenceList: popup.querySelector("#evidence-list"),
    listContent: popup.querySelector("#dropdown-list"),
    viewMode: "statements",
    statements: [],
  };

  state.setExpanded = function (isExpanded) {
    state.popup.classList.toggle("expanded", isExpanded);
    state.dialContainer.style.height = isExpanded ? "auto" : "40px";
    state.percentNumber.style.display = isExpanded ? "block" : "none";
  };

  state.showStatementsView = function () {
    state.viewMode = "statements";
    state.statementsView.classList.add("active");
    state.evidenceView.classList.remove("active");
  };

  state.renderEvidenceList = function (statement) {
    state.evidenceTitle.textContent =
      statement && statement.text ? statement.text : "Evidence details";
    state.evidenceList.innerHTML = "";

    const evidenceItems =
      statement && Array.isArray(statement.evidence) ? statement.evidence : [];

    if (!evidenceItems.length) {
      const emptyItem = document.createElement("li");
      emptyItem.className = "evidence-empty";
      emptyItem.textContent = "No evidence found.";
      state.evidenceList.appendChild(emptyItem);
      return;
    }

    for (const evidence of evidenceItems) {
      const item = document.createElement("li");
      item.className = "evidence-item";

      const titleUrl = getEvidenceUrl(evidence);
      let titleEl = null;
      if (titleUrl) {
        const link = document.createElement("a");
        link.className = "evidence-item-title";
        link.href = titleUrl;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = formatEvidenceTitle(evidence);
        titleEl = link;
      } else {
        const title = document.createElement("div");
        title.className = "evidence-item-title";
        title.textContent = formatEvidenceTitle(evidence);
        titleEl = title;
      }

      const meta = document.createElement("div");
      meta.className = "evidence-meta";

      const relevance = document.createElement("span");
      relevance.className = "evidence-meta-item";
      relevance.textContent = `Relevance: ${formatScore(evidence.relevance)}`;

      const pubType = document.createElement("span");
      pubType.className = "evidence-meta-item";
      pubType.textContent = `Type: ${formatPubType(evidence.pub_type)}`;

      const reliability = document.createElement("span");
      reliability.className = "evidence-meta-item";
      reliability.textContent = `Reliability: ${formatScore(evidence.weight)}`;

      const stance = formatStance(evidence);
      const stanceEl = document.createElement("span");
      stanceEl.className = `evidence-meta-item ${stance.className}`;
      stanceEl.textContent = `Stance: ${stance.label}`;

      meta.append(relevance, pubType, reliability, stanceEl);
      item.append(titleEl, meta);
      state.evidenceList.appendChild(item);
    }
  };

  state.showEvidenceView = function (statement) {
    state.viewMode = "evidence";
    state.statementsView.classList.remove("active");
    state.evidenceView.classList.add("active");
    state.setExpanded(true);
    state.renderEvidenceList(statement);
  };

  state.resetForNewReel = function () {
    state.setExpanded(false);
    state.showStatementsView();
    state.evidenceTitle.textContent = "";
    state.evidenceList.innerHTML = "";
    state.listContent.innerHTML = "";
    state.statements = [];
    state.setLoading();
  };

  state.setNotApplicable = function () {
    state.setExpanded(false);
    state.showStatementsView();
    state.evidenceTitle.textContent = "";
    state.evidenceList.innerHTML = "";
    state.listContent.innerHTML = "";
    state.statements = [];
    state.setScore(null);
  };

  state.setLoading = function () {
    state.needle.style.display = "block";
    state.needle.classList.add("loading-needle");
    state.percentNumber.textContent = "Loading...";
    state.percentNumber.classList.remove(
      "high-score",
      "medium-score",
      "low-score"
    );
  };

  state.setScore = function (score) {
    if (!Number.isFinite(score)) {
      state.needle.style.display = "block";
      state.needle.classList.remove("loading-needle");
      state.needle.style.transform = "rotate(-90deg)";
      state.percentNumber.textContent = "N/A";
      state.percentNumber.classList.remove(
        "high-score",
        "medium-score",
        "low-score"
      );
      return;
    }

    const needleRotation = score * 1.8 - 90;
    state.needle.style.display = "block";
    state.needle.classList.remove("loading-needle");
    state.needle.style.transform = `rotate(${needleRotation}deg)`;
    state.percentNumber.textContent = `${score}%`;
    state.percentNumber.classList.remove(
      "high-score",
      "medium-score",
      "low-score"
    );
    state.percentNumber.classList.add(getScoreColor(score));
  };

  state.setStatements = function (statements) {
    state.listContent.innerHTML = "";
    state.statements = Array.isArray(statements) ? statements : [];

    for (var i = 0; i < state.statements.length; i++) {
      const statement = state.statements[i];

      const listItem = document.createElement("li");
      listItem.className = "statement-item";

      const verdictButton = document.createElement("button");
      verdictButton.className = "feedback-button";
      verdictButton.type = "button";

      if (statement.verdict == "true") {
        verdictButton.title = "Agree with analysis";
        verdictButton.innerHTML = `<svg viewBox="0 0 24 24" fill="green">
          <path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"/>
        </svg>`;
      } else if (statement.verdict == "false") {
        verdictButton.title = "Disagree with analysis";
        verdictButton.innerHTML = `<svg viewBox="0 0 24 24" fill="red">
          <path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.58-6.59c.37-.36.59-.86.59-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"/>
        </svg>`;
      } else {
        verdictButton.title = "Disagree with analysis";
        verdictButton.innerHTML = `<svg viewBox="0 0 24 24" fill="orange" style="transform: rotate(0.25turn)">
          <path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.58-6.59c.37-.36.59-.86.59-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"/>
        </svg>`;
      }

      const statementText = document.createElement("span");
      statementText.className = "statement-text";
      statementText.textContent = statement.text || "";

      const evidenceButton = document.createElement("button");
      evidenceButton.className = "feedback-button evidence-pin";
      evidenceButton.type = "button";
      evidenceButton.title = "View evidence";
      evidenceButton.dataset.statementIndex = String(i);
      evidenceButton.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 12v-1l-2-2V5c0-1.1-.9-2-2-2H9c-1.1 0-2 .9-2 2v4l-2 2v1h6v7l1 1 1-1v-7h6z"/>
      </svg>`;

      listItem.append(verdictButton, statementText, evidenceButton);
      state.listContent.appendChild(listItem);
    }
  };

  state.dialContainer.addEventListener("click", () => {
    if (state.viewMode === "evidence") {
      state.showStatementsView();
      state.setExpanded(true);
      return;
    }
    state.setExpanded(!state.popup.classList.contains("expanded"));
  });

  state.listContent.addEventListener("click", (event) => {
    const pinButton = event.target.closest(".evidence-pin");
    if (!pinButton) return;
    const index = Number(pinButton.dataset.statementIndex);
    const statement = state.statements[index];
    if (!statement) return;
    state.showEvidenceView(statement);
  });

  popup.querySelector("#close-alert").addEventListener("click", () => {
    removePopup();
  });

  return state;
}

function ensurePopup() {
  if (popupState && currentPopup) return popupState;
  popupState = createPopupState();
  currentPopup = popupState.popup;
  return popupState;
}

function scheduleAnalysisAfterPause() {
  if (!isReelUrl()) return;
  const currentUrl = window.location.href;
  markNewReelSeen(currentUrl);
  if (scrollPauseTimer) {
    clearTimeout(scrollPauseTimer);
  }
  scrollPauseTimer = setTimeout(() => {
    const url = window.location.href;
    if (!url) return;
    if (url === lastAnalyzedUrl && !pendingAnalysisUrl) return;
    queueAnalysisForUrl(url);
  }, scrollPauseDelayMs);
}

async function runAnalysis(reelUrl) {
  if (!reelUrl || reelUrl === lastAnalyzedUrl) {
    maybeStartAnalysis();
    return;
  }
  if (analysisInFlight) return;
  analysisInFlight = true;
  const state = ensurePopup();
  state.setLoading();
  state.showStatementsView();

  const video = getActiveVideoElement();
  const skipCheck = shouldSkipPipeline(video);
  if (skipCheck.skip) {
    console.log(`Skipping pipeline for ${reelUrl}: ${skipCheck.reason}`);
    state.setNotApplicable();
    lastAnalyzedUrl = reelUrl;
    analysisInFlight = false;
    maybeStartAnalysis();
    return;
  }

  try {
    const data = await chrome.runtime.sendMessage({
      type: "getMedicalScore",
      url: reelUrl,
    });

    if (window.location.href !== reelUrl) return;

    const statements = data && data.statements ? data.statements : [];
    const rawScore = data ? data["overall_truthiness"] : null;
    const medicalScore = normalizeScore(rawScore);

    state.setStatements(statements);
    state.setScore(medicalScore);
    lastAnalyzedUrl = reelUrl;
  } catch (error) {
    console.error("Error in content script" + error);
  } finally {
    analysisInFlight = false;
    maybeStartAnalysis();
  }
}

function showPopup() {
  ensurePopup();
  const currentUrl = window.location.href;
  markNewReelSeen(currentUrl);
  if (!lastAnalyzedUrl) {
    queueAnalysisForUrl(currentUrl);
    return;
  }
  scheduleAnalysisAfterPause();
}

function removePopup() {
  if (currentPopup) {
    currentPopup.remove();
  }
  currentPopup = null;
  popupState = null;
  lastAnalyzedUrl = null;
  lastSeenUrl = null;
  pendingAnalysisUrl = null;
  analysisInFlight = false;
  if (scrollPauseTimer) {
    clearTimeout(scrollPauseTimer);
    scrollPauseTimer = null;
  }
}
