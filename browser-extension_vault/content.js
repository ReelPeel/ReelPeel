let currentPopup = null;
let popupState = null;
let lastAnalyzedUrl = null;
let lastAnalyzedReelKey = null;
let displayedReelKey = null;
let scrollPauseTimer = null;
let analysisInFlight = false;
let pendingAnalysisUrl = null;
let pendingAnalysisReelKey = null;
let lastSeenReelKey = null;
let inFlightReelKey = null;
let inquiryScrollLocked = false;
let backendPref = "local"; // "local" | "remote"
let activeLocalApiBase = "http://127.0.0.1:8765";
const scrollPauseDelayMs = 1000;
const TOOLTIP_STANCE =
  "Stance: Whether the evidence supports, refutes, or is neutral toward the statement.";
const TOOLTIP_RELEVANCE =
  "Relevance: How closely the evidence matches the claim's scope.";
const TOOLTIP_TYPE = "Type: Publication type (as listed in PubMed).";
const BOOKMARKS_STORAGE_KEY = "rp_claim_bookmarks_v1";
const BACKEND_PREF_VERSION = 2;
const LOCAL_API_PORTS = [8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775];

// Load saved backend preference
if (typeof chrome !== "undefined" && chrome.storage?.local) {
  chrome.storage.local.get(["rp_backend_pref", "rp_backend_pref_version"], (stored) => {
    if (stored?.rp_backend_pref_version !== BACKEND_PREF_VERSION) {
      backendPref = "local";
      chrome.storage.local.set({
        rp_backend_pref: "local",
        rp_backend_pref_version: BACKEND_PREF_VERSION,
      });
      return;
    }
    backendPref = stored?.rp_backend_pref || "local";
  });
}

function isInstagramPage() {
  return /(^|\.)instagram\.com$/.test(location.hostname);
}

function isLocalDemoPage() {
  return (
    location.hostname === "127.0.0.1" ||
    location.hostname === "localhost"
  );
}

function getEffectiveBackendPref() {
  if (isLocalDemoPage()) return "local";
  return backendPref || "local";
}

function getLocalApiCandidates() {
  if (isLocalDemoPage()) return [location.origin];
  return LOCAL_API_PORTS.map((port) => `http://127.0.0.1:${port}`);
}

async function resolveLocalApiBase() {
  const candidates = getLocalApiCandidates();
  for (const target of candidates) {
    try {
      const health = await chrome.runtime.sendMessage({
        type: "healthCheck",
        target,
      });
      if (health && health.ok) {
        activeLocalApiBase = target;
        return target;
      }
    } catch (_error) {
      // Try next local port.
    }
  }
  throw new Error("local backend unavailable");
}

function setBackendPref(nextPref) {
  backendPref = nextPref === "remote" ? "remote" : "local";
  if (typeof chrome !== "undefined" && chrome.storage?.local) {
    chrome.storage.local.set({
      rp_backend_pref: backendPref,
      rp_backend_pref_version: BACKEND_PREF_VERSION,
    });
  }
}

// Monitor SPA navigation. Instagram changes the current Reel through History API
// calls, which do not necessarily produce a browser navigation event.
let lastUrl = location.href;
function handleLocationChange() {
  if (location.href === lastUrl) return;
  lastUrl = location.href;
  checkForReel();
}

const observer = new MutationObserver(() => {
  handleLocationChange();
});

observer.observe(document, { subtree: true, childList: true });

["pushState", "replaceState"].forEach((method) => {
  try {
    const original = history[method];
    if (typeof original !== "function") return;
    history[method] = function (...args) {
      const result = original.apply(this, args);
      handleLocationChange();
      return result;
    };
  } catch (error) {
    // Scroll and mutation detection remain available if History is protected.
  }
});
window.addEventListener("popstate", handleLocationChange);
window.addEventListener("hashchange", handleLocationChange);
window.addEventListener("play", () => {
  handleLocationChange();
  scheduleAnalysisAfterPause();
}, true);

// Initial check
checkForReel();

window.addEventListener("scroll", scheduleAnalysisAfterPause, { passive: true, capture: true });

function blockInquiryScroll(event) {
  if (!inquiryScrollLocked) return;
  const target = event.target.nodeType === 3 ? event.target.parentNode : event.target;
  if (target && target.closest && target.closest(".reel-alert")) {
    return; // Allow scrolling inside the extension popup
  }
  event.preventDefault();
}

function setInquiryScrollLock(locked) {
  if (inquiryScrollLocked === locked) return;
  inquiryScrollLocked = locked;
  if (locked) {
    window.addEventListener("wheel", blockInquiryScroll, { passive: false, capture: true });
    window.addEventListener("touchmove", blockInquiryScroll, { passive: false, capture: true });
  } else {
    window.removeEventListener("wheel", blockInquiryScroll, { capture: true });
    window.removeEventListener("touchmove", blockInquiryScroll, { capture: true });
    window.removeEventListener("wheel", blockInquiryScroll, true);
    window.removeEventListener("touchmove", blockInquiryScroll, true);
  }
}

function forceRestorePageInteraction() {
  inquiryScrollLocked = false;
  window.removeEventListener("wheel", blockInquiryScroll, { capture: true });
  window.removeEventListener("touchmove", blockInquiryScroll, { capture: true });
  window.removeEventListener("wheel", blockInquiryScroll, true);
  window.removeEventListener("touchmove", blockInquiryScroll, true);
  document.documentElement.classList.remove("rp-inquiry-open");
  document.body.classList.remove("rp-inquiry-open");
  document.documentElement.style.cursor = "";
  document.body.style.cursor = "";
}

function checkForReel() {
  if (isReelUrl()) {
    showPopup();
  } else {
    removePopup();
  }
}

function toSortableRelevance(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  // If your backend sometimes sends 0..1, treat it as 0..100
  if (num >= 0 && num <= 1) return num * 100;
  return num;
}

function sortEvidenceByRelevanceDesc(evidenceArray) {
  return evidenceArray
    .map((evidence, originalIndex) => ({ evidence, originalIndex }))
    .sort((a, b) => {
      const aVal = toSortableRelevance(a.evidence?.relevance);
      const bVal = toSortableRelevance(b.evidence?.relevance);

      const aValid = Number.isFinite(aVal);
      const bValid = Number.isFinite(bVal);

      if (aValid && !bValid) return -1;
      if (!aValid && bValid) return 1;
      if (!aValid && !bValid) return a.originalIndex - b.originalIndex;

      if (bVal !== aVal) return bVal - aVal;
      return a.originalIndex - b.originalIndex;
    })
    .map((x) => x.evidence);
}



function formatTimeSeconds(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return "";
  const mm = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

// NOTE: Clipboard/export functions removed in the DIS Interactivity build
// to keep the artifact lightweight and reduce UI clutter.

function computeStatementGaps(statement) {
  const evidence = statement && Array.isArray(statement.evidence) ? statement.evidence : [];
  const gaps = [];

  if (!evidence.length) {
    gaps.push("No sources");
    return gaps;
  }

  // Consider only the most relevant evidence when assessing disagreement / match quality.
  const top = sortEvidenceByRelevanceDesc(evidence).slice(0, 6);
  const strong = top.filter((e) => {
    const v = toSortableRelevance(e?.relevance);
    return Number.isFinite(v) && v >= 35;
  });

  const stances = { supports: 0, refutes: 0 };
  for (const ev of strong.length ? strong : top) {
    const s = formatStance(ev).label.toLowerCase();
    if (s.startsWith("support")) stances.supports += 1;
    if (s.startsWith("refute")) stances.refutes += 1;
  }
  if (stances.supports > 0 && stances.refutes > 0) {
    gaps.push("Sources disagree");
  }

  const hasStrongMatch = strong.length > 0;
  if (!hasStrongMatch) {
    gaps.push("Weak match");
  }

  return gaps.slice(0, 2);
}

function safeNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function clamp01(value) {
  const num = safeNumber(value, 0);
  if (num < 0) return 0;
  if (num > 1) return 1;
  return num;
}

function sourceToneClass(totalSources) {
  if (!Number.isFinite(totalSources) || totalSources <= 0) return "source-tone-none";
  if (totalSources <= 3) return "source-tone-few";
  return "source-tone-many";
}

function summarizeStanceCounts(statement) {
  const evidence = statement && Array.isArray(statement.evidence) ? statement.evidence : [];
  let supports = 0;
  let refutes = 0;
  let neutral = 0;
  let unknown = 0;

  for (const item of evidence) {
    const label = String(formatStance(item).label || "").toLowerCase();
    if (label.startsWith("support")) supports += 1;
    else if (label.startsWith("refute")) refutes += 1;
    else if (label.startsWith("neutral")) neutral += 1;
    else unknown += 1;
  }

  return { supports, refutes, neutral, unknown, total: evidence.length };
}

function summarizeStanceProbabilities(statement) {
  const evidence = statement && Array.isArray(statement.evidence) ? statement.evidence : [];
  let supports = 0;
  let refutes = 0;
  let neutral = 0;
  let count = 0;

  for (const item of evidence) {
    const stance = item && item.stance;
    if (!stance || typeof stance !== "object") continue;
    supports += safeNumber(stance.abstract_p_supports, 0);
    refutes += safeNumber(stance.abstract_p_refutes, 0);
    neutral += safeNumber(stance.abstract_p_neutral, 0);
    count += 1;
  }

  if (!count) return null;
  return {
    supports: clamp01(supports / count),
    refutes: clamp01(refutes / count),
    neutral: clamp01(neutral / count),
  };
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "N/A";
  return `${Math.round(clamp01(value) * 100)}%`;
}


function isReelUrl() {
  return (
    window.location.pathname.includes("/reels/") ||
    window.location.pathname.includes("/reel/")
  );
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

function formatStatementCount(count) {
  if (!Number.isFinite(count)) return "Claims to inspect";
  const label = count === 1 ? "claim" : "claims";
  return `${count} ${label} to inspect`;
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

function normalizeReelUrl(value) {
  if (!value) return "";
  try {
    const url = new URL(value, window.location.origin);
    url.hash = "";
    return url.href;
  } catch (error) {
    return String(value);
  }
}

function isSpecificReelUrl(value) {
  try {
    const url = new URL(value);
    return /^\/reels?\/[^/]+\/?$/.test(url.pathname);
  } catch (error) {
    return false;
  }
}

function getCurrentReelKey() {
  const video = getActiveVideoElement();
  const container =
    video &&
    (video.closest("article") ||
      video.closest("section") ||
      video.closest('div[role="presentation"]'));
  const permalink = container && container.querySelector('a[href*="/reel/"], a[href*="/reels/"]');
  const mediaSource = video && (video.currentSrc || video.src || video.poster);
  const locationUrl = normalizeReelUrl(window.location.href);
  const permalinkUrl = normalizeReelUrl(permalink && permalink.href);
  const specificReelUrl = [permalinkUrl, locationUrl].find(isSpecificReelUrl);

  if (specificReelUrl) return `reel:${specificReelUrl}`;

  // The URL catches normal navigation. The permalink/media fallback also
  // distinguishes successive Reels when Instagram keeps the generic /reels/
  // route while replacing the player in place.
  return [
    locationUrl,
    permalinkUrl,
    String(mediaSource || ""),
  ].join("::");
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

function markNewReelSeen(reelKey) {
  if (!reelKey || reelKey === lastSeenReelKey) return false;
  lastSeenReelKey = reelKey;
  displayedReelKey = null;
  const state = ensurePopup();
  state.resetForNewReel();
  return true;
}

function queueAnalysisForUrl(reelUrl, reelKey = getCurrentReelKey()) {
  if (!reelUrl) return;
  
  if (reelKey === lastAnalyzedReelKey) return;
  if (reelKey === pendingAnalysisReelKey) return;
  if (analysisInFlight && reelKey === inFlightReelKey) return;

  pendingAnalysisUrl = reelUrl;
  pendingAnalysisReelKey = reelKey;
  maybeStartAnalysis();
}

function maybeStartAnalysis() {
  if (analysisInFlight) return;
  if (!pendingAnalysisUrl) return;
  const nextUrl = pendingAnalysisUrl;
  const nextReelKey = pendingAnalysisReelKey;
  pendingAnalysisUrl = null;
  pendingAnalysisReelKey = null;
  runAnalysis(nextUrl, nextReelKey);
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

function getEvidenceAbstract(evidence) {
  if (!evidence || typeof evidence !== "object") return "";
  const raw =
    evidence.abstract || evidence.text || evidence.summary || evidence.snippet || "";
  return String(raw || "").trim();
}

function getEvidenceKey(statementIndex, evidenceIndex) {
  if (Number.isFinite(statementIndex) && Number.isFinite(evidenceIndex)) {
    return `${statementIndex}:${evidenceIndex}`;
  }
  return `${statementIndex || "s"}:${evidenceIndex || "e"}`;
}

function getStatementKey(statement, fallbackIndex = null) {
  const idPart =
    statement && statement.id !== undefined && statement.id !== null
      ? String(statement.id)
      : "";
  const textPart =
    statement && statement.text !== undefined && statement.text !== null
      ? String(statement.text).trim().toLowerCase()
      : "";
  if (idPart || textPart) return `${idPart}::${textPart}`;
  return `idx::${Number.isFinite(fallbackIndex) ? fallbackIndex : "unknown"}`;
}

function getEvidenceKeyForStatement(statementKey, evidence, evidenceIndex = null) {
  const idPart =
    evidence && evidence.id !== undefined && evidence.id !== null
      ? String(evidence.id)
      : "";
  const urlPart = evidence && evidence.url ? String(evidence.url) : "";
  const titlePart = evidence ? formatEvidenceTitle(evidence) : "";
  if (idPart || urlPart || titlePart) {
    return `${statementKey}::${idPart}::${urlPart}::${titlePart}`;
  }
  return `${statementKey}::eidx::${
    Number.isFinite(evidenceIndex) ? evidenceIndex : "unknown"
  }`;
}

function snapshotStatement(statement) {
  if (!statement || typeof statement !== "object") return null;
  try {
    return JSON.parse(JSON.stringify(statement));
  } catch (_error) {
    return {
      id: statement.id ?? null,
      text: statement.text ?? "",
      verdict: statement.verdict ?? null,
      score: statement.score ?? null,
      evidence: Array.isArray(statement.evidence) ? statement.evidence.slice(0, 30) : [],
    };
  }
}

function createEvidenceItem(evidence, options = {}) {
  const item = document.createElement(options.tagName || "li");
  item.className = "evidence-item";

  const header = document.createElement("div");
  header.className = "evidence-item-header";

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

  header.appendChild(titleEl);

  if (options.includeSummaryButton) {
    const summaryButton = document.createElement("button");
    summaryButton.className = "summary-button";
    summaryButton.type = "button";
    summaryButton.title = "Show context";
    summaryButton.textContent = "?";
    if (Number.isFinite(options.statementIndex)) {
      summaryButton.dataset.statementIndex = String(options.statementIndex);
    }
    if (Number.isFinite(options.evidenceIndex)) {
      summaryButton.dataset.evidenceIndex = String(options.evidenceIndex);
    }
    if (options.statementKey) {
      summaryButton.dataset.statementKey = String(options.statementKey);
    }
    header.appendChild(summaryButton);
  }

  const meta = document.createElement("div");
  meta.className = "evidence-meta";

  const relevance = document.createElement("span");
  relevance.className = "evidence-meta-item";
  relevance.textContent = `Relevance: ${formatScore(evidence.relevance)}`;
  relevance.title = TOOLTIP_RELEVANCE;

  const pubType = document.createElement("span");
  pubType.className = "evidence-meta-item";
  pubType.textContent = `Type: ${formatPubType(evidence.pub_type)}`;
  pubType.title = TOOLTIP_TYPE;

  const stance = formatStance(evidence);
  const stanceEl = document.createElement("span");
  stanceEl.className = `evidence-meta-item ${stance.className}`;
  stanceEl.textContent = `Stance: ${stance.label}`;
  stanceEl.title = TOOLTIP_STANCE;

  meta.append(relevance, pubType, stanceEl);
  item.append(header, meta);
  return item;
}

function createPopupState() {
  const popup = document.createElement("div");
  popup.className = "reel-alert";
  popup.innerHTML = `
    <div class="inspect-container">
      <button class="inspect-button" type="button" aria-label="Hold to enter Inquiry Mode">
        <span class="inspect-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"></circle>
            <line x1="16.65" y1="16.65" x2="21" y2="21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></line>
          </svg>
        </span>
        <span id="statement-count" class="inspect-count">Finding checkable claims...</span>
        <span class="hold-hint">Hold to enter Inquiry Mode</span>
        <span class="hold-progress" aria-hidden="true"><span class="hold-progress-fill"></span></span>
      </button>
    </div>
    <div class="modebar" id="modebar">
      <nav id="breadcrumb" class="breadcrumb" aria-label="Breadcrumb"></nav>
      <button id="vault-button" class="vault-button" type="button">Vault</button>
    </div>

    <div class=\"dropdown-container\">
      <div id="statements-view" class="panel-view active">
        <div class="dropdown-content">
          <ul id="dropdown-list"></ul>
        </div>
      </div>

      <div id="evidence-view" class="panel-view">
        <div class="evidence-header">
          <div class="evidence-header-top">
            <div class="evidence-label">Sources</div>
            <button id="evidence-pin-button" class="evidence-pin-button" type="button" aria-pressed="false" title="Pin claim">
              Pin
            </button>
          </div>
          <div id="evidence-title" class="evidence-title"></div>
        </div>
        <div id="evidence-stats" class="evidence-stats"></div>
        <div class="evidence-content">
          <ul id="evidence-list"></ul>
        </div>
      </div>

      <div id="summary-view" class="panel-view">
        <div class="summary-section">
          <div class="summary-section-label">Claim</div>
          <div id="summary-statement" class="summary-statement"></div>
        </div>
        <div class="summary-section">
          <button id="summary-algo-toggle" class="summary-algo-toggle" type="button" aria-expanded="false">
            Show Algorithm Probabilities
          </button>
          <div id="summary-algo-panel" class="summary-algo-panel"></div>
        </div>
        <div class="summary-section">
          <div class="summary-section-label">Source</div>
          <div id="summary-evidence" class="summary-evidence"></div>
        </div>
        <div class="summary-section">
          <div class="summary-section-label">Context</div>
          <div id="summary-loading" class="summary-loading">
            <div class="summary-spinner"></div>
            <span>Generating context...</span>
          </div>
          <div id="summary-text" class="summary-text"></div>
        </div>
      </div>

      <div id="vault-view" class="panel-view">
        <div class="vault-header">
          <div class="vault-label">Claim Vault</div>
          <div class="vault-sub">Pinned locally in this browser</div>
        </div>
        <div class="vault-content">
          <ul id="vault-list"></ul>
        </div>
      </div>
    </div>

    <button id="close-alert" aria-label="Close">×</button>
  `;

  document.body.appendChild(popup);

  // Fullscreen inquiry backdrop (created once, activated only in Inquiry Mode).
  // Implemented as four blur segments that leave a clear "hole" for the active Reel.
  const backdrop = document.createElement("div");
  backdrop.className = "rp-inquiry-backdrop";
  backdrop.setAttribute("aria-hidden", "true");
  backdrop.innerHTML = `
    <div class="rp-inquiry-seg" data-seg="top"></div>
    <div class="rp-inquiry-seg" data-seg="left"></div>
    <div class="rp-inquiry-seg" data-seg="right"></div>
    <div class="rp-inquiry-seg" data-seg="bottom"></div>
  `;
  document.body.appendChild(backdrop);
  const backdropSegments = Array.from(backdrop.querySelectorAll(".rp-inquiry-seg"));

  // Optional spotlight frame around the active video to preserve context.
  const spotlight = document.createElement("div");
  spotlight.className = "rp-inquiry-spotlight";
  spotlight.setAttribute("aria-hidden", "true");
  document.body.appendChild(spotlight);

  // Inquiry contract (brief micro-ritual shown when entering Inquiry Mode)
  const contract = document.createElement("div");
  contract.className = "rp-inquiry-contract";
  contract.setAttribute("aria-hidden", "true");
  document.body.appendChild(contract);

  // Lightweight toast for copy/feedback affordances
  const toast = document.createElement("div");
  toast.className = "rp-toast";
  toast.setAttribute("aria-hidden", "true");
  document.body.appendChild(toast);

  const handoff = document.createElement("div");
  handoff.className = "rp-handoff";
  handoff.setAttribute("aria-hidden", "true");
  document.body.appendChild(handoff);

  // Top-centered inquiry header + breadcrumb (outside the popup).
  const inquiryTopbar = document.createElement("div");
  inquiryTopbar.className = "rp-inquiry-topbar";
  inquiryTopbar.setAttribute("aria-hidden", "true");
  inquiryTopbar.innerHTML = `
    <div class="rp-inquiry-topbar-head">
      <span class="rp-inquiry-topbar-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"></circle>
          <line x1="16.65" y1="16.65" x2="21" y2="21" stroke="currentColor" stroke-width="2" stroke-linecap="round"></line>
        </svg>
      </span>
      <span class="rp-inquiry-topbar-title">Inquiry Mode</span>
    </div>
  `;
  document.body.appendChild(inquiryTopbar);

  const offlineSettingsBox = document.createElement("div");
  offlineSettingsBox.className = "rp-offline-settings";
  offlineSettingsBox.setAttribute("aria-hidden", "true");
  document.body.appendChild(offlineSettingsBox);

  const state = {
    popup,
    backdrop,
    backdropSegments,
    spotlight,
    contract,
    toast,
    handoff,
    inquiryTopbar,
    offlineSettingsBox,
    inspectButton: popup.querySelector(".inspect-button"),
    inspectIcon: popup.querySelector(".inspect-icon"),
    statementCount: popup.querySelector("#statement-count"),
    breadcrumb: popup.querySelector("#breadcrumb"),
    vaultButton: popup.querySelector("#vault-button"),
    modeSubtitle: null,
    triageHint: null,
    statementsView: popup.querySelector("#statements-view"),
    evidenceView: popup.querySelector("#evidence-view"),
    evidencePinButton: popup.querySelector("#evidence-pin-button"),
    evidenceTitle: popup.querySelector("#evidence-title"),
    evidenceStats: popup.querySelector("#evidence-stats"),
    evidenceList: popup.querySelector("#evidence-list"),
    summaryView: popup.querySelector("#summary-view"),
    summaryStatement: popup.querySelector("#summary-statement"),
    summaryAlgoToggle: popup.querySelector("#summary-algo-toggle"),
    summaryAlgoPanel: popup.querySelector("#summary-algo-panel"),
    summaryEvidence: popup.querySelector("#summary-evidence"),
    summaryLoading: popup.querySelector("#summary-loading"),
    summaryText: popup.querySelector("#summary-text"),
    vaultView: popup.querySelector("#vault-view"),
    vaultList: popup.querySelector("#vault-list"),
    listContent: popup.querySelector("#dropdown-list"),

    viewMode: "statements",
    isInquiryMode: false,
    statements: [],
    summaryCache: new Map(),
    summaryInFlight: new Set(),
    activeStatement: null,
    activeStatementIndex: null,
    activeEvidence: null,
    activeEvidenceIndex: null,
    activeSummaryKey: null,
    activeStatementKey: null,
    summaryAlgoOpen: false,
    bookmarks: [],
    bookmarksLoaded: false,
    inquirySessionMetrics: null,
    isVaultOpen: false,
    externalStatementsByKey: new Map(),

    activeVideo: null,
    videoWasPlaying: false,
  };

  if (state.evidencePinButton) {
    state.evidencePinButton.disabled = true;
  }

  state.flashPeel = function () {
    state.popup.classList.remove("peel");
    // force reflow
    void state.popup.offsetWidth;
    state.popup.classList.add("peel");
    window.setTimeout(() => state.popup.classList.remove("peel"), 240);
  };

  state.setExpanded = function (isExpanded) {
    state.popup.classList.toggle("expanded", isExpanded);
  };

  state.initBackendToggle = function (reelUrl, reelKey) {
    state.offlineSettingsBox.style.display = "block";
    state.offlineSettingsBox.innerHTML = "";
    const effectiveBackendPref = getEffectiveBackendPref();

    const label = document.createElement("div");
    label.textContent =
      effectiveBackendPref === "local" ? "Backend: Lokal" : "Backend: Extern";
    state.offlineSettingsBox.appendChild(label);

    if (!isInstagramPage()) {
      state.offlineSettingsBox.onclick = (e) => e.stopPropagation();
      return;
    }

    const dropdown = document.createElement("div");
    dropdown.className = "rp-offline-dropdown";

    const options = [
      { id: "local", label: "Lokal (Mock Server)" },
      { id: "remote", label: "API Backend" },
    ];

    for (const opt of options) {
      const item = document.createElement("div");
      item.className = "rp-offline-dropdown-item";
      if (opt.id === effectiveBackendPref) item.classList.add("selected");
      item.textContent = opt.label;
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        dropdown.style.display = "none";
        if (opt.id === effectiveBackendPref) return;

        setBackendPref(opt.id);

        // Reset dedup state so analysis re-runs with new backend
        lastAnalyzedReelKey = null;
        displayedReelKey = null;

        removePopup();
        runAnalysis(reelUrl, reelKey);
      });
      dropdown.appendChild(item);
    }

    state.offlineSettingsBox.appendChild(dropdown);

    state.offlineSettingsBox.onclick = (e) => {
      e.stopPropagation();
      const isVisible = dropdown.style.display === "block";
      dropdown.style.display = isVisible ? "none" : "block";
    };

    document.addEventListener("click", function closeDropdown(e) {
      if (!state.offlineSettingsBox.contains(e.target)) {
        dropdown.style.display = "none";
      }
    });
  };

  state.setStatementCount = function (count) {
    state.statementCount.textContent = formatStatementCount(count);
  };

  state.setLoading = function () {
    state.statementCount.textContent = "Finding checkable claims...";
    state.setReadyPulse(false);
  };

  state.setNotApplicable = function (reason) {
    state.setExpanded(false);
    state.exitInquiryMode();
    state.setReadyPulse(false);
    state.listContent.innerHTML = "";
    state.statements = [];
    state.statementCount.textContent =
      reason === "no-audio"
        ? "No audio \u2014 nothing to inspect"
        : reason === "no-local-data"
          ? "No local data for this Reel"
          : "Nothing to inspect";
  };

  
  state.flashHoldHint = function () {
    if (!state.inquiryTopbar) return;
    state.inquiryTopbar.classList.add("hint-flash");
    window.setTimeout(() => {
      state.inquiryTopbar && state.inquiryTopbar.classList.remove("hint-flash");
    }, 800);
  };

  state.setReadyPulse = function (on) {
    if (!state.inspectIcon) return;
    if (on) {
      state.inspectIcon.classList.add("rp-ready");
    } else {
      state.inspectIcon.classList.remove("rp-ready");
    }
  };

  state.bumpAttention = function () {
    if (!state.inspectIcon) return;
    state.inspectIcon.classList.remove("rp-attn");
    void state.inspectIcon.offsetWidth;
    state.inspectIcon.classList.add("rp-attn");
    window.setTimeout(() => {
      state.inspectIcon && state.inspectIcon.classList.remove("rp-attn");
    }, 2200);
  };

  state.showToast = function (message) {
    if (!state.toast) return;
    const msg = String(message || "").trim();
    if (!msg) return;
    const rect = state.popup.getBoundingClientRect();
    const maxW = Math.min(360, window.innerWidth - 20);
    state.toast.style.maxWidth = `${maxW}px`;
    state.toast.textContent = msg;

    // Position near the panel, but keep in-frame.
    const left = Math.max(10, Math.min(rect.left, window.innerWidth - maxW - 10));
    const top = Math.max(10, rect.top - 44);
    state.toast.style.left = `${Math.round(left)}px`;
    state.toast.style.top = `${Math.round(top)}px`;

    state.toast.classList.add("active");
    window.clearTimeout(state._toastTimer);
    state._toastTimer = window.setTimeout(() => {
      state.toast && state.toast.classList.remove("active");
    }, 1600);
  };

  state.showHandoff = function (message) {
    if (!state.handoff) return;
    const msg = String(message || "").trim();
    if (!msg) return;
    state.handoff.textContent = msg;
    state.handoff.classList.remove("active");
    void state.handoff.offsetWidth;
    state.handoff.classList.add("active");
    window.clearTimeout(state._handoffTimer);
    state._handoffTimer = window.setTimeout(() => {
      state.handoff && state.handoff.classList.remove("active");
    }, 3800);
  };

  state.setVaultOpen = function (isOpen) {
    const next = Boolean(isOpen);
    state.isVaultOpen = next;
    if (state.vaultView) {
      state.vaultView.classList.toggle("active", next);
    }
    if (state.vaultButton) {
      state.vaultButton.classList.toggle("active", next);
    }
    if (next) state.renderVaultList();
  };

  state.resetInquirySessionMetrics = function () {
    state.inquirySessionMetrics = {
      claims: new Set(),
      sources: new Set(),
      agreementSources: new Set(),
      disagreementSources: new Set(),
    };
  };

  state.trackInquiryClaimCoverage = function (statement, statementIndex) {
    if (!state.isInquiryMode) return;
    if (!state.inquirySessionMetrics) state.resetInquirySessionMetrics();
    state.inquirySessionMetrics.claims.add(getStatementKey(statement, statementIndex));
  };

  state.trackInquirySummarySource = function (
    statement,
    statementIndex,
    evidence,
    evidenceIndex
  ) {
    if (!state.isInquiryMode) return;
    if (!state.inquirySessionMetrics) state.resetInquirySessionMetrics();
    const claimKey = getStatementKey(statement, statementIndex);
    const evidenceKey = getEvidenceKeyForStatement(claimKey, evidence, evidenceIndex);
    state.inquirySessionMetrics.claims.add(claimKey);
    state.inquirySessionMetrics.sources.add(evidenceKey);

    const stanceLabel = String(formatStance(evidence).label || "").toLowerCase();
    if (stanceLabel.startsWith("support")) {
      state.inquirySessionMetrics.agreementSources.add(evidenceKey);
    } else if (stanceLabel.startsWith("refute")) {
      state.inquirySessionMetrics.disagreementSources.add(evidenceKey);
    }
  };

  state.buildInquiryHandoffMessage = function () {
    const metrics = state.inquirySessionMetrics;
    if (!metrics) return "";
    return `You checked: ${metrics.claims.size} claims - ${metrics.sources.size} sources - Agreement: ${metrics.agreementSources.size} - Disagreement: ${metrics.disagreementSources.size}`;
  };

  state.statementBookmarkKey = function (statement, statementIndex = null) {
    return getStatementKey(statement, statementIndex);
  };

  state.isBookmarkedStatement = function (statement, statementIndex = null) {
    const key = state.statementBookmarkKey(statement, statementIndex);
    return state.bookmarks.some((bookmark) => bookmark && bookmark.key === key);
  };

  state.updatePinButtonState = function (button, statement, statementIndex = null) {
    if (!button) return;
    const pinned = state.isBookmarkedStatement(statement, statementIndex);
    button.classList.toggle("pinned", pinned);
    button.textContent = pinned ? "Pinned" : "Pin";
    button.setAttribute("aria-pressed", pinned ? "true" : "false");
    button.title = pinned ? "Unpin claim" : "Pin claim";
  };

  state.refreshStatementPinButtons = function () {
    if (!state.evidencePinButton || !state.activeStatement) {
      if (state.evidencePinButton) {
        state.evidencePinButton.disabled = true;
        state.evidencePinButton.classList.remove("pinned");
        state.evidencePinButton.textContent = "Pin";
        state.evidencePinButton.setAttribute("aria-pressed", "false");
        state.evidencePinButton.title = "Pin claim";
      }
      return;
    }
    state.evidencePinButton.disabled = false;
    state.updatePinButtonState(
      state.evidencePinButton,
      state.activeStatement,
      state.activeStatementIndex
    );
  };

  state.persistBookmarks = function () {
    if (!chrome?.storage?.local) return;
    chrome.storage.local.set({
      [BOOKMARKS_STORAGE_KEY]: state.bookmarks,
    });
  };

  state.renderVaultList = function () {
    if (!state.vaultList) return;
    state.vaultList.innerHTML = "";

    if (!state.bookmarks.length) {
      const empty = document.createElement("li");
      empty.className = "vault-empty";
      empty.textContent = "No pinned claims yet.";
      state.vaultList.appendChild(empty);
      return;
    }

    const sorted = state.bookmarks
      .slice()
      .sort((a, b) => Number(b?.savedAt || 0) - Number(a?.savedAt || 0));

    for (const bookmark of sorted) {
      const item = document.createElement("li");
      item.className = "vault-item";
      item.dataset.bookmarkKey = bookmark.key;

      const text = document.createElement("div");
      text.className = "vault-item-text";
      text.textContent = bookmark.text || "Unnamed claim";

      const meta = document.createElement("div");
      meta.className = "vault-item-meta";
      const gapLabel =
        Array.isArray(bookmark.gaps) && bookmark.gaps.length
          ? bookmark.gaps.join(" / ")
          : "No major gaps";
      meta.textContent = `${bookmark.sourceCount || 0} source${
        bookmark.sourceCount === 1 ? "" : "s"
      } / ${gapLabel}`;

      const actions = document.createElement("div");
      actions.className = "vault-item-actions";

      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "vault-open-item";
      openBtn.dataset.bookmarkKey = bookmark.key;
      openBtn.textContent = "Open";

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "vault-remove-item";
      removeBtn.dataset.bookmarkKey = bookmark.key;
      removeBtn.textContent = "Remove";

      actions.append(openBtn, removeBtn);
      item.append(text, meta, actions);
      state.vaultList.appendChild(item);
    }
  };

  state.toggleBookmark = function (statement, statementIndex = null) {
    if (!statement) return;
    const key = state.statementBookmarkKey(statement, statementIndex);
    const existingIndex = state.bookmarks.findIndex((bookmark) => bookmark.key === key);
    if (existingIndex >= 0) {
      state.bookmarks.splice(existingIndex, 1);
      state.showToast("Claim removed from Vault");
    } else {
      const sourceCount =
        statement && Array.isArray(statement.evidence) ? statement.evidence.length : 0;
      state.bookmarks.push({
        key,
        id: statement && statement.id !== undefined ? statement.id : null,
        text: statement && statement.text ? String(statement.text) : "",
        gaps: computeStatementGaps(statement),
        sourceCount,
        savedAt: Date.now(),
        pageUrl: window.location.href,
        statementSnapshot: snapshotStatement(statement),
      });
      state.showToast("Claim pinned to Vault");
    }
    state.persistBookmarks();
    state.refreshStatementPinButtons();
    state.renderVaultList();
  };

  state.loadBookmarks = function () {
    if (!chrome?.storage?.local) {
      state.bookmarksLoaded = true;
      state.bookmarks = [];
      state.renderVaultList();
      state.refreshStatementPinButtons();
      return;
    }
    chrome.storage.local.get([BOOKMARKS_STORAGE_KEY], (stored) => {
      const raw =
        stored && Array.isArray(stored[BOOKMARKS_STORAGE_KEY])
          ? stored[BOOKMARKS_STORAGE_KEY]
          : [];
      state.bookmarks = raw
        .filter((item) => item && typeof item === "object")
        .filter((item) => item.key && item.text)
        .map((item) => ({
          ...item,
          statementSnapshot:
            item.statementSnapshot && typeof item.statementSnapshot === "object"
              ? item.statementSnapshot
              : null,
        }))
        .slice(0, 300);
      state.bookmarksLoaded = true;
      state.renderVaultList();
      state.refreshStatementPinButtons();
    });
  };

  state.showInquiryContract = function () {
    if (!state.contract) return;

    const maxW = Math.min(520, window.innerWidth - 24);
    state.contract.style.setProperty("--rp-contract-maxw", `${maxW}px`);

    // Fullscreen, centered contract that stays until explicit confirmation.
    state.contract.innerHTML = `
      <div class="rp-contract-card" role="status" aria-live="polite">
        <div class="rp-contract-title">Inquiry Mode</div>
        <div class="rp-contract-body">
          <div class="rp-contract-line"><span class="rp-dot"></span><b>Pause scrolling.</b> Switch from watching to checking.</div>
          <div class="rp-contract-line"><span class="rp-dot"></span><b>Triage claims</b> by coverage and visible gaps.</div>
          <div class="rp-contract-line"><span class="rp-dot"></span><b>Peel into sources</b> and provenance when you choose.</div>
        </div>
        <div class="rp-contract-foot">Hold <b>background</b> to exit · Hold the <b>Reel</b> to return</div>
        <div class="rp-contract-actions">
          <button type="button" class="rp-contract-confirm">Understood</button>
        </div>
      </div>
    `;

    state.contract.classList.remove("active");
    void state.contract.offsetWidth;
    state.contract.classList.add("active");

    const confirmBtn = state.contract.querySelector(".rp-contract-confirm");
    if (confirmBtn) {
      confirmBtn.addEventListener(
        "click",
        () => {
          state.contract && state.contract.classList.remove("active");
        },
        { once: true }
      );
    }
  };


  state.renderBreadcrumbInto = function (container, crumbs, options = {}) {
    if (!container) return;
    const disableReelCrumb = Boolean(options.disableReelCrumb);
    container.innerHTML = "";

    crumbs.forEach((c, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "crumb";
      btn.textContent = c.label;

      const reelBlocked = disableReelCrumb && c.label === "Reel";
      if (c.current) {
        btn.setAttribute("aria-current", "page");
      } else if (reelBlocked) {
        btn.classList.add("crumb-disabled");
        btn.title = "Hold background to exit Inquiry Mode";
      } else {
        btn.addEventListener("click", c.onClick);
      }

      container.appendChild(btn);
      if (idx < crumbs.length - 1) {
        const sep = document.createElement("span");
        sep.className = "crumb-sep";
        sep.textContent = "›";
        container.appendChild(sep);
      }
    });
  };

  state.updateBreadcrumb = function () {
    const crumbs = [];
    if (state.vaultButton) {
      state.vaultButton.classList.toggle("active", state.isVaultOpen);
    }

    crumbs.push({
      label: "Reel",
      onClick: () => {
        if (state.isInquiryMode) {
          state.flashHoldHint();
          return;
        }
        state.setExpanded(false);
        state.showStatementsView();
      },
      current: false,
    });

    crumbs.push({
      label: "Claims",
      onClick: () => {
        state.showStatementsView();
        state.setExpanded(true);
      },
      current: state.viewMode === "statements",
    });

    if (state.viewMode === "evidence" || state.viewMode === "summary") {
      crumbs.push({
        label: "Sources",
        onClick: () => {
          if (state.activeStatement) {
            state.showEvidenceView(state.activeStatement, state.activeStatementIndex, {
              statementKey: state.activeStatementKey,
            });
          } else {
            state.showStatementsView();
          }
          state.setExpanded(true);
        },
        current: state.viewMode === "evidence",
      });
    }

    if (state.viewMode === "summary") {
      crumbs.push({
        label: "Details",
        onClick: () => {},
        current: true,
      });
    }

    state.renderBreadcrumbInto(state.breadcrumb, crumbs, {
      disableReelCrumb: state.isInquiryMode,
    });
    if (state.isInquiryMode) {
      window.requestAnimationFrame(() => {
        state.positionInquiryTopbar();
      });
    }
  };

  state.showStatementsView = function () {
    state.viewMode = "statements";
    state.setVaultOpen(false);
    state.statementsView.classList.add("active");
    state.evidenceView.classList.remove("active");
    state.summaryView.classList.remove("active");
    state.vaultView && state.vaultView.classList.remove("active");
    state.updateBreadcrumb();
  };

  state.showVaultView = function () {
    state.setVaultOpen(true);
    state.setExpanded(true);
    state.renderVaultList();
    state.updateBreadcrumb();
  };

  state.renderEvidenceList = function (statement, statementIndex, statementKey = null) {
    state.evidenceTitle.textContent =
      statement && statement.text ? statement.text : "Sources";

    state.evidenceList.innerHTML = "";

    const evidenceItemsRaw =
      statement && Array.isArray(statement.evidence) ? statement.evidence : [];

    const evidenceItems = sortEvidenceByRelevanceDesc(evidenceItemsRaw);

    if (!evidenceItems.length) {
      const emptyItem = document.createElement("li");
      emptyItem.className = "evidence-empty";
      emptyItem.textContent = "No sources found.";
      state.evidenceList.appendChild(emptyItem);
      return;
    }

    for (let i = 0; i < evidenceItems.length; i++) {
      const evidence = evidenceItems[i];
      const item = createEvidenceItem(evidence, {
        tagName: "li",
        includeSummaryButton: true,
        statementIndex,
        evidenceIndex: i,
        statementKey,
      });
      state.evidenceList.appendChild(item);
    }
  };

  state.renderEvidenceStanceOverview = function (statement) {
    if (!state.evidenceStats) return;
    const counts = summarizeStanceCounts(statement);
    const sourceTone = sourceToneClass(counts.total);
    state.evidenceStats.className = `evidence-stats ${sourceTone}`;

    const supportRefuteTotal = counts.supports + counts.refutes;
    const supportPct =
      supportRefuteTotal > 0
        ? Math.round((counts.supports / supportRefuteTotal) * 100)
        : 0;
    const refutePct =
      supportRefuteTotal > 0
        ? Math.round((counts.refutes / supportRefuteTotal) * 100)
        : 0;

    if (!counts.total) {
      state.evidenceStats.innerHTML = `
        <div class="evidence-stats-head">
          <span class="evidence-stats-icon">!</span>
          <span class="evidence-stats-title">No sources available</span>
        </div>
      `;
      return;
    }

    state.evidenceStats.innerHTML = `
      <div class="evidence-stats-head">
        <span class="evidence-stats-icon">${counts.total}</span>
        <span class="evidence-stats-title">Source stance overview</span>
      </div>
      <div class="evidence-stats-bar">
        <span class="evidence-stats-fill support" style="width:${supportPct}%"></span>
        <span class="evidence-stats-fill refute" style="width:${refutePct}%"></span>
      </div>
      <div class="evidence-stats-labels">
        <span>S ${counts.supports}</span>
        <span>R ${counts.refutes}</span>
        <span>N ${counts.neutral}</span>
      </div>
    `;
  };

  state.showEvidenceView = function (statement, statementIndex, options = {}) {
    const statementKey =
      options && options.statementKey
        ? String(options.statementKey)
        : Number.isFinite(statementIndex)
        ? state.statementBookmarkKey(statement, statementIndex)
        : null;
    state.viewMode = "evidence";
    state.setVaultOpen(false);
    state.statementsView.classList.remove("active");
    state.evidenceView.classList.add("active");
    state.summaryView.classList.remove("active");
    state.vaultView && state.vaultView.classList.remove("active");

    state.setExpanded(true);
    state.activeStatement = statement || null;
    state.activeStatementIndex = Number.isFinite(statementIndex)
      ? statementIndex
      : null;
    state.activeEvidence = null;
    state.activeEvidenceIndex = null;
    state.activeSummaryKey = null;
    state.activeStatementKey = statementKey;
    state.refreshStatementPinButtons();

    if (state.isInquiryMode) {
      state.trackInquiryClaimCoverage(statement, statementIndex);
    }

    state.renderEvidenceList(statement, statementIndex, statementKey);
    state.renderEvidenceStanceOverview(statement);
    state.updateBreadcrumb();
    state.flashPeel();
  };

  state.setSummaryLoading = function (isLoading) {
    state.summaryLoading.classList.toggle("active", isLoading);
  };

  state.setSummaryAlgoVisible = function (isVisible) {
    if (!state.summaryAlgoPanel || !state.summaryAlgoToggle) return;
    state.summaryAlgoOpen = Boolean(isVisible);
    state.summaryAlgoPanel.classList.toggle("active", state.summaryAlgoOpen);
    state.summaryAlgoToggle.setAttribute(
      "aria-expanded",
      state.summaryAlgoOpen ? "true" : "false"
    );
    state.summaryAlgoToggle.textContent = state.summaryAlgoOpen
      ? "Hide Algorithm Probabilities"
      : "Show Algorithm Probabilities";
  };

  state.renderSummaryAlgoPanel = function (statement) {
    if (!state.summaryAlgoPanel) return;
    state.summaryAlgoPanel.innerHTML = "";

    const probabilities = summarizeStanceProbabilities(statement);
    if (!probabilities) {
      const empty = document.createElement("div");
      empty.className = "summary-algo-empty";
      empty.textContent = "No probability scores available.";
      state.summaryAlgoPanel.appendChild(empty);
      return;
    }

    const rows = [
      { label: "Supports", value: probabilities.supports, tone: "support" },
      { label: "Refutes", value: probabilities.refutes, tone: "refute" },
      { label: "Neutral", value: probabilities.neutral, tone: "neutral" },
    ];

    for (const rowData of rows) {
      const row = document.createElement("div");
      row.className = `summary-algo-row ${rowData.tone}`;

      const rowHead = document.createElement("div");
      rowHead.className = "summary-algo-row-head";

      const label = document.createElement("span");
      label.className = "summary-algo-row-label";
      label.textContent = rowData.label;

      const value = document.createElement("span");
      value.className = "summary-algo-row-value";
      value.textContent = formatPercent(rowData.value);

      rowHead.append(label, value);

      const track = document.createElement("div");
      track.className = "summary-algo-track";

      const fill = document.createElement("span");
      fill.className = "summary-algo-fill";
      fill.style.width = formatPercent(rowData.value);
      track.appendChild(fill);

      row.append(rowHead, track);
      state.summaryAlgoPanel.appendChild(row);
    }
  };

  state.showSummaryView = function (
    statement,
    evidence,
    statementIndex,
    evidenceIndex,
    options = {}
  ) {
    const statementKey =
      options && options.statementKey
        ? String(options.statementKey)
        : Number.isFinite(statementIndex)
        ? state.statementBookmarkKey(statement, statementIndex)
        : state.activeStatementKey || null;
    state.viewMode = "summary";
    state.setVaultOpen(false);
    state.statementsView.classList.remove("active");
    state.evidenceView.classList.remove("active");
    state.summaryView.classList.add("active");
    state.vaultView && state.vaultView.classList.remove("active");
    state.setExpanded(true);

    state.activeStatement = statement || null;
    state.activeStatementIndex = Number.isFinite(statementIndex)
      ? statementIndex
      : null;
    state.activeEvidence = evidence || null;
    state.activeEvidenceIndex = Number.isFinite(evidenceIndex)
      ? evidenceIndex
      : null;
    state.activeStatementKey = statementKey;

    if (state.isInquiryMode) {
      state.trackInquirySummarySource(statement, statementIndex, evidence, evidenceIndex);
    }

    state.summaryStatement.textContent =
      (statement && statement.text) || "Claim details";

    state.renderSummaryAlgoPanel(statement);
    state.setSummaryAlgoVisible(false);

    state.summaryEvidence.innerHTML = "";
    if (evidence) {
      const card = createEvidenceItem(evidence, {
        tagName: "div",
        includeSummaryButton: false,
      });
      state.summaryEvidence.appendChild(card);
    }

    const key = statementKey
      ? getEvidenceKeyForStatement(statementKey, evidence, evidenceIndex)
      : getEvidenceKey(statementIndex, evidenceIndex);
    state.activeSummaryKey = key;
    const cached = state.summaryCache.get(key);
    if (cached) {
      state.setSummaryLoading(false);
      state.summaryText.textContent = cached;
      state.updateBreadcrumb();
      state.flashPeel();
      return;
    }

    state.summaryText.textContent = "";
    state.setSummaryLoading(true);
    state.updateBreadcrumb();
    state.flashPeel();
    state.requestEvidenceSummary(statement, evidence, key);
  };

  state.requestEvidenceSummary = async function (statement, evidence, key) {
    const abstract = getEvidenceAbstract(evidence);
    if (!abstract) {
      const message = "Context unavailable: no abstract provided.";
      state.summaryCache.set(key, message);
      if (state.activeSummaryKey === key) {
        state.setSummaryLoading(false);
        state.summaryText.textContent = message;
      }
      return;
    }

    if (state.summaryInFlight.has(key)) return;
    state.summaryInFlight.add(key);

    const payloadEvidence = {
      abstract,
      title:
        evidence?.title || evidence?.article_title || evidence?.paper_title || "",
      pubmed_id: evidence?.pubmed_id,
      url: evidence?.url,
      stance: evidence?.stance || null,
    };

    try {
      const summaryBackendPref = getEffectiveBackendPref();
      const summaryApiBase =
        summaryBackendPref === "local" ? await resolveLocalApiBase() : null;
      const data = await chrome.runtime.sendMessage({
        type: "getEvidenceSummary",
        statement: statement?.text || "",
        evidence: payloadEvidence,
        backendPref: summaryBackendPref,
        apiBase: summaryApiBase,
      });

      const summaryText =
        (data && data.summary) || (data && data.error) || "Context unavailable.";
      state.summaryCache.set(key, summaryText);
      if (state.activeSummaryKey === key) {
        state.summaryText.textContent = summaryText;
      }
    } catch (error) {
      const message = "Context unavailable.";
      state.summaryCache.set(key, message);
      if (state.activeSummaryKey === key) {
        state.summaryText.textContent = message;
      }
    } finally {
      state.summaryInFlight.delete(key);
      if (state.activeSummaryKey === key) {
        state.setSummaryLoading(false);
      }
    }
  };

  state.enterInquiryMode = function () {
    state.isInquiryMode = true;
    state.setVaultOpen(false);
    state.resetInquirySessionMetrics();
    state.setReadyPulse(false);
    state.popup.classList.add("inquiry-mode");
    state.inquiryTopbar && state.inquiryTopbar.classList.add("active");
    state.backdrop.classList.add("active");
    state.backdrop.style.pointerEvents = "";
    state.backdropSegments.forEach((seg) => {
      seg.style.pointerEvents = "";
      seg.style.cursor = "";
      seg.style.touchAction = "";
    });
    document.documentElement.classList.add("rp-inquiry-open");
    document.body.classList.add("rp-inquiry-open");

    // Block scrolling during Inquiry Mode.
    setInquiryScrollLock(true);

    // Pause active video to make the mode shift salient.
    const video = getActiveVideoElement();
    state.activeVideo = video;
    state.videoWasPlaying = Boolean(video && !video.paused && !video.ended);
    state.attachInquiryPauseGuard();
    state.pauseInquiryVideo(video);

    state.setReelFocus(true);
    state.positionSpotlight();
    state.positionBackdropSegments();
    state.positionInquiryPanel();
    state.positionInquiryTopbar();
    attachReelExitHold();
    attachBackdropExitHold();
    state.showInquiryContract();

    state.setExpanded(true);
    state.showStatementsView();
  };

  state.exitInquiryMode = function () {
    const handoffMessage = state.buildInquiryHandoffMessage();
    state.isInquiryMode = false;
    state.popup.classList.remove("inquiry-mode");
    state.inquiryTopbar && state.inquiryTopbar.classList.remove("active");
    state.backdrop.classList.remove("active");
    state.backdrop.classList.remove("exit-holding");
    state.backdrop.style.pointerEvents = "none";
    state.backdropSegments.forEach((seg) => {
      seg.style.pointerEvents = "none";
      seg.style.cursor = "default";
      seg.style.touchAction = "auto";
    });
    state.contract && state.contract.classList.remove("active");
    state.toast && state.toast.classList.remove("active");
    document.documentElement.classList.remove("rp-inquiry-open");
    document.body.classList.remove("rp-inquiry-open");

    setInquiryScrollLock(false);
    forceRestorePageInteraction();
    window.setTimeout(forceRestorePageInteraction, 0);
    window.dispatchEvent(new CustomEvent("reelpeel:inquiry-exit"));

    detachReelExitHold();
    detachBackdropExitHold();
    state.detachInquiryPauseGuard();

    // Resume video if we paused it.
    if (state.activeVideo && state.videoWasPlaying) {
      try {
        state.activeVideo.play();
      } catch (e) {
        // Autoplay restrictions may block; user can manually resume.
      }
    }
    state.activeVideo = null;
    state.videoWasPlaying = false;
    state.spotlight.classList.remove("active");
    state.setReelFocus(false);
    // Restore default positioning.
    state.popup.style.left = "";
    state.popup.style.top = "";
    state.popup.style.width = "";
    state.popup.style.maxHeight = "";
    if (state.inquiryTopbar) {
      state.inquiryTopbar.style.left = "";
      state.inquiryTopbar.style.top = "";
    }
    if (handoffMessage) {
      state.showHandoff(handoffMessage);
    }
    state.inquirySessionMetrics = null;
    state.updateBreadcrumb();
  };

  state.pauseInquiryVideo = function (candidate) {
    if (!state.isInquiryMode) return;

    const video =
      candidate instanceof HTMLVideoElement
        ? candidate
        : getActiveVideoElement() || state.activeVideo;
    if (!video) return;

    // Instagram can replace the video element while this tab is backgrounded.
    // Keep the currently visible player as the protected one in that case.
    state.activeVideo = video;
    try {
      if (!video.paused) video.pause();
    } catch (e) {
      // A disconnected player can throw while React replaces the Reel.
    }
  };

  state.attachInquiryPauseGuard = function () {
    if (state._inquiryPauseGuard) return;

    const pauseWhenVisible = () => {
      if (!state.isInquiryMode) return;
      state.pauseInquiryVideo();
    };
    const pauseOnPlay = (event) => {
      if (!state.isInquiryMode) return;
      const video = event && event.target;
      if (!(video instanceof HTMLVideoElement)) return;

      // Do not interfere with Instagram's offscreen/preload players. The
      // current player and the one already held by Inquiry Mode must remain
      // paused, including after a return from an external PubMed tab.
      if (video === state.activeVideo || video === getActiveVideoElement()) {
        state.pauseInquiryVideo(video);
      }
    };

    state._inquiryPauseGuard = { pauseWhenVisible, pauseOnPlay };
    window.addEventListener("focus", pauseWhenVisible);
    window.addEventListener("pageshow", pauseWhenVisible);
    document.addEventListener("visibilitychange", pauseWhenVisible);
    document.addEventListener("play", pauseOnPlay, true);
  };

  state.detachInquiryPauseGuard = function () {
    const guard = state._inquiryPauseGuard;
    if (!guard) return;
    window.removeEventListener("focus", guard.pauseWhenVisible);
    window.removeEventListener("pageshow", guard.pauseWhenVisible);
    document.removeEventListener("visibilitychange", guard.pauseWhenVisible);
    document.removeEventListener("play", guard.pauseOnPlay, true);
    state._inquiryPauseGuard = null;
  };

  state.positionSpotlight = function () {
    const video = state.activeVideo;
    if (!video) {
      state.spotlight.classList.remove("active");
      return;
    }
    const rect = video.getBoundingClientRect();
    const pad = 6;
    const left = Math.max(8, rect.left - pad);
    const top = Math.max(8, rect.top - pad);
    const width = Math.min(window.innerWidth - 16, rect.width + pad * 2);
    const height = Math.min(window.innerHeight - 16, rect.height + pad * 2);
    state.spotlight.style.left = `${left}px`;
    state.spotlight.style.top = `${top}px`;
    state.spotlight.style.width = `${width}px`;
    state.spotlight.style.height = `${height}px`;
    state.spotlight.classList.add("active");
    state.positionBackdropSegments();
  };

  state.positionBackdropSegments = function () {
    if (!state.isInquiryMode || !state.activeVideo || !state.backdropSegments || !state.backdropSegments.length) return;

    const target = state.reelFocusEl || state.activeVideo;
    const r0 = target.getBoundingClientRect();
    const pad = 2;

    const left = Math.max(0, Math.floor(r0.left - pad));
    const top = Math.max(0, Math.floor(r0.top - pad));
    const right = Math.min(window.innerWidth, Math.ceil(r0.right + pad));
    const bottom = Math.min(window.innerHeight, Math.ceil(r0.bottom + pad));

    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const segMap = {};
    state.backdropSegments.forEach((seg) => {
      const key = seg.getAttribute("data-seg");
      if (key) segMap[key] = seg;
    });

    const setSeg = (key, x, y, w, h) => {
      const seg = segMap[key];
      if (!seg) return;
      if (w <= 0 || h <= 0) {
        seg.style.display = "none";
        return;
      }
      seg.style.display = "block";
      seg.style.left = `${x}px`;
      seg.style.top = `${y}px`;
      seg.style.width = `${w}px`;
      seg.style.height = `${h}px`;
    };

    // Top / bottom fill across full width; left / right fill beside the Reel.
    setSeg("top", 0, 0, vw, top);
    setSeg("bottom", 0, bottom, vw, Math.max(0, vh - bottom));
    setSeg("left", 0, top, left, Math.max(0, bottom - top));
    setSeg("right", right, top, Math.max(0, vw - right), Math.max(0, bottom - top));
  };


  state.positionInquiryTopbar = function () {
    if (!state.isInquiryMode || !state.inquiryTopbar || !state.activeVideo) return;
    const target = state.reelFocusEl || state.activeVideo;
    const r = target.getBoundingClientRect();
    const x = Math.round((r.left + r.right) / 2);
    const y = Math.round(r.top + 10);

    // Clamp so it stays fully on-screen.
    const bw = state.inquiryTopbar.offsetWidth || 340;
    const bh = state.inquiryTopbar.offsetHeight || 56;
    const minX = Math.round(bw / 2) + 10;
    const maxX = window.innerWidth - Math.round(bw / 2) - 10;
    const minY = 10;
    const maxY = window.innerHeight - bh - 10;

    const cx = Math.max(minX, Math.min(x, maxX));
    const cy = Math.max(minY, Math.min(y, maxY));

    state.inquiryTopbar.style.left = `${cx}px`;
    state.inquiryTopbar.style.top = `${cy}px`;
  };


  
  state.setReelFocus = function (on) {
    const cleanupReelClasses = () => {
      const marked = document.querySelectorAll(".rp-reel-focus, .rp-reel-soft-blur");
      marked.forEach((el) => {
        el.classList.remove("rp-reel-focus", "rp-reel-soft-blur");
      });
    };

    const video = state.activeVideo;
    if (!video && !on) {
      cleanupReelClasses();
      state.reelFocusEl = null;
      return;
    }
    if (!video) return;
    const target = video.closest("section") || video.parentElement || video;
    if (on) {
      cleanupReelClasses();
      target.classList.add("rp-reel-focus", "rp-reel-soft-blur");
      state.reelFocusEl = target;
    } else {
      cleanupReelClasses();
      state.reelFocusEl = null;
    }
  };

  state.positionInquiryPanel = function () {
    if (!state.isInquiryMode || !state.activeVideo) return;

    const rect = state.activeVideo.getBoundingClientRect();
    const margin = 14;
    const desiredW = 360;
    const maxW = Math.max(280, Math.min(desiredW, window.innerWidth - 20));
    state.popup.style.width = `${maxW}px`;

    const spaceRight = window.innerWidth - rect.right - margin;
    const spaceLeft = rect.left - margin;

    let left;
    if (spaceRight >= maxW) {
      left = rect.right + margin;
    } else if (spaceLeft >= maxW) {
      left = rect.left - margin - maxW;
    } else {
      left = Math.max(10, window.innerWidth - maxW - 10);
    }

    const minTop = 10;
    const minPanelH = Math.min(260, window.innerHeight - 20);

    // Anchor to the Reel's top edge, but keep the panel in-frame.
    let top = rect.top;
    top = Math.max(minTop, Math.min(top, window.innerHeight - minPanelH - 10));

    const availH = Math.max(120, window.innerHeight - top - 10);
    const maxH = Math.max(minPanelH, Math.min(availH, rect.height));

    state.popup.style.left = `${Math.round(left)}px`;
    state.popup.style.top = `${Math.round(top)}px`;
    state.popup.style.maxHeight = `${Math.round(maxH)}px`;

    // Clamp horizontally after layout in case of narrow viewports.
    const finalRect = state.popup.getBoundingClientRect();
    if (finalRect.right > window.innerWidth - 10) {
      state.popup.style.left = `${Math.round(window.innerWidth - 10 - finalRect.width)}px`;
    }
    if (finalRect.left < 10) {
      state.popup.style.left = "10px";
    }
  };


state.resetForNewReel = function () {
    state.setExpanded(false);
    state.exitInquiryMode();
    state.setVaultOpen(false);
    state.showStatementsView();
    state.evidenceTitle.textContent = "";
    state.evidenceStats.innerHTML = "";
    state.evidenceList.innerHTML = "";
    state.summaryStatement.textContent = "";
    state.summaryAlgoPanel.innerHTML = "";
    state.setSummaryAlgoVisible(false);
    state.summaryEvidence.innerHTML = "";
    state.summaryText.textContent = "";
    state.setSummaryLoading(false);
    state.listContent.innerHTML = "";
    state.statements = [];
    state.summaryCache = new Map();
    state.summaryInFlight = new Set();
    state.activeStatement = null;
    state.activeStatementIndex = null;
    state.activeEvidence = null;
    state.activeEvidenceIndex = null;
    state.activeSummaryKey = null;
    state.activeStatementKey = null;
    state.externalStatementsByKey = new Map();
    state.refreshStatementPinButtons();
    state.setLoading();
    state.setReadyPulse(false);
    state.updateBreadcrumb();
  };

  state.setStatements = function (statements) {
    state.listContent.innerHTML = "";
    state.statements = Array.isArray(statements) ? statements : [];
    state.setStatementCount(state.statements.length);
    state.setReadyPulse(state.statements.length > 0 && !state.isInquiryMode);
    if (state.statements.length > 0 && !state.isInquiryMode) state.bumpAttention();

    for (const statement of state.statements) {
      if (statement && Array.isArray(statement.evidence)) {
        statement.evidence = sortEvidenceByRelevanceDesc(statement.evidence);
      }
    }

    for (let i = 0; i < state.statements.length; i++) {
      const statement = state.statements[i];

      const listItem = document.createElement("li");
      listItem.className = "statement-item";
      listItem.dataset.statementIndex = String(i);

      const evidenceTotal =
        statement && Array.isArray(statement.evidence)
          ? statement.evidence.length
          : 0;
      const sourceTone = sourceToneClass(evidenceTotal);
      listItem.classList.add(sourceTone);

      const sourceBlock = document.createElement("div");
      sourceBlock.className = `statement-stats ${sourceTone}`;
      sourceBlock.title = `Source coverage: ${evidenceTotal} source${evidenceTotal === 1 ? "" : "s"}`;

      const sourceHeader = document.createElement("div");
      sourceHeader.className = "statement-stats-head";
      const sourceIcon = document.createElement("span");
      sourceIcon.className = "statement-source-icon";
      sourceIcon.textContent = evidenceTotal > 0 ? String(evidenceTotal) : "!";
      const sourceLabel = document.createElement("span");
      sourceLabel.className = "statement-source-label";
      sourceLabel.textContent = evidenceTotal === 0 ? "No sources" : "Sources";
      sourceHeader.append(sourceIcon, sourceLabel);
      sourceBlock.appendChild(sourceHeader);

      const gaps = computeStatementGaps(statement);
      statement._rpGaps = gaps;

      const contentWrap = document.createElement("div");
      contentWrap.className = "statement-content";

      const statementText = document.createElement("div");
      statementText.className = "statement-text";
      statementText.textContent = statement.text || "";

      contentWrap.appendChild(statementText);

      if (gaps.length) {
        const tags = document.createElement("div");
        tags.className = "statement-tags";
        for (const gap of gaps) {
          const pill = document.createElement("span");
          pill.className = "gap-pill";
          const normalized = String(gap).toLowerCase();
          if (normalized.includes("no")) pill.classList.add("gap-nosources");
          else if (normalized.includes("disagree")) pill.classList.add("gap-disagree");
          else if (normalized.includes("weak")) pill.classList.add("gap-weak");
          pill.textContent = gap;
          tags.appendChild(pill);
        }
        contentWrap.appendChild(tags);
      }

      const actionButton = document.createElement("button");
      actionButton.type = "button";
      actionButton.className = `statement-action ${sourceTone}`;
      actionButton.dataset.statementIndex = String(i);

      if (evidenceTotal > 0) {
        actionButton.title = "Open sources";
        actionButton.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 18l6-6-6-6"/>
        </svg>`;
      } else {
        actionButton.title = "No sources found";
        actionButton.classList.add("disabled", "no-sources");
        actionButton.disabled = true;
        actionButton.innerHTML = `<span class="no-sources-mark" aria-hidden="true">!</span>`;
      }

      listItem.append(sourceBlock, contentWrap, actionButton);
      state.listContent.appendChild(listItem);
    }

    state.updateBreadcrumb();
  };

  // --- Interaction: quick tap vs press-and-hold ---
  let holdTimer = null;
  let holdTriggered = false;
  const holdDelayMs = 320;

  // High-friction exit from Inquiry Mode (press-and-hold)
  let exitHoldTimer = null;
  const exitHoldDelayMs = 650;

  const clearExitHold = () => {
    if (exitHoldTimer) window.clearTimeout(exitHoldTimer);
    exitHoldTimer = null;
    state.backdrop.classList.remove("exit-holding");
  };

  const triggerExit = () => {
    clearExitHold();
    if (!state.isInquiryMode) return;
    state.exitInquiryMode();
    state.setExpanded(false);
    state.showStatementsView();
  };

  const beginExitHold = () => {
    if (!state.isInquiryMode) return;
    clearExitHold();
    state.backdrop.classList.add("exit-holding");
    exitHoldTimer = window.setTimeout(() => {
      triggerExit();
    }, exitHoldDelayMs);
  };

  
const attachReelExitHold = () => {
    if (!state.activeVideo) return;
    detachReelExitHold();
    const video = state.activeVideo;
    const target = state.reelFocusEl || video;

    const onDown = (e) => {
      if (!state.isInquiryMode) return;
      // Make returning to passive viewing deliberate, and suppress platform play/pause toggles.
      if (e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.stopImmediatePropagation) e.stopImmediatePropagation();
      }
      beginExitHold();
    };
    const onUp = () => clearExitHold();

    const onClick = (e) => {
      if (!state.isInquiryMode) return;
      // Block Instagram's click-to-pause interaction while in Inquiry Mode.
      if (e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.stopImmediatePropagation) e.stopImmediatePropagation();
      }
    };

    target.addEventListener("pointerdown", onDown, true);
    target.addEventListener("pointerup", onUp, true);
    target.addEventListener("pointercancel", onUp, true);
    target.addEventListener("pointerleave", onUp, true);
    target.addEventListener("click", onClick, true);

    state._reelExitHold = { target, onDown, onUp, onClick };
  };


  const detachReelExitHold = () => {
    const h = state._reelExitHold;
    if (!h) return;
    try {
      h.target.removeEventListener("pointerdown", h.onDown, true);
      h.target.removeEventListener("pointerup", h.onUp, true);
      h.target.removeEventListener("pointercancel", h.onUp, true);
      h.target.removeEventListener("pointerleave", h.onUp, true);
      h.target.removeEventListener("click", h.onClick, true);
    } catch (e) {}
    state._reelExitHold = null;
  };

  const clearHold = () => {
    if (holdTimer) window.clearTimeout(holdTimer);
    holdTimer = null;
  };

  // High-friction exit: press-and-hold on blurred background (any blur segment).
  const attachBackdropExitHold = () => {
    detachBackdropExitHold();

    const onSegmentDown = (e) => {
      if (!state.isInquiryMode) return;
      // Avoid capturing presses that start on the panel itself.
      if (e && e.target && e.target.closest && e.target.closest(".reel-alert")) return;
      if (e && e.target && e.target.closest && e.target.closest(".rp-inquiry-topbar")) return;
      if (e) {
        e.preventDefault();
        e.stopPropagation();
      }
      beginExitHold();
    };
  
    // Fallback: treat any press-and-hold outside the Reel and outside the ReelPeel panel as a "background" exit.
    // This makes exit robust even if platform layers intercept pointer events on the blur segments.
    const onDocDown = (e) => {
      if (!state.isInquiryMode) return;
      const t = e && e.target;
      if (t && t.closest && t.closest(".reel-alert")) return;
      if (t && t.closest && t.closest(".rp-inquiry-topbar")) return;
      if (state.reelFocusEl && t && state.reelFocusEl.contains(t)) return;

      if (e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.stopImmediatePropagation) e.stopImmediatePropagation();
      }
      beginExitHold();
    };
    const onDocUp = () => clearExitHold();

    state.backdropSegments.forEach((seg) => {
      seg.addEventListener("pointerdown", onSegmentDown, true);
      seg.addEventListener("pointerup", onDocUp, true);
      seg.addEventListener("pointercancel", onDocUp, true);
      seg.addEventListener("pointerleave", onDocUp, true);
    });
    document.addEventListener("pointerdown", onDocDown, true);
    document.addEventListener("pointerup", onDocUp, true);
    document.addEventListener("pointercancel", onDocUp, true);

    state._backdropExitHold = { onSegmentDown, onDocDown, onDocUp };
  };

  const detachBackdropExitHold = () => {
    const h = state._backdropExitHold;
    if (!h) return;
    state.backdropSegments.forEach((seg) => {
      seg.removeEventListener("pointerdown", h.onSegmentDown, true);
      seg.removeEventListener("pointerup", h.onDocUp, true);
      seg.removeEventListener("pointercancel", h.onDocUp, true);
      seg.removeEventListener("pointerleave", h.onDocUp, true);
    });
    document.removeEventListener("pointerdown", h.onDocDown, true);
    document.removeEventListener("pointerup", h.onDocUp, true);
    document.removeEventListener("pointercancel", h.onDocUp, true);
    state._backdropExitHold = null;
  };
  state.detachBackdropExitHold = detachBackdropExitHold;

  state.inspectButton.addEventListener("pointerdown", (event) => {
    if (state.isInquiryMode) {
      // No scrub interaction in Inquiry Mode: use direct clicks in the panel.
      return;
    }

    holdTriggered = false;
    clearHold();
    state.inspectButton.classList.add("holding");
    holdTimer = window.setTimeout(() => {
      holdTriggered = true;
      state.inspectButton.classList.remove("holding");
      state.enterInquiryMode();
    }, holdDelayMs);
  });

  state.inspectButton.addEventListener("pointerup", () => {
    clearHold();
    state.inspectButton.classList.remove("holding");
    if (holdTriggered) return;

    if (state.isInquiryMode) {
      if (state.viewMode === "summary") {
        state.showEvidenceView(state.activeStatement, state.activeStatementIndex, {
          statementKey: state.activeStatementKey,
        });
        return;
      }
      if (state.viewMode === "evidence") {
        state.showStatementsView();
        return;
      }
      state.setExpanded(true);
      return;
    }

    // If not in inquiry mode, a single click does nothing (forces the user to press and hold).
  });

  state.inspectButton.addEventListener("pointerleave", clearHold);
  state.inspectButton.addEventListener("pointercancel", clearHold);


  // (Receipt export removed in this build.)


  // Statement -> sources
  state.listContent.addEventListener("click", (event) => {
    const actionBtn = event.target.closest(".statement-action");
    if (actionBtn && actionBtn.disabled) return;

    const row = event.target.closest(".statement-item");
    if (!row) return;
    const index = Number(row.dataset.statementIndex);
    const statement = state.statements[index];
    if (!statement) return;
    const evidenceTotal =
      statement && Array.isArray(statement.evidence) ? statement.evidence.length : 0;
    if (evidenceTotal <= 0) return;
    state.showEvidenceView(statement, index);
  });

  if (state.evidencePinButton) {
    state.evidencePinButton.addEventListener("click", () => {
      if (!state.activeStatement) return;
      state.toggleBookmark(state.activeStatement, state.activeStatementIndex);
      state.refreshStatementPinButtons();
    });
  }

  if (state.summaryAlgoToggle) {
    state.summaryAlgoToggle.addEventListener("click", () => {
      state.setSummaryAlgoVisible(!state.summaryAlgoOpen);
    });
  }

  if (state.vaultButton) {
    state.vaultButton.addEventListener("click", () => {
      state.setVaultOpen(!state.isVaultOpen);
      if (state.isVaultOpen) {
        state.setExpanded(true);
      }
      state.updateBreadcrumb();
    });
  }

  if (state.vaultList) {
    state.vaultList.addEventListener("click", (event) => {
      const removeBtn = event.target.closest(".vault-remove-item");
      if (removeBtn) {
        const key = String(removeBtn.dataset.bookmarkKey || "");
        const idx = state.bookmarks.findIndex((bookmark) => bookmark.key === key);
        if (idx >= 0) {
          state.bookmarks.splice(idx, 1);
          state.persistBookmarks();
          state.renderVaultList();
          state.refreshStatementPinButtons();
          state.showToast("Claim removed from Vault");
        }
        return;
      }

      const openBtn = event.target.closest(".vault-open-item");
      if (!openBtn) return;
      const key = String(openBtn.dataset.bookmarkKey || "");
      const bookmark = state.bookmarks.find((entry) => entry && entry.key === key);
      if (!bookmark || !bookmark.statementSnapshot) {
        state.showToast("Saved claim data missing");
        return;
      }
      const statement = bookmark.statementSnapshot;
      state.externalStatementsByKey.set(key, statement);
      state.setVaultOpen(false);
      state.setExpanded(true);
      state.showEvidenceView(statement, null, { statementKey: key });
      state.showToast("Loaded claim from Vault");
    });
  }

  // Evidence -> context
  state.evidenceList.addEventListener("click", (event) => {
    const summaryButton = event.target.closest(".summary-button");
    if (!summaryButton) return;
    const statementIndex = Number(summaryButton.dataset.statementIndex);
    const evidenceIndex = Number(summaryButton.dataset.evidenceIndex);
    const statementKey = summaryButton.dataset.statementKey
      ? String(summaryButton.dataset.statementKey)
      : null;
    const statement =
      statementKey && state.externalStatementsByKey.has(statementKey)
        ? state.externalStatementsByKey.get(statementKey)
        : state.statements[statementIndex];
    const evidence =
      statement && Array.isArray(statement.evidence)
        ? statement.evidence[evidenceIndex]
        : null;
    if (!statement || !evidence) return;
    state.showSummaryView(statement, evidence, statementIndex, evidenceIndex, {
      statementKey,
    });
  });

  // Escape: exit Inquiry Mode / collapse
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    // Keep leaving Inquiry Mode high-friction (press-and-hold).
    if (state.isInquiryMode) return;
    state.setExpanded(false);
    state.showStatementsView();
  });
window.addEventListener("resize", () => {
    if (!state.isInquiryMode) return;
    state.positionSpotlight();
    state.positionBackdropSegments();
    state.positionInquiryPanel();
    state.positionInquiryTopbar();
  });

  window.addEventListener(
    "scroll",
    () => {
      if (!state.isInquiryMode) return;
      state.positionSpotlight();
      state.positionInquiryPanel();
      state.positionInquiryTopbar();
    },
    true
  );

  // Resume button: exit Inquiry Mode and return to passive overlay.
popup.querySelector("#close-alert").addEventListener("click", () => {
    removePopup();
  });

  state.updateBreadcrumb();
  state.loadBookmarks();
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
  const currentReelKey = getCurrentReelKey();
  const isNewReel = markNewReelSeen(currentReelKey);

  if (isNewReel) {
    if (scrollPauseTimer) {
      clearTimeout(scrollPauseTimer);
      scrollPauseTimer = null;
    }
    queueAnalysisForUrl(currentUrl, currentReelKey);
    return;
  }

  if (scrollPauseTimer) {
    clearTimeout(scrollPauseTimer);
  }
  scrollPauseTimer = setTimeout(() => {
    const url = window.location.href;
    const reelKey = getCurrentReelKey();
    if (!url) return;
    markNewReelSeen(reelKey);
    if (
      reelKey === lastAnalyzedReelKey &&
      reelKey === displayedReelKey &&
      !pendingAnalysisUrl
    ) {
      return;
    }
    queueAnalysisForUrl(url, reelKey);
  }, scrollPauseDelayMs);
}

async function runAnalysis(reelUrl, reelKey = getCurrentReelKey(), forceRunId = null) {
  if (
    !reelUrl ||
    (!forceRunId && reelKey === lastAnalyzedReelKey && reelKey === displayedReelKey)
  ) {
    maybeStartAnalysis();
    return;
  }
  if (analysisInFlight) return;
  analysisInFlight = true;
  inFlightReelKey = reelKey;
  const state = ensurePopup();
  state.setLoading();
  state.showStatementsView();
  state.initBackendToggle(reelUrl, reelKey);
  const requestBackendPref = getEffectiveBackendPref();

  const video = getActiveVideoElement();
  const skipCheck = shouldSkipPipeline(video);
  if (skipCheck.skip) {
    console.log(`Skipping pipeline for ${reelUrl}: ${skipCheck.reason}`);
    state.setNotApplicable();
    lastAnalyzedUrl = reelUrl;
    lastAnalyzedReelKey = reelKey;
    displayedReelKey = reelKey;
    analysisInFlight = false;
    inFlightReelKey = null;
    maybeStartAnalysis();
    return;
  }

  // If local backend is selected, verify it is reachable first.
  let requestApiBase = null;
  if (requestBackendPref === "local") {
    try {
      requestApiBase = await resolveLocalApiBase();
    } catch (_healthErr) {
      state.setNotApplicable("no-local-data");
      state.showToast("Lokaler Server nicht erreichbar");
      lastAnalyzedUrl = reelUrl;
      lastAnalyzedReelKey = reelKey;
      displayedReelKey = reelKey;
      analysisInFlight = false;
      inFlightReelKey = null;
      maybeStartAnalysis();
      return;
    }
  }

  try {
    const data = await chrome.runtime.sendMessage({
      type: "getReelAnalysis",
      url: reelUrl,
      run_id: forceRunId,
      backendPref: requestBackendPref,
      apiBase: requestApiBase,
    });

    if (getCurrentReelKey() !== reelKey) return;

    const statements = data && data.statements ? data.statements : [];

    // If the local backend returned an error (e.g. no mock data for this reel),
    // show a clear message instead of an empty statement list.
    if (data && data.error && requestBackendPref === "local") {
      state.setNotApplicable("no-local-data");
      state.showToast("Keine lokalen Daten fuer dieses Reel vorhanden");
      lastAnalyzedUrl = reelUrl;
      lastAnalyzedReelKey = reelKey;
      displayedReelKey = reelKey;
      return;
    }

    state.setStatements(statements);
    lastAnalyzedUrl = reelUrl;
    lastAnalyzedReelKey = reelKey;
    displayedReelKey = reelKey;
  } catch (error) {
    console.error("Error in content script" + error);
  } finally {
    analysisInFlight = false;
    inFlightReelKey = null;
    maybeStartAnalysis();
  }
}

function showPopup() {
  ensurePopup();
  const currentUrl = window.location.href;
  const currentReelKey = getCurrentReelKey();
  const isNewReel = markNewReelSeen(currentReelKey);
  
  if (isNewReel || !lastAnalyzedReelKey) {
    queueAnalysisForUrl(currentUrl, currentReelKey);
    return;
  }
  scheduleAnalysisAfterPause();
}

function removePopup() {
  if (popupState && popupState.isInquiryMode) {
    popupState.exitInquiryMode();
  } else if (popupState && popupState.detachInquiryPauseGuard) {
    popupState.detachInquiryPauseGuard();
  }
  if (popupState && popupState.detachBackdropExitHold) {
    popupState.detachBackdropExitHold();
  }
  if (popupState && popupState._handoffTimer) {
    clearTimeout(popupState._handoffTimer);
  }
  if (currentPopup) {
    currentPopup.remove();
  }
  if (popupState && popupState.backdrop) {
    popupState.backdrop.remove();
  }
  if (popupState && popupState.spotlight) {
    popupState.spotlight.remove();
  }
  if (popupState && popupState.contract) {
    popupState.contract.remove();
  }
  if (popupState && popupState.toast) {
    popupState.toast.remove();
  }
  if (popupState && popupState.handoff) {
    popupState.handoff.remove();
  }
  if (popupState && popupState.inquiryTopbar) {
    popupState.inquiryTopbar.remove();
  }
  if (popupState && popupState.offlineSettingsBox) {
    popupState.offlineSettingsBox.remove();
  }
  const lingeringReelClasses = document.querySelectorAll(
    ".rp-reel-focus, .rp-reel-soft-blur"
  );
  lingeringReelClasses.forEach((el) => {
    el.classList.remove("rp-reel-focus", "rp-reel-soft-blur");
  });
  document.documentElement.classList.remove("rp-inquiry-open");
  document.body.classList.remove("rp-inquiry-open");
  setInquiryScrollLock(false);
  forceRestorePageInteraction();
  currentPopup = null;
  popupState = null;
  lastAnalyzedUrl = null;
  lastAnalyzedReelKey = null;
  displayedReelKey = null;
  lastSeenReelKey = null;
  pendingAnalysisUrl = null;
  pendingAnalysisReelKey = null;
  analysisInFlight = false;
  if (scrollPauseTimer) {
    clearTimeout(scrollPauseTimer);
    scrollPauseTimer = null;
  }
}
