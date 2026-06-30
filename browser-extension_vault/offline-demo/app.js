(() => {
  const reelsFeed = document.querySelector("#reels");
  const reels = [...document.querySelectorAll(".reel")];
  const reelById = new Map(reels.map((reel) => [reel.dataset.reelId, reel]));
  let activeId = "";

  function restoreFeedScroll() {
    if (!reelsFeed || document.body.classList.contains("rp-inquiry-open")) return;
    reelsFeed.style.overflowY = "auto";
    reelsFeed.style.pointerEvents = "";
  }

  function pathFor(reelId) {
    return `/reels/${reelId}/`;
  }

  function setActive(reel) {
    const reelId = reel.dataset.reelId;
    if (!reelId || reelId === activeId) return;

    activeId = reelId;
    history.replaceState({ reelId }, "", pathFor(reelId));

    for (const item of reels) {
      const video = item.querySelector("video");
      if (item === reel) {
        video.play().catch(() => {});
      } else {
        video.pause();
        video.currentTime = 0;
      }
    }
  }

  const observer = new IntersectionObserver(
    (entries) => {
      const mostVisible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (mostVisible && mostVisible.intersectionRatio >= 0.6) {
        setActive(mostVisible.target);
      }
    },
    { threshold: [0.6, 0.8, 1] },
  );

  reels.forEach((reel) => {
    observer.observe(reel);
    const video = reel.querySelector("video");
    const soundButton = reel.querySelector(".sound-toggle");

    video.muted = true;
    soundButton.addEventListener("click", (event) => {
      event.stopPropagation();
      video.muted = !video.muted;
      soundButton.setAttribute("aria-pressed", String(!video.muted));
      soundButton.setAttribute("aria-label", video.muted ? "Unmute video" : "Mute video");
    });
    video.addEventListener("click", () => {
      if (video.paused) video.play().catch(() => {});
      else video.pause();
    });

    reel.querySelector(".like-button").addEventListener("click", (event) => {
      const button = event.currentTarget;
      button.classList.toggle("is-active");
      button.setAttribute("aria-pressed", String(button.classList.contains("is-active")));
    });
  });

  function reelIdFromPath() {
    const match = location.pathname.match(/^\/reels\/([^/]+)\/?$/);
    return match ? match[1] : "";
  }

  function goTo(reelId, smooth = false) {
    const reel = reelById.get(reelId);
    if (!reel) return;
    reel.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
    setActive(reel);
  }

  window.addEventListener("popstate", () => goTo(reelIdFromPath(), false));
  window.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "PageDown", "PageUp"].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = Math.max(0, reels.findIndex((reel) => reel.dataset.reelId === activeId));
    const direction = ["ArrowDown", "PageDown"].includes(event.key) ? 1 : -1;
    const next = reels[Math.min(reels.length - 1, Math.max(0, currentIndex + direction))];
    next.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  new MutationObserver(restoreFeedScroll).observe(document.body, {
    attributes: true,
    attributeFilter: ["class", "style"],
  });
  window.addEventListener("reelpeel:inquiry-exit", restoreFeedScroll);
  window.addEventListener("focus", restoreFeedScroll);

  goTo(reelIdFromPath() || reels[0].dataset.reelId, false);
  restoreFeedScroll();
})();
