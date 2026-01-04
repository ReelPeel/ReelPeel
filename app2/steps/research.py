import re
from datetime import datetime
from typing import List, Tuple, Any, Optional

import requests

from ..core.base import PipelineStep
from ..core.models import PipelineState, Evidence


# -------------------------------------------------------------------------
# STEP 3: Statement to Query
# -------------------------------------------------------------------------
class StatementToQueryStep(PipelineStep):
    def execute(self, state: PipelineState) -> PipelineState:
        print(f"[{self.__class__.__name__}] Generating PubMed queries...")

        for stmt in state.statements:
            if stmt.query:
                continue  # Skip if already exists

            prompt = self.config.get('prompt_template').format(claim=stmt.text)

            try:
                resp = self.llm.call(
                    model=self.config.get('model'),
                    temperature=self.config.get('temperature', 0.2),
                    max_tokens=self.config.get('max_tokens', 64),
                    prompt=prompt,
                    stop=["\n"],
                )
                stmt.query = self._clean_query(resp, stmt.text)
                print(f"   Query for ID {stmt.id}: {stmt.query}")
            except Exception as e:
                print(f"   [Error] ID {stmt.id}: {e}")
                stmt.query = self._fallback_query(stmt.text)

        return state

    def _clean_query(self, raw: str, fallback: str) -> str:
        # Regex to find the first line starting with a letter
        m = re.search(r"^(?!\s*$).+", raw, re.MULTILINE)
        if m:
            return m.group(0).strip("`'\" ").replace("\n", " ").strip()
        return self._fallback_query(fallback)

    def _fallback_query(self, text: str) -> str:
        # Simple keyword extraction fallback
        words = re.findall(r"[A-Za-z']+", text)
        return " ".join(list(set(words))[:6])


# -------------------------------------------------------------------------
# STEP 4: Query to Link
# -------------------------------------------------------------------------
class QueryToLinkStep(PipelineStep):
    def execute(self, state: PipelineState) -> PipelineState:
        print(f"[{self.__class__.__name__}] Fetching PubMed links...")

        retmax = self.config.get("retmax", 3)
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        for stmt in state.statements:
            if not stmt.query:
                continue

            try:
                params = {
                    "db": "pubmed",
                    "term": stmt.query,
                    "retmax": retmax,
                    "retmode": "json"
                }
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()

                id_list = resp.json().get("esearchresult", {}).get("idlist", [])

                # Append new evidence items
                for pmid in id_list:
                    # Check duplicates
                    if any(e.pubmed_id == pmid for e in stmt.evidence):
                        continue

                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    stmt.evidence.append(Evidence(pubmed_id=pmid, url=url))

                print(f"   Statement {stmt.id}: Found {len(id_list)} links.")

            except Exception as e:
                print(f"   [Error] Failed to fetch links for '{stmt.query}': {e}")

        return state


# -------------------------------------------------------------------------
# STEP 5: Link to Summary (Refactored: Fetch Abstract & Types)
# -------------------------------------------------------------------------
class LinkToSummaryStep(PipelineStep):
    """
    Step 5: Fetches PubMed abstract and publication types for evidence URLs.
    Does NOT perform LLM summarization (based on new user requirements).
    """
    NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?")

    def execute(self, state: PipelineState) -> PipelineState:
        print(f"[{self.__class__.__name__}] Fetching PubMed metadata (Abstracts & Types)...")

        for stmt in state.statements:
            for ev in stmt.evidence:
                url = ev.url
                if not url:
                    continue

                try:
                    # 1. Extract PMID
                    pmid = self._extract_pmid(url)
                    ev.pubmed_id = pmid

                    # 2. Fetch Publication Types
                    pub_types = self._pubmed_fetch_types(pmid)
                    ev.pub_type = pub_types

                    # 3. Fetch Abstract
                    abstract = self._pubmed_fetch_abstract(pmid)
                    ev.abstract = abstract

                    # Optional: Map abstract to summary if summary is empty
                    if not ev.summary and abstract:
                        ev.summary = abstract[:500] + "..."  # Fallback for display

                    print(
                        f"   Processed {url} -> PMID: {pmid}, Types: {len(pub_types)}, Abstract Len: {len(abstract) if abstract else 0}")

                except Exception as e:
                    print(f"   [Warning] Could not process {url}: {e}")
                    # Keep defaults if failed

        # Update timestamp
        state.generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return state

    def _extract_pmid(self, url: str) -> str:
        """Extract numeric PMID from a PubMed URL or accept raw PMIDs."""
        if url.isdigit():
            return url
        match = self.PUBMED_URL_RE.search(url)
        if match:
            return match.group(1)
        raise ValueError(f"Unable to extract PMID from URL: {url}")

    def _pubmed_fetch_types(self, pmid: str) -> List[str]:
        """Fetch PubMed 'Publication Types' using ESummary."""
        params = {"db": "pubmed", "id": pmid, "retmode": "json"}
        r = requests.get(f"{self.NCBI_EUTILS}/esummary.fcgi", params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()

        rec = (payload.get("result") or {}).get(pmid) or {}
        pubtypes = rec.get("pubtype") or []
        # Ensure list[str]
        return [str(x) for x in pubtypes if x is not None]

    def _pubmed_fetch_abstract(self, pmid: str) -> Optional[str]:
        """Fetch abstract text using EFetch."""
        params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
        r = requests.get(f"{self.NCBI_EUTILS}/efetch.fcgi", params=params, timeout=20)
        r.raise_for_status()

        lines = [line.strip() for line in r.text.splitlines() if line.strip()]
        # Filter out uppercase metadata lines loosely
        abstract = " ".join(line for line in lines if not line.isupper())
        return abstract or None


# -------------------------------------------------------------------------
# STEP 5.1: PubType to Weight (Your New Step)
# -------------------------------------------------------------------------
class PubTypeWeightStep(PipelineStep):
    # Regex rules as class constant or loaded from config
    PUBTYPE_RULES: List[Tuple[re.Pattern, float, str]] = [
        (re.compile(r"\bmeta[- ]analysis\b"), 0.95, "meta-analysis"),
        (re.compile(r"\bsystematic review\b"), 0.95, "systematic review"),
        (re.compile(r"\brandomized controlled trial\b|\brandomised controlled trial\b"), 0.90, "RCT"),
        (re.compile(r"\bclinical trial\b"), 0.85, "clinical trial"),
        (re.compile(r"\bguideline\b"), 0.86, "guideline"),
        (re.compile(r"\bcohort\b|\bprospective\b|\bretrospective\b"), 0.75, "observational"),
        (re.compile(r"\bcase[- ]control\b"), 0.70, "case-control"),
        (re.compile(r"\breview\b"), 0.62, "narrative review"),
        (re.compile(r"\bcase reports?\b"), 0.45, "case report"),
        (re.compile(r"\beditorial\b|\bletter\b"), 0.35, "opinion"),
    ]

    def execute(self, state: PipelineState) -> PipelineState:
        print(f"[{self.__class__.__name__}] Calculating evidence weights...")
        default_weight = self.config.get("default_weight", 0.5)

        for stmt in state.statements:
            for ev in stmt.evidence:
                # Use the helper to determine weight
                w = self._weight_from_pubtypes(ev.pub_type, default=default_weight)
                ev.weight = float(w)
                # print(f"   PMID {ev.pubmed_id} ({ev.pub_type}) -> Weight: {ev.weight}")

        return state

    def _weight_from_pubtypes(self, pubtypes: Any, default: float) -> float:
        """Helper to check regex matches against pub_type string/list."""
        if not pubtypes:
            return default

        # Normalize to list of strings
        if isinstance(pubtypes, str):
            pts = [pubtypes]
        elif isinstance(pubtypes, list):
            pts = [str(x) for x in pubtypes if x]
        else:
            return default

        best_w = 0.0

        for pt in pts:
            norm_pt = re.sub(r"\s+", " ", str(pt).strip().lower())
            for rx, w, _ in self.PUBTYPE_RULES:
                if rx.search(norm_pt) and w > best_w:
                    best_w = w

        return best_w if best_w > 0.0 else default