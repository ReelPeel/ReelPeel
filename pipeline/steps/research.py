import random
import re
import time
import xml.etree.ElementTree as ET
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
            # Ensure list exists (in case of older states)
            if not hasattr(stmt, "queries") or stmt.queries is None:
                stmt.queries = []

            prompt = self.config.get("prompt_template").format(claim=stmt.text)

            try:
                resp = self.llm.call(
                    model=self.config.get("model"),
                    temperature=self.config.get("temperature", 0.2),
                    #stop=self.config.get("stop", ["\n"]),  # fine if 1 query/line
                    prompt=prompt,
                )
                raw = resp.replace("\n", " ")  # collapse to one line
                q = self.clean_pubmed_query(raw)

                # Append if not duplicate
                if q and q.lower() not in {x.lower() for x in stmt.queries}:
                    stmt.queries.append(q)

                print(f"   Statement {stmt.id}: + query: {q}")

            except Exception as e:
                print(f"   [Error] ID {stmt.id}: {e}")
                fb = self._fallback_query(stmt.text)
                if fb.lower() not in {x.lower() for x in stmt.queries}:
                    stmt.queries.append(fb)

        return state
    _ALLOWED_TAGS = {"mh", "tiab"}
    def clean_pubmed_query(self, raw: str) -> str:
        q = raw.strip()

        # If model wrapped the whole thing in quotes/backticks, remove
        q = q.strip("`").strip()
        q = q.strip('"').strip("'").strip()

        # Normalize common MeSH tag variants -> [mh]
        q = re.sub(r"\[(?:MeSH|mesh|MeSH Terms|MeSH terms|MH|mh)\]", "[mh]", q)

        # Remove quotes directly around tagged terms: "foo bar"[tiab] -> foo bar[tiab]
        q = re.sub(r'"([^"]+)"\[(mh|tiab)\]', r"\1[\2]", q)

        # Collapse whitespace to single spaces
        q = re.sub(r"\s+", " ", q).strip()

        # Uppercase boolean ops (cheap, pragmatic)
        q = re.sub(r"\band\b", "AND", q, flags=re.I)
        q = re.sub(r"\bor\b", "OR", q, flags=re.I)

        # Validate tags used
        tags = set(re.findall(r"\[([^\]]+)\]", q))
        if not tags.issubset(self._ALLOWED_TAGS):
            raise ValueError(f"Invalid tags in query: {tags - self._ALLOWED_TAGS}")

        # Basic paren balance check
        if q.count("(") != q.count(")"):
            raise ValueError("Unbalanced parentheses in query")

        return q

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
        sort = self.config.get("sort", "relevance")
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        for stmt in state.statements:
            queries = getattr(stmt, "queries", []) or []
            if not queries:
                continue

            for q in queries:
                try:
                    params = {
                        "db": "pubmed",
                        "term": q,
                        "retmax": retmax,
                        "retmode": "json",
                        "sort": sort,
                    }
                    resp = requests.get(base_url, params=params, timeout=10)
                    resp.raise_for_status()

                    id_list = resp.json().get("esearchresult", {}).get("idlist", [])

                    for pmid in id_list:
                        # Try to find existing evidence (same PMID)
                        existing = next((e for e in stmt.evidence if e.pubmed_id == pmid), None)

                        if existing is not None:
                            # Ensure queries list exists
                            if not hasattr(existing, "queries") or existing.queries is None:
                                existing.queries = []
                            # Add provenance if not already present
                            if q.lower() not in {x.lower() for x in existing.queries}:
                                existing.queries.append(q)
                            continue

                        # Create new evidence
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        stmt.evidence.append(Evidence(pubmed_id=pmid, url=url))
                  
                    print(f"   Statement {stmt.id}: Found {len(id_list)} links.")
                        ev = Evidence(pubmed_id=pmid, url=url)

                        # Attach provenance (works whether or not Evidence formally defines queries)
                        if not hasattr(ev, "queries") or ev.queries is None:
                            ev.queries = []
                        ev.queries.append(q)

                        stmt.evidence.append(ev)

                    time.sleep(random.uniform(0.5, 0.5))
                    print(f"   Statement {stmt.id}: Query '{q}' -> {len(id_list)} PMIDs.")

                except Exception as e:
                    print(f"   [Error] Failed to fetch links for '{q}': {e}")

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
                    # random sleep to avoid rate limits
                    time.sleep(random.uniform(0.5,0.5))
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
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        r = requests.get(f"{self.NCBI_EUTILS}/efetch.fcgi", params=params, timeout=20)
        r.raise_for_status()

        root = ET.fromstring(r.text)

        def collect(xpath: str) -> List[str]:
            parts = []
            for el in root.findall(xpath):
                txt = "".join(el.itertext()).strip()
                if not txt:
                    continue
                label = el.attrib.get("Label") or el.attrib.get("NlmCategory")
                parts.append(f"{label}: {txt}" if label else txt)
            return parts

        parts = collect(".//Abstract/AbstractText")
        if not parts:
            parts = collect(".//OtherAbstract/AbstractText")

        abstract = " ".join(parts).strip()
        if abstract.lower().startswith("abstract available from the publisher"):
            return None

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