"""
Research module (steps 3-5.1): turn statements into PubMed evidence.

This file defines a sequence of PipelineStep implementations that expand a
claim into PubMed queries, fetch matching PMIDs, and enrich those results with
titles, abstracts, publication types, and weights:

- StatementToQueryStep (Step 3):
  Uses an LLM to generate PubMed queries per statement, normalizes tags, and
  appends unique queries to stmt.queries.
- QueryToLinkStep (Step 4):
  Calls the PubMed ESearch API (via a local proxy) to retrieve PMIDs and
  creates PubMedEvidence entries or updates existing ones with query provenance.
- LinkToAbstractStep (Step 5):
  Batch fetches publication types (esummary) and title/abstract content
  (efetch) for all PMIDs, then attaches this metadata to evidence objects.
- PubTypeWeightStep (Step 5.1):
  Assigns evidence weights by matching publication types against regex rules
  (systematic review, RCT, etc.).

Inputs:
- state.statements populated with Statement.text.
- Each Statement should have stmt.evidence initialized (defaults to empty list).

Outputs:
- stmt.queries: normalized PubMed queries per statement.
- stmt.evidence: PubMedEvidence objects with pubmed_id, url, title, abstract,
  pub_type, and weight fields.
- state.generated_at updated after metadata enrichment.

Operational notes:
- Uses a local proxy at http://127.0.0.1:8080/proxy by default (see service_manager).
- Batches esummary/efetch requests to reduce HTTP calls.
- All network failures are caught and logged; evidence may remain partial.
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple, Any, Dict

import requests

from ..core.base import PipelineStep
from ..core.models import PipelineState, PubMedEvidence, SourceType


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
                    # stop=self.config.get("stop", ["\n"]),  # fine if 1 query/line
                    prompt=prompt,
                )
                self.log_artifact(f"Raw Output for Statement {stmt.id} Query Generation", resp)
                raw = resp.replace("\n", " ")  # collapse to one line
                q = self.clean_pubmed_query(raw)

                # Append if not duplicate
                added = False
                if q and q.lower() not in {x.lower() for x in stmt.queries}:
                    stmt.queries.append(q)
                    added = True
                if q:
                    self.log_artifact(
                        "PubMed Query",
                        {
                            "statement_id": stmt.id,
                            "query": q,
                            "source": "llm",
                            "added": added,
                        },
                    )

                print(f"   Statement {stmt.id}: + query: {q}")

            except Exception as e:
                print(f"   [Error] ID {stmt.id}: {e}")
                fb = self._fallback_query(stmt.text)
                added = False
                if fb.lower() not in {x.lower() for x in stmt.queries}:
                    stmt.queries.append(fb)
                    added = True
                self.log_artifact(
                    "PubMed Query",
                    {
                        "statement_id": stmt.id,
                        "query": fb,
                        "source": "fallback",
                        "added": added,
                        "error": str(e),
                    },
                )

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

        retmax = self.config.get("retmax", 5)
        sort = self.config.get("sort", "relevance")

        # USE PROXY: Default to local service port 8080
        # (Local variable is fine here, no 'self' needed)
        proxy_base = self.config.get("proxy_url", "http://127.0.0.1:8080/proxy")
        search_url = f"{proxy_base}/esearch.fcgi"

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
                    request_url = requests.Request("GET", search_url, params=params).prepare().url
                    self.log_artifact(
                        "PubMed ESearch Request",
                        {"statement_id": stmt.id, "query": q, "url": request_url},
                    )
                    # Send request to Proxy
                    resp = requests.get(search_url, params=params, timeout=10)
                    resp.raise_for_status()

                    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
                    print(f"   Statement {stmt.id}: Query '{q}' -> {len(id_list)} PMIDs.")

                    for pmid in id_list:
                        # Try to find existing evidence (same PMID)
                        existing = next(
                            (e for e in stmt.evidence if getattr(e, "pubmed_id", None) == pmid),
                            None,
                        )

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
                        stmt.evidence.append(PubMedEvidence(pubmed_id=pmid, url=url))

                    print(f"   Statement {stmt.id}: Found {len(id_list)} links.")

                except Exception as e:
                    print(f"   [Error] Failed to fetch links for '{q}': {e}")

        return state



# -------------------------------------------------------------------------
# STEP 5: Link to Abstract (Refactored: Fetch Abstract & Types)
# -------------------------------------------------------------------------
class LinkToAbstractStep(PipelineStep):
    """
    Step 5: Fetches PubMed metadata (Title, Abstract, Types) in BATCHES.
    Optimized to reduce HTTP calls and extract Titles for the Reranker.
    """
    PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?")

    def __init__(self, step_config: Dict[str, Any]):
        super().__init__(step_config)
        self.proxy_base = self.config.get("proxy_url", "http://127.0.0.1:8080/proxy")

    def execute(self, state: PipelineState) -> PipelineState:
        print(f"[{self.__class__.__name__}] Fetching metadata (Batch Mode)...")

        for stmt in state.statements:
            # 1. Collect all valid PMIDs for this statement
            evidence_map = {}
            for ev in stmt.evidence:
                if getattr(ev, "source_type", None) != SourceType.PUBMED:
                    continue
                try:
                    pmid = self._extract_pmid(ev.url or ev.pubmed_id or "")
                    ev.pubmed_id = pmid
                    evidence_map[pmid] = ev
                except ValueError:
                    continue

            if not evidence_map:
                continue

            pmids = list(evidence_map.keys())

            # 2. Fetch all Publication Types in ONE request
            self._batch_fetch_types(pmids, evidence_map)

            # 3. Fetch all Titles & Abstracts in ONE request
            self._batch_fetch_details(pmids, evidence_map)

        # Update timestamp
        state.generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return state

    def _extract_pmid(self, url: str) -> str:
        if url and url.isdigit():
            return url
        match = self.PUBMED_URL_RE.search(url or "")
        if match:
            return match.group(1)
        raise ValueError(f"Invalid URL/PMID: {url}")

    def _batch_fetch_types(self, pmids: List[str], ev_map: Dict[str, Any]):
        """Fetched PubTypes for multiple PMIDs in one go (esummary)."""
        if not pmids: return

        ids_str = ",".join(pmids)
        params = {"db": "pubmed", "id": ids_str, "retmode": "json"}

        try:
            r = requests.get(f"{self.proxy_base}/esummary.fcgi", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            result = data.get("result", {})

            # 'result' contains keys for each PMID (strings), plus a 'uids' list
            for pmid in pmids:
                if pmid in result:
                    rec = result[pmid]
                    ptypes = rec.get("pubtype", [])
                    # Update the evidence object
                    ev_map[pmid].pub_type = [str(x) for x in ptypes if x]

            print(f"   Fetched PubTypes for {len(pmids)} items.")

        except Exception as e:
            print(f"   [Error] Batch esummary failed: {e}")

    def _batch_fetch_details(self, pmids: List[str], ev_map: Dict[str, Any]):
        """Fetches Title and Abstract for multiple PMIDs in one go (efetch)."""
        if not pmids: return

        ids_str = ",".join(pmids)
        params = {"db": "pubmed", "id": ids_str, "retmode": "xml"}

        try:
            r = requests.get(f"{self.proxy_base}/efetch.fcgi", params=params, timeout=30)
            r.raise_for_status()

            # Parse the big XML containing multiple PubmedArticle nodes
            root = ET.fromstring(r.text)

            # Iterate over each article found in the XML
            for article in root.findall(".//PubmedArticle"):
                # 1. Identify which PMID this is
                pmid_node = article.find(".//MedlineCitation/PMID")
                if pmid_node is None: continue
                pmid = pmid_node.text

                if pmid not in ev_map: continue
                ev = ev_map[pmid]

                # 2. Extract Title
                title_node = article.find(".//Article/ArticleTitle")
                if title_node is not None and title_node.text:
                    # Reranker/Stance usually look for 'title' or 'article_title'
                    # We can store it in a standard attribute (add 'title' to Evidence model if needed)
                    # For now, we attach it dynamically or assume Evidence has a title field
                    ev.title = title_node.text.strip()

                # 3. Extract Abstract (combining parts)
                abs_texts = article.findall(".//Abstract/AbstractText")
                if not abs_texts:
                    abs_texts = article.findall(".//OtherAbstract/AbstractText")

                parts = []
                for el in abs_texts:
                    txt = "".join(el.itertext()).strip()
                    if not txt: continue
                    label = el.attrib.get("Label") or el.attrib.get("NlmCategory")
                    parts.append(f"{label}: {txt}" if label else txt)

                full_abstract = " ".join(parts).strip()
                if full_abstract and not full_abstract.lower().startswith("abstract available from"):
                    ev.abstract = full_abstract

        except Exception as e:
            print(f"   [Error] Batch efetch failed: {e}")

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
        default_weight = self.config.get("default_weight", 0.4)

        for stmt in state.statements:
            for ev in stmt.evidence:
                if getattr(ev, "source_type", None) not in (
                    SourceType.PUBMED,
                    SourceType.EPISTEMONIKOS,
                ):
                    continue

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
