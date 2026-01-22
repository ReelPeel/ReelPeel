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

import concurrent.futures
import os
import re
import socket
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Tuple, Any, Dict, Optional

import requests

from ..core.base import PipelineStep
from ..core.models import PipelineState, PubMedEvidence, SourceType


# -------------------------------------------------------------------------
# STEP 3: Statement to Query
# -------------------------------------------------------------------------
class StatementToQueryStep(PipelineStep):
    def execute(self, state: PipelineState) -> PipelineState:
        print(f"[{self.__class__.__name__}] Generating PubMed queries...")

        statements = getattr(state, "statements", None) or []
        if not statements:
            return state

        prompt_defs = self._build_prompt_defs()
        if not prompt_defs:
            raise ValueError("generate_query requires prompt_template or prompt_templates in settings.")

        prefetch_cfg = self._prefetch_links_settings()

        # Ensure list exists (in case of older states)
        for stmt in statements:
            if not hasattr(stmt, "queries") or stmt.queries is None:
                stmt.queries = []
            if not hasattr(stmt, "queries_fetched") or stmt.queries_fetched is None:
                stmt.queries_fetched = []

        tasks = []
        for stmt in statements:
            for prompt_def in prompt_defs:
                tasks.append(
                    {
                        "stmt_id": stmt.id,
                        "stmt_text": stmt.text,
                        "prompt_def": prompt_def,
                        "prefetch": prefetch_cfg,
                    }
                )

        if not tasks:
            return state

        base_urls = self._resolve_base_urls()
        if not base_urls:
            base_urls = ["http://localhost:11434/v1"]

        max_workers = self._resolve_max_workers(len(tasks), base_urls)
        results_by_task = {}

        if max_workers <= 1:
            for task_id, task in enumerate(tasks):
                base_url = base_urls[task_id % len(base_urls)]
                results_by_task[task_id] = self._run_task(task, base_url)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for task_id, task in enumerate(tasks):
                    base_url = base_urls[task_id % len(base_urls)]
                    future = executor.submit(self._run_task, task, base_url)
                    future_map[future] = task_id
                for future in concurrent.futures.as_completed(future_map):
                    task_id = future_map[future]
                    results_by_task[task_id] = future.result()

        stmt_lookup = {s.id: s for s in statements}
        existing_queries = {
            s.id: {q.lower() for q in (s.queries or [])} for s in statements
        }
        fetched_queries = {
            s.id: {q.lower() for q in (s.queries_fetched or [])} for s in statements
        }

        total_tokens = 0
        for task_id, task in enumerate(tasks):
            result = results_by_task.get(task_id)
            if not result:
                continue
            stmt = stmt_lookup.get(result["stmt_id"])
            if not stmt:
                continue

            total_tokens += int(result.get("tokens") or 0)

            q = result.get("query") or ""
            prompt_name = result.get("prompt_name") or "prompt"

            if result.get("raw_output"):
                self.log_artifact(
                    f"Raw Output for Statement {stmt.id} Query Generation ({prompt_name})",
                    result["raw_output"],
                )

            added = False
            if q:
                lower = q.lower()
                if lower not in existing_queries[stmt.id]:
                    stmt.queries.append(q)
                    existing_queries[stmt.id].add(lower)
                    added = True

            artifact = {
                "statement_id": stmt.id,
                "query": q,
                "source": result.get("source"),
                "added": added,
                "prompt": prompt_name,
            }
            if result.get("error"):
                artifact["error"] = result["error"]
            if q:
                self.log_artifact("PubMed Query", artifact)

            if q:
                print(f"   Statement {stmt.id}: + query ({prompt_name}): {q}")

            prefetch = result.get("prefetch")
            if prefetch:
                if prefetch.get("error"):
                    self.log_artifact(
                        "PubMed Prefetch",
                        {
                            "statement_id": stmt.id,
                            "query": q,
                            "error": prefetch.get("error"),
                        },
                    )
                else:
                    pmids = prefetch.get("pmids") or []
                    if pmids:
                        self._attach_prefetched_evidence(stmt, q, pmids)
                        if q.lower() not in fetched_queries[stmt.id]:
                            fetched_queries[stmt.id].add(q.lower())
                            stmt.queries_fetched.append(q)
                    self.log_artifact(
                        "PubMed Prefetch",
                        {
                            "statement_id": stmt.id,
                            "query": q,
                            "count": len(pmids),
                            "request_url": prefetch.get("request_url"),
                        },
                    )

        if total_tokens:
            self.add_step_tokens(total_tokens)

        return state

    def _build_prompt_defs(self) -> List[Dict[str, Any]]:
        prompt_templates = self.config.get("prompt_templates")
        if prompt_templates:
            defs = []
            for idx, item in enumerate(prompt_templates, 1):
                if isinstance(item, str):
                    template = item
                    name = f"prompt_{idx}"
                    overrides = {}
                elif isinstance(item, dict):
                    template = item.get("template") or item.get("prompt_template")
                    name = item.get("name") or item.get("id") or f"prompt_{idx}"
                    overrides = item
                else:
                    continue
                if not template:
                    continue
                defs.append(self._prompt_def(template, name=name, overrides=overrides))
            return defs

        template = self.config.get("prompt_template")
        if not template:
            return []
        name = self.config.get("name") or "prompt"
        return [self._prompt_def(template, name=name)]

    def _prompt_def(
        self,
        template: str,
        *,
        name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        overrides = overrides or {}
        return {
            "name": name,
            "template": template,
            "model": overrides.get("model", self.config.get("model")),
            "temperature": overrides.get("temperature", self.config.get("temperature", 0.0)),
            "max_tokens": overrides.get("max_tokens", self.config.get("max_tokens")),
            "stop": overrides.get("stop", self.config.get("stop")),
        }

    def _run_task(self, task: Dict[str, Any], base_url: str) -> Dict[str, Any]:
        prompt_def = task["prompt_def"]
        prompt = prompt_def["template"].format(claim=task["stmt_text"])
        llm_settings = self._build_llm_settings(base_url)

        raw_output = None
        error = None
        source = "llm"
        query = ""
        tokens = 0
        prefetch_info = None

        try:
            from ..core.llm import LLMService

            llm = LLMService(llm_settings, observer=self.observer)
            resp = llm.call(
                model=prompt_def.get("model"),
                temperature=prompt_def.get("temperature", 0.0),
                max_tokens=prompt_def.get("max_tokens"),
                stop=prompt_def.get("stop"),
                prompt=prompt,
            )
            raw_output = resp
            raw = resp.replace("\n", " ")
            query = self.clean_pubmed_query(raw)
            tokens = int(llm.token_usage.get("total_tokens") or 0)

            prefetch_cfg = task.get("prefetch") or {}
            if prefetch_cfg.get("enabled") and query:
                prefetch_info = self._prefetch_links_for_query(query, prefetch_cfg)
        except Exception as exc:
            error = str(exc)
            source = "fallback"
            query = self._fallback_query(task["stmt_text"])
            prefetch_cfg = task.get("prefetch") or {}
            if prefetch_cfg.get("enabled") and query:
                prefetch_info = self._prefetch_links_for_query(query, prefetch_cfg)

        return {
            "stmt_id": task["stmt_id"],
            "prompt_name": prompt_def.get("name") or "prompt",
            "query": query,
            "raw_output": raw_output,
            "error": error,
            "source": source,
            "tokens": tokens,
            "prefetch": prefetch_info,
        }

    def _resolve_max_workers(self, task_count: int, base_urls: List[str]) -> int:
        parallel = self.config.get("parallel")
        enabled = False
        max_workers = None

        if isinstance(parallel, dict):
            enabled = parallel.get("enabled", parallel.get("max_workers") is not None)
            max_workers = parallel.get("max_workers")
        elif isinstance(parallel, bool):
            enabled = parallel
        elif isinstance(parallel, int):
            enabled = parallel > 1
            max_workers = parallel

        if not enabled:
            return 1

        if max_workers is None:
            max_workers = len(base_urls)
        try:
            max_workers = int(max_workers)
        except Exception:
            max_workers = 1

        if max_workers < 1:
            return 1
        return min(max_workers, task_count)

    def _prefetch_links_settings(self) -> Dict[str, Any]:
        raw = self.config.get("prefetch_links")
        if isinstance(raw, dict):
            enabled = raw.get("enabled", True)
            return {
                "enabled": self._is_truthy(enabled),
                "retmax": raw.get("retmax", 5),
                "sort": raw.get("sort", "relevance"),
                "proxy_url": raw.get("proxy_url", "http://127.0.0.1:8080/proxy"),
            }
        if isinstance(raw, bool):
            return {
                "enabled": raw,
                "retmax": self.config.get("retmax", 5),
                "sort": self.config.get("sort", "relevance"),
                "proxy_url": self.config.get("proxy_url", "http://127.0.0.1:8080/proxy"),
            }
        return {"enabled": False}

    def _prefetch_links_for_query(self, query: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        retmax = cfg.get("retmax", 5)
        sort = cfg.get("sort", "relevance")
        proxy_base = cfg.get("proxy_url", "http://127.0.0.1:8080/proxy")
        search_url = f"{proxy_base}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": sort,
        }
        try:
            request_url = requests.Request("GET", search_url, params=params).prepare().url
            resp = requests.get(search_url, params=params, timeout=10)
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            return {"pmids": id_list, "request_url": request_url}
        except Exception as exc:
            return {"error": str(exc)}

    def _attach_prefetched_evidence(self, stmt: Any, query: str, pmids: List[str]) -> None:
        if not pmids:
            return
        for pmid in pmids:
            existing = next(
                (e for e in stmt.evidence if getattr(e, "pubmed_id", None) == pmid),
                None,
            )
            if existing is not None:
                if not hasattr(existing, "queries") or existing.queries is None:
                    existing.queries = []
                if query.lower() not in {x.lower() for x in existing.queries}:
                    existing.queries.append(query)
                continue
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            ev = PubMedEvidence(pubmed_id=pmid, url=url, queries=[query])
            stmt.evidence.append(ev)

    def _build_llm_settings(self, base_url: str) -> Dict[str, Any]:
        settings = dict(self.config.get("llm_settings", {}) or {})
        settings.pop("base_urls", None)

        if base_url:
            settings["base_url"] = base_url
        settings.setdefault(
            "api_key",
            os.environ.get("LLM_API_KEY") or os.environ.get("OLLAMA_API_KEY") or "ollama",
        )

        ctx = (
            settings.get("context_length")
            or settings.get("num_ctx")
            or settings.get("max_context")
            or os.environ.get("LLM_CONTEXT_LENGTH")
            or os.environ.get("OLLAMA_CONTEXT_LENGTH")
        )
        if ctx and "context_length" not in settings:
            settings["context_length"] = ctx

        return settings

    def _resolve_base_urls(self) -> List[str]:
        llm_settings = self.config.get("llm_settings", {}) or {}
        base_urls = llm_settings.get("base_urls")
        if isinstance(base_urls, list) and base_urls:
            return [u for u in base_urls if u]

        if self._is_truthy(os.environ.get("OLLAMA_MULTI_INSTANCE")):
            host = os.environ.get("OLLAMA_MULTI_HOST", "127.0.0.1")
            try:
                base_port = int(os.environ.get("OLLAMA_MULTI_BASE_PORT", "11434"))
            except Exception:
                base_port = 11434
            count = self._resolve_multi_count()
            return [f"http://{host}:{base_port + i}/v1" for i in range(count)]

        env_base = os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
        if env_base:
            return [env_base]

        base_url = llm_settings.get("base_url") or "http://localhost:11434/v1"
        return [base_url]

    def _resolve_multi_count(self) -> int:
        for key in ("OLLAMA_MULTI_COUNT", "OLLAMA_MULTI_INSTANCES", "OLLAMA_MULTI_WORKERS"):
            raw = os.environ.get(key)
            if not raw:
                continue
            try:
                val = int(raw)
                if val > 0:
                    return val
            except Exception:
                continue

        cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda is not None:
            ids = [p.strip() for p in cuda.split(",") if p.strip()]
            if ids:
                return len(ids)

        # Fallback: probe consecutive ports for running Ollama instances.
        host = os.environ.get("OLLAMA_MULTI_HOST", "127.0.0.1")
        try:
            base_port = int(os.environ.get("OLLAMA_MULTI_BASE_PORT", "11434"))
        except Exception:
            base_port = 11434

        max_probe = 8
        count = 0
        for i in range(max_probe):
            port = base_port + i
            if self._is_port_open(host, port):
                count += 1
            else:
                break

        return count if count > 0 else 1

    def _is_port_open(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0

    def _is_truthy(self, value: Optional[str]) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
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
            fetched = {q.lower() for q in (getattr(stmt, "queries_fetched", []) or [])}

            for q in queries:
                if q.lower() in fetched:
                    continue
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

                    if hasattr(stmt, "queries_fetched"):
                        if q.lower() not in {x.lower() for x in (stmt.queries_fetched or [])}:
                            stmt.queries_fetched.append(q)

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
         (re.compile(r"\bmeta[- ]analysis\b", re.I), 0.90, "meta-analysis"),
    (re.compile(r"\bsystematic review\b", re.I), 0.85, "systematic review"),
    (re.compile(r"\bpractice guideline\b", re.I), 0.85, "practice guideline"),

    (re.compile(r"\brandomi[sz]ed controlled trial\b", re.I), 0.80, "RCT"),

    # Clinical trials (more specific first)
    (re.compile(r"\bclinical trial,\s*phase iii\b", re.I), 0.78, "clinical trial"),
    (re.compile(r"\bcontrolled clinical trial\b", re.I), 0.75, "clinical trial"),
    (re.compile(r"\bpragmatic clinical trial\b", re.I), 0.75, "clinical trial"),
    (re.compile(r"\badaptive clinical trial\b", re.I), 0.75, "clinical trial"),
    (re.compile(r"\bequivalence trial\b", re.I), 0.75, "clinical trial"),
    (re.compile(r"\bclinical trial,\s*phase ii\b", re.I), 0.70, "clinical trial"),
    (re.compile(r"\bclinical trial,\s*phase iv\b", re.I), 0.70, "clinical trial"),
    (re.compile(r"\bclinical trial,\s*phase i\b", re.I), 0.60, "clinical trial"),
    (re.compile(r"\bclinical trial protocol\b", re.I), 0.25, "clinical trial protocol"),
    # "Clinical Trial" when unspecified (avoid matching phase/protocol)
    (re.compile(r"\bclinical trial\b", re.I), 0.70, "clinical trial"),

    # Studies / conferences / guidelines
    (re.compile(r"\bobservational stud(?:y|ies)\b", re.I), 0.60, "observational"),
    (re.compile(r"\bclinical stud(?:y|ies)\b", re.I), 0.60, "clinical study"),
    (re.compile(r"\bconsensus development conference,\s*nih\b", re.I), 0.60, "consensus conference"),
    (re.compile(r"\bconsensus development conference\b", re.I), 0.60, "consensus conference"),
    (re.compile(r"\bguideline\b", re.I), 0.60, "guideline"),
    (re.compile(r"\bcomparative stud(?:y|ies)\b", re.I), 0.60, "comparative study"),
    (re.compile(r"\bmulticenter stud(?:y|ies)\b", re.I), 0.60, "multicenter study"),
    (re.compile(r"\bevaluation stud(?:y|ies)\b", re.I), 0.60, "evaluation study"),
    (re.compile(r"\bvalidation stud(?:y|ies)\b", re.I), 0.55, "validation study"),
    (re.compile(r"\bjournal article\b", re.I), 0.50, "journal article"),
    # Reviews
    (re.compile(r"\breview\b", re.I), 0.55, "narrative review"),

    # Lower-evidence / publication types
    (re.compile(r"\bcase reports?\b", re.I), 0.40, "case report"),
    (re.compile(r"\btechnical reports?\b", re.I), 0.40, "technical report"),
    (re.compile(r"\bpreprints?\b", re.I), 0.40, "preprint"),
    (re.compile(r"\bmeeting abstracts?\b", re.I), 0.30, "meeting abstract"),

    (re.compile(r"\beditorial\b", re.I), 0.20, "opinion"),
    (re.compile(r"\bcomment\b", re.I), 0.20, "opinion"),
    (re.compile(r"\bletter\b", re.I), 0.20, "opinion"),
    (re.compile(r"\bnews\b", re.I), 0.15, "news"),

    # Integrity / retraction-related
    (re.compile(r"\bexpression of concern\b", re.I), 0.05, "expression of concern"),
    (re.compile(r"\bduplicate publication\b", re.I), 0.05, "duplicate publication"),
    (re.compile(r"\bretraction of publication\b(?:\s*\(notice\))?", re.I), 0.00, "retraction notice"),
    (re.compile(r"\bretracted publication\b", re.I), 0.00, "retracted publication"),
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
