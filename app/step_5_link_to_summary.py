import argparse
import json
import requests
import subprocess
import sys
import re
from typing import Optional, List, Dict, Any
import datetime

from .preprompts import *
from .llmconfigs import *

# Base URL for NCBI E-utilities
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Regex to find PMID in PubMed URL
PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?")
# Regex to remove trailing commas before } or ] in relaxed JSON
TRAILING_COMMAS_RE = re.compile(r",\s*(?=[\]}])")


def extract_pmid(url: str) -> str:
    """Extract numeric PMID from a PubMed URL or accept raw PMIDs."""
    if url.isdigit():
        return url
    match = PUBMED_URL_RE.search(url)
    if match:
        return match.group(1)
    raise ValueError(f"Unable to extract PMID from URL: {url}")


def pubmed_fetch_abstract(pmid: str) -> Optional[str]:
    """Fetch abstract text for given PMID (returns None if none found)."""
    params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
    r = requests.get(f"{NCBI_EUTILS}/efetch.fcgi", params=params, timeout=20)
    r.raise_for_status()
    lines = [line.strip() for line in r.text.splitlines() if line.strip()]
    abstract = " ".join(line for line in lines if not line.isupper())
    return abstract or None


def pubmed_fetch_types(pmid: str) -> List[str]:
    """
    Fetch PubMed 'Publication Types' for a PMID (returned directly as provided by PubMed).
    This uses ESummary (retmode=json) and reads the `pubtype` field.
    """
    params = {"db": "pubmed", "id": pmid, "retmode": "json"}
    r = requests.get(f"{NCBI_EUTILS}/esummary.fcgi", params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    rec = (payload.get("result") or {}).get(pmid) or {}
    pubtypes = rec.get("pubtype") or []
    # Ensure list[str]
    return [str(x) for x in pubtypes if x is not None]


def load_json_relaxed(path: str) -> Any:
    """Load JSON file, removing trailing commas to tolerate common mistakes."""
    raw = open(path, "r", encoding="utf-8").read()
    cleaned = TRAILING_COMMAS_RE.sub("", raw)
    return json.loads(cleaned)


def link_to_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    print("Starting Step5: Link to Summary")
    print("...")
    print("...")

    # Helper to process a list of evidence items
    def process_evidence_list(ev_list: List[Dict[str, Any]]):
        for ev in ev_list:
            url = ev.get("url")
            try:
                pmid = extract_pmid(url)
                ev["pubmed_id"] = pmid

                #  add publication types directly from PubMed
                ev["type"] = pubmed_fetch_types(pmid)

                abstract = pubmed_fetch_abstract(pmid)
                ev["abstract"] = abstract if abstract else None

                print(f"Processed evidence URL {url} with PMID {pmid}")
            except Exception as e:
                print(
                    f"Warning: could not process evidence URL {url}: {e}",
                    file=sys.stderr,
                )
                ev.setdefault("pubmed_id", None)
                ev.setdefault("type", None)
                ev.setdefault("abstract", None)

    # Process nested evidence under statements
    for stmt in data.get("statements", []):
        process_evidence_list(stmt.get("evidence", []))

    # Process top-level evidence list if present
    if "evidence" in data:
        process_evidence_list(data["evidence"])

    # Update generated timestamp
    data["generated_at"] = (
        datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )
    return data
