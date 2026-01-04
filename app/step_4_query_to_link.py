
import requests
import re
from typing import Any, Dict, List
from ..app2.core.preprompts import *
from .llmconfigs import *

def query_to_link(data: Dict[str, Any],) -> Dict[str, Any]:
    """
    Load statements from `input_path` (a JSON file),
    query PubMed for each statement['query'], build up to 5 URLs,
    and append one evidence entry per URL into each statement.
    Finally, write the augmented JSON to `output_path`.
    """
    print("Starting Step4: Query to PubMed Links")
    print("...")
    print("...")
    

    def get_urls(query: str, retmax: int = 4) -> List[str]:
        """Return up to `retmax` PubMed article URLs matching `query`."""
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json"
        }
        resp = requests.get(search_url, params=params)
        resp.raise_for_status()
        ids = resp.json()["esearchresult"]["idlist"]
        return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]

    # 2) For each statement, call get_urls and append to evidence
    for stmt in data.get("statements", []):
        query = stmt.get("query")
        if not query:
            continue

        try:
            urls = get_urls(query)
        except requests.RequestException as e:
            print(f"[WARN] PubMed lookup failed for '{query}': {e}")
            continue

        ev_list = stmt.setdefault("evidence", [])
        for url in urls:
            print(" --- Step4 Query to Link ---> PubmedLink --- ")
            print(f"PubmedLink: {url}")
            pmid_match = re.search(r"/(\d+)/?$", url)
            pmid = pmid_match.group(1) if pmid_match else None
            ev_list.append({
                "pubmed_id": pmid,
                "url": url,
                "abstract": "",
                "relevance": ""
            })
    return data

