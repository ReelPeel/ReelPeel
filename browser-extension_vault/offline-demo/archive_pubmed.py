"""Archive the PubMed records referenced by offline_mock for local viewing."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse


ROOT = Path(__file__).resolve().parent
MOCK_ROOT = ROOT.parent / "offline_mock"
PUBMED_ROOT = ROOT / "pubmed"
ASSET_ROOT = PUBMED_ROOT / "_assets"
USER_AGENT = "ReelPeel-Offline-Conference-Archive/1.0 (local research demo)"
ASSET_HOSTS = {"cdn.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"}
ASSET_ATTRS = {"script": "src", "img": "src", "source": "src"}
CSS_URL = re.compile(r"url\((?P<quote>['\"]?)(?P<url>[^)'\"]+)(?P=quote)\)", re.I)
PRECONNECT_TAG = re.compile(r"\s*<link\s+rel=\"preconnect\"[^>]*>\s*", re.I)


class PubMedAssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        attr_name = ASSET_ATTRS.get(tag)
        if attr_name and attributes.get(attr_name):
            self.urls.add(attributes[attr_name])
            return

        if tag != "link" or not attributes.get("href"):
            return
        rel = set((attributes.get("rel") or "").lower().split())
        if {"stylesheet", "icon"} & rel or attributes.get("as") == "font":
            self.urls.add(attributes["href"])


def pubmed_urls() -> list[str]:
    urls = set()
    for process_path in MOCK_ROOT.glob("*/process.json"):
        payload = json.loads(process_path.read_text(encoding="utf-8"))
        for statement in payload.get("statements", []):
            for evidence in statement.get("evidence", []):
                url = str(evidence.get("url") or "")
                if url.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
                    urls.add(url)
    return sorted(urls)


def local_asset_path(url: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if not suffix or len(suffix) > 12:
        suffix = ".bin"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return ASSET_ROOT / f"{digest}{suffix.lower()}"


def is_archivable(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc in ASSET_HOSTS


def relative_url(source: Path, target: Path) -> str:
    return Path("..") / target.relative_to(PUBMED_ROOT) if source.parent.parent == PUBMED_ROOT else Path(target.name)


def fetch(url: str) -> bytes:
    result = subprocess.run(
        [
            "curl.exe",
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--max-time",
            "40",
            "--connect-timeout",
            "10",
            "--retry",
            "2",
            "--retry-delay",
            "2",
            "--user-agent",
            USER_AGENT,
            url,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Download failed for {url}: {message}")
    return result.stdout


def archive_asset(url: str, page_path: Path, cached: dict[str, Path]) -> str:
    if not is_archivable(url):
        return url
    if url not in cached:
        target = local_asset_path(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                body = fetch(url)
            except RuntimeError as error:
                print(f"Skipping optional asset: {error}")
                return url
            if target.suffix == ".css":
                css = body.decode("utf-8", errors="replace")

                def replace_css_url(match):
                    raw_url = match.group("url").strip()
                    nested_url = urljoin(url, raw_url)
                    local_url = archive_asset(nested_url, target, cached)
                    return f"url('{local_url}')"

                body = CSS_URL.sub(replace_css_url, css).encode("utf-8")
            target.write_bytes(body)
        cached[url] = target
    return Path("..").joinpath("_assets", cached[url].name).as_posix()


def archive_record(record_url: str, cached: dict[str, Path]) -> str:
    pmid = record_url.rstrip("/").rsplit("/", 1)[-1]
    record_dir = PUBMED_ROOT / pmid
    record_dir.mkdir(parents=True, exist_ok=True)
    page_path = record_dir / "index.html"
    if page_path.is_file():
        return pmid
    html = fetch(record_url).decode("utf-8", errors="replace")

    parser = PubMedAssetParser()
    parser.feed(html)
    for raw_url in parser.urls:
        absolute_url = urljoin(record_url, raw_url)
        local_url = archive_asset(absolute_url, page_path, cached)
        if local_url != absolute_url:
            html = html.replace(raw_url, local_url)

    page_path.write_text(html, encoding="utf-8")
    return pmid


def remove_external_preconnects():
    """Ensure opening an archived record does not initiate external preconnects."""
    for page_path in PUBMED_ROOT.glob("*/index.html"):
        html = page_path.read_text(encoding="utf-8", errors="replace")
        sanitized = PRECONNECT_TAG.sub("\n", html)
        if sanitized != html:
            page_path.write_text(sanitized, encoding="utf-8")


def main():
    cached_assets: dict[str, Path] = {}
    record_urls = pubmed_urls()
    print(f"Archiving {len(record_urls)} PubMed records.")
    failed = []
    for index, record_url in enumerate(record_urls, start=1):
        try:
            pmid = archive_record(record_url, cached_assets)
            print(f"[{index}/{len(record_urls)}] PMID {pmid}")
        except RuntimeError as error:
            failed.append(record_url)
            print(f"[{index}/{len(record_urls)}] {error}")
    remove_external_preconnects()
    print(f"Saved {len(cached_assets)} shared PubMed assets.")
    if failed:
        raise SystemExit(f"{len(failed)} PubMed records could not be downloaded.")


if __name__ == "__main__":
    main()
