from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_ROOT = REPO_ROOT / "offline_mock" / "sweep_20260626_072546"
DEFAULT_ENDPOINT = "http://127.0.0.1:6006/evidence_summary"
OUTPUT_DIR_NAME = "evidence_summaries_with_stance"
OUTPUT_MANIFEST_NAME = "manifest_with_stance_summary.json"


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate sweep evidence summaries while passing evidence.stance."
    )
    parser.add_argument("--sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    process_files = sorted(sweep_root.glob("run_*/DT*/process.json"))

    total = 0
    written = 0
    skipped = 0
    failed = 0
    started_at = time.time()

    for process_file in process_files:
        process_dir = process_file.parent
        source_manifest_file = process_dir / "manifest.json"
        if not source_manifest_file.exists():
            continue

        process_data = json.loads(process_file.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_manifest_file.read_text(encoding="utf-8"))
        output_dir = process_dir / OUTPUT_DIR_NAME
        output_dir.mkdir(parents=True, exist_ok=True)

        output_manifest = {
            "reel_url": source_manifest.get("reel_url"),
            "reel_id": source_manifest.get("reel_id"),
            "process_response": source_manifest.get("process_response", "process.json"),
            "endpoint": args.endpoint,
            "summary_payload_includes_stance": True,
            "evidence_summaries": [],
        }

        for item in source_manifest.get("evidence_summaries", []) or []:
            total += 1
            statement_index = item["statement_index"]
            evidence_index = item["evidence_index"]
            statement = process_data["statements"][statement_index]
            evidence = statement["evidence"][evidence_index]
            source_name = Path(item["response_file"]).name
            response_file = output_dir / source_name

            output_item = dict(item)
            output_item["response_file"] = f"{OUTPUT_DIR_NAME}/{source_name}"
            output_item["payload_includes_stance"] = True
            output_manifest["evidence_summaries"].append(output_item)

            if response_file.exists() and not args.force:
                skipped += 1
                continue

            payload = {
                "statement": statement.get("text") or "",
                "evidence": {
                    "abstract": evidence.get("abstract") or "",
                    "title": (
                        evidence.get("title")
                        or evidence.get("article_title")
                        or evidence.get("paper_title")
                        or ""
                    ),
                    "pubmed_id": evidence.get("pubmed_id"),
                    "url": evidence.get("url"),
                    "stance": evidence.get("stance") or None,
                },
            }

            try:
                response_data = post_json(args.endpoint, payload, args.timeout)
                response_file.write_text(
                    json.dumps(response_data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                written += 1
            except Exception as exc:
                failed += 1
                response_file.write_text(
                    json.dumps(
                        {
                            "error": str(exc),
                            "payload_includes_stance": True,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        (process_dir / OUTPUT_MANIFEST_NAME).write_text(
            json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    elapsed = round(time.time() - started_at, 1)
    print(
        json.dumps(
            {
                "sweep_root": str(sweep_root),
                "endpoint": args.endpoint,
                "process_files": len(process_files),
                "total": total,
                "written": written,
                "skipped": skipped,
                "failed": failed,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
