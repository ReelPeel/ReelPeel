#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


QUERY_MARKER_RE = re.compile(
    r">>> \[ARTIFACT\] Raw Output for Statement (\d+) Query Generation"
)
COLUMNS = ["statement", "first_query", "second_query", "third_query"]


def _parse_statements_from_settings(lines: list[str]) -> list[dict]:
    in_mock = False
    in_settings = False
    buffer: list[str] = []

    for line in lines:
        if line.startswith("START STEP: MockStatementLoader"):
            in_mock = True
            continue
        if in_mock and line.strip() == "--- SETTINGS ---":
            in_settings = True
            buffer = []
            continue
        if in_settings:
            if line.strip() == "----------------":
                json_text = "\n".join(buffer).strip()
                if not json_text:
                    return []
                try:
                    payload = json.loads(json_text)
                except json.JSONDecodeError:
                    return []
                return payload.get("statements", [])
            buffer.append(line)

    return []


def _parse_statements_from_output_states(lines: list[str]) -> list[dict]:
    in_output = False
    buffer: list[str] = []

    for line in lines:
        if line.strip() == "--- OUTPUT STATE ---":
            in_output = True
            buffer = []
            continue
        if in_output:
            if line.startswith("="):
                json_text = "\n".join(buffer).strip()
                in_output = False
                if not json_text:
                    continue
                try:
                    payload = json.loads(json_text)
                except json.JSONDecodeError:
                    continue
                statements = payload.get("statements", [])
                if statements:
                    return statements
            else:
                buffer.append(line)

    return []


def _parse_statements_with_regex(text: str) -> list[dict]:
    statements: list[dict] = []
    seen_ids: set[int] = set()
    pattern = re.compile(r'"id":\s*(\d+),\s*"text":\s*"([^"]+)"')

    for match in pattern.finditer(text):
        statement_id = int(match.group(1))
        if statement_id in seen_ids:
            continue
        seen_ids.add(statement_id)
        statements.append({"id": statement_id, "text": match.group(2)})

    return statements


def extract_statements(lines: list[str], full_text: str) -> list[dict]:
    statements = _parse_statements_from_settings(lines)
    if statements:
        return statements

    statements = _parse_statements_from_output_states(lines)
    if statements:
        return statements

    return _parse_statements_with_regex(full_text)


def extract_queries(lines: list[str]) -> dict[int, list[str]]:
    queries_by_id: dict[int, list[str]] = defaultdict(list)
    pending_id: int | None = None

    for line in lines:
        match = QUERY_MARKER_RE.search(line)
        if match:
            pending_id = int(match.group(1))
            continue
        if pending_id is not None:
            stripped = line.strip()
            if not stripped:
                continue
            queries_by_id[pending_id].append(stripped)
            pending_id = None

    return queries_by_id


def collect_rows(log_path: Path) -> list[dict]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    statements = extract_statements(lines, text)
    queries_by_id = extract_queries(lines)

    rows: list[dict] = []
    for statement in statements:
        statement_id = statement.get("id")
        statement_text = str(statement.get("text", "")).strip()
        queries = queries_by_id.get(statement_id, [])
        rows.append(
            {
                "statement": statement_text,
                "first_query": queries[0] if len(queries) > 0 else "",
                "second_query": queries[1] if len(queries) > 1 else "",
                "third_query": queries[2] if len(queries) > 2 else "",
            }
        )
    return rows


def find_log_files(logs_dir: Path) -> list[Path]:
    if not logs_dir.exists():
        return []

    log_paths = []
    for path in logs_dir.glob("*.log"):
        name = path.name.lower()
        if "pipeline_debug_pubmed_multiquery_filter_" not in name:
            continue
        if "prompt" in name:
            continue
        log_paths.append(path)

    return sorted(log_paths)


def write_excel(rows: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    try:
        df.to_excel(output_path, index=False, engine="openpyxl")
    except Exception:
        df.to_excel(output_path, index=False, engine="xlsxwriter")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract statements and queries from PubMed MultiQuery Filter logs."
        )
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing pipeline_debug_PubMed_MultiQuery_Filter logs.",
    )
    parser.add_argument(
        "--output",
        default="pubmed_multiquery_filter_queries.xlsx",
        help="Excel output path.",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    log_files = find_log_files(logs_dir)
    if not log_files:
        raise SystemExit(f"No matching logs found under: {logs_dir}")

    rows: list[dict] = []
    for log_path in log_files:
        rows.extend(collect_rows(log_path))

    output_path = Path(args.output)
    write_excel(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
