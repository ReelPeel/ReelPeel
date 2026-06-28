from __future__ import annotations

import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.preprompts import PROMPT_TMPL_S2
from pipeline.test_configs.video_transcription_config import VIDEO_PIPELINE_CONFIG


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "downloads" / "downloads_beikost"
OUTPUT_DIR = ROOT / "evaluation" / "statements" / "beikost_first10"
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_CSV_PATH = OUTPUT_DIR / "statements.csv"
TARGET_NUMMERN = ["24", "53", "99", "80", "34", "4", "17", "56", "74", "59"]
WHISPER_MODEL = "large-v3"
SHORTCODE_RE = re.compile(r"/p/([^/]+)/")
NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package": "http://schemas.openxmlformats.org/package/2006/relationships",
}
EXTRACTION_PROMPT_30 = PROMPT_TMPL_S2.replace(
    "up to a maximum of **5**. If more than 8 are present, return the **8 most clinically important and/or potentially harmful**.",
    "up to a maximum of **30**. If more than 30 are present, return the **30 most clinically important and/or potentially harmful**.",
).replace(
    "A valid JSON array of 1–5 strings.",
    "A valid JSON array of 0-30 strings.",
).replace(
    "STRICT OUTPUT",
    "LANGUAGE RULE\nReturn each extracted claim in the same language as the input transcript. If the transcript is German, return German claims. If the transcript is English, return English claims. Do not translate claims into another language.\n\nSTRICT OUTPUT",
)


def resolve_workbook_path() -> Path:
    candidates = sorted(ROOT.glob("Stichprobe Evidenzpr* Beikost.xlsx"))
    if not candidates:
        raise FileNotFoundError("Could not find 'Stichprobe Evidenzprüfung Beikost.xlsx' in the project root.")
    return candidates[0]


def parse_xlsx_rows(path: Path) -> List[List[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = load_shared_strings(archive)
        sheet_path = resolve_first_sheet_path(archive)
        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows: List[List[str]] = []
        for row in sheet_root.findall(".//main:sheetData/main:row", NS):
            values: List[str] = []
            for cell in row.findall("main:c", NS):
                values.append(read_cell_value(cell, shared_strings))
            rows.append(values)
        return rows


def load_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("main:si", NS):
        text = "".join(node.text or "" for node in item.iterfind(".//main:t", NS))
        values.append(text)
    return values


def resolve_first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    first_sheet = workbook.find("main:sheets/main:sheet", NS)
    if first_sheet is None:
        raise ValueError("Workbook does not contain any sheets.")

    rel_id = first_sheet.attrib[f"{{{NS['office']}}}id"]
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall("package:Relationship", NS):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"]
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("Could not resolve first sheet relationship.")


def read_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("main:v", NS)
    inline = cell.find("main:is", NS)

    if cell_type == "s" and value is not None:
        return shared_strings[int(value.text or "0")]
    if cell_type == "inlineStr" and inline is not None:
        return "".join(node.text or "" for node in inline.iterfind(".//main:t", NS))
    if value is not None and value.text is not None:
        return value.text
    return ""


def extract_shortcode(url: str) -> str:
    match = SHORTCODE_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract Instagram shortcode from URL: {url}")
    return match.group(1)


def build_extraction_only_config(video_path: Path, audio_path: Path) -> Dict[str, Any]:
    cfg = {
        "name": f"beikost_statement_extraction_{video_path.stem}",
        "debug": False,
        "steps": deepcopy(VIDEO_PIPELINE_CONFIG["steps"][:3]),
    }
    cfg["steps"][0]["settings"]["video_path"] = str(video_path)
    cfg["steps"][0]["settings"]["output_path"] = str(audio_path)
    cfg["steps"][1]["settings"]["whisper_model"] = WHISPER_MODEL
    cfg["steps"][1]["settings"]["translate_non_english"] = False
    cfg["steps"][2]["settings"]["prompt_template"] = EXTRACTION_PROMPT_30
    cfg["steps"][2]["settings"]["max_statements"] = 30
    return cfg


def prepare_jobs() -> List[Dict[str, Any]]:
    workbook_path = resolve_workbook_path()
    rows = parse_xlsx_rows(workbook_path)
    rows_by_nummer: Dict[str, List[str]] = {}
    for row in rows[1:]:
        nummer = row[0] if row else ""
        if nummer:
            rows_by_nummer[nummer] = row

    missing = [nummer for nummer in TARGET_NUMMERN if nummer not in rows_by_nummer]
    if missing:
        raise ValueError(f"Workbook is missing requested Nummer IDs: {', '.join(missing)}")

    jobs: List[Dict[str, Any]] = []
    for nummer in TARGET_NUMMERN:
        row = rows_by_nummer[nummer]
        link = row[1]
        sample_rank = row[2] if len(row) > 2 else ""
        shortcode = extract_shortcode(link)
        video_path = VIDEO_DIR / f"{shortcode}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for shortcode {shortcode}: {video_path}")
        jobs.append(
            {
                "nummer": nummer,
                "link": link,
                "sample_rank": sample_rank,
                "shortcode": shortcode,
                "video_path": video_path,
                "workbook_path": workbook_path,
            }
        )
    return jobs


def serialize_state(job: Dict[str, Any], state: PipelineState, audio_path: Path) -> Dict[str, Any]:
    return {
        "nummer": job["nummer"],
        "link": job["link"],
        "sample_rank": job["sample_rank"],
        "shortcode": job["shortcode"],
        "video_path": str(job["video_path"].resolve()),
        "audio_path": str(audio_path.resolve()),
        "workbook_path": str(job["workbook_path"].resolve()),
        "generated_at": state.generated_at,
        "transcript": state.transcript or "",
        "statement_count": len(state.statements),
        "statements": [statement.model_dump(mode="json") for statement in state.statements],
    }


def build_csv_rows(job: Dict[str, Any], payload: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    base_row = {
        "index": index,
        "nummer": job["nummer"],
        "sample_rank": job["sample_rank"],
        "shortcode": job["shortcode"],
        "link": job["link"],
        "video_path": str(job["video_path"].resolve()),
        "audio_path": payload["audio_path"],
        "generated_at": payload["generated_at"],
        "statement_count": payload["statement_count"],
        "transcript": payload["transcript"],
    }
    statements = payload["statements"]
    if not statements:
        return [{**base_row, "statement_id": "", "statement": ""}]

    rows: List[Dict[str, Any]] = []
    for statement in statements:
        rows.append(
            {
                **base_row,
                "statement_id": statement.get("id", ""),
                "statement": statement.get("text", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "index",
        "nummer",
        "sample_rank",
        "shortcode",
        "link",
        "video_path",
        "audio_path",
        "generated_at",
        "statement_count",
        "statement_id",
        "statement",
        "transcript",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    jobs = prepare_jobs()
    manifest: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for index, job in enumerate(jobs, start=1):
        audio_path = OUTPUT_AUDIO_DIR / f"{job['shortcode']}.wav"
        output_json = OUTPUT_DIR / f"{index:02d}_{job['shortcode']}.json"
        config = build_extraction_only_config(job["video_path"], audio_path)
        state = PipelineOrchestrator(config).run(PipelineState(video_path=str(job["video_path"])))
        payload = serialize_state(job, state, audio_path)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_rows.extend(build_csv_rows(job, payload, index))
        manifest.append(
            {
                "index": index,
                "nummer": job["nummer"],
                "sample_rank": job["sample_rank"],
                "shortcode": job["shortcode"],
                "link": job["link"],
                "video_path": str(job["video_path"].resolve()),
                "audio_path": str(audio_path.resolve()),
                "output_json": str(output_json.resolve()),
                "statement_count": payload["statement_count"],
            }
        )

    write_csv(OUTPUT_CSV_PATH, csv_rows)
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(manifest)} statement files and CSV to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
