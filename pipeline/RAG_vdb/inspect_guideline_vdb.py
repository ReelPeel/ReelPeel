#!/usr/bin/env python3
"""Print corpus diagnostics for a guideline VDB."""
import argparse
import sqlite3
from pathlib import Path

def scalar(con, sql, params=()):
    return int(con.execute(sql, params).fetchone()[0])

def inspect_db(db_path: Path) -> str:
    with sqlite3.connect(str(db_path)) as con:
        lines = [f"Documents: {scalar(con, 'SELECT COUNT(*) FROM documents')}", f"Pages: {scalar(con, 'SELECT COUNT(*) FROM pages')}", f"Low-text pages: {scalar(con, 'SELECT COUNT(*) FROM pages WHERE low_text_warning=1')}", "Chunks:"]
        for chunk_type, count in con.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type ORDER BY chunk_type"):
            lines.append(f"  {chunk_type}: {count}")
        lines.append("Embeddings:")
        for model, count in con.execute("SELECT embed_model, COUNT(*) FROM embeddings GROUP BY embed_model ORDER BY embed_model"):
            lines.append(f"  {model}: {count}")
        lines.append(f"Recommendations: {scalar(con, 'SELECT COUNT(*) FROM recommendations')}")
        lines.append("Top documents by chunk count:")
        for source, count in con.execute("SELECT d.source_path, COUNT(c.chunk_id) FROM documents d LEFT JOIN chunks c ON c.doc_id=d.doc_id GROUP BY d.doc_id ORDER BY COUNT(c.chunk_id) DESC LIMIT 10"):
            lines.append(f"  {Path(source).name}: {count}")
        warnings = []
        for source, low_count, page_count, rec_count in con.execute("SELECT d.source_path, (SELECT COUNT(*) FROM pages p WHERE p.doc_id=d.doc_id AND p.low_text_warning=1), (SELECT COUNT(*) FROM pages p WHERE p.doc_id=d.doc_id), (SELECT COUNT(*) FROM recommendations r WHERE r.doc_id=d.doc_id) FROM documents d"):
            if rec_count == 0:
                warnings.append(f"  {Path(source).name} has no extracted recommendations")
            if low_count and (low_count >= 3 or low_count / max(page_count, 1) >= 0.2):
                warnings.append(f"  {Path(source).name} has {low_count} low-text pages")
        lines.append("Warnings:")
        lines.extend(warnings or ["  none"])
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db_path", default="guidelines_vdb.sqlite")
    args = parser.parse_args()
    path = Path(args.db_path).expanduser().resolve()
    if not path.is_file():
        parser.error(f"Database not found: {path}")
    try:
        print(inspect_db(path))
    except sqlite3.Error as exc:
        raise SystemExit(f"Failed to inspect database: {exc}") from exc

if __name__ == "__main__":
    main()
