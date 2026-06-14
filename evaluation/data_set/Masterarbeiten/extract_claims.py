#!/usr/bin/env python3
import argparse
import csv
import json


def extract_third_column(
    input_csv: str,
    output_file: str,
    skip_header: bool = False,
    encoding: str = "utf-8-sig",
) -> None:
    statements = []

    with open(input_csv, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=";")

        if skip_header:
            next(reader, None)

        for row in reader:
            if len(row) < 3:
                continue

            statement = row[2]

            if not statement:
                continue

            statements.append({
                "statement": statement,
                "id": len(statements)
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(statements, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract the third column from a CSV and write it as JSON-style text."
    )

    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to the output text file")
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first row of the CSV"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV file encoding. Try latin-1 or cp1252 if UTF-8 fails."
    )

    args = parser.parse_args()

    extract_third_column(
        input_csv=args.input_csv,
        output_file=args.output_file,
        skip_header=args.skip_header,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()