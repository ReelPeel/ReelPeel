#!/usr/bin/env python3
import json
import random
import argparse
from collections import defaultdict, Counter

def allocate_counts(n, ratios):
    """
    Allocate n items into k buckets according to ratios (sum ~ 1),
    using the largest-remainder method to keep totals exact.
    """
    raw = [n * r for r in ratios]
    base = [int(x) for x in raw]  # floor
    remainder = n - sum(base)

    # Distribute remaining items to buckets with largest fractional parts
    fracs = [(raw[i] - base[i], i) for i in range(len(ratios))]
    fracs.sort(reverse=True)
    for _, idx in fracs[:remainder]:
        base[idx] += 1
    return base  # sums to n

def stratified_split(data, train_ratio, test_ratio, seed=42):
    if train_ratio + test_ratio <= 0:
        raise ValueError("Ratios must sum to a positive number.")

    # Normalize ratios to sum to 1.0
    s = train_ratio + test_ratio
    ratios = (train_ratio / s, test_ratio / s)

    rng = random.Random(seed)

    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)

    train, test = [], []

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train, n_test = allocate_counts(n, ratios)

        train.extend(items[:n_train])
        test.extend(items[n_train:n_train + n_test])

    # Shuffle within each split so labels are mixed
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test

def write_json_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def label_counts(items):
    return dict(Counter(item["label"] for item in items))

def main():
    ap = argparse.ArgumentParser(description="Stratified train/test split for JSON-list data.")
    ap.add_argument("--input", default="claims_statements.txt", help="Input file (JSON list).")
    ap.add_argument("--train", type=float, default=0.8, help="Train ratio (default: 0.8).")
    ap.add_argument("--test", type=float, default=0.1, help="Test ratio (default: 0.1).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("--out_prefix", default="claims_statements", help="Output prefix (default: claims_statements).")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    train, test = stratified_split(data, args.train, args.test, seed=args.seed)

    train_path = f"{args.out_prefix}_train.txt"
    test_path  = f"{args.out_prefix}_test.txt"

    write_json_list(train_path, train)
    write_json_list(test_path, test)

    print("Wrote:")
    print(f"  {train_path} ({len(train)} items) label counts: {label_counts(train)}")
    print(f"  {test_path}  ({len(test)} items) label counts: {label_counts(test)}")
    print(f"Total          ({len(data)} items) label counts: {label_counts(data)}")

if __name__ == "__main__":
    main()
