import numpy as np
import json

PATH="evaluation/data_set/claims_statements_train.txt"

def load_evaluation_data(path: str) -> list[dict]:
    """Load evaluation data from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

if __name__ == "__main__":
    data = load_evaluation_data(PATH)
    print(f"Loaded {len(data)} evaluation items.")