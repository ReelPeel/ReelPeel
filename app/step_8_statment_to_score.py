from typing import Dict, Any

from ..app2.core.preprompts import *
from .llmconfigs import *

threshold = 0.15

def statement_to_score(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the weighted average of statement confidences,
    giving triple weight to values below the threshold.
    Updates the data dict in-place by setting 'overall_truthiness',
    rounded to two decimal places (zwei Nachkommastellen).
    """
    print("Starting Step8: Statement to Score")
    print("...")
    print("...")

    confidences = [stmt["confidence"] for stmt in data["statements"]]
    weights = [3 if conf < threshold else 1 for conf in confidences]

    weighted_sum = sum(conf * w for conf, w in zip(confidences, weights))
    total_weight = sum(weights)

    overall = round(weighted_sum / total_weight, 2) if total_weight else 0.0
    print(f"Overall truthiness score: {overall}")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    data["overall_truthiness"] = overall
    return data
