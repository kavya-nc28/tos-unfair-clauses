from __future__ import annotations
from src.inference.severity_mapping import severity_label
import numpy as np
from typing import List, Dict
import math


# Adjust this list to match your UNFAIR-ToS label order
LABEL_NAMES = [
    "arbitration",
    "unilateral_change",
    "content_removal",
    "jurisdiction",
    "choice_of_law",
    "limitation_of_liability",
    "unilateral_termination",
    "contract_by_using",
]


def prob_to_band(score: float) -> str:
    """
    Map max probability for a clause into a coarse risk band.
    HIGH    : very likely unfair
    MEDIUM  : somewhat suspicious
    SAFE    : low evidence of unfairness
    """
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "SAFE"


def prob_to_severity(score: float) -> int:
    """
    Map [0,1] probability to integer severity score 1 to 10.
    """
    return max(1, min(10, math.ceil(score * 10)))


def explain_labels(predicted_labels: List[str]) -> str:
    """
    Turn predicted label ids into a human-readable explanation string.
    """
    if not predicted_labels:
        return "No strong unfair-clause pattern detected."

    label_map = {
        "arbitration": "This clause may force disputes into arbitration.",
        "unilateral_change": "This clause may allow the service to change terms unilaterally.",
        "content_removal": "This clause may allow the platform to remove content broadly.",
        "jurisdiction": "This clause may force legal disputes into a specific court.",
        "choice_of_law": "This clause may impose a specific governing law.",
        "limitation_of_liability": "This clause may reduce the provider’s legal responsibility.",
        "unilateral_termination": "This clause may allow account termination at the provider’s discretion.",
        "contract_by_using": "This clause may bind users simply by continued use.",
    }

    return " ".join(label_map.get(lbl, lbl) for lbl in predicted_labels)


def build_clause_results(
    clauses: List[Dict],
    probs,
    threshold: float,
) -> List[Dict]:
    """
    clauses: list of dicts from preprocessing:
        {
          "clause_id": int,
          "text": str,
          "start_char": int,
          "end_char": int,
        }
    probs: numpy array [num_clauses, num_labels] with sigmoid outputs
    threshold: probability threshold for deciding which labels are 'active'
    """
    results = []

    for clause, row in zip(clauses, probs):
        row = row.tolist()
        max_prob = max(row)

        # labels whose probability crosses the global threshold
        pred_ids = [i for i, p in enumerate(row) if p >= threshold]
        pred_labels = [LABEL_NAMES[i] for i in pred_ids]

        severity_score = prob_to_severity(float(max_prob))
        severity_band = prob_to_band(float(max_prob))

        result = {
            **clause,
            "probability": round(float(max_prob), 4),
            "predicted_labels": pred_labels,
            "severity_score": severity_score,
            "severity_band": severity_band,
            "verdict": severity_label(severity_score),
            "explanation": explain_labels(pred_labels),
            "raw_scores": {
                LABEL_NAMES[i]: round(float(row[i]), 4)
                for i in range(len(LABEL_NAMES))
            },
        }

        results.append(result)

    return results

def overall_safety_score(probs_binary: np.ndarray) -> int:
    """
    Returns 0–100 where 100 = fully safe, 0 = fully unsafe.
    Driven by the model's binary unfairness probabilities.
    """
    mean_unfair = float(np.mean(probs_binary))   # avg prob of being unfair
    safety = round((1 - mean_unfair) * 100)
    return max(0, min(100, safety))

def summarize_document(results: List[Dict]) -> Dict:

    """
    Aggregate per-clause results into a document-level summary.
    """
    num_clauses = len(results)
    critical    = sum(r["severity_band"] == "CRITICAL" for r in results)
    high_risk   = sum(r["severity_band"] == "HIGH"     for r in results)
    medium_risk = sum(r["severity_band"] == "MEDIUM"   for r in results)
    safe        = sum(r["severity_band"] == "SAFE"     for r in results)

    if critical > 0:
        overall = "Critical unfair clauses detected. Do NOT agree."
    elif high_risk > 0:
        overall = "High risk Terms of Service detected."
    elif medium_risk > 0:
        overall = "Some potentially problematic clauses detected."
    else:
        overall = "Mostly safe according to the current model."

    return {
        "num_clauses":     num_clauses,
        "critical":        critical,
        "high_risk":       high_risk,
        "medium_risk":     medium_risk,
        "safe":            safe,
        "overall_verdict": overall,
    }