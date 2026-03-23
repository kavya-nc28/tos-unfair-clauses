"""
Map raw model outputs to severity scores and human-readable labels.
"""

from typing import Tuple


def logits_to_severity(logits) -> float:
    """
    TODO:
    - Decide how to convert logits/probabilities into a 1–10 severity score.
      Example (placeholder): severity = 10 * p(unfair).
    """
    raise NotImplementedError("Implement logits_to_severity().")


def severity_to_label(severity: float) -> str:
    """
    Map 1–10 severity to plain-English verdicts.
    """
    if severity <= 2:
        return "You are good to go."
    if severity <= 5:
        return "Needs another look."
    if severity <= 8:
        return "This might be trouble."
    return "DO NOT AGREE TO THIS."
