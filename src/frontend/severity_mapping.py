"""
Map raw model outputs to severity scores and human-readable labels.
"""

from typing import Tuple
import math


def logits_to_severity(prob: float) -> int:
    """Map sigmoid probability [0,1] to severity score 1–10."""
    return max(1, min(10, math.ceil(prob * 10)))


def severity_label(score: int) -> str:
    if score <= 3:
        return "You are good to go"
    if score <= 5:
        return "Needs another look"
    if score <= 7:
        return "This might be trouble"
    return "DO NOT AGREE TO THIS"
