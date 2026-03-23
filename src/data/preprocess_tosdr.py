"""
Preprocessing utilities for the ToS;DR dataset (optional).

TODO:
- Load ToS;DR CSVs from data/raw.
- Map good/bad/blocker/neutral to numeric labels.
- Return a cleaned DataFrame or Dataset.
"""

from pathlib import Path
import pandas as pd

from src.config import RAW_DIR


def load_and_preprocess_tosdr():
    """
    Load ToS;DR policies dataset and return a cleaned DataFrame.

    TODO:
    - Decide which CSV(s) to use (raw vs clean).
    - Implement basic cleaning: strip whitespace, drop missing, etc.
    """
    # Example path (adjust):
    # csv_path = RAW_DIR / "tosdr_policies_clean.csv"
    # df = pd.read_csv(csv_path)
    # TODO: map 'classification' column to numeric labels
    raise NotImplementedError("Implement load_and_preprocess_tosdr()")
