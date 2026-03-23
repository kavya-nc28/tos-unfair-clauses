"""
Loading the Lippi et al. (2019) / UNFAIR-ToS dataset.

TODO:
- Decide whether to load from HuggingFace `datasets` or local CSV.
- Implement a function that returns train/val/test splits as HF Dataset or pandas.
"""

from pathlib import Path
from typing import Tuple

from datasets import load_dataset   # TODO: pin version in requirements if needed


def load_unfair_tos_splits() -> Tuple:
    """
    Load train/validation/test splits for the unfair ToS dataset.

    Returns
    -------
    train_ds, val_ds, test_ds
        HF Datasets or pandas DataFrames with fields like:
        - 'text' (clause)
        - 'labels' (multi-hot unfair types or binary)
    """
    # TODO: adjust to actual dataset name / config
    # Example if using LexGLUE:
    # ds = load_dataset("lex_glue", "unfair_tos")
    # train_ds = ds["train"]
    # val_ds = ds["validation"]
    # test_ds = ds["test"]

    raise NotImplementedError("Implement load_unfair_tos_splits()")
