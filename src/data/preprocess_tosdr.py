"""
Preprocessing utilities for the ToS;DR dataset.

Assumptions about the CSV (you can adapt as needed):
- Lives in `data/raw/` as `tosdr_policies_clean.csv`
- Has at least:
    - `text` or `excerpt` column with the clause text
    - `classification` column with values like:
        'good', 'bad', 'blocker', 'neutral'

We produce:
- label_ordinal: 0 = good, 1 = neutral, 2 = bad, 3 = blocker
- label_binary : 0 = good/neutral (fair-ish), 1 = bad/blocker (unfair)

and then tokenize with LegalBERT, returning a HF Dataset with
PyTorch tensors similar to UNFAIR-ToS.
"""

from typing import Tuple

from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.config import RAW_DIR, BASE_MODEL_NAME


# TODO: change this filename if your CSV has a different name
TOSDR_FILE = RAW_DIR / "tosdr_policies_clean.csv"

# Mapping from ToS;DR string labels to ordinal severity
TOSDR_ORDINAL_MAP = {
    "good": 0,
    "neutral": 1,
    "bad": 2,
    "blocker": 3,
}


def load_raw_tosdr(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load the raw ToS;DR CSV into a pandas DataFrame and normalize columns.

    Parameters
    ----------
    csv_path : Path, optional
        If None, uses the default TOSDR_FILE under data/raw.

    Returns
    -------
    df : pd.DataFrame
        Columns:
        - text           : cleaned clause text
        - classification : original ToS;DR label string
    """
    if csv_path is None:
        csv_path = TOSDR_FILE

    if not csv_path.exists():
        raise FileNotFoundError(f"ToS;DR CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Try to find the text column: prefer 'text', otherwise 'excerpt'
    if "text" in df.columns:
        text_col = "text"
    elif "excerpt" in df.columns:
        text_col = "excerpt"
    else:
        raise ValueError(
            "Expected a 'text' or 'excerpt' column in ToS;DR CSV, "
            f"found: {list(df.columns)}"
        )

    # Keep only the columns we need
    if "classification" not in df.columns:
        raise ValueError(
            "Expected a 'classification' column in ToS;DR CSV "
            "(values like good/bad/blocker/neutral)."
        )

    df = df[[text_col, "classification"]].rename(
    columns={text_col: "text"}  # type: ignore[call-overload]
    )

    # Basic cleaning: strip whitespace, drop empty texts
    df["text"] = df["text"].astype(str).str.strip()
    df["classification"] = df["classification"].astype(str).str.lower().str.strip()

    df = df[(df["text"] != "") & df["classification"].isin(TOSDR_ORDINAL_MAP.keys())]
    df = df.reset_index(drop=True)

    return df


def prepare_tosdr_dataset(
    max_length: int = 256,
) -> Tuple[Dataset, AutoTokenizer]:
    """
    Load and preprocess ToS;DR into a tokenized Dataset.

    Parameters
    ----------
    max_length : int
        Max sequence length for tokenizer padding / truncation.

    Returns
    -------
    ds : datasets.Dataset
        Columns:
        - input_ids      : LongTensor [seq_len]
        - attention_mask : LongTensor [seq_len]
        - label_ordinal  : LongTensor [] in {0,1,2,3}
        - label_binary   : LongTensor [] in {0,1}
    tokenizer : AutoTokenizer
        Tokenizer corresponding to BASE_MODEL_NAME.
    """
    df = load_raw_tosdr()

    # Map classification → ordinal severity [0..3]
    df["label_ordinal"] = df["classification"].map(TOSDR_ORDINAL_MAP)

    # Binary unfair label: bad/blocker = 1, good/neutral = 0
    df["label_binary"] = df["label_ordinal"].apply(lambda x: 1 if x >= 2 else 0)

    # Convert to HF Dataset
    ds = Dataset.from_pandas(df[["text", "label_ordinal", "label_binary"]])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def tokenize_fn(example):
        enc = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["label_ordinal"] = int(example["label_ordinal"])
        enc["label_binary"] = int(example["label_binary"])
        return enc

    ds = ds.map(tokenize_fn, batched=False)

    # Set format for PyTorch
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label_ordinal", "label_binary"],
        output_all_columns=False,
    )

    return ds, tokenizer