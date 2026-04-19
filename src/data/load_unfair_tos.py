"""
Data loading and preprocessing for the UNFAIR-ToS task (LexGLUE).

- Uses Hugging Face `datasets` to load the `lex_glue`, `unfair_tos` config.
- Each example has:
    text        : clause text (string)
    labels      : list of 8 integers (0/1) for each unfair clause type
    label_binary: 0 if all labels==0, 1 if at least one unfair type

- `prepare_unfair_tos_datasets` additionally tokenizes with LegalBERT
  and sets the format to PyTorch tensors ready for DataLoader.
"""

from typing import Tuple, Dict

import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.config import BASE_MODEL_NAME


UNFAIR_TOS_CONFIG = "unfair_tos"
NUM_LABELS = 8  # limitation of liability, unilateral termination, ..., arbitration


def load_unfair_tos_raw() -> DatasetDict:
    """
    Download and return the raw UNFAIR-ToS splits from LexGLUE.

    Returns
    -------
    ds : DatasetDict
        Keys: "train", "validation", "test"
        Each split has columns:
        - 'text'   : clause text
        - 'labels' : list[int] length 8 with 0/1 flags
    """
    ds = load_dataset("lex_glue", UNFAIR_TOS_CONFIG)
    return ds


def _add_binary_label(example: Dict) -> Dict:
    """
    Add a binary label: 1 if any unfair tag is present, else 0.
    """
    labels = example["labels"]
    # labels is list[int] length 8
    example["label_binary"] = int(any(labels))
    return example


def prepare_unfair_tos_datasets(
    max_length: int = 256,
) -> Tuple[DatasetDict, AutoTokenizer]:
    """
    Load UNFAIR-ToS, add binary labels, tokenize with LegalBERT,
    and set format to PyTorch tensors.

    Parameters
    ----------
    max_length : int
        Max sequence length for tokenizer padding / truncation.

    Returns
    -------
    tokenized_ds : DatasetDict
        Splits "train", "validation", "test" with columns:
        - input_ids        : LongTensor [seq_len]
        - attention_mask   : LongTensor [seq_len]
        - labels           : FloatTensor [8] (multi-label, 0/1)
        - label_binary     : LongTensor []  (0/1)
    tokenizer : AutoTokenizer
        Tokenizer corresponding to BASE_MODEL_NAME.
    """
    ds = load_unfair_tos_raw()

    # 1) add binary label to all splits
    ds = ds.map(_add_binary_label)

    # 2) load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def tokenize_fn(example: Dict) -> Dict:
        enc = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # example["labels"] is a list of indices, e.g. [] or [0, 3]
        label_ids = example["labels"]

        multi_hot = [0.0] * NUM_LABELS
        for idx in label_ids:
            if isinstance(idx, int) and 0 <= idx < NUM_LABELS:
                multi_hot[idx] = 1.0

        enc["labels"] = multi_hot
        enc["label_binary"] = int(len(label_ids) > 0)

        return enc

    tokenized_ds = ds.map(tokenize_fn, batched=False)

    # 3) set PyTorch format
    cols = ["input_ids", "attention_mask", "labels", "label_binary"]
    for split in tokenized_ds.keys():
        tokenized_ds[split].set_format(
            type="torch",
            columns=cols,
            output_all_columns=False,
        )
        # Optional: light check but more robust
        # first = tokenized_ds[split][0]["labels"]
        # print(split, first.shape)

    return tokenized_ds, tokenizer