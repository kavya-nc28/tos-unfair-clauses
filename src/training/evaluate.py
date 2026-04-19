# src/training/evaluate.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.config import BASE_MODEL_NAME, BATCH_SIZE, MODELS_DIR
from src.data.load_unfair_tos import prepare_unfair_tos_datasets
from src.models.baseline_legalbert import BaselineLegalBert
from src.training.train_baseline import collate_fn  # reuse exact same batching


NUM_LABELS = 8
MAX_LENGTH = 256


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_checkpoint(checkpoint_path: Path, device: torch.device) -> None:
    print(f"Loading model from {checkpoint_path}")

    # model must match the one used in training
    model = BaselineLegalBert(num_labels=NUM_LABELS, use_binary_head=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Data – same preprocessing as training
    ds, tokenizer = prepare_unfair_tos_datasets(max_length=MAX_LENGTH)
    test_ds = ds["test"]

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_logits = []
    all_labels = []
    all_label_binary = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_binary = batch["label_binary"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                label_binary=None,
            )
            # BaselineLegalBert returns dict-like; logits are under "logits"
            logits = outputs["logits"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_label_binary.append(label_binary.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_true_bin = torch.cat(all_label_binary, dim=0).numpy()

    probs = sigmoid(logits)
    y_pred = (probs >= 0.5).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    probs_bin = probs.max(axis=1)
    y_pred_bin = (probs_bin >= 0.5).astype(int)

    f1_bin = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        roc_auc_bin = roc_auc_score(y_true_bin, probs_bin)
    except ValueError:
        roc_auc_bin = float("nan")
    try:
        ap_bin = average_precision_score(y_true_bin, probs_bin)
    except ValueError:
        ap_bin = float("nan")

    print("\n=== UNFAIR-ToS test metrics ===")
    print(f"Multi-label macro F1: {macro_f1:.4f}")
    print(f"Multi-label micro F1: {micro_f1:.4f}")
    print(f"Binary unfair-vs-fair F1: {f1_bin:.4f}")
    print(f"Binary ROC-AUC: {roc_auc_bin:.4f}")
    print(f"Binary PR-AUC: {ap_bin:.4f}")
    print("================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline LegalBERT on UNFAIR-ToS test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "baseline_legal_bert.pt"),
        help=(
            "Path to the trained model .pt file "
            "(must match what train_baseline.py saved)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Make sure train_baseline.py saved a model with this path, "
            "or pass --checkpoint explicitly."
        )

    evaluate_checkpoint(checkpoint_path, device)


if __name__ == "__main__":
    main()