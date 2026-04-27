from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing import cast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from src.config import MODELS_DIR, BATCH_SIZE
from src.data.load_unfair_tos import prepare_unfair_tos_datasets
from src.models.contrastive_legalbert import ContrastiveLegalBert
from src.training.train_contrastive import collate_fn


NUM_LABELS = 8
MAX_LENGTH = 256


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = ContrastiveLegalBert(num_labels=NUM_LABELS)
    state_dict = torch.load(
        MODELS_DIR / "contrastive_legal_bert.pt", map_location=device
    )
    model.load_state_dict(state_dict, strict =False)
    model.to(device)
    model.eval()

    # Load data
    ds, _ = prepare_unfair_tos_datasets(max_length=MAX_LENGTH)
    test_loader = DataLoader(
        cast(TorchDataset, ds["test"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load multi-label threshold
    threshold = 0.15
    binary_threshold = 0.45
    threshold_path = MODELS_DIR / "contrastive_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
            threshold = data["threshold"]
            binary_threshold = data.get("binary_threshold", 0.45)
    print(f"Using multi-label threshold: {threshold}")

    # Run inference
    all_logits        = []
    all_labels        = []
    all_label_binary  = []
    all_binary_logits = []   # binary head logits

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_logits.append(outputs["logits"].cpu())
            all_binary_logits.append(outputs["logits_binary"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_label_binary.append(batch["label_binary"].cpu())

    logits     = torch.cat(all_logits).numpy()
    y_true     = torch.cat(all_labels).numpy()
    y_true_bin = torch.cat(all_label_binary).numpy()

    # Multi-label predictions
    probs  = sigmoid(logits)
    y_pred = (probs >= threshold).astype(int)

    # Binary predictions using binary head logits
    binary_logits    = torch.cat(all_binary_logits).numpy().reshape(-1)
    probs_bin        = 1.0 / (1.0 + np.exp(-binary_logits))
    y_pred_bin       = (probs_bin >= binary_threshold).astype(int)

    # Compute metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_bin   = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_bin, probs_bin)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true_bin, probs_bin)
    except ValueError:
        pr_auc = float("nan")

    print("\n=== Contrastive Model Test Metrics ===")
    print(f"Multi-label Macro F1 : {macro_f1:.4f}")
    print(f"Multi-label Micro F1 : {micro_f1:.4f}")
    print(f"Binary F1            : {f1_bin:.4f}")
    print(f"Binary ROC-AUC       : {roc_auc:.4f}")
    print(f"Binary PR-AUC        : {pr_auc:.4f}")
    print("\n=======================================\n")

    # Save results
    results = {
        "macro_f1":         round(float(macro_f1), 4),
        "micro_f1":         round(float(micro_f1), 4),
        "binary_f1":        round(float(f1_bin),   4),
        "roc_auc":          round(float(roc_auc),  4),
        "pr_auc":           round(float(pr_auc),   4),
        "threshold":        threshold,
        "binary_threshold": binary_threshold,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "contrastive_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → reports/contrastive_metrics.json")

if __name__ == "__main__":
    main()