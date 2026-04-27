"""
Training script for ContrastiveLegalBert.
- Same data pipeline as baseline for fair comparison
- pos_weight computed exactly like train_baseline.py
- Combined classification + contrastive + binary loss
"""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from src.config import (
    MODELS_DIR,
    CONTRASTIVE_EPOCHS,
    CONTRASTIVE_LR,
    BATCH_SIZE,
    SEED,
)
from src.data.load_unfair_tos import prepare_unfair_tos_datasets
from src.models.contrastive_legalbert import ContrastiveLegalBert

torch.manual_seed(SEED)


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
        "labels":         torch.stack([b["labels"]          for b in batch]).float(),
        "label_binary":   torch.stack([b["label_binary"]    for b in batch]),
    }


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs["loss"].item()
    return total_loss / len(loader)


@torch.no_grad()
def find_best_threshold(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].cpu().numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs["logits"].cpu().numpy()
        probs   = 1 / (1 + np.exp(-logits))
        all_probs.append(probs)
        all_y.append(labels)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (all_probs >= thr).astype(int)
        f1    = f1_score(all_y, preds, average="micro", zero_division=0)
        print(f"  thr={thr:.2f}  micro_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"Best threshold: {best_thr:.2f}  micro_F1={best_f1:.4f}")
    return float(best_thr), float(best_f1)

# after line 107, ADD this entire new function:
@torch.no_grad()
def find_best_binary_threshold(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_bin     = batch["label_binary"].cpu().numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs["binary_logits"].cpu().numpy().squeeze()
        probs   = 1 / (1 + np.exp(-logits))
        all_probs.append(probs)
        all_y.append(labels_bin)

    all_probs = np.concatenate(all_probs)
    all_y     = np.concatenate(all_y)

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (all_probs >= thr).astype(int)
        f1    = f1_score(all_y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"Best binary threshold: {best_thr:.2f}  F1={best_f1:.4f}")
    return float(best_thr)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds, _ = prepare_unfair_tos_datasets(max_length=256)

    train_loader = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute capped pos_weight exactly like train_baseline.py
    all_labels = np.array(ds["train"]["labels"])
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(ds["train"]) - pos_counts
    pos_weight = torch.clamp(
        torch.tensor(
            neg_counts / np.clip(pos_counts, 1, None),
            dtype=torch.float
        ),
        max=10.0
    ).to(device)
    print("pos_weight (capped):", pos_weight)

    model = ContrastiveLegalBert(
        num_labels=8,
        proj_dim=128,
        lambda_cls=1.0,
        lambda_con=0.5,
        pos_weight=pos_weight,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONTRASTIVE_LR)
    num_training_steps = CONTRASTIVE_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting contrastive training...")
    best_val_loss = float("inf")
    for epoch in range(1, CONTRASTIVE_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss   = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{CONTRASTIVE_EPOCHS}: "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "contrastive_legal_bert.pt")
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            
    print("Loading best checkpoint for threshold tuning...")
    model.load_state_dict(torch.load(MODELS_DIR / "contrastive_legal_bert.pt", map_location=device), strict=False)
    model.eval()

    print("\nFinding best threshold on validation set...")
    best_thr, best_f1 = find_best_threshold(model, val_loader, device)
    best_bin_thr = find_best_binary_threshold(model, val_loader, device)

    with open(MODELS_DIR / "contrastive_threshold.json", "w") as f:
        json.dump({"threshold": best_thr, "binary_threshold":  best_bin_thr, "best_val_micro_f1": best_f1}, f, indent=2)
    print(f"Saved threshold → {MODELS_DIR}/contrastive_threshold.json")

if __name__ == "__main__":
    main()