import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from src.config import MODELS_DIR
from src.data.load_unfair_tos import prepare_unfair_tos_datasets
from src.models.baseline_legalbert import BaselineLegalBert


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "label_binary": torch.stack([b["label_binary"] for b in batch]),
    }


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs["loss"].item()
    return total_loss / len(loader)


def find_best_threshold(model, loader, device):
    model.eval()
    all_probs, all_y = [], []

    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"].detach().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            all_probs.append(probs)
            all_y.append(y)

    all_probs = np.concatenate(all_probs, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (all_probs >= thr).astype(int)
        f1 = f1_score(all_y, preds, average="micro", zero_division=0)
        print(f"thr={thr:.2f}  micro_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"Best threshold: {best_thr:.2f} with micro F1={best_f1:.4f}")
    return best_thr, best_f1

def find_best_binary_threshold(model, loader, device):
    model.eval()
    all_probs_bin, all_y_bin = [], []

    with torch.no_grad():
        for batch in loader:
            y_bin = batch["label_binary"].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits_bin = outputs["logits_binary"].detach().cpu().numpy()
            probs_bin = 1 / (1 + np.exp(-logits_bin))
            all_probs_bin.append(probs_bin)
            all_y_bin.append(y_bin)

    all_probs_bin = np.concatenate(all_probs_bin)
    all_y_bin = np.concatenate(all_y_bin)

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = (all_probs_bin >= thr).astype(int)
        f1 = f1_score(all_y_bin, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"Best binary threshold: {best_thr:.2f} with F1={best_f1:.4f}")
    return best_thr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds, tokenizer = prepare_unfair_tos_datasets(max_length=256)
    train_ds = ds["train"]
    all_labels = np.array(train_ds["labels"])
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(train_ds) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float)
    pos_weight = torch.clamp(pos_weight, max=10.0)
    print("pos_weight:", pos_weight)

    val_ds = ds["validation"]

    batch_size = 16
    num_epochs = 3

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BaselineLegalBert(num_labels=8, use_binary_head=True, pos_weight=pos_weight)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    save_path = MODELS_DIR / "baseline_legal_bert.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f" Best model saved (val_loss={val_loss:.4f})")

    best_thr_multi, _ = find_best_threshold(model, val_loader, device)
    best_thr_binary = find_best_binary_threshold(model, val_loader, device)

    print(f"Chosen multi threshold from validation: {best_thr_multi:.2f}")
    print(f"Chosen binary threshold from validation: {best_thr_binary:.2f}")

    threshold_path = MODELS_DIR / "baseline_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold": float(best_thr_multi),
            "binary_threshold": float(best_thr_binary)
        }, f)
    print(f"Saved threshold to {threshold_path}")



if __name__ == "__main__":
    main()