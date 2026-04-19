#from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from src.config import MODELS_DIR

from src.data.load_unfair_tos import prepare_unfair_tos_datasets
from src.models.baseline_legalbert import BaselineLegalBert


def collate_fn(batch):
    # datasets with set_format("torch") already give you tensors;
    # just stack into a dict for the model.
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
        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    # later: accumulate predictions and compute F1
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, tokenizer = prepare_unfair_tos_datasets(max_length=256)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    # DEBUG: use tiny subsets so training is fast
    train_ds_small = train_ds.select(range(700))   # first 2000 examples (sucess with 500, but let's try a bit more)
    val_ds_small = val_ds.select(range(150))   # first 500 examples (success with 100, but let's try a bit more)

    batch_size = 6          # instead of 16 (used this to save CPU memory, but let's try a bit more)
    #num_epochs = 1          # instead of 3

    train_loader = DataLoader(train_ds_small, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds_small, batch_size=batch_size , shuffle=False, collate_fn=collate_fn)

    model = BaselineLegalBert(num_labels=8, use_binary_head=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 2
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )


    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # later: save model, compute F1 etc.

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "baseline_legal_bert.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()