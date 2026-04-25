
"""
LegalBERT with projection head and supervised contrastive loss.
Positives: same unfair type
Negatives: different unfair type or fair vs unfair (hard negatives)
"""

from typing import Optional
import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from src.config import BASE_MODEL_NAME, CONTRASTIVE_TEMPERATURE


class ContrastiveLegalBert(nn.Module):
    def __init__(self, num_labels: int = 8, proj_dim: int = 128,
                 lambda_cls: float = 1.0, lambda_con: float = 0.5):
        super().__init__()
        self.config = AutoConfig.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(BASE_MODEL_NAME, config=self.config)
        self.dropout = nn.Dropout(0.1)

        # Classification head — same as baseline
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Projection head for contrastive loss
        # 768 -> 256 -> proj_dim as professor suggested
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

        self.temperature  = CONTRASTIVE_TEMPERATURE
        self.lambda_cls   = lambda_cls
        self.lambda_con   = lambda_con
        self.loss_fct     = nn.BCEWithLogitsLoss()

    def _supervised_contrastive_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Supervised contrastive loss.
        Positive pairs  : two clauses sharing at least one unfair label type
        Negative pairs  : fair vs unfair  OR  two clauses with NO shared label
        Hard negatives  : two unfair clauses from completely different categories
        """
        # z      : (B, proj_dim) already L2-normalised
        # labels : (B, num_labels) multi-hot float

        B = z.size(0)
        device = z.device

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(z, z.T) / self.temperature          # (B, B)

        # Positive mask: at least one label in common AND not self
        label_overlap = (labels @ labels.T) > 0                # (B, B) bool
        mask_self     = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask      = label_overlap & mask_self               # (B, B)

        # If no positives exist in this batch skip contrastive term
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Log-softmax over all non-self pairs
        exp_sim  = torch.exp(sim) * mask_self                   # zero out diagonal
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        loss = -(log_prob * pos_mask).sum() / pos_mask.sum()
        return loss

    def forward(self, input_ids, attention_mask=None,
                labels: Optional[torch.Tensor] = None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state[:, 0]   # CLS token
        pooled  = self.dropout(pooled)

        logits  = self.classifier(pooled)            # (B, num_labels)
        z       = self.projection(pooled)            # (B, proj_dim)
        z       = nn.functional.normalize(z, dim=-1)

        loss = None
        if labels is not None:
            labels   = labels.float()
            cls_loss = self.loss_fct(logits, labels)
            con_loss = self._supervised_contrastive_loss(z, labels)
            loss     = self.lambda_cls * cls_loss + self.lambda_con * con_loss

        return {"logits": logits, "embeddings": z, "loss": loss}
