"""
LegalBERT with an additional projection head and supervised contrastive loss.

TODO:
- Implement projection to contrastive space.
- Implement supervised contrastive loss with temperature.
"""

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from src.config import BASE_MODEL_NAME, CONTRASTIVE_TEMPERATURE


class ContrastiveLegalBert(nn.Module):
    def __init__(self, num_labels: int, proj_dim: int = 128):
        super().__init__()
        self.config = AutoConfig.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(BASE_MODEL_NAME, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.projection = nn.Linear(self.config.hidden_size, proj_dim)
        self.temperature = CONTRASTIVE_TEMPERATURE

    def forward(self, input_ids, attention_mask=None, labels: Optional[torch.Tensor] = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        z = self.projection(pooled)  # embeddings for contrastive loss
        z = nn.functional.normalize(z, dim=-1)

        # TODO: compute classification loss + supervised contrastive loss
        # and return both or their weighted sum.

        return {"logits": logits, "embeddings": z}
