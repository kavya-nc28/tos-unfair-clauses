"""
Baseline LegalBERT classifier for unfair clause detection.

TODO:
- Implement a simple sequence classification head on top of LegalBERT.
- Support binary (unfair vs fair) and/or multi-label for 8 unfair types.
"""

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from src.config import BASE_MODEL_NAME


class BaselineLegalBert(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(BASE_MODEL_NAME, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels: Optional[torch.Tensor] = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            # TODO: choose BCEWithLogitsLoss (multi-label) or CrossEntropyLoss (single-label)
            raise NotImplementedError("Add loss computation here.")

        return {"logits": logits, "loss": loss}
