from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from src.config import BASE_MODEL_NAME


class BaselineLegalBert(nn.Module):
    def __init__(self, num_labels: int = 8, use_binary_head: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(BASE_MODEL_NAME, config=self.config)
        self.dropout = nn.Dropout(0.1)

        # Multi-label head for 8 unfair types
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Optional extra binary unfair/fair head
        self.use_binary_head = use_binary_head
        if self.use_binary_head:
            self.binary_classifier = nn.Linear(self.config.hidden_size, 1)

        self.loss_fct_multi = nn.BCEWithLogitsLoss()
        if self.use_binary_head:
            self.loss_fct_binary = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,          # shape [bs, 8]
        label_binary: Optional[torch.Tensor] = None,    # shape [bs]
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)  # [bs, 8]
        logits_binary = None
        if self.use_binary_head:
            logits_binary = self.binary_classifier(pooled).squeeze(-1)  # [bs]

        loss = None
        if labels is not None:
            # ensure float for BCEWithLogitsLoss
            labels = labels.float()
            loss_multi = self.loss_fct_multi(logits, labels)

            if self.use_binary_head and label_binary is not None:
                label_binary = label_binary.float()
                loss_bin = self.loss_fct_binary(logits_binary, label_binary)
                loss = loss_multi + loss_bin  # equal weight for baseline
            else:
                loss = loss_multi

        return {
            "logits": logits,
            "logits_binary": logits_binary,
            "loss": loss,
        }