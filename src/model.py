import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import MODEL_NAME, NUM_CLASSES


class RoBERTa(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.output = nn.Linear(
            self.encoder.config.hidden_size,
            NUM_CLASSES
        )

    def embedding_with_masks(self, token_emb, mask):
        mask = mask.unsqueeze(-1).expand(token_emb.size()).float()

        sum_emb = torch.sum(token_emb * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        res = sum_emb / sum_mask
        return res

    def forward(self, x, mask):
        out = self.encoder(x, attention_mask=mask)

        hidden = out.last_hidden_state
        pooled = self.embedding_with_masks(hidden, mask)

        logits = self.output(pooled)
        return logits