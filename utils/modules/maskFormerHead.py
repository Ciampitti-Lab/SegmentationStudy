import torch.nn as nn
from torch import Tensor
import torch
from typing import Dict, Tuple
import torch.nn.functional as F

# ------------------------------
# MaskFormer Head (classification + mask projection)
# ------------------------------
class MaskFormerHead(nn.Module):
    """
    Semantic-only head:
      - class logits per query (B, Q, C)
      - query mask logits (B, Q, H, W)
      - aggregated per-class segmentation logits (B, C, H, W)
    """
    def __init__(self, embed_dim=256, num_queries=100, num_classes=21):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.class_embed = nn.Linear(embed_dim, num_classes)  # no background class
        self.mask_embed = nn.Linear(embed_dim, embed_dim)
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, decoder_output: Tensor, pixel_memory: Tensor, pixel_shape: Tuple[int,int]) -> Dict[str, Tensor]:
        B, Q, C = decoder_output.shape
        H, W = pixel_shape

        # Per-query classification
        logits_q = self.class_embed(decoder_output)  # (B, Q, C)

        # Per-query mask logits from dot(mask_embedding, pixel_features)
        mask_embeddings = self.refine(self.mask_embed(decoder_output))  # (B, Q, C)
        pixel_memory_t = pixel_memory.permute(0, 2, 1)  # (B, C, N)
        pred_masks = torch.einsum("bqc,bcn->bqn", mask_embeddings, pixel_memory_t)  # (B, Q, N)
        pred_masks = pred_masks.view(B, Q, H, W)  # (B, Q, H, W)

        # Aggregate queries into per-class segmentation logits
        class_probs = F.softmax(logits_q, dim=-1)                # (B, Q, C)
        seg_logits = torch.einsum("bqc,bqhw->bchw", class_probs, pred_masks)  # (B, C, H, W)

        return {"seg_logits": seg_logits}