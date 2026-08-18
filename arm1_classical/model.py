"""Arm 1 — frozen Swin-Tiny + frozen BERT (MuRIL), residual fusion head.

Both backbones are frozen (feature extractors only); only the fusion head
and the two classification heads (level1, level2) are trained.
"""
import timm
import torch
import torch.nn as nn
from transformers import AutoModel

TEXT_MODEL_NAME = "google/muril-base-cased"
VISION_MODEL_NAME = "swin_tiny_patch4_window7_224"
FUSION_DIM = 512


class ResidualFusionBlock(nn.Module):
    """fused = x + MLP(x), i.e. a residual refinement of the concatenated
    image+text projection — matches the "residual fusion" architecture
    named in instructions.md / the design review."""

    def __init__(self, dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class ClassicalBaseline(nn.Module):
    def __init__(self, n_level1=2, n_level2=3, fusion_dim=FUSION_DIM, n_fusion_blocks=2):
        super().__init__()

        self.vision = timm.create_model(VISION_MODEL_NAME, pretrained=True, num_classes=0)
        self.text = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.text.parameters():
            p.requires_grad = False
        self.vision.eval()
        self.text.eval()

        vision_dim = self.vision.num_features  # 768 for swin-tiny
        text_dim = self.text.config.hidden_size  # 768 for muril-base

        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.fusion_blocks = nn.ModuleList(
            [ResidualFusionBlock(fusion_dim) for _ in range(n_fusion_blocks)]
        )

        self.level1_head = nn.Linear(fusion_dim, n_level1)
        self.level2_head = nn.Linear(fusion_dim, n_level2)

    def train(self, mode=True):
        # keep frozen backbones in eval() (disables their dropout/BN updates)
        # even when the rest of the module is set to train()
        super().train(mode)
        self.vision.eval()
        self.text.eval()
        return self

    def forward(self, image, input_ids, attention_mask):
        with torch.no_grad():
            img_feat = self.vision(image)  # [B, vision_dim]
            text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = text_out.last_hidden_state[:, 0, :]  # [CLS], [B, text_dim]

        img_proj = self.vision_proj(img_feat)
        text_proj = self.text_proj(text_feat)

        fused = img_proj + text_proj  # residual: sum of both modality projections
        for block in self.fusion_blocks:
            fused = block(fused)

        return {
            "level1_logits": self.level1_head(fused),
            "level2_logits": self.level2_head(fused),
        }

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
