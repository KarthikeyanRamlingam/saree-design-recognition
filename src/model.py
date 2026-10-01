from __future__ import annotations

import math

import timm
import torch
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class SareeEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        embedding_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )
        feature_dim = self.backbone.num_features
        self.pool = GeM()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return F.normalize(self.projection(self.pool(features)), dim=1)


class ArcMarginHead(nn.Module):
    def __init__(
        self, embedding_dim: int, classes: int, scale: float = 30.0, margin: float = 0.3
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)


def load_encoder(checkpoint: str, device: torch.device) -> SareeEncoder:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SareeEncoder(**payload["model_config"], pretrained=False)
    model.load_state_dict(payload["encoder"])
    return model.to(device).eval()

