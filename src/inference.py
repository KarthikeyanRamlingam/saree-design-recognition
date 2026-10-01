from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data import eval_transform
from src.model import load_encoder


class SareeSearch:
    def __init__(self, checkpoint: str, gallery_csv: str, image_size: int = 320) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_encoder(checkpoint, self.device)
        self.frame = pd.read_csv(gallery_csv)
        self.transform = eval_transform(image_size)
        cache = Path(gallery_csv).with_suffix(".embeddings.pt")
        if cache.exists():
            self.embeddings = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            self.embeddings = torch.cat([self.embed(path) for path in self.frame.image_path])
            torch.save(self.embeddings, cache)

    @torch.inference_mode()
    def embed(self, image_or_path) -> torch.Tensor:
        if isinstance(image_or_path, (str, Path)):
            with Image.open(image_or_path) as opened:
                image = opened.convert("RGB")
        else:
            image = image_or_path.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(tensor).cpu()

    def search(self, image: Image.Image, top_k: int = 5):
        query = self.embed(image)
        scores = (query @ self.embeddings.T).squeeze(0)
        values, indices = scores.topk(min(top_k, len(scores)))
        return [
            {**self.frame.iloc[index].to_dict(), "similarity": float(score)}
            for score, index in zip(values, indices)
        ]

    def verify(self, first: Image.Image, second: Image.Image, threshold: float):
        score = float(self.embed(first) @ self.embed(second).T)
        return {"same_design": score >= threshold, "similarity": score, "threshold": threshold}

