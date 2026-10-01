from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PaletteIntervention:
    """Strong color changes that preserve the motif's spatial structure."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            image = ImageOps.grayscale(image).convert("RGB")
        else:
            channels = list(image.split())
            random.shuffle(channels)
            image = Image.merge("RGB", channels)
            image = ImageEnhance.Color(image).enhance(random.uniform(0.1, 2.5))
            image = TF.adjust_hue(image, random.uniform(-0.5, 0.5))
        if random.random() < 0.25:
            image = ImageOps.posterize(image, random.choice((3, 4, 5)))
        return image


def train_transform(size: int, palette: bool = False):
    ops = [
        transforms.RandomResizedCrop(size, scale=(0.70, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ]
    if palette:
        ops += [PaletteIntervention()]
    else:
        ops += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.08)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ]
    return transforms.Compose(ops)


def eval_transform(size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.10)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SareeDataset(Dataset):
    def __init__(self, csv_path: str, size: int, train: bool = False) -> None:
        self.frame = pd.read_csv(csv_path)
        required = {"image_path", "design_id"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        self.frame["design_id"] = self.frame["design_id"].astype(str)
        designs = sorted(self.frame["design_id"].unique())
        self.class_to_idx = {name: i for i, name in enumerate(designs)}
        self.train = train
        self.first = train_transform(size, palette=False) if train else eval_transform(size)
        self.second = train_transform(size, palette=True) if train else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as opened:
            image = opened.convert("RGB")
        label = self.class_to_idx[str(row.design_id)]
        if self.train:
            return self.first(image), self.second(image), label, str(row.image_path)
        return self.first(image), label, str(row.image_path), str(row.design_id)


class PKSampler(Sampler[int]):
    def __init__(self, frame: pd.DataFrame, identities: int = 8, samples: int = 2) -> None:
        self.groups = {
            str(label): group.index.tolist()
            for label, group in frame.reset_index(drop=True).groupby("design_id")
        }
        self.identities = min(identities, len(self.groups))
        self.samples = samples
        self.length = max(len(frame), self.identities * self.samples)

    def __iter__(self):
        output = []
        labels = list(self.groups)
        while len(output) < self.length:
            chosen = random.sample(labels, self.identities)
            for label in chosen:
                indices = self.groups[label]
                output.extend(random.choices(indices, k=self.samples))
        return iter(output[: self.length])

    def __len__(self) -> int:
        return self.length

