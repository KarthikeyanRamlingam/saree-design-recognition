from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PKSampler, SareeDataset
from src.losses import invariance_loss, supervised_contrastive_loss
from src.model import ArcMarginHead, SareeEncoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def retrieval_recall_at_1(model, loader, device) -> float:
    model.eval()
    vectors, labels = [], []
    for images, batch_labels, _, _ in loader:
        vectors.append(model(images.to(device)).cpu())
        labels.append(batch_labels)
    vectors = torch.cat(vectors)
    labels = torch.cat(labels)
    if len(vectors) < 2:
        return 0.0
    similarities = vectors @ vectors.T
    similarities.fill_diagonal_(-float("inf"))
    nearest = similarities.argmax(1)
    valid = torch.stack([(labels == label).sum() > 1 for label in labels])
    return float((labels[nearest][valid] == labels[valid]).float().mean()) if valid.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["train"]
    train_set = SareeDataset(data_cfg["train_csv"], data_cfg["image_size"], train=True)
    val_set = SareeDataset(data_cfg["val_csv"], data_cfg["image_size"], train=False)
    sampler = PKSampler(train_set.frame, identities=8, samples=2)
    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], sampler=sampler,
        num_workers=data_cfg["workers"], pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["workers"], pin_memory=device.type == "cuda",
    )

    encoder = SareeEncoder(**model_cfg).to(device)
    head = ArcMarginHead(
        model_cfg["embedding_dim"], len(train_set.class_to_idx),
        train_cfg["arc_scale"], train_cfg["arc_margin"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.backbone.parameters(), "lr": train_cfg["backbone_lr"]},
            {"params": encoder.pool.parameters(), "lr": train_cfg["head_lr"]},
            {"params": encoder.projection.parameters(), "lr": train_cfg["head_lr"]},
            {"params": head.parameters(), "lr": train_cfg["head_lr"]},
        ],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"] and device.type == "cuda")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []

    for epoch in range(1, train_cfg["epochs"] + 1):
        encoder.train()
        head.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for original, recolored, labels, _ in progress:
            original, recolored, labels = original.to(device), recolored.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                first = encoder(original)
                second = encoder(recolored)
                arc = F.cross_entropy(head(first, labels), labels)
                embeddings = torch.cat((first, second))
                repeated_labels = torch.cat((labels, labels))
                supcon = supervised_contrastive_loss(
                    embeddings, repeated_labels, train_cfg["temperature"]
                )
                consistent = invariance_loss(first, second)
                loss = arc + train_cfg["supcon_weight"] * supcon + train_cfg["invariance_weight"] * consistent
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        recall = retrieval_recall_at_1(encoder, val_loader, device)
        record = {"epoch": epoch, "loss": running / max(1, len(train_loader)), "val_recall_at_1": recall}
        history.append(record)
        print(json.dumps(record))
        if recall > best:
            best, stale = recall, 0
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "model_config": {
                        "backbone": model_cfg["backbone"],
                        "embedding_dim": model_cfg["embedding_dim"],
                    },
                    "class_to_idx": train_set.class_to_idx,
                    "best_val_recall_at_1": best,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= train_cfg["patience"]:
                print("Early stopping")
                break
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation Recall@1: {best:.4f}; checkpoint: {output / 'best.pt'}")


if __name__ == "__main__":
    main()

