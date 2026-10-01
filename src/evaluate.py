from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.data import eval_transform
from src.model import load_encoder


@torch.inference_mode()
def embed_paths(model, paths, transform, device):
    vectors = []
    for path in tqdm(paths, desc="embedding"):
        with Image.open(path) as opened:
            tensor = transform(opened.convert("RGB")).unsqueeze(0).to(device)
        vectors.append(model(tensor).cpu())
    return torch.cat(vectors)


def identification(gallery_frame, gallery_embeddings, query_frame, query_embeddings):
    # Average all gallery photographs into one normalized prototype per design.
    gallery_labels, prototypes = [], []
    for label, group in gallery_frame.groupby("design_id"):
        prototype = gallery_embeddings[group.index].mean(0)
        prototypes.append(torch.nn.functional.normalize(prototype, dim=0))
        gallery_labels.append(str(label))
    gallery_labels = np.asarray(gallery_labels)
    gallery = torch.stack(prototypes)
    common = query_frame["design_id"].astype(str).isin(set(gallery_labels)).to_numpy()
    queries = query_embeddings[common]
    query_labels = query_frame.loc[common, "design_id"].astype(str).to_numpy()
    if not len(queries):
        raise ValueError("Gallery and query CSVs contain no common design IDs")
    similarities = queries @ gallery.T
    ranking = similarities.argsort(dim=1, descending=True).numpy()
    recall_1 = np.mean(gallery_labels[ranking[:, 0]] == query_labels)
    k = min(5, len(gallery_labels))
    recall_5 = np.mean([label in gallery_labels[row[:k]] for label, row in zip(query_labels, ranking)])
    reciprocal = []
    for label, row in zip(query_labels, ranking):
        positions = np.flatnonzero(gallery_labels[row] == label)
        reciprocal.append(1.0 / (positions[0] + 1) if len(positions) else 0.0)
    return {"queries": len(queries), "gallery_designs": len(gallery), "recall_at_1": float(recall_1), "recall_at_5": float(recall_5), "mrr": float(np.mean(reciprocal))}


def verification(gallery_frame, gallery_embeddings, query_frame, query_embeddings, seed: int = 42):
    rng = np.random.default_rng(seed)
    positives, negatives = [], []
    gallery_groups = {str(k): v.index.to_numpy() for k, v in gallery_frame.groupby("design_id")}
    query_groups = {str(k): v.index.to_numpy() for k, v in query_frame.groupby("design_id")}
    labels = sorted(set(gallery_groups) & set(query_groups))
    for label in labels:
        combinations = [(a, b) for a in gallery_groups[label] for b in query_groups[label]]
        rng.shuffle(combinations)
        positives.extend(combinations[:5])
    if not positives or len(labels) < 2:
        raise ValueError("Verification requires at least two shared designs across gallery and query")
    while len(negatives) < len(positives):
        first, second = rng.choice(labels, 2, replace=False)
        negatives.append((rng.choice(gallery_groups[first]), rng.choice(query_groups[second])))
    pairs = positives + negatives
    targets = np.array([1] * len(positives) + [0] * len(negatives))
    scores = np.array([float(gallery_embeddings[a] @ query_embeddings[b]) for a, b in pairs])
    fpr, tpr, thresholds = roc_curve(targets, scores)
    eer_index = np.nanargmin(np.abs(fpr - (1.0 - tpr)))
    far_index = np.flatnonzero(fpr <= 0.01)
    tar_at_far = tpr[far_index[-1]] if len(far_index) else 0.0
    return {"pairs": len(pairs), "roc_auc": float(roc_auc_score(targets, scores)), "eer": float((fpr[eer_index] + 1.0 - tpr[eer_index]) / 2), "tar_at_far_1pct": float(tar_at_far), "suggested_threshold": float(thresholds[eer_index])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery-csv", required=True)
    parser.add_argument("--query-csv", required=True)
    parser.add_argument("--image-size", type=int, default=320)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery_frame = pd.read_csv(args.gallery_csv).reset_index(drop=True)
    query_frame = pd.read_csv(args.query_csv).reset_index(drop=True)
    gallery_frame["design_id"] = gallery_frame["design_id"].astype(str)
    query_frame["design_id"] = query_frame["design_id"].astype(str)
    model = load_encoder(args.checkpoint, device)
    transform = eval_transform(args.image_size)
    gallery_embeddings = embed_paths(model, gallery_frame.image_path, transform, device)
    query_embeddings = embed_paths(model, query_frame.image_path, transform, device)
    results = {
        "identification": identification(gallery_frame, gallery_embeddings, query_frame, query_embeddings),
        "verification": verification(gallery_frame, gallery_embeddings, query_frame, query_embeddings),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

