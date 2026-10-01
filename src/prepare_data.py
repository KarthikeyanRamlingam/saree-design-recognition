from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(root: Path) -> pd.DataFrame:
    """Expected layout: root/design_id/colorway_id/image.jpg."""
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            raise ValueError(f"Expected design/colorway/image layout, got: {relative}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            print(f"Skipping corrupt image {path}: {exc}")
            continue
        rows.append(
            {
                "image_path": str(path.resolve()),
                "design_id": relative.parts[0],
                "colorway_id": relative.parts[1],
                "product_id": path.stem,
                "source": root.name,
                "sha256": image_sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No images found under {root}")
    duplicates = frame.duplicated("sha256", keep="first")
    if duplicates.any():
        print(f"Removing {int(duplicates.sum())} exact duplicate files")
        frame = frame.loc[~duplicates].reset_index(drop=True)
    return frame


def split_by_colorway(frame: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    insufficient = 0
    for _, design in frame.groupby("design_id"):
        colorways = design["colorway_id"].astype(str).unique().copy()
        rng.shuffle(colorways)
        if len(colorways) < 3:
            train_parts.append(design)
            insufficient += 1
            continue
        test_colorway, val_colorway = colorways[0], colorways[1]
        test_parts.append(design[design["colorway_id"].astype(str) == test_colorway])
        val_parts.append(design[design["colorway_id"].astype(str) == val_colorway])
        train_parts.append(design[~design["colorway_id"].astype(str).isin((test_colorway, val_colorway))])
    if insufficient:
        print(f"{insufficient} designs have fewer than 3 colorways and are train-only")
    if not val_parts or not test_parts:
        raise ValueError("Strict evaluation requires at least one design with 3+ colorways")
    return (
        pd.concat(train_parts).sample(frac=1, random_state=seed),
        pd.concat(val_parts).sample(frac=1, random_state=seed),
        pd.concat(test_parts).sample(frac=1, random_state=seed),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = scan(args.root)
    train, val, test = split_by_colorway(frame, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "manifest.csv", index=False)
    train.to_csv(args.output / "train.csv", index=False)
    val.to_csv(args.output / "val.csv", index=False)
    test.to_csv(args.output / "test.csv", index=False)
    print(f"Images: {len(frame)} | train={len(train)} val={len(val)} test={len(test)}")
    print(f"Designs: {frame.design_id.nunique()} | colorways={frame.colorway_id.nunique()}")


if __name__ == "__main__":
    main()
