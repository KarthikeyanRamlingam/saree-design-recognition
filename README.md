# Color-Invariant Saree Design Recognition

An end-to-end PyTorch retrieval system that matches saree surface designs across different
color palettes. It supports identification (query-to-gallery ranking), verification
(same/different design), strict held-out-colorway evaluation, and a local Gradio demo.

## Why this implementation

The system uses a pretrained ConvNeXt-Tiny, GeM pooling, and 256-dimensional normalized
embeddings. ArcFace learns design identities, supervised contrastive loss pulls together
different examples of the same design, and a consistency loss aligns original images with
aggressively recolored views. Color is therefore treated as a nuisance variable while motif
geometry remains useful.

## 1. Prepare the images

Do not commit or redistribute the proprietary corpus. Arrange a working copy as:

```text
data/raw/
├── DESIGN_001/
│   ├── red_gold/
│   │   ├── product_001_a.jpg
│   │   └── product_001_b.jpg
│   ├── blue_silver/
│   │   └── product_014.jpg
│   └── green_gold/
│       └── product_208.jpg
└── DESIGN_002/
    └── ...
```

The first folder is the **exact design identity**, not a broad pattern category. The second
folder is the colorway. A strict train/validation/test result requires at least three colorways
for an evaluated design. Designs with fewer than three are retained for training only.

If filenames do not identify physical products, edit the generated `product_id` column before
training. Multiple crops/photos of one physical saree must share one `product_id`.

## 2. Install and validate

Use Python 3.10 or 3.11. On the 8 GB laptop, perform only data preparation and inference.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.prepare_data --root data/raw --output data/splits
```

The command rejects the wrong folder shape, skips corrupt images, removes byte-identical
duplicates, and creates deterministic held-out-colorway splits.

Inspect these before training:

```text
data/splits/manifest.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

## 3. Train on Kaggle

1. Create a Kaggle notebook with a GPU accelerator.
2. Add the private image dataset as notebook input.
3. Upload this repository as a Kaggle dataset or clone it from GitHub.
4. Set `DATA_ROOT` in `notebooks/kaggle_quickstart.ipynb`.
5. Run all cells.

Equivalent commands are:

```bash
pip install -q timm gradio
python -m src.prepare_data --root /kaggle/input/YOUR-DATASET/images --output data/splits
python -m src.train --config configs/base.yaml
python -m src.evaluate \
  --checkpoint outputs/convnext_tiny/best.pt \
  --gallery-csv data/splits/val.csv \
  --query-csv data/splits/test.csv
```

If Kaggle runs out of memory, change `batch_size` to 8 in `configs/base.yaml`. Do not reduce
image size until trying the smaller batch because motif detail matters.

## 4. Understand the split

For every design with at least three colorways:

- training receives all but two colorways;
- validation receives one unseen colorway;
- test receives another unseen colorway;
- validation images form the identification gallery;
- test images form the queries.

This means gallery and query palettes do not overlap for a design. Exact design identities are
shared, which is appropriate for closed-set identification. Verification is calculated from
test images only. For publication-quality verification, additionally create a design-disjoint
test split and select the decision threshold on validation data.

## 5. Run the demo

Download `best.pt` and the validation CSV/images from Kaggle, then run:

```powershell
python app.py --checkpoint outputs/convnext_tiny/best.pt `
  --gallery-csv data/splits/val.csv --threshold 0.50
```

Replace `0.50` with the threshold selected on validation data. Open the local URL printed by
Gradio. Gallery embeddings are cached next to the CSV after the first launch.

## 6. What to report

Report Recall@1, Recall@5, MRR, ROC-AUC, EER, and TAR at 1% FAR. Run at least these ablations:

| Model | Palette intervention | SupCon | Consistency |
|---|---:|---:|---:|
| Pretrained embedding baseline | No | No | No |
| ArcFace baseline | No | No | No |
| Color-invariant model | Yes | Yes | Yes |

Include successful matches and two failure groups: same design/different palette, and different
design/similar palette. Never tune a threshold or architectural choice on the test set.

## Reproducibility and disclosure

- Framework: PyTorch
- Backbone: `timm` ConvNeXt-Tiny with ImageNet pretrained weights
- Random seed: 42
- Input: 320×320 RGB
- Embedding: 256-D, L2 normalized
- Similarity: cosine similarity
- Proprietary images are ignored by Git and must be deleted after the exercise

