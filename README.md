# Retinal Fundus Dataset Distributional Analysis Toolkit

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/mousamoradi/Anomaly_detection)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible, open-source toolkit for **distributional analysis of publicly available retinal fundus photograph datasets** using [RETFound](https://github.com/rmaphoh/RETFound_MAE) (ViT-Large, 1024-D features). The toolkit covers 29 datasets (~140,000 images) and provides feature extraction, diversity scoring, KL divergence analysis, dimensionality reduction visualisation (t-SNE / UMAP), and an interactive Flask/HuggingFace web application for querying new images against the reference collection.

---

## 🔬 Overview

This codebase accompanies the paper:

> **Distributional Analysis of Publicly Available Retinal Fundus Photograph Datasets Using RETFound**  
> Mousa Moradi, Nazlee Zebardast et al. — Massachusetts Eye and Ear / Harvard Medical School

The repository enables you to:
- Extract 1024-D CLS-token features from fundus images using the RETFound ViT-Large backbone
- Compute per-dataset diversity metrics (effective rank, pairwise cosine distance, KL divergence, composite diversity score)
- Visualise the feature space with t-SNE and UMAP
- Compare any new dataset (e.g. an institutional cohort) against the 29 reference datasets using Mahalanobis distance and symmetric KL divergence
- Query individual images through an interactive web app deployed on HuggingFace Spaces

---

## 📁 Repository Structure

```
├── retfound_extractor.py       # RETFound feature extraction (ViT-Large CLS token)
├── precompute_stats.py         # Fit shared PCA, compute per-dataset Gaussian stats
├── build_embeddings.py         # Run t-SNE + UMAP and save embeddings.pkl
├── app.py                      # Flask web application
├── diversity_analysis.py       # Per-dataset diversity metrics + 5 publication figures
├── tsne_umap_visualization.py  # t-SNE / UMAP side-by-side plots
├── dataset_distribution.py     # Dataset image count distribution figure
├── mee_vs_datasets.py          # Institutional dataset (MEE) vs reference analysis
├── templates/
│   └── index.html              # Web app front-end
├── static/
│   └── precomputed/            # shared_pca.pkl, dataset_stats.pkl, embeddings.pkl
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Mousamoradi/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Core dependencies:**
```
torch torchvision timm
scikit-learn numpy pandas matplotlib seaborn
openTSNE umap-learn
flask pillow
adjustText
```

For GPU-accelerated t-SNE / UMAP (optional, recommended for large datasets):
```bash
pip install cuml-cu11   # or cuml-cu12 depending on your CUDA version
```

---

## 🚀 Usage

### 1. Feature Extraction

Place your RETFound checkpoint at the path defined in `retfound_extractor.py` (or update `RETFOUND_CKPT`). Download the weights from the [RETFound repository](https://github.com/rmaphoh/RETFound_MAE).

```python
from retfound_extractor import extract_features
from PIL import Image

images = [Image.open("fundus_01.jpg"), Image.open("fundus_02.jpg")]
features = extract_features(images)   # (N, 1024) float32
```

### 2. Precompute Shared Statistics

Run once before launching the web app or diversity analysis:

```bash
python precompute_stats.py
python build_embeddings.py   # GPU recommended: CUDA_VISIBLE_DEVICES=1 python build_embeddings.py
```

### 3. Diversity Analysis

```bash
python diversity_analysis.py
```

Produces 5 publication-quality figures in `Outputs/Diversity/`:
- `diversity_ranking.png` — composite diversity score bar chart
- `diversity_heatmap.png` — normalized metrics heatmap
- `diversity_scatter.png` — effective rank vs cosine distance scatter
- `pca_variance_curves.png` — cumulative PCA explained variance per dataset
- `kl_divergence_heatmap.png` — pairwise symmetric KL divergence matrix

### 4. t-SNE / UMAP Visualisation

```bash
CUDA_VISIBLE_DEVICES=1 python tsne_umap_visualization.py
```

### 5. Institutional Dataset Comparison (MEE example)

```bash
python mee_vs_datasets.py
```

Produces 4 figures comparing your dataset against all 29 references:
- UMAP embedding with closest-dataset annotation
- Mahalanobis distance ranking
- Symmetric KL divergence ranking
- Violin plot of per-sample distance distributions

### 6. Web Application

```bash
python app.py   # runs on http://0.0.0.0:5000
```

Or visit the live HuggingFace Space: [mousamoradi/Anomaly_detection](https://huggingface.co/spaces/mousamoradi/Anomaly_detection)

---

## 📊 Datasets

The toolkit was evaluated on 29 publicly available retinal fundus datasets including AIROGS, ODIR, APTOS2019, REFUGE, DRIVE, STARE, CHASEDB1, and others. Each dataset's `.pkl` feature file follows the naming convention:

```
SavedFeatures_Retfound_<DatasetName>.pkl
```

with the internal structure:
```python
{'Features': np.ndarray}   # shape (N, 1024), dtype float32
```

---

## 🌐 HuggingFace Deployment

The web app is deployed at [mousamoradi/Anomaly_detection](https://huggingface.co/spaces/mousamoradi/Anomaly_detection). Pre-computed embeddings (float16, ~55 MB) are stored via HuggingFace LFS. To redeploy:

```bash
# Slim embeddings to float16 before pushing (required for LFS limits)
import numpy as np, pickle
with open('embeddings.pkl', 'rb') as f: d = pickle.load(f)
d['emb_umap'] = d['emb_umap'].astype(np.float16)
with open('embeddings.pkl', 'wb') as f: pickle.dump(d, f, protocol=4)
```

---

## 📄 Citation

If you use this toolkit in your research, please cite:

```bibtex
TBD
```

Also cite the original RETFound paper:
```bibtex
@article{zhou2023retfound,
  title   = {A foundation model for generalizable disease detection from retinal images},
  author  = {Zhou, Yukun and others},
  journal = {Nature},
  year    = {2023}
}
```

---

## 🤝 Acknowledgements

- [RETFound](https://github.com/rmaphoh/RETFound_MAE) — Zhou et al., Nature 2023
- Massachusetts Eye and Ear Infirmary / Harvard Medical School
- Mass General Brigham

---

## 📬 Contact

**Mousa Moradi**  
Postdoctoral Research Fellow, Schepens Eye Research Institute  
Massachusetts Eye and Ear / Harvard Medical School  
mmoradi2@meei.harvard.edu  
GitHub: [Mousamoradi](https://github.com/Mousamoradi)  
HuggingFace: [mousamoradi](https://huggingface.co/mousamoradi)
