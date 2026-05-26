# Shifting Cultivation Mapping

Code for the paper:
> **Revealing the Widespread Mosaic of Shifting Cultivation Across the Tropics**
> *[Author list, Journal, Year]*

---

## Overview

This repository contains all code needed to reproduce the shifting cultivation map, accuracy assessment, and analyses presented in the paper. The pipeline has three stages:

1. **Model training** — train the EfficientNet-B1 classifier on labelled PlanetScope patches
2. **Prediction** — run inference over NICFI basemap tiles to produce the pantropical map
3. **Analysis** — accuracy assessment, country statistics, and environmental correlates

---

## Repository structure

```
sc-mapping/
├── main.py                          Training entry point
├── predict.py                       Inference entry point
├── requirements.txt
├── configs/
│   ├── sc_classification.yaml       Training configuration
│   └── sc_prediction.yaml           Inference configuration
├── model/
│   ├── solver.py                    Model architecture, training and validation loops
│   ├── trainer.py                   Data split management and Solver orchestration
│   └── predictor.py                 TTA inference and two-date fusion
├── data/
│   ├── data_loader.py               Training / validation DataLoader
│   └── pred_loader.py               Sliding-window inference DataLoader
├── utils/
│   ├── preprocessing.py             Per-channel CLAHE equalisation (equalize)
│   ├── model_checkpoint.py          Best-model checkpoint saving
│   └── wandb_utils.py               Optional Weights & Biases logging
└── analysis/
    ├── validation/
    │   ├── gee/
    │   │   └── stratified_sampling.js   GEE stratified sample design
    │   └── validation.ipynb             Olofsson et al. accuracy assessment
    └── notebooks/
        ├── country_statistic_with_uncertainty.ipynb
        ├── country_statistics_fig1.ipynb
        ├── cropland_sc_GDP_per_capita.ipynb
        ├── elevation_slope.ipynb
        ├── environment_economic.ipynb
        ├── transition.ipynb
        ├── tree_loss_1km_display.ipynb
        └── tree_loss_percentile_countries.ipynb
```

---

## Installation

```bash
git clone https://github.com/ywant/sc-mapping.git
cd sc-mapping
pip install -r requirements.txt
```

Tested on Python 3.9, PyTorch 2.0, CUDA 12.1.

---

## Data

### Input imagery
Predictions use **NICFI PlanetScope basemap** tiles (4-band R/G/B/NIR, 4.77 m)  Two mosaic composite are used for the final prediction: December,2019-May,2020 and June-August,2020.

### Model weights
Pre-trained model weights, example training patches, and a sample label CSV are deposited on Zenodo: **[DOI placeholder — update after publication]**

| File | Size | MD5 | Description |
|------|------|-----|-------------|
| `bestF1.pkl` | 30.67 MB | `518e72769ea02bf5ec6b8ca1ca98e272` | Trained EfficientNet-B1 weights |
| `samples_example.zip` | 35.07 MB | `f4dcfa442158421093ef079de2ef6a7d` | Example PlanetScope training patches (.npy) |
| `sample_csv_demo.csv` | — | `223fb02bedc35d26be2af0c7a9704284` | Example label CSV with required column format |

Download `bestF1.pkl` and place it at `saved_models/bestF1.pkl` before running inference.

### Training labels
A sample label CSV (`sample_csv_demo.csv`) is deposited on Zenodo showing the required format. Required columns: `path`, `label`, `useCase`, `augProb`, `id`. Please replact the path to your own.

### Class legend
| Index | Class |
|-------|-------|
| 0 | High tree-cover woodland |
| 1 | Shifting cultivation |
| 2 | Others |
| 3 | Conventional agriculture |
| 4 | Mixed woody plantation |

---

## Model training

Edit `configs/sc_classification.yaml` to set your data paths, then run:

```bash
python main.py --config configs/sc_classification.yaml
```

Training saves three checkpoints to `saved_models/`:
- `bestLoss.pkl` — lowest validation loss
- `bestF1.pkl` — highest validation macro-F1 *(used for prediction)*
- `epoch_N.pkl` — periodic snapshots every 30 epochs

To monitor training with Weights & Biases, set `use_wandb: 1` in the config and run `wandb login` beforehand. W&B is entirely optional; set `use_wandb: 0` to skip.

---

## Prediction

Edit `configs/sc_prediction.yaml` to set `model_path` and `input_image_dir`, then run:

```bash
python predict.py --config configs/sc_prediction.yaml
```

To loop over multiple years without editing the config:

```bash
for year in 2019 2020 2021 2022; do
    python predict.py --config configs/sc_prediction.yaml \
                      --input_dir data/nicfi/${year}/ \
                      --output_dir results/predictions/${year}/
done
```

### Output rasters
Two spatial resolutions are saved per input tile:

| Folder | Resolution | Description |
|--------|-----------|-------------|
| `pred_ori/` | Native (~4.77 m) | Predicted class index (0–4), one pixel per input pixel |
| `pred_pat/` | ~1 km | Predicted class index aggregated per 234×234 patch |
| `conf_ori/` | Native | Maximum softmax probability (confidence) |
| `conf_pat/` | ~1 km | Confidence aggregated per patch |

The `pred_pat/` outputs at 1 km resolution are the basis for the pantropical map reported in the paper.

### Prediction methodology
Inference applies **test-time augmentation (TTA)**: each patch is classified under 8 augmentations (horizontal flip × vertical flip × 4 rotations) and predictions are averaged. For each 1-degree tile, predictions are generated independently for the December 2019 and June 2020 NICFI basemaps; the element-wise maximum of the two softmax probability maps is taken as the final prediction (**two-date fusion**).

---

## Accuracy assessment

### 1. Stratified sample design (`analysis/validation/gee/stratified_sampling.js`)

A Google Earth Engine script that generates 400 stratified random validation points across four strata designed to capture both commission and omission errors:

| Stratum | Definition | n |
|---------|-----------|---|
| 1 — Confirmed SC | Model predicts SC and WRI GDM agrees | 100 |
| 2 — Commission check | Model predicts SC, WRI GDM does not | 100 |
| 3 — Omission check | Intermediate probability, not predicted SC, WRI GDM says SC | 100 |
| 4 — Background | All remaining pixels in the ROI | 100 |

To run: open [code.earthengine.google.com](https://code.earthengine.google.com), paste the script, and submit the export task. The output CSV is used as input to the validation notebook.

### 2. Area-adjusted accuracy (`analysis/validation/validation.ipynb`)

Implements the **Olofsson et al. (2014)** area-adjusted accuracy framework. Given the stratified sampling design, simple counts cannot be used directly — this notebook corrects for unequal sampling intensities across strata.

Inputs: validation point GeoPackage (`val_points_strata.gpkg`) and the strata raster (`validation_strata.tif`).

Outputs: area-weighted error matrix, per-class User Accuracy and Producer Accuracy, Overall Accuracy, and area-adjusted class extent estimates. Results are saved to `results/accuracy_summary.csv`.

Reference: Olofsson, P. et al. (2014). *Remote Sensing of Environment*, 148, 42–57.

---

## Analysis notebooks

All notebooks are in `analysis/notebooks/`. Hardcoded paths at the top of each notebook should be updated to point to your local data before running.

### Country-level statistics

**`country_statistic_with_uncertainty.ipynb`**
Computes per-country SC area statistics from prediction pixel centroids joined to country boundaries. Applies area-adjusted uncertainty from the validation results (User Accuracy) to estimate the confidence range on national SC extent estimates. Outputs a GeoPackage with country-level class areas and uncertainty columns used by the figure notebooks.

**`country_statistics_fig1.ipynb`**
Produces the country-level figures in the paper, including: a choropleth map of national SC extent; a ranked bar chart of the top-20 countries by SC area; and stacked bars showing SC as a share of total agricultural land. Inputs the GeoPackage from `country_statistic_with_uncertainty.ipynb`.

### Environmental and socioeconomic correlates

**`cropland_sc_GDP_per_capita.ipynb`**
Analyses the relationship between national SC extent and GDP per capita (World Bank, 2020). Merges SC country statistics with World Bank GDP data and plots SC area against GDP per capita across tropical countries.

**`environment_economic.ipynb`**
Examines the association between SC land share and the Human Development Index (HDI). Countries are grouped into HDI classes (Low / Medium / High / Very High) and SC land share is compared across groups using violin plots and Pearson correlation.

**`elevation_slope.ipynb`**
Characterises the topographic context of SC using elevation and slope values extracted at SC pixel centroids. Produces per-continent histograms and summary statistics (median, IQR) for elevation and slope distributions across Africa, Asia-Oceania, and the Americas.

**`transition.ipynb`**
Analyses the spatial proximity of SC pixels to plantation agriculture and conventional cropland using nearest-neighbour distance calculations. Reports median distances by continent and produces violin plots comparing plantation vs cropland proximity, providing evidence on agricultural transition dynamics.

### Tree cover loss

**`tree_loss_1km_display.ipynb`**
Categorises 1-degree SC grid cells by Hansen tree cover loss intensity into five activity classes (No loss / Low / Moderate / High / Very high) using percentile thresholds. Produces a boxplot of loss distributions by activity class and saves the classified grid for cartographic display.

**`tree_loss_percentile_countries.ipynb`**
Aggregates tree cover loss activity classes to the country level and computes the percentage of SC grid cells in high-activity classes per country. Calculates Spearman rank correlation between national SC extent and high-activity loss fraction, and produces a labelled scatter plot of the top countries.

---

## Reproducibility notes

- All random seeds are fixed at 42 (training, data splitting, GEE sampling).
- The train/validation split is saved as a JSON file in `data/splits/` on first run, ensuring the same split is used for any subsequent training run.
- The GEE stratified sampling script exports a CSV with a fixed seed; the deposited validation GeoPackage on Zenodo is the one used in the paper.
- Model weights (`bestF1.pkl`, MD5: `518e72769ea02bf5ec6b8ca1ca98e272`) deposited on Zenodo correspond to the `bestF1.pkl` checkpoint from the training run described in the paper. The MD5 hash can be used to verify download integrity.

---

## Citation

If you use this code or the SC map, please cite:

```
[Citation placeholder — to be filled after acceptance]
```

---

## License

Code: MIT License. See `LICENSE`.
Map outputs: CC BY 4.0.
