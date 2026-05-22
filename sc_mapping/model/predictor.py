"""
Inference engine for the shifting cultivation classifier.

Implements patch-based prediction over NICFI PlanetScope GeoTIFF basemap
tiles using test-time augmentation (TTA) and optional two-date fusion.

Two-date fusion
---------------
For each 1-degree tile, predictions are generated independently for two
NICFI basemap epochs (December 2019 and June 2020).  The final class
probability map is the element-wise maximum of the two softmax outputs,
which retains the clearest signal across the two acquisition dates.

Test-time augmentation
----------------------
Each patch is classified under eight augmentations (horizontal flip,
vertical flip, and 0/90/180/270° rotations) via the ``ttach`` library.
Predictions are averaged (``merge_mode='mean'``) before the argmax.

Output rasters
--------------
Two resolution versions are saved per tile:

* **Original resolution** (``pred_ori/``, ``conf_ori/``) — one pixel per
  input pixel, preserving the full 4.77 m NICFI ground sampling distance.
* **Patch resolution** (``pred_pat/``, ``conf_pat/``) — one pixel per
  234×234 patch (≈ 1.1 km), used for the 1 km SC map in the paper.

Usage::

    from model.predictor import Predictor
    from omegaconf import OmegaConf

    config = OmegaConf.load('configs/sc_prediction.yaml')
    pred = Predictor(config)
    pred.predict_all()
"""

import os
import gc
import glob
import logging
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.efficientnet as efficientnet
import rasterio
from rasterio.transform import Affine
from torchinfo import summary
from tqdm import tqdm
import ttach as tta

from data.pred_loader import get_pred_loader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

class SoftmaxWrapper(nn.Module):
    """Wrap a classification model to return softmax probabilities.

    Used at inference time so all downstream code works with normalised
    class probabilities rather than raw logits.

    Args:
        model: Any classification model whose forward() returns logits
               of shape (B, num_classes).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def build_efficientnet_b1(output_ch: int, pretrained: bool = True) -> nn.Module:
    """Build the EfficientNet-B1 classifier used in the SC paper.

    Architecture:
        - 4-channel input conv (R, G, B, NIR)
        - ImageNet-pretrained feature extractor
        - 4-layer dense classification head → output_ch logits

    Args:
        output_ch:  Number of output classes (5 for the SC paper).
        pretrained: Load ImageNet weights for the feature extractor.

    Returns:
        Configured EfficientNet-B1 model (not yet on any device).
    """
    weights = efficientnet.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b1(weights=weights)

    # Adapt first conv layer for 4-band RGBNIR input
    model.features[0] = nn.Conv2d(
        in_channels=4, out_channels=32,
        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False,
    )
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 640),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(640, 320),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(320, 64),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(64, output_ch),
    )
    return model


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_prediction_rasters(
    mask: np.ndarray,
    src_profile: dict,
    path_orig: str,
    path_patch: str,
    patch_size: int = 234,
    stride: int = 234,
    dtype: str = "uint8",
    compress: str = "lzw",
):
    """Write a prediction mask at two spatial resolutions.

    Saves two GeoTIFFs:

    1. **Original resolution** at ``path_orig``: one output pixel per input
       pixel, preserving the NICFI ground sampling distance.
    2. **Patch resolution** at ``path_patch``: one output pixel per
       ``patch_size × patch_size`` block, aggregated by mean.  This is the
       resolution used for the 1 km SC map (patch_size = 234 px ≈ 1.1 km).

    Args:
        mask:        2D float or int array of shape (H, W).
        src_profile: Rasterio profile dict copied from the source GeoTIFF.
        path_orig:   Output path for the original-resolution raster.
        path_patch:  Output path for the patch-resolution raster.
        patch_size:  Patch size in pixels used during prediction.
        stride:      Stride used during tiling.
        dtype:       Output data type string (e.g. ``'int32'``, ``'float32'``).
        compress:    GeoTIFF compression algorithm.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}.")

    nodata = src_profile.get("nodata", -1)
    height, width = mask.shape

    # ---- 1. Original resolution ----
    profile_orig = src_profile.copy()
    profile_orig.update(dtype=dtype, count=1, compress=compress,
                        driver="GTiff", nodata=nodata)
    os.makedirs(os.path.dirname(path_orig) or ".", exist_ok=True)
    with rasterio.open(path_orig, "w", **profile_orig) as dst:
        dst.write(mask.astype(dtype), 1)
    log.info(f"Saved (original res): {path_orig}  shape={mask.shape}")

    # ---- 2. Patch resolution ----
    y_coords = list(range(0, height - patch_size + 1, stride))
    if not y_coords or y_coords[-1] + patch_size < height:
        y_coords.append(height - patch_size)

    x_coords = list(range(0, width - patch_size + 1, stride))
    if not x_coords or x_coords[-1] + patch_size < width:
        x_coords.append(width - patch_size)

    out = np.full((len(y_coords), len(x_coords)), nodata, dtype=np.float32)
    for yi, y in enumerate(y_coords):
        for xi, x in enumerate(x_coords):
            chip = mask[y:y + patch_size, x:x + patch_size]
            if chip.size > 0:
                out[yi, xi] = chip.mean()

    profile_patch = src_profile.copy()
    profile_patch.update(
        height=out.shape[0],
        width=out.shape[1],
        transform=src_profile["transform"] * Affine.scale(patch_size, patch_size),
        dtype=dtype,
        count=1,
        compress=compress,
        driver="GTiff",
        nodata=nodata,
    )
    os.makedirs(os.path.dirname(path_patch) or ".", exist_ok=True)
    with rasterio.open(path_patch, "w", **profile_patch) as dst:
        dst.write(out.astype(dtype), 1)
    log.info(f"Saved (patch res):    {path_patch}  shape={out.shape}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def load_imagery(config) -> list:
    """Find input GeoTIFF tiles that have not yet been predicted.

    Skips any tile whose ``_conf_ori`` output file already exists in
    ``config.output_dir/conf_ori/``, enabling interrupted runs to resume.

    Args:
        config: Config namespace with input_image_dir and output_dir.

    Returns:
        List of absolute paths to unpredicted GeoTIFF tiles.
    """
    input_dirs = config.input_image_dir
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    all_images = []
    for d in input_dirs:
        all_images.extend(glob.glob(os.path.join(d, "*.tif")))

    predicted = {
        os.path.basename(p).replace("_conf_ori.tif", ".tif")
        for p in glob.glob(os.path.join(config.output_dir, "conf_ori", "*.tif"))
    }
    image_map = {os.path.basename(p): p for p in all_images}
    remaining = [image_map[fn] for fn in image_map if fn not in predicted]

    log.info(f"Total tiles: {len(all_images)} | "
             f"Already predicted: {len(predicted)} | "
             f"Remaining: {len(remaining)}")
    return remaining


def load_test_files(config) -> tuple[list, list]:
    """Load file paths and labels for the held-out test set.

    Args:
        config: Config namespace with classification_label_fn and test_path.

    Returns:
        Tuple of (file_paths, labels).
    """
    gt_df = pd.read_csv(config.classification_label_fn,
                        usecols=["path", "label", "useCase"])
    test_df = gt_df[gt_df["useCase"] == "test"]

    all_files = (glob.glob(f"{config.test_path1}/**/*.npy", recursive=True)
                 + glob.glob(f"{config.test_path2}/**/*.npy", recursive=True))
    test_set  = set(test_df["path"].tolist())
    paths     = [f for f in all_files if f in test_set]
    labels    = test_df.set_index("path").loc[paths, "label"].tolist()

    log.info(f"Test set: {len(paths)} samples | Classes: {set(labels)}")
    return paths, labels


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Sliding-window inference engine for the SC classifier.

    Supports three prediction modes set via ``config.pred_mode``:

    * ``'pred'``   — full tile prediction over NICFI GeoTIFF basemaps.
    * ``'test'``   — prediction on labelled test-set patches (.npy files).

    Args:
        config: Config namespace loaded from ``configs/sc_prediction.yaml``.
    """

    # Patch size (px) — must match the value used during training
    CHIP_SIZE: int = 234

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")

        if config.pred_mode == "pred":
            self.all_files = load_imagery(config)
        elif config.pred_mode == "test":
            self.all_files, self.test_labels = load_test_files(config)
        else:
            raise ValueError(f"Unsupported pred_mode: '{config.pred_mode}'. "
                             "Use 'pred' or 'test'.")

        self._build_model()
        self._load_weights()
        os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _build_model(self):
        """Construct the EfficientNet-B1 architecture and print a summary."""
        self.nnet = build_efficientnet_b1(
            output_ch=self.config.output_ch,
            pretrained=self.config.pretrained,
        )
        self.nnet.to(self.device)

        num_params = sum(p.numel() for p in self.nnet.parameters())
        log.info(f"Model: EfficientNet-B1 | Parameters: {num_params:,}")

    def _load_weights(self):
        """Load trained weights from config.model_path.

        Handles state dicts saved with or without a ``'nnet.'`` prefix
        (the prefix is stripped automatically if present).

        Wraps the model in SoftmaxWrapper so forward() returns class
        probabilities rather than raw logits.
        """
        path = self.config.model_path
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Download the weights from Zenodo and set model_path in the config."
            )

        state_dict = torch.load(path, map_location=self.device)

        # Strip 'nnet.' prefix if weights were saved from a Solver instance
        if any(k.startswith("nnet.") for k in state_dict):
            state_dict = {k.replace("nnet.", ""): v for k, v in state_dict.items()}

        self.nnet.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded weights: {path}")

        # Wrap with softmax for inference
        self.nnet = SoftmaxWrapper(self.nnet)
        self.nnet.eval()
        self.nnet.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_all(self):
        """Dispatch to the appropriate prediction method based on pred_mode."""
        if self.config.pred_mode == "pred":
            self._predict_tiles()
        elif self.config.pred_mode == "test":
            self._predict_test_set()

    # ------------------------------------------------------------------
    # Tile prediction (pred mode)
    # ------------------------------------------------------------------

    def _predict_tiles(self):
        """Run full-tile prediction with TTA and two-date fusion.

        For each tile, predictions are generated for the primary date
        (``input_image_dir``) and optionally fused with a second date
        (``imageB_dir``).  The element-wise maximum of the two softmax
        probability maps is used as the final prediction.

        Two output GeoTIFFs are saved per tile (original and patch
        resolution) for both the predicted class index and the maximum
        class probability (confidence).
        """
        # Build TTA model once
        tta_model = tta.ClassificationTTAWrapper(
            self.nnet.eval(),
            tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ]),
            merge_mode="mean",
        )

        # Setup output directories
        dirs = {
            "pred_ori": os.path.join(self.config.output_dir, "pred_ori"),
            "pred_pat": os.path.join(self.config.output_dir, "pred_pat"),
            "conf_ori": os.path.join(self.config.output_dir, "conf_ori"),
            "conf_pat": os.path.join(self.config.output_dir, "conf_pat"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # Build filename → path lookup for the second-date (B) tiles
        b_file_map = {}
        if hasattr(self.config, "imageB_dir") and self.config.imageB_dir:
            b_tiles = glob.glob(
                os.path.join(self.config.imageB_dir, "**", "2020-06_2020-08", "*.tif"),
                recursive=True,
            )
            b_file_map = {os.path.basename(f): f for f in b_tiles}

        # Tile selection
        files = self._select_tiles()
        cooldown_secs = getattr(self.config, "cooldown_seconds", 30)
        cooldown_interval = getattr(self.config, "cooldown_interval", 1)

        torch.set_grad_enabled(False)

        for idx, tile_path in enumerate(tqdm(files, desc="Predicting tiles")):
            name = os.path.splitext(os.path.basename(tile_path))[0]

            # Derive second-date tile name (2019-12 → 2020-06)
            b_name = name.replace("2019_12", "2020_06") + ".tif"
            path_b = b_file_map.get(b_name)
            use_b  = path_b is not None
            log.info(f"[{name}] B image: {'found' if use_b else 'not found'}")

            with rasterio.open(tile_path) as src:
                h, w       = src.height, src.width
                src_profile = src.profile.copy()
            src_profile["dtype"] = "float32"

            # Run inference on primary date
            logits_a = self._run_tta_on_tile(tile_path, tta_model, h, w)

            # Optionally run on second date and fuse
            if use_b:
                logits_b = self._run_tta_on_tile(path_b, tta_model, h, w)
                logits   = np.maximum(logits_a, logits_b)
            else:
                logits = logits_a

            prediction  = np.argmax(logits, axis=0)[:h, :w].astype(np.int32)
            confidence  = np.max(logits,   axis=0)[:h, :w].astype(np.float32)

            # Save prediction and confidence at both resolutions
            write_prediction_rasters(
                prediction, src_profile,
                os.path.join(dirs["pred_ori"], f"{name}_pred_ori.tif"),
                os.path.join(dirs["pred_pat"], f"{name}_pred_pat.tif"),
                patch_size=self.CHIP_SIZE, stride=self.CHIP_SIZE,
                dtype="int32",
            )
            write_prediction_rasters(
                confidence, src_profile,
                os.path.join(dirs["conf_ori"], f"{name}_conf_ori.tif"),
                os.path.join(dirs["conf_pat"], f"{name}_conf_pat.tif"),
                patch_size=self.CHIP_SIZE, stride=self.CHIP_SIZE,
                dtype="float32",
            )

            log.info(f"Finished: {name}")
            del logits_a, logits, prediction, confidence
            torch.cuda.empty_cache()
            gc.collect()

            if (idx + 1) % cooldown_interval == 0 and cooldown_secs > 0:
                log.info(f"GPU cooldown: {cooldown_secs}s")
                time.sleep(cooldown_secs)

    def _run_tta_on_tile(
        self,
        tile_path: str,
        tta_model: nn.Module,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Run TTA inference on all patches of one GeoTIFF tile.

        Args:
            tile_path: Path to the input GeoTIFF.
            tta_model: TTA-wrapped model.
            height:    Tile height in pixels.
            width:     Tile width in pixels.

        Returns:
            Float32 array of shape (output_ch, height, width) with
            averaged class probabilities from all TTA passes.
        """
        loader  = get_pred_loader(tile_path, self.config)
        logits  = np.zeros((self.config.output_ch, height, width), dtype=np.float32)

        for patches, coords in loader:
            patches = patches.to(self.device)
            with torch.inference_mode():
                probs = tta_model(patches)          # (B, C)
            probs_np = probs.cpu().numpy()

            for i, (y, x) in enumerate(coords):
                y, x = int(y), int(x)
                # Broadcast scalar class vector over the patch spatial extent
                logits[:, y:y + self.CHIP_SIZE, x:x + self.CHIP_SIZE] = (
                    probs_np[i][:, np.newaxis, np.newaxis]
                )

            del patches, probs, probs_np
            torch.cuda.empty_cache()
            gc.collect()

        return logits

    def _select_tiles(self) -> list:
        """Return the subset of tiles to predict based on config.pred_tile.

        Returns:
            List of GeoTIFF file paths.
        """
        if self.config.pred_tile == "all":
            return self.all_files

        elif self.config.pred_tile == "selective":
            gdf      = gpd.read_file(self.config.pred_tile_pgkg)
            tile_ids = gdf["id"].tolist()
            selected = []
            for tile_id in tile_ids:
                try:
                    lat, lon   = map(int, tile_id.split(","))
                    target     = f"psbm_2019_12_{lat:05}_{lon:05}.tif"
                    selected  += [f for f in self.all_files
                                  if os.path.basename(f) == target]
                except ValueError:
                    log.warning(f"Invalid tile ID: {tile_id}")
            return selected

        else:
            raise ValueError(
                f"config.pred_tile must be 'all' or 'selective', "
                f"got '{self.config.pred_tile}'."
            )

    # ------------------------------------------------------------------
    # Test-set evaluation (test mode)
    # ------------------------------------------------------------------

    def _predict_test_set(self):
        """Run inference on labelled test patches and save predictions to CSV.

        Outputs a CSV with columns: name, labels, pred (softmax vector).
        The CSV path is ``config.output_dir / config.output_suffix + '.csv'``.
        """
        output_file = os.path.join(
            self.config.output_dir,
            f"{self.config.output_suffix}{self.config.output_table_type}",
        )
        loader = get_pred_loader(self.all_files, self.config)

        pd_list, image_files, labels_list = [], [], []

        with torch.no_grad():
            for images, labels, filenames in tqdm(loader, desc="Test set"):
                images = images.to(self.device)
                probs  = self.nnet(images).cpu().numpy()
                pd_list.extend(probs)
                image_files.extend(filenames)
                labels_list.extend(labels.numpy())

        df = pd.DataFrame({
            "name":   image_files,
            "labels": labels_list,
            "pred":   [p.tolist() for p in pd_list],
        })
        df.to_csv(output_file, index=False)
        log.info(f"Test predictions saved to {output_file}")
