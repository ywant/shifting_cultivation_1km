"""
DataLoader for patch-based inference over NICFI PlanetScope GeoTIFF tiles.

A full GeoTIFF tile (typically ~5000×5000 px) is loaded into memory and
divided into overlapping 234×234 patches using a sliding window. Each patch
is returned alongside its (y, x) top-left pixel coordinate so that
predictions can be reassembled into a spatially referenced output raster.

Typical usage::

    dataset    = PredDataset(tile_path, config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.num_workers)

    for patches, coords in dataloader:
        preds = model(patches.to(device))
        # use coords to write preds back into the output raster
"""

import numpy as np
import torch
from torch.utils import data
import rasterio

from utils.preprocessing import equalize


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PredDataset(data.Dataset):
    """Sliding-window patch dataset for inference on a single GeoTIFF tile.

    Loads the full tile into memory at initialisation, then serves individual
    234×234 patches on demand.  Each item is a (patch, coord) pair where
    ``coord`` is the (y, x) top-left pixel position of the patch — needed to
    reconstruct the prediction raster after inference.

    Args:
        tile_path:  Absolute path to a 4-band GeoTIFF (R, G, B, NIR).
        config:     Config namespace.  Required fields:
                        input_size  (int)  — patch spatial size in pixels
                        BATCH_SIZE  (int)  — batch size for the DataLoader
                        HEIGHT      (int)  — sliding window height (== input_size)
                        WIDTH       (int)  — sliding window width  (== input_size)
                        STRIDE      (int)  — stride between patches (≤ input_size)
    """

    def __init__(self, tile_path: str, config):
        self.tile_path  = tile_path
        self.config     = config
        self.patch_size = config.input_size

        self.image = self._load_tile(tile_path)
        print(f"Loaded tile: {tile_path}  shape: {self.image.shape}")

        self.chip_coordinates = self._compute_patch_coords(
            self.image, stride=config.STRIDE
        )
        print(
            f"Generated {len(self.chip_coordinates)} patches "
            f"(patch_size={self.patch_size}, stride={config.STRIDE})"
        )

    # ------------------------------------------------------------------
    # Tile loading
    # ------------------------------------------------------------------

    def _load_tile(self, path: str) -> np.ndarray:
        """Load bands 1–4 of a GeoTIFF into a (4, H, W) float32 array.

        Args:
            path: Path to the input GeoTIFF.

        Returns:
            numpy array of shape (4, H, W), dtype float32.
        """
        with rasterio.open(path) as src:
            image = src.read([1, 2, 3, 4]).astype(np.float32)
        return image

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    def _compute_patch_coords(self, image: np.ndarray, stride: int) -> list:
        """Compute top-left (y, x) coordinates for all sliding-window patches.

        Patches are placed on a regular grid with the given stride.  An extra
        row/column of patches is added at the bottom/right edge to ensure
        complete coverage even when the tile dimensions are not divisible by
        the stride.

        Args:
            image:  Array of shape (C, H, W).
            stride: Step size between consecutive patches (px).

        Returns:
            List of (y, x) tuples.
        """
        _, height, width = image.shape
        p = self.patch_size

        if height < p or width < p:
            raise ValueError(
                f"Tile ({height}×{width}) is smaller than patch size ({p}×{p})."
            )

        y_coords = list(range(0, height - p + 1, stride))
        if not y_coords or y_coords[-1] + p < height:
            y_coords.append(height - p)

        x_coords = list(range(0, width - p + 1, stride))
        if not x_coords or x_coords[-1] + p < width:
            x_coords.append(width - p)

        return [(y, x) for y in y_coords for x in x_coords]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chip_coordinates)

    def __getitem__(self, index: int):
        """Extract, preprocess, and return one patch.

        Args:
            index: Patch index.

        Returns:
            patch: Float32 array of shape (4, patch_size, patch_size),
                   scaled to [0, 255].
            coord: Int array of shape (2,) containing (y, x) top-left
                   pixel coordinates within the source tile.
        """
        y, x = self.chip_coordinates[index]
        p = self.patch_size

        patch = self.image[:, y:y + p, x:x + p].copy()
        patch = equalize(patch)         # per-channel histogram equalisation
        patch = patch.astype(np.float32) * 255.0

        return patch, np.array([y, x], dtype=np.int64)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_pred_loader(tile_path: str, config) -> data.DataLoader:
    """Create a DataLoader for sliding-window inference over one GeoTIFF tile.

    Args:
        tile_path: Path to the input GeoTIFF.
        config:    Config namespace (see PredDataset for required fields).

    Returns:
        DataLoader that yields (patches, coords) batches.
    """
    dataset = PredDataset(tile_path, config)
    return data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
