"""
Image preprocessing utilities for PlanetScope NICFI patches.

These functions are used by both the training DataLoader and the inference
pipeline to normalise 4-band (R, G, B, NIR) patches before they are fed
into the classifier.

Key function
------------
equalize(image)
    Per-channel CLAHE equalisation + percentile intensity rescaling.
    This is the only preprocessing step applied at both training and
    inference time, so consistency here is critical for reproducibility.
"""

import numpy as np
from skimage.exposure import equalize_adapthist, rescale_intensity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum percentage of pixels allowed in any histogram bin during CLAHE.
# Lower values produce stronger contrast enhancement.
CLIP_LIMIT = 0.01

# Percentile used to clip intensity outliers before rescaling.
# e.g. 1 means the bottom 1 % and top 1 % of pixel values are clipped.
OUTLIER_PERCENTAGE = 1


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def equalize(image: np.ndarray) -> np.ndarray:
    """Apply per-channel CLAHE equalisation to a 4-band patch.

    Processing steps:
        1. Min-max normalise each channel to [0, 1].
        2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
           to the RGB channels jointly and the NIR channel separately.
        3. Rescale intensities to the [OUTLIER_PERCENTAGE, 100-OUTLIER_PERCENTAGE]
           percentile range to clip outlier values.

    RGB and NIR are processed separately because their dynamic ranges and
    scene statistics differ substantially in tropical forest imagery.

    Args:
        image: Float or int array of shape (4, H, W) in channel-first order
               (R=0, G=1, B=2, NIR=3).

    Returns:
        Float32 array of shape (4, H, W) with values in [0, 1].
    """
    # Step 1: per-channel min-max normalisation → [0, 1]
    amin = np.amin(image, axis=(1, 2), keepdims=True)
    amax = np.amax(image, axis=(1, 2), keepdims=True)
    image = (image.astype(np.float64) - amin) / (amax - amin + 1e-10)

    # skimage expects channel-last (H, W, C)
    image = image.transpose(1, 2, 0)

    # Step 2a: CLAHE on RGB channels jointly
    rgb_eq = equalize_adapthist(image[:, :, :3], clip_limit=CLIP_LIMIT)
    plow, phigh = np.percentile(rgb_eq, (OUTLIER_PERCENTAGE, 100 - OUTLIER_PERCENTAGE))
    rgb_eq = rescale_intensity(rgb_eq, in_range=(plow, phigh))
    rgb_eq = rgb_eq.transpose(2, 0, 1).astype(np.float32)   # (3, H, W)

    # Step 2b: CLAHE on NIR channel separately
    nir_eq = equalize_adapthist(image[:, :, 3], clip_limit=CLIP_LIMIT)
    plow, phigh = np.percentile(nir_eq, (OUTLIER_PERCENTAGE, 100 - OUTLIER_PERCENTAGE))
    nir_eq = rescale_intensity(nir_eq, in_range=(plow, phigh))
    nir_eq = nir_eq[np.newaxis, :, :].astype(np.float32)    # (1, H, W)

    return np.concatenate((rgb_eq, nir_eq), axis=0)          # (4, H, W)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute the Normalised Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Red-band array, any shape, any numeric dtype.
        nir: NIR-band array, same shape as red.

    Returns:
        Float32 NDVI array in [-1, 1], same shape as inputs.
        Division-by-zero pixels are set to 0.
    """
    np.seterr(divide="ignore", invalid="ignore")
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    ndvi = (nir - red) / (nir + red)
    return np.nan_to_num(ndvi, nan=0.0).astype(np.float32)
