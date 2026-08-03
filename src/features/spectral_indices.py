"""
Spectral indices computation for marine debris detection.

Implements NDVI, FDI (Floating Debris Index), and NDWI from
Sentinel-2 multispectral bands.
"""

import numpy as np

# Sentinel-2 central wavelengths in nm (needed for FDI computation)
WAVELENGTHS = {
    "B04": 665.0,   # Red
    "B06": 740.0,   # Red Edge (reference, not in our 6 bands)
    "B08": 842.0,   # NIR
    "B11": 1610.0,  # SWIR1
}


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def compute_fdi(nir: np.ndarray, red: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Floating Debris Index (FDI):
    FDI = NIR - (RED + (SWIR1 - RED) * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED))
    """
    lam_nir  = WAVELENGTHS["B08"]
    lam_red  = WAVELENGTHS["B04"]
    lam_swir = WAVELENGTHS["B11"]

    slope = (lam_nir - lam_red) / (lam_swir - lam_red)
    baseline = red + (swir1 - red) * slope
    fdi = nir - baseline
    return fdi.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (GREEN - NIR) / (GREEN + NIR) -- useful to separate water"""
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    return ndwi.astype(np.float32)


def extract_features_from_patch(bands: np.ndarray) -> np.ndarray:
    """
    Compute spectral features for a single patch.

    Args:
        bands: np.ndarray of shape (6, H, W) -- order [B02, B03, B04, B08, B11, B12]

    Returns:
        features: np.ndarray of shape (H*W, 9) -- 6 raw bands + NDVI + FDI + NDWI,
                  flattened to one row per pixel.
    """
    B02, B03, B04, B08, B11, B12 = bands  # unpack along channel axis

    ndvi = compute_ndvi(nir=B08, red=B04)
    fdi  = compute_fdi(nir=B08, red=B04, swir1=B11)
    ndwi = compute_ndwi(green=B03, nir=B08)

    # Stack all features along the channel axis -> shape (9, H, W)
    feature_stack = np.stack([B02, B03, B04, B08, B11, B12, ndvi, fdi, ndwi], axis=0)

    # Reshape to (H*W, 9): each pixel becomes one sample
    features = feature_stack.reshape(9, -1).T  # shape (H*W, 9)

    return features
