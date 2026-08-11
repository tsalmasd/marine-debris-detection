"""
Spectral indices computation for marine debris detection.

Implements NDVI, FDI (Floating Debris Index), and NDWI from
Sentinel-2 multispectral bands.
"""

import numpy as np

# Sentinel-2 central wavelengths in nm. Per Biermann et al. (2020), the FDI
# interpolation coefficient is defined by the RED, NIR and SWIR1 wavelengths;
# the red-edge B06 reflectance supplies the baseline anchor but not a wavelength.
WAVELENGTHS = {
    "B04": 665.0,    # Red
    "B06": 740.0,    # Red Edge 2 (baseline anchor for the FDI)
    "B08": 842.0,    # NIR
    "B11": 1610.0,   # SWIR1
}


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def compute_fdi(nir: np.ndarray, re2: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Floating Debris Index (FDI), Biermann et al. (2020).

    FDI = R_NIR - R'_NIR, where the baseline is linearly interpolated from the
    red-edge 2 (B06) reflectance towards SWIR1 (B11):

        R'_NIR = R_RE2 + (R_SWIR1 - R_RE2) * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    The red-edge 2 band replaces the red band as the baseline anchor (more
    selective for floating macroplastic, less confounded by chlorophyll), while
    the interpolation coefficient still uses the RED/NIR/SWIR1 wavelengths, per
    the original formulation.

    Args:
        nir:   NIR reflectance (B08).
        re2:   red-edge 2 reflectance (B06), the baseline anchor.
        swir1: SWIR1 reflectance (B11).
    """
    lam_nir  = WAVELENGTHS["B08"]
    lam_red  = WAVELENGTHS["B04"]
    lam_swir = WAVELENGTHS["B11"]

    slope = (lam_nir - lam_red) / (lam_swir - lam_red)
    baseline = re2 + (swir1 - re2) * slope
    fdi = nir - baseline
    return fdi.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (GREEN - NIR) / (GREEN + NIR) -- useful to separate water"""
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    return ndwi.astype(np.float32)


def extract_features_from_patch(bands: np.ndarray) -> np.ndarray:
    """
    Compute the Random Forest per-pixel feature vector for a single patch.

    Args:
        bands: np.ndarray of shape (7, H, W) in the order
            [B02, B03, B04, B06, B08, B11, B12] (``dataset_loader.RF_BANDS``).

    Returns:
        features: np.ndarray of shape (H*W, 9) -- the 7 raw bands plus NDVI and
                  the red-edge FDI, flattened to one row per pixel:
                  [B02, B03, B04, B06, B08, B11, B12, NDVI, FDI].
    """
    B02, B03, B04, B06, B08, B11, B12 = bands  # unpack along channel axis

    ndvi = compute_ndvi(nir=B08, red=B04)
    fdi  = compute_fdi(nir=B08, re2=B06, swir1=B11)

    # Stack all features along the channel axis -> shape (9, H, W)
    feature_stack = np.stack([B02, B03, B04, B06, B08, B11, B12, ndvi, fdi], axis=0)

    # Reshape to (H*W, 9): each pixel becomes one sample
    features = feature_stack.reshape(9, -1).T  # shape (H*W, 9)

    return features


# Human-readable names for the 9 RF features, aligned with the columns produced
# by :func:`extract_features_from_patch` (used for feature-importance reporting).
RF_FEATURE_NAMES = ["B02", "B03", "B04", "B06", "B08", "B11", "B12", "NDVI", "FDI"]
