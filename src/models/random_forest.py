"""
Random Forest baseline for marine debris detection.

Trains a pixel-wise binary classifier (debris vs non-debris) using
Sentinel-2 bands and spectral indices as features.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

from src.features.spectral_indices import extract_features_from_patch

DEBRIS_LABEL = 1  # The positive class in MARIDA (Marine Debris)


def prepare_rf_data(
    all_bands: list,
    all_labels: list,
    debris_classes: set = {1},
    max_pixels_per_patch: int = 500,
    exclude_nodata: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of patches into flat (N_pixels, N_features) arrays for RF.

    Binary classification: debris=1, non-debris=0.
    Pixels are subsampled per patch to keep RAM usage manageable.

    Args:
        all_bands: list of np.ndarray (6, H, W)
        all_labels: list of np.ndarray (H, W) with original MARIDA class IDs
        debris_classes: set of class IDs to consider as positive (debris)
        max_pixels_per_patch: cap on pixels sampled per patch
        exclude_nodata: drop pixels labeled 0 (nodata)

    Returns:
        X: np.ndarray of shape (N, 9) -- features
        y: np.ndarray of shape (N,)   -- binary labels (0/1)
    """
    X_list, y_list = [], []

    for bands, label in zip(all_bands, all_labels):
        features = extract_features_from_patch(bands)  # (H*W, 9)
        flat_label = label.flatten()                   # (H*W,)

        # Exclude nodata pixels (label == 0)
        if exclude_nodata:
            valid_mask = flat_label > 0
            features = features[valid_mask]
            flat_label = flat_label[valid_mask]

        if len(features) == 0:
            continue

        # Binary labels
        binary_label = np.where(np.isin(flat_label, list(debris_classes)), 1, 0)

        # Subsample to keep training tractable.
        # replace=False: never duplicate pixels — with-replacement subsampling
        # would inject repeated rows and distort any metric computed on the result.
        if len(features) > max_pixels_per_patch:
            features, binary_label = resample(
                features, binary_label,
                n_samples=max_pixels_per_patch,
                replace=False,
                random_state=42
            )

        X_list.append(features)
        y_list.append(binary_label)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest with class balancing.

    Returns:
        clf: trained RandomForestClassifier
        scaler: fitted StandardScaler (must be reused at inference time)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",   # critical: debris is a rare class
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    clf.fit(X_scaled, y_train)
    return clf, scaler


def predict(
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    bands: np.ndarray
) -> np.ndarray:
    """
    Run inference on a single patch.

    Args:
        clf: trained RandomForestClassifier
        scaler: fitted StandardScaler from training
        bands: np.ndarray of shape (6, H, W)

    Returns:
        Binary prediction mask of shape (H, W).
    """
    H, W = bands.shape[1], bands.shape[2]
    features = extract_features_from_patch(bands)        # (H*W, 9)
    features_scaled = scaler.transform(features)
    preds = clf.predict(features_scaled)                 # (H*W,)
    return preds.reshape(H, W)


def save_model(
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    path_prefix: str
) -> None:
    """Persist the model and scaler to disk."""
    joblib.dump(clf, f"{path_prefix}_rf.joblib")
    joblib.dump(scaler, f"{path_prefix}_scaler.joblib")


def load_model(path_prefix: str) -> tuple[RandomForestClassifier, StandardScaler]:
    """Load a previously saved model and scaler."""
    clf = joblib.load(f"{path_prefix}_rf.joblib")
    scaler = joblib.load(f"{path_prefix}_scaler.joblib")
    return clf, scaler
