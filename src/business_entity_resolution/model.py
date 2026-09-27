"""
src/business_entity_resolution/model.py
=======================================
ML matching classifiers for Amazon ML Challenge 2026.
Supports Logistic Regression baseline, HistGradientBoostingClassifier,
and LightGBM, with model persistence and probability scoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from business_entity_resolution.config import config
from business_entity_resolution.features import FEATURE_NAMES

logger = logging.getLogger("BER.Model")


def get_classifier(model_type: str = config.model_type, random_seed: int = config.random_seed):
    """Factory to instantiate the appropriate classifier pipeline."""
    if model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight=config.class_weight,
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ])
    elif model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=250,
            max_depth=6,
            min_samples_leaf=20,
            learning_rate=0.05,
            class_weight=config.class_weight,
            random_state=random_seed,
        )
    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                class_weight=config.class_weight,
                random_state=random_seed,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            logger.warning("[MODEL] LightGBM not installed, falling back to HistGradientBoostingClassifier.")
            return HistGradientBoostingClassifier(
                max_iter=250,
                max_depth=6,
                min_samples_leaf=20,
                learning_rate=0.05,
                class_weight=config.class_weight,
                random_state=random_seed,
            )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class MatchClassifier:
    """Wrapper for training, persistence, and probability scoring."""

    def __init__(self, model_type: str = config.model_type):
        self.model_type = model_type
        self.clf = get_classifier(model_type)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the matching model on pairwise features."""
        logger.info("[MODEL] Training %s on %d samples (%d features)...", self.model_type, len(y), X.shape[1])
        self.clf.fit(X, y)
        self.is_fitted = True
        logger.info("[MODEL] Training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability P(match=1) for each pair."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        # Returns 1D array of positive class probabilities
        probas = self.clf.predict_proba(X)
        return probas[:, 1]

    def save(self, model_path: Optional[Path] = None, meta_path: Optional[Path] = None):
        """Save model artifact and feature metadata."""
        mp = model_path or config.model_artifact_path
        mp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, mp)

        metadata = {
            "model_type": self.model_type,
            "feature_names": FEATURE_NAMES,
            "num_features": len(FEATURE_NAMES),
            "class_weight": config.class_weight,
        }
        meta_p = meta_path or config.feature_metadata_path
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("[MODEL] Saved model to %s and metadata to %s", mp, meta_p)

    @classmethod
    def load(cls, model_path: Optional[Path] = None, meta_path: Optional[Path] = None):
        """Load trained model and metadata."""
        mp = model_path or config.model_artifact_path
        meta_p = meta_path or config.feature_metadata_path

        if not mp.is_file():
            raise FileNotFoundError(f"Model artifact not found at {mp}")

        with open(meta_p, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        instance = cls(model_type=metadata.get("model_type", "hist_gradient_boosting"))
        instance.clf = joblib.load(mp)
        instance.is_fitted = True
        logger.info("[MODEL] Loaded model from %s", mp)
        return instance
