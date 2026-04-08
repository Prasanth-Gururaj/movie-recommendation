"""Abstract base ranker for learning-to-rank models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """LTR model interface.

    Subclasses wrap XGBoost or LightGBM ranking objectives and expose a
    uniform ``fit / predict / save / load`` contract so the rest of the
    pipeline stays model-agnostic.
    """

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: list[str],
        label_col: str = "label",
        group_col: str = "userId",
    ) -> None:
        """Train the ranker.

        Parameters
        ----------
        train_df / val_df:
            DataFrames that include *feature_columns*, *label_col*, and
            *group_col*.  Rows within each user group must be contiguous.
        feature_columns:
            Ordered list of feature names — must match the column order used
            at inference time.
        label_col:
            Binary relevance label (1 = positive, 0 = negative).
        group_col:
            Column that identifies the query (user) group.
        """

    @abstractmethod
    def predict(self, features_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        """Return a 1-D score array aligned with *features_df* rows."""

    @abstractmethod
    def save_artifacts(self, output_dir: Path) -> None:
        """Persist model artefacts to *output_dir*."""

    @abstractmethod
    def load_artifacts(self, output_dir: Path) -> None:
        """Restore model artefacts from *output_dir*."""

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature-name → importance mapping (normalized to sum 1)."""

    @abstractmethod
    def log_to_mlflow(self) -> None:
        """Log hyperparameters and metrics to the active MLflow run."""
