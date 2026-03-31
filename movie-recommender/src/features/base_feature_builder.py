"""Abstract base class for all feature builders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class BaseFeatureBuilder(ABC):
    """Base for every feature builder in the pipeline.

    Subclasses must implement ``build`` and ``get_feature_names``.
    Each subclass stores ``_train_max_ts`` (Unix int) when ``build()`` is
    called, enabling ``validate_no_leakage`` to detect accidental use of
    val / test data.
    """

    _train_max_ts: int | None = None

    # ── abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def build(
        self,
        data: dict[str, pd.DataFrame],
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features from ``train_df`` only and return the feature table.

        Parameters
        ----------
        data:
            Keyed dict of DataFrames (e.g. ``"movies"``, ``"genome_scores"``).
            Subclasses may also look for ``"user_features"`` / ``"item_features"``
            when building interaction or time features.
        train_df:
            Training split — the ONLY source of statistical truth.
        """

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return the ordered list of model-facing feature column names.

        Auxiliary / routing columns (e.g. ``user_tier``, entity IDs) are
        excluded.
        """

    # ── concrete helpers ───────────────────────────────────────────────────

    def validate_no_leakage(
        self,
        features_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> None:
        """Raise ``ValueError`` if features reference data beyond train_df.

        Compares the stored ``_train_max_ts`` (set during ``build()``) against
        ``train_df``'s max timestamp.  A larger stored value means ``build()``
        was called on val / test data instead of (or in addition to) train.
        """
        if self._train_max_ts is None:
            logger.warning(
                "%s.validate_no_leakage: build() has not been called yet — "
                "skipping check.",
                self.__class__.__name__,
            )
            return

        train_max = int(train_df["timestamp"].max())
        if self._train_max_ts > train_max:
            raise ValueError(
                f"Leakage detected in {self.__class__.__name__}: "
                f"features were built with max_timestamp={self._train_max_ts} "
                f"but train_df max_timestamp={train_max}. "
                f"Features may have been computed from val or test data."
            )

        logger.debug(
            "%s: no leakage detected (builder_ts=%d, train_max_ts=%d).",
            self.__class__.__name__,
            self._train_max_ts,
            train_max,
        )

    def log_feature_stats(self, features_df: pd.DataFrame) -> None:
        """Log shape, null counts, and numeric summary statistics."""
        logger.info(
            "%s: shape=%s  n_features=%d",
            self.__class__.__name__,
            features_df.shape,
            len(features_df.columns),
        )
        null_counts = features_df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols):
            logger.warning(
                "%s: null values — %s",
                self.__class__.__name__,
                null_cols.to_dict(),
            )
        else:
            logger.info("%s: no null values.", self.__class__.__name__)

        numeric = features_df.select_dtypes(include="number").columns.tolist()
        if numeric:
            stats = features_df[numeric].describe().loc[["mean", "std", "min", "max"]]
            logger.debug(
                "%s: numeric stats:\n%s",
                self.__class__.__name__,
                stats.to_string(),
            )
