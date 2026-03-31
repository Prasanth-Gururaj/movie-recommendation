"""Time-based feature builder for (userId, movieId) pairs.

Uses ``data["user_features"]`` and ``data["item_features"]`` (which must contain
a ``last_timestamp`` column) to compute recency features relative to training data.
All day-based features are clipped at 0.0 — no negative values.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config.experiment_config import ExperimentConfig
from src.features.base_feature_builder import BaseFeatureBuilder

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY: int = 86_400


class TimeFeatureBuilder(BaseFeatureBuilder):
    """Builds time features for each (userId, movieId) pair.

    Features
    --------
    interaction_month      — calendar month (1–12) of the pair's timestamp
    interaction_dayofweek  — day-of-week (0=Monday … 6=Sunday)
    days_since_user_active — (pair_ts − user last_ts in train) / 86400, clipped ≥ 0
    days_since_item_rated  — (pair_ts − item last_ts in train) / 86400, clipped ≥ 0
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    # ── build ──────────────────────────────────────────────────────────────

    def build(
        self,
        data: dict[str, pd.DataFrame],
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        user_feat: pd.DataFrame = data["user_features"]
        item_feat: pd.DataFrame = data["item_features"]

        train_max_ts: int = int(train_df["timestamp"].max())
        self._train_max_ts = train_max_ts

        # One row per unique (userId, movieId) pair — keep timestamp
        pairs = (
            train_df[["userId", "movieId", "timestamp"]]
            .drop_duplicates(subset=["userId", "movieId"])
            .reset_index(drop=True)
            .copy()
        )

        # Calendar features from the pair's own timestamp
        dt = pd.to_datetime(pairs["timestamp"], unit="s")
        pairs["interaction_month"] = dt.dt.month.astype(int)
        pairs["interaction_dayofweek"] = dt.dt.dayofweek.astype(int)

        # User / item last-seen timestamps from train
        u_last = (
            user_feat[["userId", "last_timestamp"]]
            .rename(columns={"last_timestamp": "_u_last"})
        )
        i_last = (
            item_feat[["movieId", "last_timestamp"]]
            .rename(columns={"last_timestamp": "_i_last"})
        )

        pairs = pairs.merge(u_last, on="userId", how="left")
        pairs = pairs.merge(i_last, on="movieId", how="left")

        pairs["days_since_user_active"] = (
            (pairs["timestamp"] - pairs["_u_last"].fillna(pairs["timestamp"]))
            / _SECONDS_PER_DAY
        ).clip(lower=0.0)

        pairs["days_since_item_rated"] = (
            (pairs["timestamp"] - pairs["_i_last"].fillna(pairs["timestamp"]))
            / _SECONDS_PER_DAY
        ).clip(lower=0.0)

        pairs = pairs.drop(columns=["timestamp", "_u_last", "_i_last"])

        self.log_feature_stats(pairs)
        return pairs

    # ── feature names ──────────────────────────────────────────────────────

    def get_feature_names(self) -> list[str]:
        return [
            "interaction_month",
            "interaction_dayofweek",
            "days_since_user_active",
            "days_since_item_rated",
        ]
