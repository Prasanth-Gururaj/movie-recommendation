"""User-level feature builder — all statistics computed from train_df only.

log1p applied to: log_total_ratings, log_positive_count, activity_30d, activity_90d.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.experiment_config import ExperimentConfig
from src.features.base_feature_builder import BaseFeatureBuilder

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).parents[2] / "configs"
_GENRE_COLUMNS_PATH: Path = _CONFIGS_DIR / "genre_columns.json"
_SECONDS_PER_DAY: int = 86_400


class UserFeatureBuilder(BaseFeatureBuilder):
    """Builds one-row-per-user feature table from training data only.

    All count-based features use log1p transformation:
      - log_total_ratings  = log1p(total rating count)
      - log_positive_count = log1p(count of ratings >= relevance_threshold)
      - activity_30d       = log1p(ratings in last 30 days of train)
      - activity_90d       = log1p(ratings in last 90 days of train)

    ``user_tier`` is a string routing column and is NOT included in
    ``get_feature_names()`` — it is not fed to the ranker.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        with _GENRE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
            self._genre_columns: list[str] = json.load(f)

    # ── build ──────────────────────────────────────────────────────────────

    def build(
        self,
        data: dict[str, pd.DataFrame],
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        movies_df: pd.DataFrame = data["movies"]
        threshold: float = self._config.data.relevance_threshold
        cold_thresh: int = self._config.data.cold_user_threshold
        short_win: int = self._config.feature.activity_short_window   # 30
        medium_win: int = self._config.feature.activity_medium_window  # 90

        train_max_ts: int = int(train_df["timestamp"].max())
        self._train_max_ts = train_max_ts

        # ── basic per-user stats ──────────────────────────────────────────
        user_stats = (
            train_df.groupby("userId")
            .agg(
                total_ratings=("rating", "count"),
                mean_rating=("rating", "mean"),
                rating_variance=("rating", "var"),
                last_timestamp=("timestamp", "max"),
            )
            .reset_index()
        )
        user_stats["rating_variance"] = user_stats["rating_variance"].fillna(0.0)
        user_stats["log_total_ratings"] = np.log1p(user_stats["total_ratings"])

        # ── positive counts ───────────────────────────────────────────────
        positives = train_df[train_df["rating"] >= threshold]
        pos_counts = (
            positives.groupby("userId")
            .size()
            .rename("positive_count")
            .reset_index()
        )
        user_stats = user_stats.merge(pos_counts, on="userId", how="left")
        user_stats["positive_count"] = user_stats["positive_count"].fillna(0).astype(int)
        user_stats["log_positive_count"] = np.log1p(user_stats["positive_count"])

        # ── temporal activity windows ─────────────────────────────────────
        user_stats["days_since_active"] = (
            (train_max_ts - user_stats["last_timestamp"]) / _SECONDS_PER_DAY
        ).clip(lower=0.0)

        cutoff_short = train_max_ts - short_win * _SECONDS_PER_DAY
        cutoff_medium = train_max_ts - medium_win * _SECONDS_PER_DAY

        act_short = (
            train_df[train_df["timestamp"] >= cutoff_short]
            .groupby("userId")
            .size()
            .rename("_act_short")
            .reset_index()
        )
        act_medium = (
            train_df[train_df["timestamp"] >= cutoff_medium]
            .groupby("userId")
            .size()
            .rename("_act_medium")
            .reset_index()
        )
        user_stats = user_stats.merge(act_short, on="userId", how="left")
        user_stats = user_stats.merge(act_medium, on="userId", how="left")
        user_stats["activity_30d"] = np.log1p(user_stats["_act_short"].fillna(0))
        user_stats["activity_90d"] = np.log1p(user_stats["_act_medium"].fillna(0))
        user_stats = user_stats.drop(columns=["_act_short", "_act_medium"])

        # ── user tier (routing — excluded from model features) ────────────
        user_stats["user_tier"] = user_stats["positive_count"].apply(
            lambda c: "warm" if c >= cold_thresh else "light"
        )

        # ── genre affinity (all train positives) ──────────────────────────
        genre_cols = self._genre_columns
        affinity_all = self._compute_genre_affinity(
            positives, movies_df, genre_cols, prefix="genre_affinity_"
        )
        user_stats = user_stats.merge(affinity_all, on="userId", how="left")
        for g in genre_cols:
            user_stats[f"genre_affinity_{g}"] = user_stats[f"genre_affinity_{g}"].fillna(0.0)

        # ── recent genre affinity (last medium_win days of train) ─────────
        recent_pos = positives[positives["timestamp"] >= cutoff_medium]
        affinity_recent = self._compute_genre_affinity(
            recent_pos, movies_df, genre_cols, prefix="recent_genre_affinity_"
        )
        user_stats = user_stats.merge(affinity_recent, on="userId", how="left")
        for g in genre_cols:
            user_stats[f"recent_genre_affinity_{g}"] = (
                user_stats[f"recent_genre_affinity_{g}"].fillna(0.0)
            )

        self.log_feature_stats(user_stats)
        return user_stats

    # ── helpers ────────────────────────────────────────────────────────────

    def _compute_genre_affinity(
        self,
        positives_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        genre_cols: list[str],
        prefix: str,
    ) -> pd.DataFrame:
        """Fraction of each user's positives that fall in each genre.

        Returns a DataFrame with columns: ``userId``, ``{prefix}{genre}``.
        """
        out_cols = ["userId"] + [f"{prefix}{g}" for g in genre_cols]

        if len(positives_df) == 0:
            return pd.DataFrame(columns=out_cols)

        avail = [c for c in genre_cols if c in movies_df.columns]
        if not avail:
            return pd.DataFrame(columns=out_cols)

        merged = positives_df[["userId", "movieId"]].merge(
            movies_df[["movieId"] + avail], on="movieId", how="left"
        )
        merged[avail] = merged[avail].fillna(0)

        genre_sums = merged.groupby("userId")[avail].sum()
        pos_counts = merged.groupby("userId").size().rename("_n")
        affinity = genre_sums.div(pos_counts, axis=0).reset_index()
        affinity = affinity.rename(columns={g: f"{prefix}{g}" for g in avail})

        # Fill any genres not in movies_df
        for g in genre_cols:
            col = f"{prefix}{g}"
            if col not in affinity.columns:
                affinity[col] = 0.0

        return affinity[out_cols]

    # ── feature names (model-facing only) ─────────────────────────────────

    def get_feature_names(self) -> list[str]:
        g = self._genre_columns
        return [
            "log_total_ratings",
            "log_positive_count",
            "mean_rating",
            "rating_variance",
            *[f"genre_affinity_{genre}" for genre in g],
            *[f"recent_genre_affinity_{genre}" for genre in g],
            "days_since_active",
            "activity_30d",
            "activity_90d",
        ]
