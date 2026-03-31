"""Item-level feature builder — computes features from train_df and cleaned movies_df.

log1p applied to: log_rating_count, recent_pop_30d, log_movie_age.

Genome tag selection (top-N by variance) is saved to configs/genome_tag_columns.json
after build() so downstream inference can reconstruct the same feature schema.
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
_GENOME_TAG_COLUMNS_PATH: Path = _CONFIGS_DIR / "genome_tag_columns.json"
_SECONDS_PER_DAY: int = 86_400


class ItemFeatureBuilder(BaseFeatureBuilder):
    """Builds one-row-per-movie feature table.

    All count-based features use log1p:
      - log_rating_count = log1p(total ratings in train)
      - recent_pop_30d   = log1p(ratings in last 30 days of train)
      - log_movie_age    = log1p(movie_age)

    Genre columns in output are prefixed ``genre_`` (e.g. ``genre_Action``)
    to avoid clashing with user ``genre_affinity_*`` columns in the matrix.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        with _GENRE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
            self._genre_columns: list[str] = json.load(f)
        self._selected_tag_ids: list[int] = []
        self._tag_feature_names: list[str] = []

    # ── build ──────────────────────────────────────────────────────────────

    def build(
        self,
        data: dict[str, pd.DataFrame],
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        movies_df: pd.DataFrame = data["movies"]
        genome_scores: pd.DataFrame | None = data.get("genome_scores")
        cfg_data = self._config.data
        cfg_feat = self._config.feature

        train_max_ts: int = int(train_df["timestamp"].max())
        self._train_max_ts = train_max_ts

        # ── rating stats from train only ──────────────────────────────────
        item_stats = (
            train_df.groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean"),
                rating_variance=("rating", "var"),
                last_timestamp=("timestamp", "max"),
            )
            .reset_index()
        )
        item_stats["rating_variance"] = item_stats["rating_variance"].fillna(0.0)
        item_stats["log_rating_count"] = np.log1p(item_stats["rating_count"])
        item_stats["popularity_pct"] = item_stats["rating_count"].rank(pct=True)

        # recent popularity (last 30 days of train)
        cutoff_30d = train_max_ts - 30 * _SECONDS_PER_DAY
        recent = (
            train_df[train_df["timestamp"] >= cutoff_30d]
            .groupby("movieId")
            .size()
            .rename("_recent")
            .reset_index()
        )
        item_stats = item_stats.merge(recent, on="movieId", how="left")
        item_stats["recent_pop_30d"] = np.log1p(item_stats["_recent"].fillna(0))
        item_stats = item_stats.drop(columns=["_recent"])

        # cold flag
        item_stats["is_cold"] = (
            item_stats["rating_count"] < cfg_data.cold_item_threshold
        ).astype(int)

        # ── movie metadata ─────────────────────────────────────────────────
        genre_cols = self._genre_columns
        meta_cols = ["movieId", "release_year", "movie_age", "has_genre"] + genre_cols
        avail_meta = [c for c in meta_cols if c in movies_df.columns]

        item_stats = item_stats.merge(
            movies_df[avail_meta].drop_duplicates("movieId"),
            on="movieId",
            how="left",
        )

        # Rename raw genre cols → genre_{g} to avoid name conflicts downstream
        rename_map = {g: f"genre_{g}" for g in genre_cols if g in item_stats.columns}
        item_stats = item_stats.rename(columns=rename_map)

        # log_movie_age
        if "movie_age" in item_stats.columns:
            item_stats["log_movie_age"] = np.log1p(
                item_stats["movie_age"].clip(lower=0).fillna(0)
            )
        else:
            item_stats["log_movie_age"] = 0.0

        # ── genome features ───────────────────────────────────────────────
        if genome_scores is not None and len(genome_scores) > 0:
            n_tags = cfg_feat.n_genome_tags
            tag_var = genome_scores.groupby("tagId")["relevance"].var()
            n_available = min(n_tags, len(tag_var))
            top_tags = tag_var.nlargest(n_available).index.tolist()
            self._selected_tag_ids = sorted(int(t) for t in top_tags)

            # Persist selection
            with _GENOME_TAG_COLUMNS_PATH.open("w", encoding="utf-8") as f:
                json.dump(self._selected_tag_ids, f, indent=2)

            # Pivot: movieId × tagId → relevance
            gs_filt = genome_scores[genome_scores["tagId"].isin(self._selected_tag_ids)]
            if len(gs_filt) > 0:
                genome_pivot = (
                    gs_filt.pivot_table(
                        index="movieId",
                        columns="tagId",
                        values="relevance",
                        fill_value=0.0,
                    )
                    .reset_index()
                )
                genome_pivot.columns.name = None
                genome_pivot = genome_pivot.rename(
                    columns={t: f"genome_tag_{t}" for t in self._selected_tag_ids}
                )
                self._tag_feature_names = [
                    f"genome_tag_{t}" for t in self._selected_tag_ids
                ]
                item_stats = item_stats.merge(genome_pivot, on="movieId", how="left")
                for col in self._tag_feature_names:
                    item_stats[col] = item_stats[col].fillna(0.0)

            genome_movie_ids = set(genome_scores["movieId"].unique())
            item_stats["has_genome"] = item_stats["movieId"].isin(genome_movie_ids).astype(int)
        else:
            self._tag_feature_names = []
            item_stats["has_genome"] = 0

        self.log_feature_stats(item_stats)
        return item_stats

    # ── feature names (model-facing only) ─────────────────────────────────

    def get_feature_names(self) -> list[str]:
        genre_cols = self._genre_columns
        base = [
            "log_rating_count",
            "avg_rating",
            "rating_variance",
            "popularity_pct",
            "recent_pop_30d",
            "is_cold",
            "has_genre",
            "release_year",
            "movie_age",
            "log_movie_age",
            "has_genome",
        ]
        genre_features = [f"genre_{g}" for g in genre_cols]
        return base + genre_features + self._tag_feature_names
