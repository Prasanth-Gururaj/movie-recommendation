"""User-item interaction feature builder.

Requires ``data["user_features"]`` and ``data["item_features"]`` to be present
(populated by FeatureStore before this builder is called).
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


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two same-shape 2-D arrays."""
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    denom = (norm_a * norm_b).ravel()
    dot = (a * b).sum(axis=1)
    return np.where(denom > 0, dot / denom, 0.0).astype(float)


class InteractionFeatureBuilder(BaseFeatureBuilder):
    """Builds one-row-per-(userId, movieId) interaction feature table.

    Features
    --------
    genre_overlap_score  — dot(user_affinity, item_genre) / (sum(user_affinity) + ε), clipped to [0,1]
    tag_profile_similarity — cosine(user_tag_profile, item_genome_vector)
    rating_gap           — user mean_rating − item avg_rating  (can be negative)
    genre_history_count  — log1p(# user positives sharing item's primary genre)
    mf_score             — placeholder 0.0 (filled after ALS training)
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
        user_feat: pd.DataFrame = data["user_features"]
        item_feat: pd.DataFrame = data["item_features"]
        threshold: float = self._config.data.relevance_threshold

        train_max_ts: int = int(train_df["timestamp"].max())
        self._train_max_ts = train_max_ts

        # Unique pairs from train
        pairs = (
            train_df[["userId", "movieId"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )

        genre_cols = self._genre_columns
        ua_cols = [f"genre_affinity_{g}" for g in genre_cols]
        ig_cols = [f"genre_{g}" for g in genre_cols]
        avail_ua = [c for c in ua_cols if c in user_feat.columns]
        avail_ig = [c for c in ig_cols if c in item_feat.columns]

        # ── genre_overlap_score ───────────────────────────────────────────
        tmp = pairs.merge(
            user_feat[["userId"] + avail_ua].rename(
                columns={c: f"_ua_{c}" for c in avail_ua}
            ),
            on="userId",
            how="left",
        ).merge(
            item_feat[["movieId"] + avail_ig].rename(
                columns={c: f"_ig_{c}" for c in avail_ig}
            ),
            on="movieId",
            how="left",
        )

        if avail_ua and avail_ig:
            ua_mat = tmp[[f"_ua_{c}" for c in avail_ua]].fillna(0.0).values
            ig_mat = tmp[[f"_ig_{c}" for c in avail_ig]].fillna(0.0).values
            raw_dot = (ua_mat * ig_mat).sum(axis=1)
            ua_sum = ua_mat.sum(axis=1)
            pairs["genre_overlap_score"] = np.where(
                ua_sum > 0, raw_dot / (ua_sum + 1e-8), 0.0
            ).clip(0.0, 1.0)
        else:
            pairs["genre_overlap_score"] = 0.0

        # ── rating_gap ────────────────────────────────────────────────────
        has_mr = "mean_rating" in user_feat.columns
        has_ar = "avg_rating" in item_feat.columns
        if has_mr and has_ar:
            tmp2 = pairs.merge(
                user_feat[["userId", "mean_rating"]].rename(
                    columns={"mean_rating": "_u_mr"}
                ),
                on="userId",
                how="left",
            ).merge(
                item_feat[["movieId", "avg_rating"]].rename(
                    columns={"avg_rating": "_i_ar"}
                ),
                on="movieId",
                how="left",
            )
            pairs["rating_gap"] = (
                tmp2["_u_mr"].fillna(0.0).values - tmp2["_i_ar"].fillna(0.0).values
            )
        else:
            pairs["rating_gap"] = 0.0

        # ── tag_profile_similarity ────────────────────────────────────────
        genome_cols = [c for c in item_feat.columns if c.startswith("genome_tag_")]
        pairs["tag_profile_similarity"] = 0.0

        if genome_cols:
            positives = train_df[train_df["rating"] >= threshold][["userId", "movieId"]]
            if len(positives) > 0:
                pos_genome = positives.merge(
                    item_feat[["movieId"] + genome_cols], on="movieId", how="left"
                )
                pos_genome[genome_cols] = pos_genome[genome_cols].fillna(0.0)

                # User tag profile = mean genome vector across positives
                u_profiles = (
                    pos_genome.groupby("userId")[genome_cols]
                    .mean()
                    .reset_index()
                    .rename(columns={c: f"_up_{c}" for c in genome_cols})
                )
                up_cols = [f"_up_{c}" for c in genome_cols]

                tmp3 = pairs.merge(u_profiles, on="userId", how="left").merge(
                    item_feat[["movieId"] + genome_cols].rename(
                        columns={c: f"_ip_{c}" for c in genome_cols}
                    ),
                    on="movieId",
                    how="left",
                )
                ip_cols = [f"_ip_{c}" for c in genome_cols]
                u_mat = tmp3[up_cols].fillna(0.0).values
                i_mat = tmp3[ip_cols].fillna(0.0).values
                pairs["tag_profile_similarity"] = _cosine_rows(u_mat, i_mat)

        # ── genre_history_count ───────────────────────────────────────────
        pairs["genre_history_count"] = 0.0

        if avail_ig:
            positives = train_df[train_df["rating"] >= threshold][["userId", "movieId"]]
            if len(positives) > 0:
                pos_g = positives.merge(
                    item_feat[["movieId"] + avail_ig], on="movieId", how="left"
                )
                pos_g[avail_ig] = pos_g[avail_ig].fillna(0)

                # Per-user positive counts per genre
                ugc = pos_g.groupby("userId")[avail_ig].sum()

                # Item primary genre: first avail_ig column with value 1
                item_primary: dict[int, str | None] = {}
                for _, row in item_feat[["movieId"] + avail_ig].iterrows():
                    for gc in avail_ig:
                        if row[gc] == 1:
                            item_primary[int(row["movieId"])] = gc
                            break
                    else:
                        item_primary[int(row["movieId"])] = None

                ugc_dict: dict[str, dict] = ugc.to_dict()

                def _ghc(row: pd.Series) -> float:
                    pg = item_primary.get(int(row["movieId"]))
                    if pg is None or pg not in ugc_dict:
                        return 0.0
                    return float(ugc_dict[pg].get(row["userId"], 0.0) or 0.0)

                pairs_w = pairs.copy()
                pairs["genre_history_count"] = np.log1p(
                    pairs_w.apply(_ghc, axis=1).values
                )

        # ── mf_score placeholder ──────────────────────────────────────────
        pairs["mf_score"] = 0.0

        self.log_feature_stats(pairs)
        return pairs

    # ── feature names ──────────────────────────────────────────────────────

    def get_feature_names(self) -> list[str]:
        return [
            "genre_overlap_score",
            "tag_profile_similarity",
            "rating_gap",
            "genre_history_count",
            "mf_score",
        ]
