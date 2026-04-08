"""Non-ML baseline recommenders used for benchmarking."""

from __future__ import annotations

import logging

import pandas as pd

from src.ranking.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """Recommend the globally most popular unseen movies."""

    def __init__(self, item_features_df: pd.DataFrame) -> None:
        self._df = item_features_df.copy()

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        candidates = self._df[~self._df["movieId"].isin(rated_movie_ids)]
        candidates = candidates.sort_values("log_rating_count", ascending=False)
        results = []
        for _, row in candidates.head(n).iterrows():
            results.append(
                {
                    "movie_id": int(row["movieId"]),
                    "score": float(row["log_rating_count"]),
                    "reason_code": "popular",
                }
            )
        return results


class GenrePopularityRecommender(BaseRecommender):
    """Popularity-based recommender that re-weights by user genre affinity."""

    def __init__(self, item_features_df: pd.DataFrame) -> None:
        self._df = item_features_df.copy()
        self._genre_cols = [c for c in item_features_df.columns if c.startswith("genre_")]

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        candidates = self._df[~self._df["movieId"].isin(rated_movie_ids)].copy()

        # Build genre affinity vector from user_features
        affinity = {}
        for col in self._genre_cols:
            genre = col[len("genre_"):]
            aff_key = f"genre_affinity_{genre}"
            affinity[col] = float(user_features.get(aff_key, 0.0))

        # Score = log_rating_count + sum of (affinity * genre indicator)
        genre_score = sum(
            candidates[col] * weight
            for col, weight in affinity.items()
            if col in candidates.columns
        )
        candidates = candidates.copy()
        candidates["_score"] = candidates["log_rating_count"] + genre_score
        candidates = candidates.sort_values("_score", ascending=False)

        results = []
        for _, row in candidates.head(n).iterrows():
            results.append(
                {
                    "movie_id": int(row["movieId"]),
                    "score": float(row["_score"]),
                    "reason_code": "genre_popular",
                }
            )
        return results


class CFRecommender(BaseRecommender):
    """Baseline that wraps a :class:`~src.candidates.CFCandidateGenerator`."""

    def __init__(self, cf_gen, item_features_df: pd.DataFrame) -> None:
        self._cf = cf_gen
        self._popularity = {
            int(row["movieId"]): float(row["log_rating_count"])
            for _, row in item_features_df.iterrows()
        }

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        candidates = self._cf.generate(user_id, user_features, n * 3, rated_movie_ids)
        # Score by position (first = most similar)
        results = []
        for rank, mid in enumerate(candidates[:n]):
            results.append(
                {
                    "movie_id": mid,
                    "score": float(len(candidates) - rank),
                    "reason_code": "similar_to_watched",
                }
            )
        return results


class ALSRecommender(BaseRecommender):
    """Baseline that wraps an :class:`~src.candidates.ALSCandidateGenerator`."""

    def __init__(self, als_gen) -> None:
        self._als = als_gen

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        candidates = self._als.generate(user_id, user_features, n * 3, rated_movie_ids)
        scores = self._als.get_mf_scores(user_id, candidates)
        ranked = sorted(candidates, key=lambda mid: scores.get(mid, 0.0), reverse=True)
        return [
            {
                "movie_id": mid,
                "score": float(scores.get(mid, 0.0)),
                "reason_code": "als",
            }
            for mid in ranked[:n]
        ]
