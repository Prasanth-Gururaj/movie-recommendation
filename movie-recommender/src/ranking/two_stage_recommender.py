"""Two-stage recommender: candidate retrieval → LTR ranking."""

from __future__ import annotations

import logging

import pandas as pd

from src.ranking.base_recommender import BaseRecommender
from src.ranking.base_ranker import BaseRanker

logger = logging.getLogger(__name__)


class TwoStageRecommender(BaseRecommender):
    """Full retrieval-then-ranking pipeline.

    Stage 1 — candidate generation via :class:`HybridCandidateGenerator`.
    Stage 2 — feature assembly via :class:`FeatureStore` + LTR scoring.

    Falls back to the popularity fallback list when the ranker produces
    fewer than *n* results.

    Parameters
    ----------
    hybrid_gen:
        Fitted ``HybridCandidateGenerator``.
    feature_store:
        Fitted ``FeatureStore`` (has ``.feature_columns`` and
        ``.assemble_inference_features()``).
    ranker:
        Fitted ``BaseRanker`` subclass.
    item_features_df:
        Item features for fallback and score enrichment.
    config:
        ``ExperimentConfig`` — used for ``candidate_pool_size``.
    """

    def __init__(
        self,
        hybrid_gen,
        feature_store,
        ranker: BaseRanker,
        item_features_df: pd.DataFrame,
        config,
    ) -> None:
        self._hybrid = hybrid_gen
        self._feature_store = feature_store
        self._ranker = ranker
        self._item_df = item_features_df.set_index("movieId") if "movieId" in item_features_df.columns else item_features_df
        self._config = config
        # Pre-compute popularity fallback order
        self._fallback_order: list[int] = (
            item_features_df.sort_values("log_rating_count", ascending=False)["movieId"]
            .astype(int)
            .tolist()
        )

    # ── recommend ─────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        pool_size = self._config.training.candidate_pool_size

        # Stage 1: retrieve candidates
        candidates = self._hybrid.generate(
            user_id,
            user_features,
            n=pool_size,
            rated_movie_ids=rated_movie_ids,
        )

        if not candidates:
            logger.debug(
                "TwoStageRecommender: no candidates for user %d — using fallback.", user_id
            )
            return self._fallback(rated_movie_ids, n)

        # Stage 2: assemble features and score
        try:
            item_feature_list = self._build_item_feature_list(candidates)
            features_df = self._feature_store.assemble_inference_features(
                user_features=user_features,
                item_features_list=item_feature_list,
                request_context={},
            )
            scores = self._ranker.predict(features_df, self._feature_store.feature_columns)
        except Exception as exc:
            logger.warning(
                "TwoStageRecommender: ranking failed for user %d (%s) — using fallback.",
                user_id,
                exc,
            )
            return self._fallback(rated_movie_ids, n)

        # Sort by score descending, excluding any rated items that slipped through
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        results = [
            {
                "movie_id": mid,
                "score": float(score),
                "reason_code": "ranked",
            }
            for mid, score in ranked
            if mid not in rated_movie_ids
        ][:n]

        # Pad with fallback if needed
        if len(results) < n:
            seen = {r["movie_id"] for r in results}
            for mid in self._fallback_order:
                if mid not in seen and mid not in rated_movie_ids:
                    results.append(
                        {"movie_id": mid, "score": 0.0, "reason_code": "popular_fallback"}
                    )
                if len(results) >= n:
                    break

        return results

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_item_feature_list(self, movie_ids: list[int]) -> list[dict]:
        rows = []
        for mid in movie_ids:
            if mid in self._item_df.index:
                rows.append(self._item_df.loc[mid].to_dict())
            else:
                rows.append({"movieId": mid})
        return rows

    def _fallback(self, rated_movie_ids: set[int], n: int) -> list[dict]:
        results = []
        for mid in self._fallback_order:
            if mid not in rated_movie_ids:
                results.append(
                    {"movie_id": mid, "score": 0.0, "reason_code": "popular_fallback"}
                )
            if len(results) >= n:
                break
        return results
