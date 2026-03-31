"""Popularity-based candidate generator.

Ranks items by log_rating_count.  Supports optional genre-aware blending:
70 % global popularity + 30 % genre-specific top items for the user's top-3
genres (by affinity score).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.candidates.base_candidate_generator import BaseCandidateGenerator
from src.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)

_GLOBAL_WEIGHT: float = 0.70
_GENRE_WEIGHT: float = 0.30
_TOP_GENRE_COUNT: int = 3
_GENRE_AFFINITY_PREFIX: str = "genre_affinity_"


class PopularityCandidateGenerator(BaseCandidateGenerator):
    """Retrieves top-N popular items, with optional genre-affinity blending.

    Parameters
    ----------
    item_features_df:
        Cleaned item feature table — must contain ``movieId`` and
        ``log_rating_count``.  Genre columns are expected to be named
        ``genre_{GenreName}`` (as produced by ItemFeatureBuilder).
    config:
        TrainingConfig that supplies ``n_candidates_pop`` (default cap).
    """

    def __init__(
        self,
        item_features_df: pd.DataFrame,
        config: TrainingConfig,
    ) -> None:
        self._config = config

        # Global popularity ranking (descending log_rating_count)
        self._global_rank: list[int] = (
            item_features_df.sort_values("log_rating_count", ascending=False)["movieId"]
            .tolist()
        )

        # Per-genre top-item lists  {genre_name: [movie_id, ...]}
        self._genre_top: dict[str, list[int]] = {}
        genre_cols = [
            c for c in item_features_df.columns if c.startswith("genre_")
        ]
        for col in genre_cols:
            genre_name = col[len("genre_"):]
            top = (
                item_features_df[item_features_df[col] == 1]
                .sort_values("log_rating_count", ascending=False)["movieId"]
                .tolist()
            )
            if top:
                self._genre_top[genre_name] = top

        logger.info(
            "PopularityCandidateGenerator: %d items, %d genres indexed.",
            len(self._global_rank),
            len(self._genre_top),
        )

    # ── generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        user_id: int,
        user_features: dict,
        n: int = 100,
        rated_movie_ids: set[int] | None = None,
    ) -> list[int]:
        """Return up to *n* popular candidates not in *rated_movie_ids*.

        If *user_features* contains ``genre_affinity_*`` keys, the result
        is a 70/30 blend of global and genre-specific popular items.
        """
        rated = rated_movie_ids or set()

        # Identify user's top-3 genres by affinity score
        affinity: dict[str, float] = {
            key[len(_GENRE_AFFINITY_PREFIX):]: val
            for key, val in user_features.items()
            if key.startswith(_GENRE_AFFINITY_PREFIX) and float(val) > 0
        }

        if affinity:
            top_genres = sorted(affinity, key=affinity.get, reverse=True)[  # type: ignore[arg-type]
                :_TOP_GENRE_COUNT
            ]
            candidates = self._blend_candidates(top_genres, rated, n)
        else:
            candidates = self.filter_rated(self._global_rank, rated)

        candidates = self.deduplicate(candidates)
        return candidates[:n]

    # ── helpers ────────────────────────────────────────────────────────────

    def _blend_candidates(
        self,
        top_genres: list[str],
        rated: set[int],
        n: int,
    ) -> list[int]:
        """70 % global + 30 % genre-specific, interleaved proportionally."""
        n_global = max(1, int(n * _GLOBAL_WEIGHT))
        n_genre = max(1, n - n_global)
        n_per_genre = max(1, n_genre // max(len(top_genres), 1))

        global_part = self.filter_rated(self._global_rank, rated)[:n_global]

        genre_part: list[int] = []
        for genre in top_genres:
            items = self._genre_top.get(genre, [])
            genre_part.extend(self.filter_rated(items, rated)[:n_per_genre])

        # Interleave: global first (higher weight), then genre
        return global_part + genre_part
