"""Hybrid candidate generator — union of popularity, CF, and ALS candidates.

Deduplication order: popularity → CF → ALS (priority given to earlier sources).
Falls back to extra popularity candidates when CF or ALS returns empty
(cold-user handling).
"""

from __future__ import annotations

import logging

from src.candidates.base_candidate_generator import BaseCandidateGenerator
from src.candidates.collaborative import CFCandidateGenerator
from src.candidates.matrix_factorization import ALSCandidateGenerator
from src.candidates.popularity import PopularityCandidateGenerator
from src.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class HybridCandidateGenerator(BaseCandidateGenerator):
    """Union-based retrieval from three complementary sources.

    Parameters
    ----------
    popularity_gen / cf_gen / als_gen:
        Pre-built generators.
    config:
        TrainingConfig — supplies ``n_candidates_pop``, ``n_candidates_cf``,
        ``n_candidates_mf``, and ``candidate_pool_size``.
    """

    def __init__(
        self,
        popularity_gen: PopularityCandidateGenerator,
        cf_gen: CFCandidateGenerator,
        als_gen: ALSCandidateGenerator,
        config: TrainingConfig,
    ) -> None:
        self._pop = popularity_gen
        self._cf = cf_gen
        self._als = als_gen
        self._config = config

    # ── generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        user_id: int,
        user_features: dict,
        n: int = 300,
        rated_movie_ids: set[int] | None = None,
    ) -> list[int]:
        """Return up to *n* deduplicated candidates (no rated movies).

        Retrieval budget per source (from config):
          - popularity : n_candidates_pop  (100)
          - CF         : n_candidates_cf   (100)
          - ALS        : n_candidates_mf   (100)

        Cold-user fallback: if CF or ALS returns empty, the missing quota
        is filled by extra popularity candidates.
        """
        rated = rated_movie_ids or set()
        n_pop = self._config.n_candidates_pop
        n_cf = self._config.n_candidates_cf
        n_mf = self._config.n_candidates_mf

        pop_candidates = self._pop.generate(user_id, user_features, n_pop, rated)
        cf_candidates = self._cf.generate(user_id, user_features, n_cf, rated)
        als_candidates = self._als.generate(user_id, user_features, n_mf, rated)

        extra_needed = 0
        if not cf_candidates:
            logger.debug(
                "HybridCandidateGenerator: CF empty for user %d — using extra popularity.",
                user_id,
            )
            extra_needed += n_cf
        if not als_candidates:
            logger.debug(
                "HybridCandidateGenerator: ALS empty for user %d — using extra popularity.",
                user_id,
            )
            extra_needed += n_mf

        if extra_needed > 0:
            total_pop = n_pop + extra_needed
            pop_candidates = self._pop.generate(user_id, user_features, total_pop, rated)

        # Union: popularity first (highest priority), then CF, then ALS
        combined = pop_candidates + cf_candidates + als_candidates
        combined = self.deduplicate(combined)
        combined = self.filter_rated(combined, rated)
        return combined[:n]
