"""Abstract base class for all candidate generators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseCandidateGenerator(ABC):
    """Base for every candidate generator in Stage 1 retrieval.

    Subclasses implement ``generate()``.  Concrete helpers ``filter_rated``
    and ``deduplicate`` are provided here so every generator behaves
    identically when cleaning candidates.
    """

    @abstractmethod
    def generate(
        self,
        user_id: int,
        user_features: dict,
        n: int = 300,
        rated_movie_ids: set[int] | None = None,
    ) -> list[int]:
        """Return up to *n* candidate movie IDs for *user_id*.

        Implementations must:
        - Never include movie IDs in *rated_movie_ids*.
        - Never return more than *n* candidates.
        - Return an empty list (not raise) for cold / unknown users.
        """

    # ── concrete helpers ───────────────────────────────────────────────────

    def filter_rated(
        self,
        candidates: list[int],
        rated_movie_ids: set[int],
    ) -> list[int]:
        """Remove any movie in *rated_movie_ids* from *candidates*.

        Preserves the original order of surviving elements.
        """
        return [mid for mid in candidates if mid not in rated_movie_ids]

    def deduplicate(self, candidates: list[int]) -> list[int]:
        """Remove duplicates from *candidates* while preserving insertion order."""
        seen: set[int] = set()
        result: list[int] = []
        for mid in candidates:
            if mid not in seen:
                seen.add(mid)
                result.append(mid)
        return result
