"""Abstract base recommender."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """All recommenders return a ranked list of dicts.

    Each dict: ``{"movie_id": int, "score": float, "reason_code": str}``

    Implementations must:
    - Never raise exceptions.
    - Never return an empty list when a fallback exists.
    - Never include already-rated movies.
    - Return exactly *n* results when possible.
    """

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        """Return up to *n* ranked recommendations for *user_id*."""
