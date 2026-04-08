"""Cold-start routing and A/B experiment routing."""

from __future__ import annotations

import hashlib
import logging
import math

from src.ranking.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)

# User tiers
_TIER_WARM = "warm"
_TIER_LIGHT = "light"
_TIER_NEW = "new"


class ColdStartRouter:
    """Route users to the appropriate recommender based on interaction history.

    Tier classification uses ``log_positive_count`` from ``user_features``
    (stored as ``log1p(positive_rating_count)``).  We reverse with
    ``math.expm1`` to get the actual count.

    Tiers
    -----
    warm:
        ``actual_count >= warm_threshold`` — use full two-stage ranker.
    light:
        ``1 <= actual_count < warm_threshold`` — use CF or ALS baseline.
    new:
        ``actual_count < 1`` — use popularity baseline.

    Parameters
    ----------
    warm_recommender:
        Full two-stage recommender.
    light_recommender:
        Recommender for users with a small number of interactions.
    new_recommender:
        Pure popularity recommender for brand-new users.
    warm_threshold:
        Minimum positive-rating count to be considered warm (default 20,
        matching ``DataConfig.cold_user_threshold``).
    """

    def __init__(
        self,
        warm_recommender: BaseRecommender,
        light_recommender: BaseRecommender,
        new_recommender: BaseRecommender,
        warm_threshold: int = 20,
    ) -> None:
        self._warm = warm_recommender
        self._light = light_recommender
        self._new = new_recommender
        self._threshold = warm_threshold

    def get_tier(self, user_features: dict) -> str:
        """Return the tier string for *user_features*."""
        log_pos = float(user_features.get("log_positive_count", 0.0))
        actual_count = math.expm1(log_pos)
        if actual_count >= self._threshold:
            return _TIER_WARM
        if actual_count >= 1:
            return _TIER_LIGHT
        return _TIER_NEW

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        """Route to the appropriate recommender and return results."""
        tier = self.get_tier(user_features)
        logger.debug("ColdStartRouter: user %d → tier %s.", user_id, tier)

        if tier == _TIER_WARM:
            recommender = self._warm
        elif tier == _TIER_LIGHT:
            recommender = self._light
        else:
            recommender = self._new

        results = recommender.recommend(user_id, user_features, rated_movie_ids, n)

        # Attach tier metadata
        for rec in results:
            rec["tier"] = tier

        return results


class ABRouter:
    """Deterministic A/B routing using MD5 hash of user_id.

    The bucket is computed as ``int(md5(str(user_id))) % 100``.
    Users with bucket >= 50 → "treatment"; bucket < 50 → "control".

    Parameters
    ----------
    control_recommender:
        Baseline recommender (control arm).
    treatment_recommender:
        New recommender under test (treatment arm).
    treatment_fraction:
        Fraction of users routed to treatment (default 0.50 → bucket >= 50).
    """

    def __init__(
        self,
        control_recommender: BaseRecommender,
        treatment_recommender: BaseRecommender,
        treatment_fraction: float = 0.50,
    ) -> None:
        self._control = control_recommender
        self._treatment = treatment_recommender
        self._treatment_bucket = int(round(100 * (1.0 - treatment_fraction)))

    @staticmethod
    def hash_bucket(user_id: int) -> int:
        """Return a deterministic bucket in [0, 100)."""
        hex_digest = hashlib.md5(str(user_id).encode()).hexdigest()
        return int(hex_digest, 16) % 100

    def get_variant(self, user_id: int) -> str:
        """Return "treatment" or "control" for *user_id*."""
        return "treatment" if self.hash_bucket(user_id) >= self._treatment_bucket else "control"

    def recommend(
        self,
        user_id: int,
        user_features: dict,
        rated_movie_ids: set[int],
        n: int = 10,
    ) -> list[dict]:
        """Route to the correct arm and return recommendations."""
        variant = self.get_variant(user_id)
        logger.debug("ABRouter: user %d → variant %s.", user_id, variant)

        recommender = self._treatment if variant == "treatment" else self._control
        results = recommender.recommend(user_id, user_features, rated_movie_ids, n)

        for rec in results:
            rec["variant"] = variant

        return results
