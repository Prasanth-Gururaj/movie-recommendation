"""Tests for ColdStartRouter, ABRouter, and TwoStageRecommender."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ranking.cold_start import ColdStartRouter, ABRouter
from src.ranking.base_recommender import BaseRecommender


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pop_recommender(movie_ids: list[int], score: float = 1.0) -> BaseRecommender:
    """Return a mock BaseRecommender that returns a fixed list."""
    mock = MagicMock(spec=BaseRecommender)
    mock.recommend.side_effect = lambda user_id, user_features, rated_movie_ids, n=10: [
        {"movie_id": mid, "score": score, "reason_code": "pop"}
        for mid in movie_ids[:n]
        if mid not in rated_movie_ids
    ]
    return mock


def _log1p_count(n: int) -> float:
    """Return log1p(n) — matches how user_features stores positive count."""
    return math.log1p(n)


# ── ColdStartRouter ───────────────────────────────────────────────────────────

class TestColdStartRouterTiers:
    """Test tier classification at and around the warm_threshold boundary."""

    @pytest.fixture
    def router(self):
        warm = _make_pop_recommender(list(range(100, 120)))
        light = _make_pop_recommender(list(range(200, 220)))
        new = _make_pop_recommender(list(range(300, 320)))
        return ColdStartRouter(
            warm_recommender=warm,
            light_recommender=light,
            new_recommender=new,
            warm_threshold=20,
        )

    def test_zero_positives_is_new(self, router):
        features = {"log_positive_count": _log1p_count(0)}
        assert router.get_tier(features) == "new"

    def test_one_positive_is_light(self, router):
        features = {"log_positive_count": _log1p_count(1)}
        assert router.get_tier(features) == "light"

    def test_19_positives_is_light(self, router):
        """19 positives < threshold 20 → light."""
        features = {"log_positive_count": _log1p_count(19)}
        assert router.get_tier(features) == "light"

    def test_20_positives_is_warm(self, router):
        """Exactly at threshold → warm."""
        features = {"log_positive_count": _log1p_count(20)}
        assert router.get_tier(features) == "warm"

    def test_100_positives_is_warm(self, router):
        features = {"log_positive_count": _log1p_count(100)}
        assert router.get_tier(features) == "warm"

    def test_missing_feature_is_new(self, router):
        """Missing log_positive_count defaults to 0 → new tier."""
        assert router.get_tier({}) == "new"

    def test_warm_user_routed_to_warm_recommender(self, router):
        features = {"log_positive_count": _log1p_count(50)}
        results = router.recommend(1, features, set(), n=5)
        # warm recommender returns movies 100-119
        assert all(r["movie_id"] in range(100, 120) for r in results)

    def test_light_user_routed_to_light_recommender(self, router):
        features = {"log_positive_count": _log1p_count(5)}
        results = router.recommend(2, features, set(), n=5)
        assert all(r["movie_id"] in range(200, 220) for r in results)

    def test_new_user_routed_to_new_recommender(self, router):
        features = {"log_positive_count": _log1p_count(0)}
        results = router.recommend(3, features, set(), n=5)
        assert all(r["movie_id"] in range(300, 320) for r in results)

    def test_tier_attached_to_result(self, router):
        features = {"log_positive_count": _log1p_count(50)}
        results = router.recommend(1, features, set(), n=3)
        assert all("tier" in r for r in results)
        assert results[0]["tier"] == "warm"

    def test_rated_movies_excluded(self, router):
        features = {"log_positive_count": _log1p_count(50)}
        rated = {100, 101, 102}
        results = router.recommend(1, features, rated, n=5)
        for r in results:
            assert r["movie_id"] not in rated


# ── ABRouter ─────────────────────────────────────────────────────────────────

class TestABRouter:
    @pytest.fixture
    def ab_router(self):
        control = _make_pop_recommender(list(range(400, 420)))
        treatment = _make_pop_recommender(list(range(500, 520)))
        return ABRouter(
            control_recommender=control,
            treatment_recommender=treatment,
            treatment_fraction=0.50,
        )

    def test_get_variant_returns_control_or_treatment(self, ab_router):
        for uid in range(50):
            variant = ab_router.get_variant(uid)
            assert variant in ("control", "treatment")

    def test_hash_bucket_deterministic(self):
        """Same user_id always maps to same bucket."""
        for uid in [1, 42, 999, 12345]:
            b1 = ABRouter.hash_bucket(uid)
            b2 = ABRouter.hash_bucket(uid)
            assert b1 == b2

    def test_hash_bucket_in_range(self):
        for uid in range(200):
            b = ABRouter.hash_bucket(uid)
            assert 0 <= b < 100

    def test_assignment_is_stable(self, ab_router):
        """Same user always gets same variant."""
        for uid in [7, 42, 100, 9999]:
            v1 = ab_router.get_variant(uid)
            v2 = ab_router.get_variant(uid)
            assert v1 == v2

    def test_roughly_50_50_split(self):
        """With 50% fraction, approximately half of users in each arm."""
        control = _make_pop_recommender([1])
        treatment = _make_pop_recommender([2])
        router = ABRouter(control, treatment, treatment_fraction=0.50)
        variants = [router.get_variant(uid) for uid in range(1000)]
        treatment_frac = variants.count("treatment") / 1000
        # Allow generous tolerance — MD5 distribution is uniform enough
        assert 0.40 <= treatment_frac <= 0.60

    def test_variant_attached_to_results(self, ab_router):
        results = ab_router.recommend(1, {}, set(), n=3)
        assert all("variant" in r for r in results)

    def test_known_user_bucket(self):
        """Spot-check a specific user bucket for regression."""
        import hashlib
        uid = 42
        expected = int(hashlib.md5(str(uid).encode()).hexdigest(), 16) % 100
        assert ABRouter.hash_bucket(uid) == expected


# ── TwoStageRecommender ───────────────────────────────────────────────────────

def _make_item_df(n: int = 50) -> pd.DataFrame:
    """Create a minimal item features DataFrame."""
    return pd.DataFrame({
        "movieId": list(range(1, n + 1)),
        "log_rating_count": [float(n - i) for i in range(n)],
        "is_cold": [0] * n,
    })


def _make_config(pool_size: int = 10):
    cfg = MagicMock()
    cfg.training.candidate_pool_size = pool_size
    return cfg


class TestTwoStageRecommender:
    """TwoStageRecommender integration-style tests with mocked dependencies."""

    def _build_recommender(
        self,
        candidate_ids: list[int],
        scores: list[float] | None = None,
        item_df: pd.DataFrame | None = None,
        n_items: int = 50,
        fail_ranker: bool = False,
    ):
        from src.ranking.two_stage_recommender import TwoStageRecommender

        if item_df is None:
            item_df = _make_item_df(n_items)

        # Hybrid generator mock
        hybrid = MagicMock()
        hybrid.generate.return_value = candidate_ids

        # Feature store mock
        feature_store = MagicMock()
        feature_store.feature_columns = ["log_rating_count", "is_cold"]
        n_cands = len(candidate_ids)
        feature_store.assemble_inference_features.return_value = pd.DataFrame(
            {"log_rating_count": [1.0] * n_cands, "is_cold": [0] * n_cands}
        )

        # Ranker mock
        ranker = MagicMock()
        if fail_ranker:
            ranker.predict.side_effect = RuntimeError("ranker failed")
        elif scores is not None:
            ranker.predict.return_value = np.array(scores)
        else:
            ranker.predict.return_value = np.arange(n_cands, 0, -1, dtype=float)

        config = _make_config(pool_size=max(n_cands, 10))
        return TwoStageRecommender(hybrid, feature_store, ranker, item_df, config)

    def test_returns_exactly_n_results(self):
        from src.ranking.two_stage_recommender import TwoStageRecommender

        rec = self._build_recommender(candidate_ids=list(range(1, 21)))
        results = rec.recommend(1, {}, set(), n=10)
        assert len(results) == 10

    def test_no_rated_movies_in_results(self):
        rated = {1, 2, 3}
        rec = self._build_recommender(candidate_ids=list(range(1, 21)))
        results = rec.recommend(1, {}, rated, n=10)
        result_ids = {r["movie_id"] for r in results}
        assert result_ids.isdisjoint(rated)

    def test_results_have_required_keys(self):
        rec = self._build_recommender(candidate_ids=list(range(1, 11)))
        results = rec.recommend(1, {}, set(), n=5)
        for r in results:
            assert "movie_id" in r
            assert "score" in r
            assert "reason_code" in r

    def test_fallback_on_ranker_failure(self):
        """When ranker raises, falls back to popularity order."""
        rec = self._build_recommender(
            candidate_ids=list(range(1, 11)),
            fail_ranker=True,
        )
        results = rec.recommend(1, {}, set(), n=5)
        # Should still return results without raising
        assert len(results) > 0
        assert all(r["reason_code"] == "popular_fallback" for r in results)

    def test_fallback_on_empty_candidates(self):
        """When no candidates generated, falls back to popularity."""
        rec = self._build_recommender(candidate_ids=[])
        results = rec.recommend(1, {}, set(), n=5)
        assert len(results) > 0
        assert all(r["reason_code"] == "popular_fallback" for r in results)

    def test_higher_scored_items_ranked_first(self):
        candidates = [10, 20, 30]
        # Give item 30 the highest score
        scores = [1.0, 2.0, 5.0]
        rec = self._build_recommender(candidate_ids=candidates, scores=scores)
        results = rec.recommend(1, {}, set(), n=3)
        assert results[0]["movie_id"] == 30

    def test_pads_with_fallback_when_too_few_candidates(self):
        """If ranker returns fewer than n, pad with popularity fallback."""
        item_df = _make_item_df(20)
        rec = self._build_recommender(
            candidate_ids=[1, 2],  # only 2 candidates
            item_df=item_df,
        )
        results = rec.recommend(1, {}, set(), n=10)
        assert len(results) == 10
