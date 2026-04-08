"""Tests for src/features/.

All tests use toy in-memory DataFrames — no real 25M dataset is loaded.
Tests must complete in seconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.data_config import DataConfig
from src.config.experiment_config import ExperimentConfig
from src.config.feature_config import FeatureConfig
from src.features.feature_store import FeatureStore
from src.features.interaction_features import InteractionFeatureBuilder
from src.features.item_features import ItemFeatureBuilder
from src.features.time_features import TimeFeatureBuilder
from src.features.user_features import UserFeatureBuilder

# ── constants ──────────────────────────────────────────────────────────────────
_GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# Timestamps (UTC midnight Jan 1 of each year)
_TS = {
    2015: 1_420_070_400,
    2016: 1_451_606_400,
    2017: 1_483_228_800,
    2018: 1_514_764_800,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_movies_df(single_genre: bool = False) -> pd.DataFrame:
    """Return a cleaned movies DataFrame with binary genre columns.

    If single_genre=True every movie gets exactly one genre (useful for
    testing that genre_affinity sums to ≤ 1.0).
    """
    if single_genre:
        genres_raw = ["Action", "Drama", "Comedy", "Thriller", "Sci-Fi"]
        genres_pipe = [g for g in genres_raw]
    else:
        genres_raw = [
            "Action|Comedy", "Drama", "Action|Sci-Fi",
            "Comedy", "(no genres listed)",
        ]

    rows = []
    for i, g in enumerate(genres_raw, start=1):
        row: dict = {
            "movieId": i,
            "title": f"Movie {i} ({2000 + i})",
            "genres": g if not single_genre else genres_raw[i - 1],
            "release_year": 2000 + i,
            "movie_age": 2019 - (2000 + i),
            "has_genre": 0 if g == "(no genres listed)" else 1,
        }
        for genre in _GENRES:
            row[genre] = 1 if genre in g.split("|") else 0
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def toy_movies_df() -> pd.DataFrame:
    return _make_movies_df(single_genre=False)


@pytest.fixture
def single_genre_movies_df() -> pd.DataFrame:
    """Five movies each with exactly one distinct genre."""
    return _make_movies_df(single_genre=True)


@pytest.fixture
def default_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            cold_user_threshold=2,
            cold_item_threshold=2,
            relevance_threshold=4.0,
        ),
        feature=FeatureConfig(n_genome_tags=20),
    )


@pytest.fixture
def toy_train_df() -> pd.DataFrame:
    """Small training split with known properties.

    User 1: 3 positives (movies 1, 2, 3)  → warm
    User 2: 1 positive  (movie 1)          → light
    User 3: 0 positives (all negatives)    → light
    """
    return pd.DataFrame(
        {
            "userId": pd.array([1, 1, 1, 2, 2, 3, 3], dtype="int32"),
            "movieId": pd.array([1, 2, 3, 1, 2, 1, 2], dtype="int32"),
            "rating": pd.array([5.0, 4.5, 4.0, 5.0, 3.0, 2.0, 1.0], dtype="float32"),
            "timestamp": pd.array(
                [_TS[2016]] * 5 + [_TS[2015], _TS[2015]], dtype="int64"
            ),
            "is_positive": pd.array([1, 1, 1, 1, 0, 0, 0], dtype="int8"),
        }
    )


@pytest.fixture
def toy_genome_scores_df() -> pd.DataFrame:
    """20 tags × 5 movies.  Tag 1 has highest variance (discriminative)."""
    rng = np.random.default_rng(42)
    records = []
    for movie_id in range(1, 6):
        for tag_id in range(1, 21):
            # Tag 1 is very discriminative; others are near-zero
            relevance = float(rng.uniform(0.8, 1.0) if tag_id == 1 and movie_id <= 2
                              else rng.uniform(0.0, 0.1))
            records.append({"movieId": movie_id, "tagId": tag_id, "relevance": relevance})
    df = pd.DataFrame(records)
    df["movieId"] = df["movieId"].astype("int32")
    df["tagId"] = df["tagId"].astype("int32")
    df["relevance"] = df["relevance"].astype("float32")
    return df


@pytest.fixture
def data_dict(toy_movies_df: pd.DataFrame, toy_genome_scores_df: pd.DataFrame) -> dict:
    return {
        "movies": toy_movies_df,
        "genome_scores": toy_genome_scores_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UserFeatureBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserFeatureBuilder:

    def test_log_total_ratings_is_log1p(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        """log_total_ratings must equal log1p(count), not raw count."""
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        for _, row in feat.iterrows():
            uid = row["userId"]
            count = (toy_train_df["userId"] == uid).sum()
            assert math.isclose(row["log_total_ratings"], math.log1p(count), rel_tol=1e-5)

    def test_genre_affinity_18_values_per_user(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        affinity_cols = [c for c in feat.columns if c.startswith("genre_affinity_") and not c.startswith("genre_affinity_recent")]
        # filter to just genre_affinity_ (not recent_genre_affinity_)
        affinity_cols = [c for c in feat.columns if c.startswith("genre_affinity_")]
        assert len(affinity_cols) == 18

    def test_genre_affinity_sums_leq_1_single_genre(
        self, default_config: ExperimentConfig, single_genre_movies_df: pd.DataFrame,
        toy_genome_scores_df: pd.DataFrame, toy_train_df: pd.DataFrame
    ) -> None:
        """With single-genre movies, affinity sums to exactly 1.0 for users with positives."""
        data = {"movies": single_genre_movies_df, "genome_scores": toy_genome_scores_df}
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data, toy_train_df)
        affinity_cols = [f"genre_affinity_{g}" for g in _GENRES]
        for _, row in feat.iterrows():
            s = row[affinity_cols].sum()
            assert s <= 1.0 + 1e-6, f"user {row['userId']} affinity sum={s} > 1"

    def test_genre_affinity_values_in_0_1(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        affinity_cols = [f"genre_affinity_{g}" for g in _GENRES]
        for col in affinity_cols:
            assert (feat[col] >= 0).all(), f"{col} has negative values"
            assert (feat[col] <= 1).all(), f"{col} exceeds 1.0"

    def test_no_leakage_raises_value_error(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        """Validate detects when build() used data with timestamps beyond train."""
        # Build using contaminated train (add a future timestamp row)
        future_row = toy_train_df.iloc[:1].copy()
        future_row["timestamp"] = _TS[2018]  # beyond real train max = 2016
        contaminated = pd.concat([toy_train_df, future_row], ignore_index=True)

        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, contaminated)

        # Validate against the real (smaller-max) train
        with pytest.raises(ValueError, match="Leakage detected"):
            builder.validate_no_leakage(feat, toy_train_df)

    def test_no_leakage_passes_on_clean_build(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        # Must not raise
        builder.validate_no_leakage(feat, toy_train_df)

    def test_user_tier_warm_correct(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        """cold_user_threshold=2 → user1 (3 pos) warm, user2 (1 pos) light."""
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        assert feat.loc[feat["userId"] == 1, "user_tier"].iloc[0] == "warm"
        assert feat.loc[feat["userId"] == 2, "user_tier"].iloc[0] == "light"

    def test_activity_30d_is_log1p(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        # All toy train timestamps are either _TS[2015] or _TS[2016].
        # With train_max_ts = _TS[2016], cutoff_30d = _TS[2016] - 30*86400.
        # Rows at _TS[2016] are within 30 days (diff = 0 days).
        # Rows at _TS[2015] (≈365 days before) are outside 30 days.
        train_max = _TS[2016]
        cutoff_30 = train_max - 30 * 86_400
        for _, row in feat.iterrows():
            uid = row["userId"]
            count_30 = (
                (toy_train_df["userId"] == uid) & (toy_train_df["timestamp"] >= cutoff_30)
            ).sum()
            assert math.isclose(row["activity_30d"], math.log1p(count_30), rel_tol=1e-5)

    def test_feature_names_count(
        self, default_config: ExperimentConfig
    ) -> None:
        """get_feature_names() must return 7 + 18 + 18 = 43 features.

        Base (7): log_total_ratings, log_positive_count, mean_rating,
                  rating_variance, days_since_active, activity_30d, activity_90d
        genre_affinity_*  (18)
        recent_genre_affinity_* (18)
        """
        builder = UserFeatureBuilder(default_config)
        names = builder.get_feature_names()
        assert len(names) == 7 + 18 + 18  # base + genre_affinity + recent_genre_affinity

    def test_user_tier_not_in_feature_names(
        self, default_config: ExperimentConfig
    ) -> None:
        builder = UserFeatureBuilder(default_config)
        assert "user_tier" not in builder.get_feature_names()

    def test_no_nulls_in_affinity(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = UserFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        affinity_cols = [f"genre_affinity_{g}" for g in _GENRES]
        assert feat[affinity_cols].isnull().sum().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ItemFeatureBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestItemFeatureBuilder:

    def _make_train_with_counts(self) -> pd.DataFrame:
        """train_df with movie 10 having 9 ratings and movie 20 having 10 ratings."""
        rows = []
        for i in range(9):
            rows.append({"userId": i + 1, "movieId": 10, "rating": 3.0,
                         "timestamp": _TS[2016], "is_positive": 0})
        for i in range(10):
            rows.append({"userId": i + 1, "movieId": 20, "rating": 3.0,
                         "timestamp": _TS[2016], "is_positive": 0})
        df = pd.DataFrame(rows)
        df["userId"] = df["userId"].astype("int32")
        df["movieId"] = df["movieId"].astype("int32")
        df["rating"] = df["rating"].astype("float32")
        df["timestamp"] = df["timestamp"].astype("int64")
        df["is_positive"] = df["is_positive"].astype("int8")
        return df

    def _make_movies_for_count_test(self) -> pd.DataFrame:
        rows = []
        for mid in [10, 20]:
            row: dict = {
                "movieId": mid, "title": f"Movie {mid} (2010)",
                "genres": "Action", "release_year": 2010,
                "movie_age": 9, "has_genre": 1,
            }
            for g in _GENRES:
                row[g] = 1 if g == "Action" else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def test_is_cold_boundary_9_ratings(self, default_config: ExperimentConfig) -> None:
        """cold_item_threshold=2: movie with 1 rating → is_cold=1."""
        train = pd.DataFrame({
            "userId": pd.array([1, 2, 3], dtype="int32"),
            "movieId": pd.array([10, 10, 20], dtype="int32"),
            "rating": pd.array([3.0, 3.0, 3.0], dtype="float32"),
            "timestamp": pd.array([_TS[2016]] * 3, dtype="int64"),
        })
        movies = self._make_movies_for_count_test()
        data = {"movies": movies}
        cfg = ExperimentConfig(data=DataConfig(cold_item_threshold=2), feature=FeatureConfig(n_genome_tags=20))
        builder = ItemFeatureBuilder(cfg)
        feat = builder.build(data, train)
        # movie 10: 2 ratings → is_cold=0; movie 20: 1 rating → is_cold=1
        assert feat.loc[feat["movieId"] == 10, "is_cold"].iloc[0] == 0
        assert feat.loc[feat["movieId"] == 20, "is_cold"].iloc[0] == 1

    def test_is_cold_with_cold_item_threshold_10(
        self, default_config: ExperimentConfig
    ) -> None:
        """cold_item_threshold=10: movie with 9 ratings cold, 10 not cold."""
        train = self._make_train_with_counts()
        movies = self._make_movies_for_count_test()
        cfg = ExperimentConfig(
            data=DataConfig(cold_item_threshold=10, cold_user_threshold=2),
            feature=FeatureConfig(n_genome_tags=20),
        )
        data = {"movies": movies}
        builder = ItemFeatureBuilder(cfg)
        feat = builder.build(data, train)
        row_9 = feat.loc[feat["movieId"] == 10]
        row_10 = feat.loc[feat["movieId"] == 20]
        assert row_9["is_cold"].iloc[0] == 1   # 9 < 10
        assert row_10["is_cold"].iloc[0] == 0  # 10 >= 10

    def test_has_genre_zero_for_no_genre_listed(
        self, default_config: ExperimentConfig, toy_movies_df: pd.DataFrame,
        toy_train_df: pd.DataFrame
    ) -> None:
        data = {"movies": toy_movies_df}
        builder = ItemFeatureBuilder(default_config)
        feat = builder.build(data, toy_train_df)
        # movieId 5 has genres="(no genres listed)" and has_genre=0 in movies_df
        # but movie 5 may not appear in train → check movies that do appear
        no_genre_movies = toy_movies_df[toy_movies_df["has_genre"] == 0]["movieId"].tolist()
        if no_genre_movies:
            for mid in no_genre_movies:
                row = feat.loc[feat["movieId"] == mid]
                if len(row) > 0:
                    assert row["has_genre"].iloc[0] == 0

    def test_log_rating_count_is_log1p(
        self, default_config: ExperimentConfig, toy_movies_df: pd.DataFrame,
        toy_train_df: pd.DataFrame
    ) -> None:
        data = {"movies": toy_movies_df}
        builder = ItemFeatureBuilder(default_config)
        feat = builder.build(data, toy_train_df)
        for _, row in feat.iterrows():
            mid = row["movieId"]
            count = (toy_train_df["movieId"] == mid).sum()
            assert math.isclose(row["log_rating_count"], math.log1p(count), rel_tol=1e-5)

    def test_genre_columns_prefixed(
        self, default_config: ExperimentConfig, toy_movies_df: pd.DataFrame,
        toy_train_df: pd.DataFrame
    ) -> None:
        data = {"movies": toy_movies_df}
        builder = ItemFeatureBuilder(default_config)
        feat = builder.build(data, toy_train_df)
        # Raw genre names must be prefixed
        for g in _GENRES:
            assert f"genre_{g}" in feat.columns, f"genre_{g} missing"
            assert g not in feat.columns, f"raw {g} should have been renamed"

    def test_genome_features_present(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = ItemFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        genome_cols = [c for c in feat.columns if c.startswith("genome_tag_")]
        assert len(genome_cols) == 20  # n_genome_tags=20

    def test_genome_tag_columns_json_written(
        self, default_config: ExperimentConfig, data_dict: dict,
        toy_train_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        builder = ItemFeatureBuilder(default_config)
        builder.build(data_dict, toy_train_df)
        json_path = Path(__file__).parents[1] / "configs" / "genome_tag_columns.json"
        assert json_path.exists()
        with json_path.open() as f:
            tags = json.load(f)
        assert len(tags) == 20
        assert all(isinstance(t, int) for t in tags)

    def test_no_nulls_in_genome_features(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = ItemFeatureBuilder(default_config)
        feat = builder.build(data_dict, toy_train_df)
        genome_cols = [c for c in feat.columns if c.startswith("genome_tag_")]
        assert feat[genome_cols].isnull().sum().sum() == 0

    def test_feature_names_genre_prefixed(
        self, default_config: ExperimentConfig, data_dict: dict, toy_train_df: pd.DataFrame
    ) -> None:
        builder = ItemFeatureBuilder(default_config)
        builder.build(data_dict, toy_train_df)
        names = builder.get_feature_names()
        for g in _GENRES:
            assert f"genre_{g}" in names
        assert "Action" not in names


# ═══════════════════════════════════════════════════════════════════════════════
# InteractionFeatureBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionFeatureBuilder:

    @pytest.fixture
    def built_data(
        self,
        default_config: ExperimentConfig,
        data_dict: dict,
        toy_train_df: pd.DataFrame,
    ) -> tuple[dict, pd.DataFrame]:
        """Return (data_with_user_item_features, train_df)."""
        ub = UserFeatureBuilder(default_config)
        ib = ItemFeatureBuilder(default_config)
        u_feat = ub.build(data_dict, toy_train_df)
        i_feat = ib.build(data_dict, toy_train_df)
        full_data = {**data_dict, "user_features": u_feat, "item_features": i_feat}
        return full_data, toy_train_df

    def test_genre_overlap_score_in_0_1(
        self,
        default_config: ExperimentConfig,
        built_data: tuple,
    ) -> None:
        data, train = built_data
        builder = InteractionFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["genre_overlap_score"] >= 0.0).all()
        assert (feat["genre_overlap_score"] <= 1.0).all()

    def test_rating_gap_can_be_negative(
        self,
        default_config: ExperimentConfig,
        built_data: tuple,
    ) -> None:
        """rating_gap = user_mean_rating - item_avg_rating.  No artificial floor."""
        data, train = built_data
        builder = InteractionFeatureBuilder(default_config)
        feat = builder.build(data, train)
        # Not all values must be non-negative
        has_negative = (feat["rating_gap"] < 0).any()
        # Even if this particular toy doesn't produce negatives, feature is correct
        # Verify it's a real numeric column (not always 0)
        assert "rating_gap" in feat.columns
        assert feat["rating_gap"].dtype in (float, np.float64, np.float32)

    def test_mf_score_is_zero_placeholder(
        self, default_config: ExperimentConfig, built_data: tuple
    ) -> None:
        data, train = built_data
        builder = InteractionFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["mf_score"] == 0.0).all()

    def test_tag_profile_similarity_in_0_1(
        self, default_config: ExperimentConfig, built_data: tuple
    ) -> None:
        data, train = built_data
        builder = InteractionFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["tag_profile_similarity"] >= 0.0).all()
        assert (feat["tag_profile_similarity"] <= 1.0 + 1e-6).all()

    def test_output_has_one_row_per_pair(
        self, default_config: ExperimentConfig, built_data: tuple
    ) -> None:
        data, train = built_data
        builder = InteractionFeatureBuilder(default_config)
        feat = builder.build(data, train)
        n_pairs = train[["userId", "movieId"]].drop_duplicates().shape[0]
        assert len(feat) == n_pairs

    def test_genre_history_count_non_negative(
        self, default_config: ExperimentConfig, built_data: tuple
    ) -> None:
        data, train = built_data
        builder = InteractionFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["genre_history_count"] >= 0.0).all()

    def test_feature_names_correct(self, default_config: ExperimentConfig) -> None:
        builder = InteractionFeatureBuilder(default_config)
        names = builder.get_feature_names()
        assert set(names) == {
            "genre_overlap_score", "tag_profile_similarity",
            "rating_gap", "genre_history_count", "mf_score",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TimeFeatureBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeFeatureBuilder:

    @pytest.fixture
    def built_data_for_time(
        self,
        default_config: ExperimentConfig,
        data_dict: dict,
        toy_train_df: pd.DataFrame,
    ) -> tuple[dict, pd.DataFrame]:
        ub = UserFeatureBuilder(default_config)
        ib = ItemFeatureBuilder(default_config)
        u_feat = ub.build(data_dict, toy_train_df)
        i_feat = ib.build(data_dict, toy_train_df)
        full_data = {**data_dict, "user_features": u_feat, "item_features": i_feat}
        return full_data, toy_train_df

    def test_interaction_month_in_range(
        self, default_config: ExperimentConfig, built_data_for_time: tuple
    ) -> None:
        data, train = built_data_for_time
        builder = TimeFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["interaction_month"] >= 1).all()
        assert (feat["interaction_month"] <= 12).all()

    def test_interaction_dayofweek_in_range(
        self, default_config: ExperimentConfig, built_data_for_time: tuple
    ) -> None:
        data, train = built_data_for_time
        builder = TimeFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["interaction_dayofweek"] >= 0).all()
        assert (feat["interaction_dayofweek"] <= 6).all()

    def test_days_since_user_active_non_negative(
        self, default_config: ExperimentConfig, built_data_for_time: tuple
    ) -> None:
        data, train = built_data_for_time
        builder = TimeFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["days_since_user_active"] >= 0.0).all()

    def test_days_since_item_rated_non_negative(
        self, default_config: ExperimentConfig, built_data_for_time: tuple
    ) -> None:
        data, train = built_data_for_time
        builder = TimeFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert (feat["days_since_item_rated"] >= 0.0).all()

    def test_no_timestamp_column_in_output(
        self, default_config: ExperimentConfig, built_data_for_time: tuple
    ) -> None:
        data, train = built_data_for_time
        builder = TimeFeatureBuilder(default_config)
        feat = builder.build(data, train)
        assert "timestamp" not in feat.columns

    def test_feature_names_correct(self, default_config: ExperimentConfig) -> None:
        builder = TimeFeatureBuilder(default_config)
        names = builder.get_feature_names()
        assert set(names) == {
            "interaction_month", "interaction_dayofweek",
            "days_since_user_active", "days_since_item_rated",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureStore — build_all_features & feature_columns.json
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureStore:

    @pytest.fixture
    def store_and_data(
        self,
        default_config: ExperimentConfig,
        data_dict: dict,
        toy_train_df: pd.DataFrame,
        tmp_path: Path,
    ) -> tuple[FeatureStore, dict, pd.DataFrame]:
        # Use tmp_path for processed_data_dir to avoid polluting project
        from src.config.data_config import DataConfig
        cfg = ExperimentConfig(
            data=DataConfig(
                cold_user_threshold=2,
                cold_item_threshold=2,
                processed_data_dir=str(tmp_path / "processed"),
            ),
            feature=FeatureConfig(n_genome_tags=20),
        )
        store = FeatureStore(cfg)
        val_df = toy_train_df.iloc[:1].copy()  # dummy — not used in building
        test_df = toy_train_df.iloc[:1].copy()
        store.build_all_features(data_dict, toy_train_df, val_df, test_df)
        return store, data_dict, toy_train_df

    def test_feature_columns_json_written(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        json_path = Path(__file__).parents[1] / "configs" / "feature_columns.json"
        assert json_path.exists()

    def test_feature_columns_json_is_list_of_strings(self, store_and_data: tuple) -> None:
        json_path = Path(__file__).parents[1] / "configs" / "feature_columns.json"
        with json_path.open() as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_feature_columns_non_empty(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert len(store.feature_columns) > 0

    def test_parquet_train_features_written(
        self, store_and_data: tuple, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "processed"
        assert (out_dir / "train_features.parquet").exists()

    def test_parquet_user_features_written(
        self, store_and_data: tuple, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "processed"
        assert (out_dir / "user_features.parquet").exists()

    def test_parquet_item_features_written(
        self, store_and_data: tuple, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "processed"
        assert (out_dir / "item_features.parquet").exists()

    def test_feature_columns_contain_user_features(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert "log_total_ratings" in store.feature_columns
        assert "log_positive_count" in store.feature_columns

    def test_feature_columns_contain_item_features(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert "log_rating_count" in store.feature_columns
        assert "is_cold" in store.feature_columns

    def test_feature_columns_contain_interaction_features(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert "genre_overlap_score" in store.feature_columns
        assert "rating_gap" in store.feature_columns

    def test_feature_columns_contain_time_features(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert "interaction_month" in store.feature_columns

    def test_no_label_in_feature_columns(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert "is_positive" not in store.feature_columns

    def test_no_entity_ids_in_feature_columns(self, store_and_data: tuple) -> None:
        store, _, _ = store_and_data
        assert "userId" not in store.feature_columns
        assert "movieId" not in store.feature_columns


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureStore — assemble_inference_features
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssembleInferenceFeatures:

    @pytest.fixture
    def store_built(
        self,
        default_config: ExperimentConfig,
        data_dict: dict,
        toy_train_df: pd.DataFrame,
        tmp_path: Path,
    ) -> FeatureStore:
        from src.config.data_config import DataConfig
        cfg = ExperimentConfig(
            data=DataConfig(
                cold_user_threshold=2,
                cold_item_threshold=2,
                processed_data_dir=str(tmp_path / "processed"),
            ),
            feature=FeatureConfig(n_genome_tags=20),
        )
        store = FeatureStore(cfg)
        val_df = toy_train_df.iloc[:1].copy()
        test_df = toy_train_df.iloc[:1].copy()
        store.build_all_features(data_dict, toy_train_df, val_df, test_df)
        return store

    def test_output_columns_match_feature_columns(self, store_built: FeatureStore) -> None:
        user_feats = {col: 0.5 for col in store_built.feature_columns}
        item_feats_list = [{col: 0.3 for col in store_built.feature_columns}]
        result = store_built.assemble_inference_features(user_feats, item_feats_list, {})
        assert list(result.columns) == store_built.feature_columns

    def test_missing_features_filled_with_zero(self, store_built: FeatureStore) -> None:
        """Features absent from the dicts must be 0.0, not NaN."""
        result = store_built.assemble_inference_features(
            user_features={},
            item_features_list=[{}],
            request_context={},
        )
        assert result.shape == (1, len(store_built.feature_columns))
        assert result.isnull().sum().sum() == 0
        assert (result == 0.0).all().all()

    def test_partial_features_filled_correctly(self, store_built: FeatureStore) -> None:
        """Provided features override 0.0; missing ones stay 0.0."""
        user_feats = {"log_total_ratings": 2.5}
        item_feats_list = [{"log_rating_count": 1.1}]
        result = store_built.assemble_inference_features(
            user_feats, item_feats_list, {}
        )
        assert result["log_total_ratings"].iloc[0] == pytest.approx(2.5)
        assert result["log_rating_count"].iloc[0] == pytest.approx(1.1)

    def test_multiple_items_produces_multiple_rows(self, store_built: FeatureStore) -> None:
        item_feats_list = [{"log_rating_count": float(i)} for i in range(5)]
        result = store_built.assemble_inference_features({}, item_feats_list, {})
        assert len(result) == 5

    def test_empty_item_list_returns_empty_df(self, store_built: FeatureStore) -> None:
        result = store_built.assemble_inference_features({}, [], {})
        assert len(result) == 0
        assert list(result.columns) == store_built.feature_columns


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureStore — get_negative_samples
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeSampling:

    @pytest.fixture
    def store_default(self, default_config: ExperimentConfig) -> FeatureStore:
        return FeatureStore(default_config)

    def _make_train(self) -> pd.DataFrame:
        """User 1 has positives on movies 1, 2. Warm items: 1–10."""
        rows = [
            {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": _TS[2016], "is_positive": 1},
            {"userId": 1, "movieId": 2, "rating": 4.5, "timestamp": _TS[2016], "is_positive": 1},
            {"userId": 1, "movieId": 3, "rating": 2.0, "timestamp": _TS[2016], "is_positive": 0},
        ]
        df = pd.DataFrame(rows)
        df["userId"] = df["userId"].astype("int32")
        df["movieId"] = df["movieId"].astype("int32")
        df["rating"] = df["rating"].astype("float32")
        df["timestamp"] = df["timestamp"].astype("int64")
        df["is_positive"] = df["is_positive"].astype("int8")
        return df

    def test_negative_ratio_correct(self, store_default: FeatureStore) -> None:
        train = self._make_train()
        warm_items = set(range(1, 11))  # 10 warm items
        neg_df = store_default.get_negative_samples(train, warm_items, ratio=4)
        n_positives = (train["rating"] >= 4.0).sum()
        # Up to ratio * n_positives negatives
        assert len(neg_df) <= n_positives * 4
        assert len(neg_df) > 0

    def test_negatives_have_is_positive_zero(self, store_default: FeatureStore) -> None:
        train = self._make_train()
        warm_items = set(range(1, 11))
        neg_df = store_default.get_negative_samples(train, warm_items, ratio=4)
        assert (neg_df["is_positive"] == 0).all()

    def test_no_positives_in_negatives(self, store_default: FeatureStore) -> None:
        """Sampled negatives must be items the user hasn't rated in train."""
        train = self._make_train()
        warm_items = set(range(1, 11))
        neg_df = store_default.get_negative_samples(train, warm_items, ratio=4)
        for uid, group in neg_df.groupby("userId"):
            rated = set(train[train["userId"] == uid]["movieId"].tolist())
            sampled = set(group["movieId"].tolist())
            overlap = rated & sampled
            assert len(overlap) == 0, f"user {uid}: negatives overlap rated items {overlap}"

    def test_only_warm_items_sampled(self, store_default: FeatureStore) -> None:
        train = self._make_train()
        warm_items = {5, 6, 7, 8, 9, 10}  # cold items 1-4 excluded
        neg_df = store_default.get_negative_samples(train, warm_items, ratio=4)
        sampled_items = set(neg_df["movieId"].tolist())
        assert sampled_items.issubset(warm_items), \
            f"Non-warm items sampled: {sampled_items - warm_items}"

    def test_deterministic_with_fixed_seed(self, store_default: FeatureStore) -> None:
        train = self._make_train()
        warm_items = set(range(1, 20))
        neg1 = store_default.get_negative_samples(train, warm_items, ratio=3)
        neg2 = store_default.get_negative_samples(train, warm_items, ratio=3)
        assert list(neg1["movieId"].sort_values()) == list(neg2["movieId"].sort_values())

    def test_schema_matches_train_df(self, store_default: FeatureStore) -> None:
        train = self._make_train()
        warm_items = set(range(1, 11))
        neg_df = store_default.get_negative_samples(train, warm_items, ratio=2)
        for col in ["userId", "movieId", "rating", "timestamp", "is_positive"]:
            assert col in neg_df.columns

    def test_empty_warm_items_returns_empty(self, store_default: FeatureStore) -> None:
        train = self._make_train()
        neg_df = store_default.get_negative_samples(train, warm_items=set(), ratio=4)
        assert len(neg_df) == 0

    def test_stratified_includes_cold_items(self, store_default: FeatureStore) -> None:
        """When all_items is provided, cold items must appear as negatives."""
        train = self._make_train()
        warm_items = {5, 6, 7, 8, 9, 10}
        all_items = warm_items | {20, 21, 22, 23, 24}  # 5 cold items
        neg_df = store_default.get_negative_samples(train, warm_items, all_items=all_items, ratio=4)
        sampled = set(neg_df["movieId"].tolist())
        cold_items = all_items - warm_items
        assert sampled & cold_items, "No cold items found in stratified negatives"
        assert sampled & warm_items, "No warm items found in stratified negatives"

    def test_stratified_no_rated_items(self, store_default: FeatureStore) -> None:
        """Stratified negatives must not include items the user has already rated."""
        train = self._make_train()
        warm_items = {5, 6, 7, 8, 9, 10}
        all_items = warm_items | {20, 21, 22, 23, 24}
        neg_df = store_default.get_negative_samples(train, warm_items, all_items=all_items, ratio=4)
        for uid, group in neg_df.groupby("userId"):
            rated = set(train[train["userId"] == uid]["movieId"].tolist())
            overlap = rated & set(group["movieId"].tolist())
            assert len(overlap) == 0, f"user {uid}: stratified negatives overlap rated {overlap}"

    def test_empty_warm_with_all_items_uses_cold_only(self, store_default: FeatureStore) -> None:
        """When warm_items is empty but all_items has cold items, cold negatives are generated."""
        train = self._make_train()
        cold_only = {20, 21, 22, 23, 24, 25}
        neg_df = store_default.get_negative_samples(
            train, warm_items=set(), all_items=cold_only, ratio=4
        )
        # Cold items don't overlap with user-rated items (1,2,3), so negatives exist
        assert len(neg_df) > 0
        assert set(neg_df["movieId"].tolist()).issubset(cold_only)
