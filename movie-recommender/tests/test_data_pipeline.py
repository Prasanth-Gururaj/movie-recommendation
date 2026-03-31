"""Tests for src/ingestion: loader, cleaner, splitter.

All tests use toy in-memory DataFrames (or small temp CSV files).
The actual 25 M-row files are never loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.data_config import DataConfig
from src.ingestion.cleaner import (
    build_genre_vector,
    clean_all,
    clean_movies,
    clean_ratings,
    parse_release_year,
)
from src.ingestion.loader import (
    load_genome_scores,
    load_genome_tags,
    load_movies,
    load_ratings,
    load_tags,
)
from src.ingestion.splitter import (
    get_warm_items,
    get_warm_users,
    save_splits,
    split_ratings,
)

# ── shared genre list (mirrors configs/genre_columns.json) ──────────────────
_GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# ── Unix timestamps for specific years (UTC midnight Jan 1) ─────────────────
_TS = {
    2015: 1_420_070_400,
    2016: 1_451_606_400,
    2017: 1_483_228_800,
    2018: 1_514_764_800,
    2019: 1_546_300_800,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def toy_movies_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.array([1, 2, 3, 4, 5], dtype="int32"),
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Unknown Movie",           # no year in title
                "The Matrix (1999)",
                "No Genre Movie (2000)",
            ],
            "genres": [
                "Animation|Children|Comedy",
                "Adventure|Children|Fantasy",
                "Drama",
                "Action|Sci-Fi",
                "(no genres listed)",
            ],
        }
    )


@pytest.fixture
def toy_ratings_df() -> pd.DataFrame:
    """Ratings spanning 2015–2018 with a mix of positive / negative scores."""
    return pd.DataFrame(
        {
            "userId": pd.array([1, 1, 2, 2, 3, 3, 4, 4], dtype="int32"),
            "movieId": pd.array([10, 20, 10, 30, 20, 30, 10, 40], dtype="int32"),
            "rating": pd.array([5.0, 3.0, 4.0, 2.0, 4.5, 3.9, 1.0, 4.0], dtype="float32"),
            "timestamp": pd.array(
                [
                    _TS[2015],  # user 1, movie 10 → train
                    _TS[2016],  # user 1, movie 20 → train
                    _TS[2017],  # user 2, movie 10 → val
                    _TS[2017],  # user 2, movie 30 → val
                    _TS[2018],  # user 3, movie 20 → test
                    _TS[2018],  # user 3, movie 30 → test
                    _TS[2015],  # user 4, movie 10 → train
                    _TS[2016],  # user 4, movie 40 → train
                ],
                dtype="int64",
            ),
        }
    )


@pytest.fixture
def data_config_small() -> DataConfig:
    """DataConfig with low cold thresholds suitable for toy data."""
    return DataConfig(
        cold_user_threshold=1,
        cold_item_threshold=1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Loader tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoader:
    """Tests for loader.py — uses temp CSV files instead of the real 25M dataset."""

    def _make_ratings_csv(self, tmp_path: Path) -> Path:
        p = tmp_path / "ratings.csv"
        p.write_text(
            "userId,movieId,rating,timestamp\n"
            "1,10,5.0,1420070400\n"
            "2,20,3.5,1451606400\n",
            encoding="utf-8",
        )
        return tmp_path

    def _make_movies_csv(self, tmp_path: Path) -> Path:
        p = tmp_path / "movies.csv"
        p.write_text(
            "movieId,title,genres\n"
            "10,Toy Story (1995),Animation|Children|Comedy\n"
            "20,Jumanji (1995),Adventure|Children|Fantasy\n",
            encoding="utf-8",
        )
        return tmp_path

    def _make_tags_csv(self, tmp_path: Path, include_null: bool = False) -> Path:
        p = tmp_path / "tags.csv"
        rows = "userId,movieId,tag,timestamp\n1,10,fun,1420070400\n2,20,classic,1451606400\n"
        if include_null:
            rows += "3,20,,1483228800\n"  # null tag
        p.write_text(rows, encoding="utf-8")
        return tmp_path

    def _make_genome_scores_csv(self, tmp_path: Path) -> Path:
        p = tmp_path / "genome-scores.csv"
        p.write_text(
            "movieId,tagId,relevance\n10,1,0.95\n10,2,0.30\n",
            encoding="utf-8",
        )
        return tmp_path

    def _make_genome_tags_csv(self, tmp_path: Path) -> Path:
        p = tmp_path / "genome-tags.csv"
        p.write_text("tagId,tag\n1,funny\n2,dark\n", encoding="utf-8")
        return tmp_path

    # ── ratings ──────────────────────────────────────────────────────────────

    def test_ratings_dtypes(self, tmp_path: Path) -> None:
        data_dir = str(self._make_ratings_csv(tmp_path))
        df = load_ratings(data_dir)
        assert df["userId"].dtype == "int32"
        assert df["movieId"].dtype == "int32"
        assert df["rating"].dtype == "float32"
        assert df["timestamp"].dtype == "int64"

    def test_ratings_columns(self, tmp_path: Path) -> None:
        df = load_ratings(str(self._make_ratings_csv(tmp_path)))
        assert list(df.columns) == ["userId", "movieId", "rating", "timestamp"]

    def test_ratings_row_count(self, tmp_path: Path) -> None:
        df = load_ratings(str(self._make_ratings_csv(tmp_path)))
        assert len(df) == 2

    # ── movies ───────────────────────────────────────────────────────────────

    def test_movies_dtypes(self, tmp_path: Path) -> None:
        df = load_movies(str(self._make_movies_csv(tmp_path)))
        assert df["movieId"].dtype == "int32"

    def test_movies_columns(self, tmp_path: Path) -> None:
        df = load_movies(str(self._make_movies_csv(tmp_path)))
        assert list(df.columns) == ["movieId", "title", "genres"]

    # ── tags ─────────────────────────────────────────────────────────────────

    def test_tags_columns(self, tmp_path: Path) -> None:
        df = load_tags(str(self._make_tags_csv(tmp_path)))
        assert list(df.columns) == ["userId", "movieId", "tag", "timestamp"]

    def test_tags_dtypes(self, tmp_path: Path) -> None:
        df = load_tags(str(self._make_tags_csv(tmp_path)))
        assert df["userId"].dtype == "int32"
        assert df["movieId"].dtype == "int32"
        assert df["timestamp"].dtype == "int64"

    def test_tags_null_rows_dropped(self, tmp_path: Path) -> None:
        df = load_tags(str(self._make_tags_csv(tmp_path, include_null=True)))
        assert df["tag"].isna().sum() == 0
        assert len(df) == 2  # 3 rows written, 1 null dropped

    def test_tags_no_null_rows_unchanged(self, tmp_path: Path) -> None:
        df = load_tags(str(self._make_tags_csv(tmp_path, include_null=False)))
        assert len(df) == 2

    # ── genome scores ─────────────────────────────────────────────────────────

    def test_genome_scores_dtypes(self, tmp_path: Path) -> None:
        df = load_genome_scores(str(self._make_genome_scores_csv(tmp_path)))
        assert df["movieId"].dtype == "int32"
        assert df["tagId"].dtype == "int32"
        assert df["relevance"].dtype == "float32"

    def test_genome_scores_columns(self, tmp_path: Path) -> None:
        df = load_genome_scores(str(self._make_genome_scores_csv(tmp_path)))
        assert list(df.columns) == ["movieId", "tagId", "relevance"]

    # ── genome tags ──────────────────────────────────────────────────────────

    def test_genome_tags_dtypes(self, tmp_path: Path) -> None:
        df = load_genome_tags(str(self._make_genome_tags_csv(tmp_path)))
        assert df["tagId"].dtype == "int32"

    def test_genome_tags_columns(self, tmp_path: Path) -> None:
        df = load_genome_tags(str(self._make_genome_tags_csv(tmp_path)))
        assert list(df.columns) == ["tagId", "tag"]


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaner — parse_release_year
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseReleaseYear:
    def test_year_extracted_correctly(self, toy_movies_df: pd.DataFrame) -> None:
        df = parse_release_year(toy_movies_df)
        assert df.loc[df["movieId"] == 1, "release_year"].iloc[0] == 1995
        assert df.loc[df["movieId"] == 4, "release_year"].iloc[0] == 1999

    def test_missing_year_filled_with_median(self, toy_movies_df: pd.DataFrame) -> None:
        df = parse_release_year(toy_movies_df)
        # movieId=3 has no year in title → filled with median
        # years present: 1995, 1995, 1999, 2000 → median = (1995+1999)/2 = 1997
        filled_year = df.loc[df["movieId"] == 3, "release_year"].iloc[0]
        known_years = [1995, 1995, 1999, 2000]
        expected_median = int(pd.Series(known_years).median())
        assert filled_year == expected_median

    def test_no_missing_years_remain(self, toy_movies_df: pd.DataFrame) -> None:
        df = parse_release_year(toy_movies_df)
        assert df["release_year"].isna().sum() == 0

    def test_movie_age_computed(self, toy_movies_df: pd.DataFrame) -> None:
        df = parse_release_year(toy_movies_df)
        assert df.loc[df["movieId"] == 1, "movie_age"].iloc[0] == 2019 - 1995

    def test_release_year_is_int(self, toy_movies_df: pd.DataFrame) -> None:
        df = parse_release_year(toy_movies_df)
        assert df["release_year"].dtype in (int, "int64", "int32")

    def test_original_df_not_mutated(self, toy_movies_df: pd.DataFrame) -> None:
        original_cols = list(toy_movies_df.columns)
        parse_release_year(toy_movies_df)
        assert list(toy_movies_df.columns) == original_cols


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaner — build_genre_vector
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildGenreVector:
    def test_exactly_18_genre_columns(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        genre_cols = [c for c in df.columns if c in _GENRES]
        assert len(genre_cols) == 18

    def test_binary_values_only(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        for col in _GENRES:
            assert set(df[col].unique()).issubset({0, 1}), f"{col} has non-binary values"

    def test_imax_not_in_columns(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        assert "IMAX" not in df.columns

    def test_genre_assignment_correct(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        # movieId=1: Animation|Children|Comedy
        row = df[df["movieId"] == 1].iloc[0]
        assert row["Animation"] == 1
        assert row["Children"] == 1
        assert row["Comedy"] == 1
        assert row["Action"] == 0
        assert row["Drama"] == 0

    def test_has_genre_flag_present(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        assert "has_genre" in df.columns

    def test_has_genre_zero_for_no_genre_listed(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        no_genre_row = df[df["movieId"] == 5].iloc[0]
        assert no_genre_row["has_genre"] == 0

    def test_has_genre_one_for_movies_with_genres(self, toy_movies_df: pd.DataFrame) -> None:
        df = build_genre_vector(toy_movies_df, _GENRES)
        for mid in [1, 2, 3, 4]:
            assert df[df["movieId"] == mid].iloc[0]["has_genre"] == 1

    def test_matches_genre_columns_json(self, toy_movies_df: pd.DataFrame) -> None:
        """Genre columns used in tests must match the project JSON exactly."""
        json_path = Path(__file__).parents[1] / "configs" / "genre_columns.json"
        with json_path.open() as f:
            json_genres = json.load(f)
        assert json_genres == _GENRES


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaner — clean_ratings
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanRatings:
    def test_is_positive_boundary_below(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        # rating = 3.9 → is_positive = 0; use range filter for float32 safety
        row = df[(df["rating"] > 3.8) & (df["rating"] < 4.0)]
        assert len(row) == 1
        assert row["is_positive"].iloc[0] == 0

    def test_is_positive_boundary_at_threshold(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        # rating = 4.0 → is_positive = 1
        rows = df[df["rating"] == pytest.approx(4.0)]
        assert (rows["is_positive"] == 1).all()

    def test_is_positive_above_threshold(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        rows = df[df["rating"] == pytest.approx(5.0)]
        assert (rows["is_positive"] == 1).all()

    def test_is_positive_below_threshold(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        rows = df[df["rating"] == pytest.approx(2.0)]
        assert (rows["is_positive"] == 0).all()

    def test_rated_at_column_added(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        assert "rated_at" in df.columns

    def test_rated_at_is_datetime(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        assert pd.api.types.is_datetime64_any_dtype(df["rated_at"])

    def test_no_rows_dropped(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        assert len(df) == len(toy_ratings_df)

    def test_is_positive_dtype_int8(self, toy_ratings_df: pd.DataFrame) -> None:
        df = clean_ratings(toy_ratings_df)
        assert df["is_positive"].dtype == "int8"

    def test_original_df_not_mutated(self, toy_ratings_df: pd.DataFrame) -> None:
        original_cols = set(toy_ratings_df.columns)
        clean_ratings(toy_ratings_df)
        assert set(toy_ratings_df.columns) == original_cols


# ═══════════════════════════════════════════════════════════════════════════════
# Cleaner — clean_movies (integration: parse_year + genre_vector)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanMovies:
    def test_all_new_columns_present(self, toy_movies_df: pd.DataFrame) -> None:
        df = clean_movies(toy_movies_df)
        assert "release_year" in df.columns
        assert "movie_age" in df.columns
        assert "has_genre" in df.columns
        for genre in _GENRES:
            assert genre in df.columns

    def test_exactly_18_genre_columns(self, toy_movies_df: pd.DataFrame) -> None:
        df = clean_movies(toy_movies_df)
        genre_cols = [c for c in df.columns if c in _GENRES]
        assert len(genre_cols) == 18


# ═══════════════════════════════════════════════════════════════════════════════
# Splitter — split_ratings
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitRatings:
    def test_train_year_range(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        train, _, _ = split_ratings(toy_ratings_df, data_config_small)
        years = pd.to_datetime(train["timestamp"], unit="s").dt.year
        assert (years <= 2016).all()

    def test_val_year_range(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        _, val, _ = split_ratings(toy_ratings_df, data_config_small)
        years = pd.to_datetime(val["timestamp"], unit="s").dt.year
        assert (years == 2017).all()

    def test_test_year_range(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        _, _, test = split_ratings(toy_ratings_df, data_config_small)
        years = pd.to_datetime(test["timestamp"], unit="s").dt.year
        assert (years >= 2018).all()

    def test_no_val_rows_in_train(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        """Critical leakage check: no val-year timestamp appears in train."""
        train, val, _ = split_ratings(toy_ratings_df, data_config_small)
        train_ts = set(train["timestamp"].tolist())
        val_ts = set(val["timestamp"].tolist())
        # Since timestamps are distinct per year in toy data, sets must not overlap
        # More robust: check year membership
        train_years = set(pd.to_datetime(train["timestamp"], unit="s").dt.year.tolist())
        assert data_config_small.val_year not in train_years

    def test_no_test_rows_in_train(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        """Critical leakage check: no test-year timestamp appears in train."""
        train, _, _ = split_ratings(toy_ratings_df, data_config_small)
        train_years = set(pd.to_datetime(train["timestamp"], unit="s").dt.year.tolist())
        assert data_config_small.test_start_year not in train_years

    def test_exhaustive_row_coverage(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        """Every row in the input must appear in exactly one split."""
        train, val, test = split_ratings(toy_ratings_df, data_config_small)
        total = len(train) + len(val) + len(test)
        assert total == len(toy_ratings_df)

    def test_all_columns_retained(
        self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig
    ) -> None:
        train, val, test = split_ratings(toy_ratings_df, data_config_small)
        for split in (train, val, test):
            assert set(split.columns) == set(toy_ratings_df.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Splitter — get_warm_users / get_warm_items
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarmSets:
    @pytest.fixture
    def cleaned_train(self) -> pd.DataFrame:
        """A toy train split with known warm users and warm items."""
        # user 1: 3 positives (movies 10, 20, 30)
        # user 2: 1 positive  (movie 10)
        # item 10: 3 ratings, item 20: 2 ratings, item 30: 1 rating, item 40: 1 rating
        return pd.DataFrame(
            {
                "userId": pd.array([1, 1, 1, 2, 2], dtype="int32"),
                "movieId": pd.array([10, 20, 30, 10, 20], dtype="int32"),
                "rating": pd.array([5.0, 4.0, 4.5, 3.0, 3.0], dtype="float32"),
                "timestamp": pd.array([_TS[2015]] * 5, dtype="int64"),
                "is_positive": pd.array([1, 1, 1, 0, 0], dtype="int8"),
            }
        )

    def test_warm_users_meet_threshold(self, cleaned_train: pd.DataFrame) -> None:
        cfg = DataConfig(cold_user_threshold=3, cold_item_threshold=1)
        warm = get_warm_users(cleaned_train, cfg)
        # user 1 has 3 positives → warm; user 2 has 0 positives → cold
        assert 1 in warm
        assert 2 not in warm

    def test_warm_users_all_have_enough_positives(self, cleaned_train: pd.DataFrame) -> None:
        cfg = DataConfig(cold_user_threshold=2, cold_item_threshold=1)
        warm = get_warm_users(cleaned_train, cfg)
        positives = cleaned_train[cleaned_train["is_positive"] == 1]
        user_counts = positives.groupby("userId")["is_positive"].count()
        for uid in warm:
            assert user_counts.get(uid, 0) >= cfg.cold_user_threshold

    def test_warm_items_meet_threshold(self, cleaned_train: pd.DataFrame) -> None:
        cfg = DataConfig(cold_user_threshold=1, cold_item_threshold=2)
        warm = get_warm_items(cleaned_train, cfg)
        # item 10: 2 ratings, item 20: 2 ratings → both warm
        # item 30: 1 rating → cold
        assert 10 in warm
        assert 20 in warm
        assert 30 not in warm

    def test_warm_items_all_have_enough_ratings(self, cleaned_train: pd.DataFrame) -> None:
        cfg = DataConfig(cold_user_threshold=1, cold_item_threshold=2)
        warm = get_warm_items(cleaned_train, cfg)
        counts = cleaned_train.groupby("movieId")["rating"].count()
        for mid in warm:
            assert counts.get(mid, 0) >= cfg.cold_item_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Splitter — save_splits
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveSplits:
    @pytest.fixture
    def split_data(self, toy_ratings_df: pd.DataFrame, data_config_small: DataConfig):
        cleaned = clean_ratings(toy_ratings_df)
        train, val, test = split_ratings(cleaned, data_config_small)
        return train, val, test, data_config_small

    def test_parquet_files_created(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        assert (tmp_path / "train_pairs.parquet").exists()
        assert (tmp_path / "val_pairs.parquet").exists()
        assert (tmp_path / "test_pairs.parquet").exists()

    def test_metadata_file_created(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        assert (tmp_path / "splits_metadata.json").exists()

    def test_metadata_has_required_keys(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        required = {
            "row_counts", "date_ranges", "warm_user_count",
            "warm_item_count", "checksums", "data_config",
        }
        assert required.issubset(set(meta.keys()))

    def test_row_counts_correct(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        assert meta["row_counts"]["train"] == len(train)
        assert meta["row_counts"]["val"] == len(val)
        assert meta["row_counts"]["test"] == len(test)

    def test_checksums_are_valid_sha256(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        for name, digest in meta["checksums"].items():
            # SHA-256 hex digest is always 64 hex chars
            assert len(digest) == 64, f"checksum for {name!r} is not 64 chars"
            assert all(c in "0123456789abcdef" for c in digest)

    def test_checksums_match_files(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        for name, digest in meta["checksums"].items():
            p = tmp_path / f"{name}_pairs.parquet"
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            assert h.hexdigest() == digest, f"checksum mismatch for {name}"

    def test_parquet_readable_with_pyarrow(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        loaded_train = pd.read_parquet(tmp_path / "train_pairs.parquet", engine="pyarrow")
        assert len(loaded_train) == len(train)

    def test_date_ranges_in_metadata(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        for split_name in ("train", "val", "test"):
            assert "min" in meta["date_ranges"][split_name]
            assert "max" in meta["date_ranges"][split_name]

    def test_data_config_in_metadata(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        assert "train_end_year" in meta["data_config"]
        assert meta["data_config"]["train_end_year"] == 2016

    def test_warm_counts_in_metadata(
        self, split_data: tuple, tmp_path: Path
    ) -> None:
        train, val, test, cfg = split_data
        save_splits(train, val, test, str(tmp_path), cfg)
        with (tmp_path / "splits_metadata.json").open() as f:
            meta = json.load(f)
        assert "warm_user_count" in meta
        assert "warm_item_count" in meta
        assert isinstance(meta["warm_user_count"], int)
        assert isinstance(meta["warm_item_count"], int)


# ═══════════════════════════════════════════════════════════════════════════════
# Genre vector — integration with configs/genre_columns.json
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenreVectorIntegration:
    def test_genre_columns_json_has_18_entries(self) -> None:
        json_path = Path(__file__).parents[1] / "configs" / "genre_columns.json"
        with json_path.open() as f:
            genres = json.load(f)
        assert len(genres) == 18

    def test_imax_not_in_json(self) -> None:
        json_path = Path(__file__).parents[1] / "configs" / "genre_columns.json"
        with json_path.open() as f:
            genres = json.load(f)
        assert "IMAX" not in genres

    def test_genre_vector_binary_from_json(self, toy_movies_df: pd.DataFrame) -> None:
        df = clean_movies(toy_movies_df)
        json_path = Path(__file__).parents[1] / "configs" / "genre_columns.json"
        with json_path.open() as f:
            genres = json.load(f)
        for col in genres:
            assert set(df[col].unique()).issubset({0, 1})

    def test_sci_fi_column_correct(self, toy_movies_df: pd.DataFrame) -> None:
        df = clean_movies(toy_movies_df)
        # movieId=4 is "Action|Sci-Fi"
        row = df[df["movieId"] == 4].iloc[0]
        assert row["Sci-Fi"] == 1
        assert row["Action"] == 1
        assert row["Drama"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# clean_all integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanAll:
    def test_clean_all_returns_dict(
        self, toy_ratings_df: pd.DataFrame, toy_movies_df: pd.DataFrame
    ) -> None:
        genome_scores = pd.DataFrame(
            {"movieId": pd.array([1], dtype="int32"),
             "tagId": pd.array([1], dtype="int32"),
             "relevance": pd.array([0.5], dtype="float32")}
        )
        genome_tags = pd.DataFrame(
            {"tagId": pd.array([1], dtype="int32"), "tag": ["funny"]}
        )
        tags = pd.DataFrame(
            {"userId": pd.array([1], dtype="int32"),
             "movieId": pd.array([1], dtype="int32"),
             "tag": ["fun"],
             "timestamp": pd.array([_TS[2016]], dtype="int64")}
        )
        data = {
            "ratings": toy_ratings_df,
            "movies": toy_movies_df,
            "tags": tags,
            "genome_scores": genome_scores,
            "genome_tags": genome_tags,
        }
        cleaned = clean_all(data)
        assert set(cleaned.keys()) == {"ratings", "movies", "tags", "genome_scores", "genome_tags"}
        assert "is_positive" in cleaned["ratings"].columns
        assert "release_year" in cleaned["movies"].columns
