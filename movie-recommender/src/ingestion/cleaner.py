"""Cleaning transformations for raw MovieLens data.

All genre names are loaded from ``configs/genre_columns.json`` — never hardcoded.
The reference year for ``movie_age`` is 2019 (last year of data in the dataset).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ── path constants ──────────────────────────────────────────────────────────
# src/ingestion/cleaner.py → two parents up → movie-recommender/ → configs/
_CONFIGS_DIR: Path = Path(__file__).parents[2] / "configs"
_GENRE_COLUMNS_PATH: Path = _CONFIGS_DIR / "genre_columns.json"

# Reference year for movie_age = reference_year - release_year (LOCKED from EDA)
_REFERENCE_YEAR: int = 2019

# Relevance threshold for positive label (LOCKED from EDA)
_RELEVANCE_THRESHOLD: float = 4.0

# Compiled regex for release year extraction
_YEAR_RE: re.Pattern[str] = re.compile(r"\((\d{4})\)$")


# ── individual cleaning functions ───────────────────────────────────────────

def parse_release_year(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Extract release year from movie title and compute movie_age.

    - Searches for the pattern ``(YYYY)`` at the end of each title.
    - Fills the 412 missing years (confirmed in EDA) with the median year.
    - Adds ``release_year`` (int) and ``movie_age`` = 2019 − release_year.
    """
    df = movies_df.copy()

    def _extract(title: Any) -> int | None:
        m = _YEAR_RE.search(str(title).strip())
        return int(m.group(1)) if m else None

    raw_years: pd.Series = df["title"].apply(_extract)
    median_year = int(raw_years.dropna().median())
    df["release_year"] = raw_years.fillna(median_year).astype(int)
    df["movie_age"] = _REFERENCE_YEAR - df["release_year"]
    return df


def build_genre_vector(
    movies_df: pd.DataFrame,
    genre_columns: list[str],
) -> pd.DataFrame:
    """Create binary genre indicator columns and a ``has_genre`` flag.

    Parameters
    ----------
    movies_df:
        DataFrame that must contain a ``genres`` column with pipe-separated
        genre strings (e.g. ``"Action|Adventure"``).
    genre_columns:
        Ordered list of genre names to use — typically loaded from
        ``configs/genre_columns.json``.  IMAX must NOT be in this list.

    Returns
    -------
    DataFrame with one binary column per genre plus ``has_genre``.
    """
    df = movies_df.copy()

    # Split once and cache as sets for O(1) membership testing
    genre_sets: list[set[str]] = [
        set(str(g).split("|")) for g in df["genres"]
    ]

    for genre in genre_columns:
        df[genre] = [1 if genre in gs else 0 for gs in genre_sets]

    # has_genre = 0 only when genres is the sentinel "(no genres listed)"
    df["has_genre"] = (df["genres"] != "(no genres listed)").astype(int)

    return df


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all movie cleaning steps.

    1. Parse release year and compute movie_age.
    2. Build 18-column binary genre vector (list read from genre_columns.json).

    Returns the enriched DataFrame.
    """
    with _GENRE_COLUMNS_PATH.open("r", encoding="utf-8") as fh:
        genre_columns: list[str] = json.load(fh)

    df = parse_release_year(movies_df)
    df = build_genre_vector(df, genre_columns)
    return df


def clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_positive`` label and ``rated_at`` datetime column.

    - ``is_positive``: 1 if ``rating >= 4.0`` else 0 (int8).
    - ``rated_at``: UTC datetime derived from Unix timestamp.
    - No rows are dropped — ratings are clean per EDA.
    """
    df = ratings_df.copy()
    df["is_positive"] = (df["rating"] >= _RELEVANCE_THRESHOLD).astype("int8")
    df["rated_at"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def clean_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply all cleaning functions and return a cleaned dict.

    Passthrough keys: ``"tags"``, ``"genome_scores"``, ``"genome_tags"``.
    """
    return {
        "ratings": clean_ratings(data["ratings"]),
        "movies": clean_movies(data["movies"]),
        "tags": data["tags"],
        "genome_scores": data["genome_scores"],
        "genome_tags": data["genome_tags"],
    }
