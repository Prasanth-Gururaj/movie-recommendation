"""Raw data loaders for the MovieLens 25M dataset.

Each loader enforces explicit dtypes matching the EDA schema.
``load_all`` returns a keyed dict and logs shape of every frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_ratings(data_dir: str) -> pd.DataFrame:
    """Load ratings.csv with enforced dtypes.

    Columns: userId (int32), movieId (int32), rating (float32), timestamp (int64).
    """
    path = Path(data_dir) / "ratings.csv"
    df = pd.read_csv(
        path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    logger.info("ratings loaded — shape: %s", df.shape)
    return df


def load_movies(data_dir: str) -> pd.DataFrame:
    """Load movies.csv with enforced dtypes.

    Columns: movieId (int32), title (str), genres (str).
    """
    path = Path(data_dir) / "movies.csv"
    df = pd.read_csv(
        path,
        dtype={"movieId": "int32", "title": str, "genres": str},
    )
    logger.info("movies loaded — shape: %s", df.shape)
    return df


def load_tags(data_dir: str) -> pd.DataFrame:
    """Load tags.csv, drop the 16 null-tag rows confirmed in EDA.

    Columns: userId (int32), movieId (int32), tag (str), timestamp (int64).
    """
    path = Path(data_dir) / "tags.csv"
    df = pd.read_csv(
        path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "timestamp": "int64",
        },
    )
    n_before = len(df)
    df = df.dropna(subset=["tag"]).reset_index(drop=True)
    dropped = n_before - len(df)
    logger.info("tags loaded — dropped %d null-tag rows — shape: %s", dropped, df.shape)
    return df


def load_genome_scores(data_dir: str) -> pd.DataFrame:
    """Load genome-scores.csv with enforced dtypes.

    Columns: movieId (int32), tagId (int32), relevance (float32).
    """
    path = Path(data_dir) / "genome-scores.csv"
    df = pd.read_csv(
        path,
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    )
    logger.info("genome_scores loaded — shape: %s", df.shape)
    return df


def load_genome_tags(data_dir: str) -> pd.DataFrame:
    """Load genome-tags.csv with enforced dtypes.

    Columns: tagId (int32), tag (str).
    """
    path = Path(data_dir) / "genome-tags.csv"
    df = pd.read_csv(
        path,
        dtype={"tagId": "int32", "tag": str},
    )
    logger.info("genome_tags loaded — shape: %s", df.shape)
    return df


def load_all(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load all five required files and return a keyed dict.

    Keys: ``"ratings"``, ``"movies"``, ``"tags"``,
    ``"genome_scores"``, ``"genome_tags"``.

    ``links.csv`` is intentionally excluded — not used in this pipeline.
    """
    return {
        "ratings": load_ratings(data_dir),
        "movies": load_movies(data_dir),
        "tags": load_tags(data_dir),
        "genome_scores": load_genome_scores(data_dir),
        "genome_tags": load_genome_tags(data_dir),
    }
