"""Temporal train / val / test splitting and metadata persistence.

Split logic (LOCKED from EDA):
  train : year <= 2016
  val   : year == 2017
  test  : year >= 2018

Year is always extracted from the Unix ``timestamp`` column via
``pd.to_datetime(timestamp, unit='s').dt.year`` — never from ``rated_at``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config.data_config import DataConfig


# ── public splitting functions ──────────────────────────────────────────────

def split_ratings(
    ratings_df: pd.DataFrame,
    data_config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split into train, val, test using year from Unix timestamp.

    Guarantees no val / test rows leak into train.

    Parameters
    ----------
    ratings_df:
        Full ratings DataFrame containing at minimum ``timestamp`` (int64).
    data_config:
        Provides ``train_end_year``, ``val_year``, ``test_start_year``.

    Returns
    -------
    (train, val, test) — each a copy with all original columns retained.
    """
    year: pd.Series = pd.to_datetime(ratings_df["timestamp"], unit="s").dt.year

    train = ratings_df[year <= data_config.train_end_year].copy()
    val   = ratings_df[year == data_config.val_year].copy()
    test  = ratings_df[year >= data_config.test_start_year].copy()

    return train, val, test


def get_warm_users(
    train_df: pd.DataFrame,
    data_config: DataConfig,
) -> set[int]:
    """Return user IDs with >= ``cold_user_threshold`` positive ratings in train.

    Requires an ``is_positive`` column (added by ``clean_ratings``).
    """
    positives = train_df[train_df["is_positive"] == 1]
    counts: pd.Series = positives.groupby("userId")["is_positive"].count()
    warm_mask = counts >= data_config.cold_user_threshold
    return set(counts[warm_mask].index.tolist())


def get_warm_items(
    train_df: pd.DataFrame,
    data_config: DataConfig,
) -> set[int]:
    """Return item IDs with >= ``cold_item_threshold`` total ratings in train."""
    counts: pd.Series = train_df.groupby("movieId")["rating"].count()
    warm_mask = counts >= data_config.cold_item_threshold
    return set(counts[warm_mask].index.tolist())


# ── persistence ─────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _timestamp_date_range(df: pd.DataFrame) -> dict[str, str]:
    """Return {min, max} date strings derived from the ``timestamp`` column."""
    dt = pd.to_datetime(df["timestamp"], unit="s")
    return {"min": str(dt.min().date()), "max": str(dt.max().date())}


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: str,
    data_config: DataConfig,
) -> None:
    """Persist train / val / test as Parquet files plus a JSON metadata file.

    Output files (all written to ``processed_dir``):
      - ``train_pairs.parquet``
      - ``val_pairs.parquet``
      - ``test_pairs.parquet``
      - ``splits_metadata.json``

    Metadata fields:
      - ``row_counts``       : rows in each split
      - ``date_ranges``      : {min, max} date per split (from timestamp)
      - ``warm_user_count``  : users with >= cold_user_threshold positives in train
      - ``warm_item_count``  : items with >= cold_item_threshold ratings in train
      - ``checksums``        : SHA-256 hex digest of each parquet file
      - ``data_config``      : flat dict of DataConfig values used
    """
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits: dict[str, pd.DataFrame] = {"train": train, "val": val, "test": test}
    parquet_paths: dict[str, Path] = {}

    for name, df in splits.items():
        p = out / f"{name}_pairs.parquet"
        df.to_parquet(p, engine="pyarrow", index=False)
        parquet_paths[name] = p

    warm_users = get_warm_users(train, data_config)
    warm_items = get_warm_items(train, data_config)

    metadata: dict = {
        "row_counts": {name: len(df) for name, df in splits.items()},
        "date_ranges": {
            name: _timestamp_date_range(df) for name, df in splits.items()
        },
        "warm_user_count": len(warm_users),
        "warm_item_count": len(warm_items),
        "checksums": {
            name: _sha256_file(parquet_paths[name]) for name in splits
        },
        "data_config": data_config.to_dict(),
    }

    meta_path = out / "splits_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
