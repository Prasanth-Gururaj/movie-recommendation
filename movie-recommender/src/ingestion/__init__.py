"""Public API for src.ingestion."""

from src.ingestion.cleaner import clean_all, clean_movies, clean_ratings
from src.ingestion.loader import load_all, load_movies, load_ratings, load_tags
from src.ingestion.splitter import (
    get_warm_items,
    get_warm_users,
    save_splits,
    split_ratings,
)

__all__ = [
    "clean_all",
    "clean_movies",
    "clean_ratings",
    "get_warm_items",
    "get_warm_users",
    "load_all",
    "load_movies",
    "load_ratings",
    "load_tags",
    "save_splits",
    "split_ratings",
]
