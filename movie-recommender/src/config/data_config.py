"""Data split and sampling configuration — all EDA-locked values."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    # ── time splits (LOCKED from EDA) ────────────────────────────────────────
    train_end_year: int = 2016
    val_year: int = 2017
    test_start_year: int = 2018

    # ── relevance (LOCKED from EDA) ──────────────────────────────────────────
    relevance_threshold: float = 4.0

    # ── cold thresholds (LOCKED from EDA) ────────────────────────────────────
    cold_user_threshold: int = 20
    cold_item_threshold: int = 10

    # ── sampling ─────────────────────────────────────────────────────────────
    negative_sample_ratio: int = 4   # 1:4 positive-to-negative ratio
    random_seed: int = 42

    # ── paths ─────────────────────────────────────────────────────────────────
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    def validate(self) -> None:
        assert self.train_end_year < self.val_year, (
            f"train_end_year ({self.train_end_year}) must be less than "
            f"val_year ({self.val_year})"
        )
        assert self.val_year < self.test_start_year, (
            f"val_year ({self.val_year}) must be less than "
            f"test_start_year ({self.test_start_year})"
        )
        assert 0.5 <= self.relevance_threshold <= 5.0, (
            f"relevance_threshold ({self.relevance_threshold}) must be between 0.5 and 5.0"
        )
        assert self.cold_user_threshold > 0, (
            f"cold_user_threshold ({self.cold_user_threshold}) must be > 0"
        )
        assert self.cold_item_threshold > 0, (
            f"cold_item_threshold ({self.cold_item_threshold}) must be > 0"
        )
        assert self.negative_sample_ratio > 0, (
            f"negative_sample_ratio ({self.negative_sample_ratio}) must be > 0"
        )
