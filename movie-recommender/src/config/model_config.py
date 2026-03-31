"""Ranker model configurations: XGBoost and LightGBM."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.base_config import BaseConfig


@dataclass
class XGBConfig(BaseConfig):
    model_type: str = "xgboost"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping: int = 50
    random_seed: int = 42
    objective: str = "rank:pairwise"

    def validate(self) -> None:
        assert 0.001 <= self.learning_rate <= 1.0, (
            f"learning_rate ({self.learning_rate}) must be between 0.001 and 1.0"
        )
        assert self.n_estimators > 0, (
            f"n_estimators ({self.n_estimators}) must be > 0"
        )
        assert self.max_depth > 0, (
            f"max_depth ({self.max_depth}) must be > 0"
        )
        assert 0.1 <= self.subsample <= 1.0, (
            f"subsample ({self.subsample}) must be between 0.1 and 1.0"
        )
        assert 0.1 <= self.colsample_bytree <= 1.0, (
            f"colsample_bytree ({self.colsample_bytree}) must be between 0.1 and 1.0"
        )


@dataclass
class LGBMConfig(BaseConfig):
    model_type: str = "lightgbm"
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping: int = 50
    lambdarank_truncation_level: int = 10
    random_seed: int = 42

    def validate(self) -> None:
        assert 0.001 <= self.learning_rate <= 1.0, (
            f"learning_rate ({self.learning_rate}) must be between 0.001 and 1.0"
        )
        assert self.n_estimators > 0, (
            f"n_estimators ({self.n_estimators}) must be > 0"
        )
        assert self.num_leaves > 0, (
            f"num_leaves ({self.num_leaves}) must be > 0"
        )
        assert 0.1 <= self.subsample <= 1.0, (
            f"subsample ({self.subsample}) must be between 0.1 and 1.0"
        )
        assert 0.1 <= self.colsample_bytree <= 1.0, (
            f"colsample_bytree ({self.colsample_bytree}) must be between 0.1 and 1.0"
        )
